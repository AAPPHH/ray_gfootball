import os
import sys
import json
import math
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "8"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFIG = {
    "updates": 200,
    "timeout_s": 3600,
    "first_get_timeout_s": 300,
    "driver_threads": 8,
    "out_dir": str(HERE.parent / "smoke_out"),
    "ray": {"address": "auto", "namespace": None, "worker": {"num_cpus": 1, "num_gpus": 0, "memory": 2 * 1024 ** 3}},
    "learner": {"num_workers": 8, "rollout_len": 128, "batch_size": 8, "goal_traj_len": 64, "golden_capacity": 256},
    "warmstart": None,
    "expert_parquet": None,
}


class SmokeDone(Exception):
    pass


class FirstCallTimeout:
    def __init__(self, ray, timeout, state):
        self._ray, self._timeout, self._state = ray, timeout, state

    def __getattr__(self, name):
        return getattr(self._ray, name)

    def get(self, refs, timeout=None):
        if timeout is None and not self._state["first_get"]:
            timeout = self._timeout
        out = self._ray.get(refs, timeout=timeout)
        self._state["first_get"] = True
        return out

    def wait(self, refs, **kw):
        if not self._state["first_wait"]:
            kw.setdefault("timeout", self._timeout)
        ready, rest = self._ray.wait(refs, **kw)
        if not self._state["first_wait"] and not ready:
            raise TimeoutError(f"no rollout within {self._timeout}s")
        self._state["first_wait"] = True
        return ready, rest


def make_learner_class(fl, state):
    class SmokeLearner(fl.LeagueLearner):
        def _update_sil_gpu(self, batch):
            state["sil_calls"] += 1
            return super()._update_sil_gpu(batch)

        def _update(self, rollouts):
            stats = super()._update(rollouts)
            state["done"] += 1
            state["finite"] = state["finite"] and all(math.isfinite(float(v)) for v in stats.values())
            state["last"] = {k: float(v) for k, v in stats.items()}
            if state["done"] >= CONFIG["updates"]:
                raise SmokeDone
            return stats

    return SmokeLearner


def run():
    import ray
    import torch
    import impala_dreamer
    import full_league as fl
    out = Path(CONFIG["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    state = {"done": 0, "finite": True, "last": None, "sil_calls": 0, "first_get": False, "first_wait": False}
    torch.set_num_threads(CONFIG["driver_threads"])
    summary = {"updates_requested": CONFIG["updates"], "workers": CONFIG["learner"]["num_workers"], "ok": False, "error": None}
    t0 = time.time()
    fl.CONFIG["ray"].update({"address": CONFIG["ray"]["address"], "worker": CONFIG["ray"]["worker"]})
    ray.init(address=CONFIG["ray"]["address"], namespace=CONFIG["ray"]["namespace"], ignore_reinit_error=True, logging_level="WARNING",
             runtime_env={"py_modules": [str(Path(impala_dreamer.__file__).resolve().parent)], "working_dir": str(HERE), "excludes": ["__pycache__", "smoke_out"],
                          "env_vars": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}})
    learner = None
    need = CONFIG["learner"]["num_workers"] * CONFIG["ray"]["worker"]["num_cpus"]
    free = ray.available_resources().get("CPU", 0)
    if free < need:
        ray.shutdown()
        summary["error"] = f"insufficient CPU on the shared Ray cluster: {free} available, {need} needed; smoke not started"
        print(json.dumps(summary))
        return 1
    fl.ray = FirstCallTimeout(ray, CONFIG["first_get_timeout_s"], state)
    try:
        learner = make_learner_class(fl, state)(checkpoint_dir=str(out / "checkpoints"), warmstart_path=CONFIG["warmstart"], expert_parquet=CONFIG["expert_parquet"], **{**fl.CONFIG["learner"], **CONFIG["learner"]})
        try:
            learner.train(max_time=CONFIG["timeout_s"])
        except SmokeDone:
            pass
        summary["ok"] = state["done"] >= CONFIG["updates"] and state["finite"]
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
    finally:
        if learner is not None:
            learner.close()
        ray.shutdown()
    summary.update({"updates_done": state["done"], "total_steps": learner.total_steps if learner else 0, "losses_finite": state["finite"], "sil_path_ran": state["sil_calls"] > 0, "sil_calls": state["sil_calls"], "last_stats": state["last"], "wall_s": round(time.time() - t0, 1)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(run())
