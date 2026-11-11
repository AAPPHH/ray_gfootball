import os
import sys
import json
import math
import shutil
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "8"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFIG = {
    "updates": 40,
    "timeout_s": 3600,
    "call_timeout_s": 300,
    "poll_s": 5,
    "driver_threads": 4,
    "min_free_cpus": 8,
    "out_dir": str(HERE.parent / "smoke_out"),
    "ray": {"address": "auto", "namespace": None, "worker": {"num_cpus": 1, "num_gpus": 0, "memory": 2 * 1024 ** 3}},
    "learner": {"num_workers": 4, "rollout_len": 512, "batch_size": 4, "goal_traj_len": 64, "golden_capacity": 256},
    "eval": {"every": 0},
    "warmstart": None,
    "expert_parquet": None,
}


class SmokeDone(Exception):
    pass


class BoundedRay:
    def __init__(self, ray, call_timeout, poll, deadline):
        self._ray, self._call_timeout, self._poll, self._deadline = ray, call_timeout, poll, deadline

    def __getattr__(self, name):
        return getattr(self._ray, name)

    def get(self, refs, timeout=None):
        return self._ray.get(refs, timeout=self._call_timeout if timeout is None else timeout)

    def wait(self, refs, **kw):
        if kw.get("timeout") is not None:
            return self._ray.wait(refs, **kw)
        stall = time.time() + self._call_timeout
        while True:
            ready, rest = self._ray.wait(refs, **{**kw, "timeout": self._poll})
            if ready:
                return ready, rest
            if time.time() > min(stall, self._deadline):
                raise TimeoutError(f"no rollout within {self._call_timeout}s or past the {CONFIG['timeout_s']}s limit")


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
                self.updates += 1
                self._log_progress(stats)
                raise SmokeDone
            return stats

    return SmokeLearner


def run():
    import ray
    import torch
    import actor_critic
    import full_league as fl
    out = Path(CONFIG["out_dir"])
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    state = {"done": 0, "finite": True, "last": None, "sil_calls": 0}
    torch.set_num_threads(CONFIG["driver_threads"])
    summary = {"updates_requested": CONFIG["updates"], "workers": CONFIG["learner"]["num_workers"], "ok": False, "error": None}
    t0 = time.time()
    fl.CONFIG["eval"].update(CONFIG["eval"])
    fl.CONFIG["ray"].update({"address": CONFIG["ray"]["address"], "worker": CONFIG["ray"]["worker"]})
    ray.init(address=CONFIG["ray"]["address"], namespace=CONFIG["ray"]["namespace"], ignore_reinit_error=True, logging_level="WARNING",
             runtime_env={"py_modules": [str(Path(actor_critic.__file__).resolve().parent)], "working_dir": str(HERE), "excludes": ["__pycache__", "smoke_out"],
                          "env_vars": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}})
    learner = None
    need = max(CONFIG["min_free_cpus"], CONFIG["learner"]["num_workers"] * CONFIG["ray"]["worker"]["num_cpus"])
    free = ray.available_resources().get("CPU", 0)
    if free < need:
        ray.shutdown()
        summary["error"] = f"insufficient CPU on the shared Ray cluster: {free} available, {need} needed; smoke not started"
        print(json.dumps(summary))
        return 1
    fl.ray = BoundedRay(ray, CONFIG["call_timeout_s"], CONFIG["poll_s"], t0 + CONFIG["timeout_s"])
    try:
        learner = make_learner_class(fl, state)(checkpoint_dir=str(out / "checkpoints"), warmstart_path=CONFIG["warmstart"], expert_parquet=CONFIG["expert_parquet"], **{**fl.CONFIG["learner"], **CONFIG["learner"]})
        try:
            learner.train(max_time=CONFIG["timeout_s"])
        except SmokeDone:
            pass
        summary["ok"] = state["done"] >= CONFIG["updates"] and state["finite"] and state["last"] is not None
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
    finally:
        if learner is not None:
            stats = learner.league.get_stats()
            summary["league"] = {"games_by_opp": stats["games_by_opp"], "snap_games": stats["snap_games"], "total_games": stats["total_games"],
                                 "EMH": {tag: stats["bots"][name]["games"] for tag, name in (("E", "bot_easy"), ("M", "bot_medium"), ("H", "bot_hard"))}}
            learner.close()
        ray.shutdown()
    summary.update({"updates_done": state["done"], "total_steps": learner.total_steps if learner else 0, "losses_finite": state["finite"], "sil_path_ran": state["sil_calls"] > 0, "sil_calls": state["sil_calls"], "last_stats": state["last"], "wall_s": round(time.time() - t0, 1)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(run())
