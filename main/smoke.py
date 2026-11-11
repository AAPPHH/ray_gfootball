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

DREAMER_SMALL = {
    "batch": 4, "T": 16, "replay_cap": 40_000, "prefill": 1024, "train_ratio": 4, "max_updates": 4,
    "model": {"deter": 64, "hidden": 64, "units": 64, "stoch": 8, "discrete": 8, "horizon": 5, "bins": 63, "warmup": 1},
}
CONFIG = {
    "agent": "impala",
    "updates": 40,
    "timeout_s": 3600,
    "call_timeout_s": 300,
    "poll_s": 5,
    "min_free_cpus": 8,
    "out_dir": None,
    "overrides": {
        "ray": {"address": "auto", "worker": {"num_cpus": 1, "num_gpus": 0, "memory": 2 * 1024 ** 3}, "env_vars": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}},
        "learner": {"num_workers": 4, "rollout_len": 512, "batch_size": 4, "goal_traj_len": 64, "driver_threads": 4},
        "impala": {"golden_capacity": 256},
        "dreamer": DREAMER_SMALL,
        "eval": {"every": 0},
        "paths": {"warmstart": None, "expert_parquet": None},
    },
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


def make_trainer_class(state):
    import train

    class SmokeTrainer(train.LeagueTrainer):
        def _update(self, rollouts):
            stats = super()._update(rollouts)
            state["done"] += 1
            state["model_updates"] += stats.get("model_updates", 0.0)
            state["finite"] = state["finite"] and all(math.isfinite(float(v)) for v in stats.values())
            state["last"] = {k: float(v) for k, v in stats.items()}
            if state["done"] >= CONFIG["updates"]:
                self.updates += 1
                self._log_progress(stats)
                raise SmokeDone
            return stats

    return SmokeTrainer


def run():
    import copy
    import ray
    import torch
    import league
    import train
    cfg = train.deep_merge(copy.deepcopy(train.CONFIG), {**copy.deepcopy(CONFIG["overrides"]), "agent": CONFIG["agent"]})
    out = Path(CONFIG["out_dir"] or Path(train.GF_RUNS) / cfg["agent"] / "smoke")
    cfg["paths"]["checkpoint_dir"] = str(out / "checkpoints")
    cfg["min_free_cpus"] = CONFIG["min_free_cpus"]
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    state = {"done": 0, "finite": True, "last": None, "sil_calls": 0, "model_updates": 0.0}
    workers = cfg["learner"]["num_workers"]
    torch.set_num_threads(cfg["learner"]["driver_threads"])
    summary = {"agent": cfg["agent"], "updates_requested": CONFIG["updates"], "workers": workers, "ok": False, "error": None}
    t0 = time.time()
    train.ray_init(cfg)
    trainer = None
    error = train.free_cpu_error(cfg)
    if error:
        ray.shutdown()
        summary["error"] = error
        print(json.dumps(summary))
        return 1
    bounded = BoundedRay(ray, CONFIG["call_timeout_s"], CONFIG["poll_s"], t0 + CONFIG["timeout_s"])
    train.ray = league.ray = bounded
    try:
        trainer = make_trainer_class(state)(cfg)
        if hasattr(trainer.agent, "_update_sil_gpu"):
            sil = trainer.agent._update_sil_gpu
            def counted(batch):
                state["sil_calls"] += 1
                return sil(batch)
            trainer.agent._update_sil_gpu = counted
        try:
            trainer.train(max_time=CONFIG["timeout_s"])
        except SmokeDone:
            pass
        summary["ok"] = state["done"] >= CONFIG["updates"] and state["finite"] and state["last"] is not None and (cfg["agent"] != "dreamer" or state["model_updates"] > 0)
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
    finally:
        if trainer is not None:
            stats = trainer.league.get_stats()
            summary["league"] = {"games_by_opp": stats["games_by_opp"], "snap_games": stats["snap_games"], "total_games": stats["total_games"],
                                 "EMH": {tag: stats["bots"][name]["games"] for tag, name in (("E", "bot_easy"), ("M", "bot_medium"), ("H", "bot_hard"))}}
            trainer.close()
        ray.shutdown()
    summary.update({"updates_done": state["done"], "total_steps": trainer.total_steps if trainer else 0, "losses_finite": state["finite"], "sil_path_ran": state["sil_calls"] > 0, "sil_calls": state["sil_calls"], "model_updates": state["model_updates"], "last_stats": state["last"], "wall_s": round(time.time() - t0, 1)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        CONFIG.update(json.loads(sys.argv[1]))
    sys.exit(run())
