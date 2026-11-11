import os
import sys
import json
import math
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFIG = {
    "updates": 200,
    "timeout_s": 3600,
    "out_dir": str(HERE.parent / "smoke_out"),
    "ray": {"address": "auto", "namespace": None, "worker": {"num_cpus": 1, "num_gpus": 0, "memory": 2 * 1024 ** 3}},
    "learner": {"num_workers": 8, "rollout_len": 128, "batch_size": 8, "goal_traj_len": 64, "golden_capacity": 256},
    "warmstart": None,
    "expert_parquet": None,
}


class SmokeDone(Exception):
    pass


def make_learner_class(fl, state):
    class SmokeLearner(fl.LeagueLearner):
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
    import impala_dreamer
    import full_league as fl
    out = Path(CONFIG["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    state = {"done": 0, "finite": True, "last": None}
    summary = {"updates_requested": CONFIG["updates"], "workers": CONFIG["learner"]["num_workers"], "ok": False, "error": None}
    t0 = time.time()
    fl.CONFIG["ray"].update({"address": CONFIG["ray"]["address"], "worker": CONFIG["ray"]["worker"]})
    ray.init(address=CONFIG["ray"]["address"], namespace=CONFIG["ray"]["namespace"], ignore_reinit_error=True, logging_level="WARNING",
             runtime_env={"py_modules": [str(Path(impala_dreamer.__file__).resolve().parent)], "working_dir": str(HERE), "excludes": ["__pycache__", "smoke_out"],
                          "env_vars": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}})
    learner = None
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
    summary.update({"updates_done": state["done"], "total_steps": learner.total_steps if learner else 0, "losses_finite": state["finite"], "last_stats": state["last"], "wall_s": round(time.time() - t0, 1)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(run())
