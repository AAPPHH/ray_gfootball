import os
import re
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from deep_rl_architectures.lake import Lake

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "lake:"
CKPT = re.compile(r"^ckpt_(\d+)\.pt$")

def is_ref(value) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)

def split_ref(ref: str):
    name, _, pin = ref[len(PREFIX):].partition("@")
    return name, pin or None

def read_table(path) -> pa.Table:
    if isinstance(path, pa.Table):
        return path
    p = Path(path)
    if p.is_dir():
        shards = sorted(p.glob("shard_*.parquet"))
        if not shards:
            raise FileNotFoundError(f"no parquet shards in {p}")
        return pa.concat_tables([pq.read_table(f) for f in shards])
    if not p.exists():
        raise FileNotFoundError(f"data file {p} not found")
    return pq.read_table(p)

class GameLake:
    def __init__(self, root=None):
        root = Path(root or "lake")
        self.root = root if root.is_absolute() else ROOT / root
        self.lake = Lake(self.root)

    def metas(self) -> list:
        return sorted(self.lake.runs() if (self.root / "runs").is_dir() else [], key=lambda m: m.get("created_at", 0))

    def new_run_id(self, prefix: str, preset: str) -> str:
        base = f"{prefix}_{preset}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_id, n = base, 0
        while (self.lake.run_dir(run_id) / "run_meta.json").exists():
            n += 1
            run_id = f"{base}_{n}"
        return run_id

    def open(self, cfg: dict, kind: str, streams=("metrics", "games", "eval"), run_id: str = None, resume_source: str = None):
        run_id = run_id or cfg.get("run_id") or self.new_run_id(cfg["agent"] if kind == "train" else kind, cfg.get("preset") or "default")
        if resume_source is None and (self.lake.run_dir(run_id) / "run_meta.json").exists():
            raise FileExistsError(f"run {run_id} already exists in {self.root}; use resume to continue it")
        run = self.lake.run(run_id, cfg, {s: None for s in streams}, resume_source=resume_source)
        run.update_meta(kind=kind, agent=cfg["agent"], preset=cfg.get("preset"))
        return run

    def latest_ckpt(self, run_id: str):
        ckpt_dir = self.lake.run_dir(run_id) / "checkpoints"
        found = sorted(((int(m[1]), f) for f in ckpt_dir.glob("ckpt_*.pt") if (m := CKPT.match(f.name))), key=lambda x: x[0])
        if found:
            return found[-1][1]
        final = ckpt_dir / "final.pt"
        return final if final.exists() else None

    def plan_resume(self, cfg: dict):
        target = cfg.get("resume")
        if not target:
            return None, None, None
        runs = {m["run_id"]: m for m in self.metas()}
        if target == "latest":
            target = next((m["run_id"] for m in reversed(list(runs.values())) if m.get("kind") == "train" and m.get("agent") == cfg["agent"] and self.latest_ckpt(m["run_id"])), None)
            if target is None:
                raise FileNotFoundError(f"resume=latest: no {cfg['agent']} run with a checkpoint in {self.root}")
        if target in runs:
            ckpt = self.latest_ckpt(target)
            if ckpt is None:
                raise FileNotFoundError(f"run {target} has no checkpoint to resume from")
            return target, ckpt, str(ckpt)
        if not Path(target).is_file():
            raise FileNotFoundError(f"resume target {target!r} is neither a run id in {self.root} nor a checkpoint file")
        return None, Path(target), str(target)

    def resolve_warmstart(self, ref):
        if ref is None:
            return None
        if not is_ref(ref):
            if not Path(ref).is_file():
                raise FileNotFoundError(f"warmstart {ref} not found")
            return {"ref": ref, "path": str(ref)}
        kind, pin = split_ref(ref)
        runs = [m for m in self.metas() if m.get("kind") == kind and (self.lake.run_dir(m["run_id"]) / "checkpoints" / "final.pt").is_file()]
        if pin:
            runs = [m for m in runs if m["run_id"] == pin]
        if not runs:
            raise FileNotFoundError(f"warmstart {ref}: no {kind} run{' ' + pin if pin else ''} with a final.pt in {self.root}")
        run_id = runs[-1]["run_id"]
        return {"ref": ref, "run_id": run_id, "path": str(self.lake.run_dir(run_id) / "checkpoints" / "final.pt")}

    def resolve_dataset(self, ref):
        if ref is None:
            return None
        if not is_ref(ref):
            if not Path(ref).exists():
                raise FileNotFoundError(f"expert data {ref} not found")
            return {"ref": ref, "path": str(ref)}
        name, pin = split_ref(ref)
        try:
            found = self.lake.dataset(name, int(pin) if pin else None)
        except KeyError as e:
            raise FileNotFoundError(f"expert data {ref}: {e.args[0]} in {self.root}") from None
        return {"ref": ref, "name": name, "version": found["version"], "path": str(found["path"])}

    def sources(self, cfg: dict) -> dict:
        return {"warmstart": self.resolve_warmstart(cfg["paths"]["warmstart"]), "expert": self.resolve_dataset(cfg["paths"]["expert"])}

    def write_dataset(self, name: str, table: pa.Table, meta: dict) -> dict:
        self.lake.write_dataset(name, table, meta=meta)
        return self.resolve_dataset(f"{PREFIX}{name}")
