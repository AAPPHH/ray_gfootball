import contextlib
import copy
import io
import os
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents as ag
import football_env as fe
import league as lg
import train as tr
from football_env import GameLake
from test_agents import FakeRay, FakeWorker, FakeWorkerClass, Stop, small_cfg
from test_dispatch_eval import FakeActor
from test_dispatch_eval import FakeRay as EvalFakeRay


class WinningWorkerClass(FakeWorkerClass):
    def remote(self, wid, spec, rollout_len, goal_traj_len):
        worker = FakeWorker(lg.SelfPlayCore(wid, spec, rollout_len, goal_traj_len))
        collect = worker.collect.fn

        def with_game(*a, **k):
            r = collect(*a, **k)
            r["episodes"] = [{"return": 1.0, "won": True, "drawn": False, "length": 5, "opponent": a[1], "goals_scored": 2, "goals_conceded": 1}]
            return r
        worker.collect.fn = with_game
        return worker


def fake_game(episode_id, n=20):
    rng = np.random.default_rng(episode_id)
    rows = [(rng.standard_normal(fe.OBS_DIM).astype(np.float32).tobytes(), int(rng.integers(fe.NUM_ACTIONS)), 0.0, 0, episode_id, i) for i in range(n)]
    return {"opponent": "bot_easy", "episode_id": episode_id, "goals_for": 1, "goals_against": 0, "rows": rows}


def write_expert(store, games=2):
    return store.write_dataset("expert", fe.ExpertGenerator.table([fake_game(i) for i in range(games)]), {"source": "test"})


def short_cfg(kind, lake, **over):
    cfg = small_cfg(kind)
    cfg.update(lake=lake, eval={**cfg["eval"], "every": 0}, checkpoint_every=2, log_every=1, **over)
    cfg["learner"].update(rollout_len=40, num_workers=2, batch_size=2)
    return cfg


class Patched(unittest.TestCase):
    worker_class = FakeWorkerClass

    def setUp(self):
        self.old = tr.ray, lg.ray, tr.SelfPlayWorker
        tr.ray = lg.ray = FakeRay()
        tr.SelfPlayWorker = self.worker_class()

    def tearDown(self):
        tr.ray, lg.ray, tr.SelfPlayWorker = self.old

    def trainer(self, cfg, stop_at=2, finish=True):
        class Short(tr.LeagueTrainer):
            def _update(self, batch):
                stats = super()._update(batch)
                if self.updates + 1 >= stop_at:
                    if finish:
                        self.start = 0.0
                    else:
                        self.updates += 1
                        raise Stop
                return stats
        return Short(cfg)


class RunLayout(Patched):
    worker_class = WinningWorkerClass

    def test_training_run_writes_meta_checkpoints_and_streams(self):
        for kind in ("impala",):
            with self.subTest(kind):
                lake = tempfile.mkdtemp()
                cfg = short_cfg(kind, lake, preset="P1")
                t = self.trainer(cfg)
                with contextlib.redirect_stdout(io.StringIO()):
                    t.train(max_time=600)
                t.close()
                store = GameLake(lake)
                run_id = t.run.run_id
                self.assertTrue(run_id.startswith(f"{kind}_P1_"))
                run_dir = store.lake.run_dir(run_id)
                meta = store.lake.runs()[0]
                self.assertEqual((meta["kind"], meta["agent"], meta["preset"]), ("train", kind, "P1"))
                self.assertEqual(meta["config"]["learner"]["num_workers"], 2)
                self.assertEqual(meta["config"]["lake"], lake)
                self.assertTrue((run_dir / "checkpoints" / "ckpt_2.pt").exists())
                self.assertTrue((run_dir / "checkpoints" / "final.pt").exists())
                self.assertTrue((run_dir / "league.pkl").exists())
                metrics = store.lake.read(run_id, "metrics")
                self.assertEqual(len(metrics), 2)
                self.assertTrue({"update", "frames", "wall_s"} <= set(metrics.columns))
                games = store.lake.read(run_id, "games")
                self.assertGreater(len(games), 0)
                self.assertEqual(set(games["result"]), {"win"})
                self.assertEqual(set(games["score"]), {"2:1"})
                self.assertTrue({"opponent", "player", "goals_for"} <= set(games.columns))

    def test_eval_rounds_land_in_the_eval_stream(self):
        lake = tempfile.mkdtemp()
        store = GameLake(lake)
        cfg = small_cfg("impala")
        run = store.open(cfg, "train")
        fake, old = EvalFakeRay(), lg.ray
        lg.ray = fake
        try:
            r = lg.EvalRunner.__new__(lg.EvalRunner)
            r.cfg, r.sink = {"every": 50, "games": 4}, lambda rows: run.log("eval", rows)
            r.actors = {"bot_easy": FakeActor("a"), "bot_hard": FakeActor("b")}
            r.refs, r.results, r.errors, r.meta, r.last = {}, {}, {}, None, {}
            r.start(50, 1000, {})
            fake.ready.update(["a", "b"])
            with contextlib.redirect_stdout(io.StringIO()):
                r.poll()
        finally:
            lg.ray = old
        run.close()
        df = store.lake.read(run.run_id, "eval").set_index("bot")
        self.assertEqual(set(df.index), {"bot_easy", "bot_hard"})
        self.assertEqual((df.loc["bot_easy", "win"], df.loc["bot_easy", "n"], df.loc["bot_easy", "update"]), (4, 4, 50))
        self.assertTrue({"win_rate", "ci_lo", "ci_hi", "goal_diff", "draw", "loss"} <= set(df.columns))
        self.assertEqual(len(store.lake.query("select * from eval where bot = 'bot_hard'")), 1)

    def test_existing_run_id_is_not_overwritten_without_resume(self):
        lake = tempfile.mkdtemp()
        cfg = short_cfg("impala", lake, run_id="fixed")
        self.trainer(cfg).close()
        with self.assertRaises(FileExistsError):
            self.trainer(cfg)


class Resume(Patched):
    def test_resume_by_run_id_continues_the_run_and_by_path_starts_a_new_one(self):
        for kind in ("impala", "dreamer"):
            with self.subTest(kind):
                lake = tempfile.mkdtemp()
                cfg = short_cfg(kind, lake)
                first = self.trainer(cfg, stop_at=2, finish=False)
                with self.assertRaises(Stop), contextlib.redirect_stdout(io.StringIO()):
                    first.train(max_time=600)
                first._save_checkpoint()
                first.close()
                run_id = first.run.run_id
                with contextlib.redirect_stdout(io.StringIO()):
                    same = self.trainer({**cfg, "resume": run_id})
                self.assertEqual((same.run.run_id, same.updates, same.total_steps), (run_id, first.updates, first.total_steps))
                ckpt = str(first.run.ckpt_path(first.updates))
                self.assertEqual(same.run.meta["resumes"][-1]["resume_source"], ckpt)
                same.close()
                with contextlib.redirect_stdout(io.StringIO()):
                    latest = self.trainer({**cfg, "resume": "latest"})
                self.assertEqual(latest.run.run_id, run_id)
                latest.close()
                with contextlib.redirect_stdout(io.StringIO()):
                    forked = self.trainer({**cfg, "resume": ckpt})
                self.assertNotEqual(forked.run.run_id, run_id)
                self.assertEqual((forked.updates, forked.run.meta["resume_source"]), (first.updates, ckpt))
                forked.close()

    def finished(self, lake, **over):
        cfg = short_cfg("impala", lake, **over)
        t = self.trainer(cfg, stop_at=3)
        with contextlib.redirect_stdout(io.StringIO()):
            t.train(max_time=600)
        t.close()
        return cfg, t

    def test_final_newer_than_last_periodic_checkpoint_wins(self):
        cfg, t = self.finished(tempfile.mkdtemp())
        self.assertEqual(t.updates, 3)
        self.assertEqual(sorted(p.name for p in (t.run.run_dir / "checkpoints").glob("*.pt")), ["ckpt_2.pt", "final.pt"])
        with contextlib.redirect_stdout(io.StringIO()):
            again = self.trainer({**cfg, "resume": t.run.run_id})
        self.assertEqual((again.updates, Path(again.run.meta["resumes"][-1]["resume_source"]).name), (3, "final.pt"))
        self.assertGreater(again.wall_offset, 0.0)
        again.close()

    def test_rejected_resume_leaves_run_meta_untouched(self):
        cfg, t = self.finished(tempfile.mkdtemp())
        before = (t.run.run_dir / "run_meta.json").read_text()
        with self.assertRaises(ValueError):
            tr.LeagueTrainer({**short_cfg("dreamer", cfg["lake"]), "resume": t.run.run_id})
        self.assertEqual((t.run.run_dir / "run_meta.json").read_text(), before)

    def test_resume_keeps_kind_and_original_sources(self):
        lake = tempfile.mkdtemp()
        store = GameLake(lake)
        write_expert(store, 1)
        cfg, t = self.finished(lake, paths={"warmstart": None, "expert": "lake:expert"})
        original = GameLake(lake).meta(t.run.run_id)["sources"]
        write_expert(store, 2)
        with contextlib.redirect_stdout(io.StringIO()):
            again = self.trainer({**cfg, "resume": t.run.run_id})
        again.close()
        meta = GameLake(lake).meta(t.run.run_id)
        self.assertEqual((meta["kind"], meta["sources"]), ("train", original))
        self.assertEqual(original["expert"]["version"], 1)

    def test_bc_run_cannot_be_resumed_as_train(self):
        lake = tempfile.mkdtemp()
        store = GameLake(lake)
        with store.open({"agent": "impala", "preset": "x"}, "bc", streams=("metrics",)) as run:
            run.ckpt_path("final").write_bytes(b"x")
        with self.assertRaises(ValueError):
            store.plan_resume({"agent": "impala", "resume": run.run_id})

    def test_fork_copies_snapshots_and_never_prunes_the_source_run(self):
        cfg, t = self.finished(tempfile.mkdtemp())
        rating = lg.TS_ENV.create_rating
        t = self.trainer({**cfg, "run_id": "src2"}, stop_at=2)
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(3):
                p = t.checkpoint_dir / f"old_{i}.pt"
                t.agent.save_snapshot(p)
                t.league.add_snapshot(f"old_{i}", str(p), t.agent.weights(), rating(mu=10.0 + i, sigma=1.0))
            t.league.save()
            t._save_checkpoint("final")
        t.close()
        src = sorted(p.name for p in t.checkpoint_dir.glob("old_*.pt"))
        with contextlib.redirect_stdout(io.StringIO()):
            f = self.trainer({**cfg, "resume": str(t.run.ckpt_path("final"))})
            f.league.max_snapshots = 4
            for i in range(8):
                p = f.checkpoint_dir / f"new_{i}.pt"
                f.agent.save_snapshot(p)
                f.league.add_snapshot(f"new_{i}", str(p), f.agent.weights(), rating(mu=40.0 + 5 * i, sigma=1.0))
        self.assertEqual(sorted(p.name for p in t.checkpoint_dir.glob("old_*.pt")), src)
        self.assertTrue(all(Path(m.weights_path).parent == f.checkpoint_dir for m in f.league.members.values() if m.weights_path))
        f.close()

    def test_close_kills_workers_even_if_run_close_raises(self):
        t = self.trainer(short_cfg("impala", tempfile.mkdtemp()))
        killed = []
        tr.ray.kill = lambda w: killed.append(w)
        def boom():
            raise OSError("disk full")
        t.run.close = boom
        with contextlib.redirect_stdout(io.StringIO()):
            t.close()
        self.assertEqual(len(killed), len(t.workers))

    def test_run_ids_are_unique_within_one_second(self):
        store = GameLake(tempfile.mkdtemp())
        ids = {store.new_run_id("impala", "p") for _ in range(5)}
        self.assertEqual(len(ids), 5)

    def test_resume_target_that_does_not_exist_fails(self):
        lake = tempfile.mkdtemp()
        for target in ("latest", "nope", "/nonexistent/ckpt.pt"):
            with self.assertRaises(FileNotFoundError):
                tr.LeagueTrainer({**short_cfg("impala", lake), "resume": target})


class Resolution(unittest.TestCase):
    def test_expert_resolves_to_latest_or_pinned_version(self):
        store = GameLake(tempfile.mkdtemp())
        v1, v2 = write_expert(store, 1), write_expert(store, 2)
        self.assertEqual((v1["version"], v2["version"]), (1, 2))
        self.assertEqual(store.resolve_dataset("lake:expert")["version"], 2)
        self.assertEqual(store.resolve_dataset("lake:expert@1")["version"], 1)
        self.assertEqual(ag.ExpertBufferGPU(store.resolve_dataset("lake:expert@1")["path"], torch.device("cpu"), 64).num_episodes, 1)
        self.assertEqual(ag.ExpertBufferGPU(store.resolve_dataset("lake:expert")["path"], torch.device("cpu"), 64).num_episodes, 2)
        self.assertIsNone(store.resolve_dataset(None))
        plain = Path(tempfile.mkdtemp()) / "e.parquet"
        import pyarrow.parquet as pq
        pq.write_table(fe.ExpertGenerator.table([fake_game(0)]), plain)
        self.assertEqual(store.resolve_dataset(str(plain))["path"], str(plain))

    def test_bc_resolves_to_newest_final_or_pinned_run(self):
        store = GameLake(tempfile.mkdtemp())
        ids = []
        for i in range(2):
            run = store.open(small_cfg("impala"), "bc", streams=("metrics",))
            torch.save({"model": {}}, run.ckpt_path("final"))
            run.close()
            ids.append(run.run_id)
            time.sleep(0.02)
        unfinished = store.open(small_cfg("impala"), "bc", streams=("metrics",))
        unfinished.close()
        other = store.open(small_cfg("impala"), "train")
        torch.save({"model": {}}, other.ckpt_path("final"))
        other.close()
        self.assertEqual(store.resolve_warmstart("lake:bc")["run_id"], ids[1])
        self.assertEqual(store.resolve_warmstart(f"lake:bc@{ids[0]}")["run_id"], ids[0])
        self.assertTrue(store.resolve_warmstart("lake:bc")["path"].endswith("final.pt"))
        self.assertIsNone(store.resolve_warmstart(None))

    def test_missing_sources_fail_with_clear_errors(self):
        store = GameLake(tempfile.mkdtemp())
        for call, ref in ((store.resolve_dataset, "lake:expert"), (store.resolve_dataset, "/nonexistent/e.parquet"),
                          (store.resolve_warmstart, "lake:bc"), (store.resolve_warmstart, "/nonexistent/bc.pt")):
            with self.assertRaisesRegex(FileNotFoundError, "not found|no bc run|expert data|warmstart"):
                call(ref)
        write_expert(store)
        with self.assertRaisesRegex(FileNotFoundError, "version 9"):
            store.resolve_dataset("lake:expert@9")
        with self.assertRaisesRegex(FileNotFoundError, "bc run nope"):
            store.resolve_warmstart("lake:bc@nope")

    def test_reading_does_not_create_anything(self):
        root = Path(tempfile.mkdtemp()) / "lake"
        store = GameLake(root)
        with self.assertRaises(FileNotFoundError):
            store.resolve_dataset("lake:expert")
        with self.assertRaises(FileNotFoundError):
            store.resolve_warmstart("lake:bc")
        self.assertFalse(root.exists())


class FailFast(Patched):
    def test_trainer_refuses_to_start_on_unresolvable_sources(self):
        lake = tempfile.mkdtemp()
        for key, ref in (("warmstart", "lake:bc"), ("warmstart", "/nonexistent/w.pt"), ("expert", "lake:expert"), ("expert", "/nonexistent/e.parquet")):
            with self.subTest(key=key, ref=ref):
                cfg = short_cfg("impala", lake)
                cfg["paths"][key] = ref
                with self.assertRaises(FileNotFoundError):
                    tr.LeagueTrainer(cfg)
        self.assertEqual(GameLake(lake).metas(), [])

    def test_architecture_mismatch_raises(self):
        lake = tempfile.mkdtemp()
        path = Path(lake) / "w.pt"
        torch.save({"model": ag.Net(32, 32).state_dict()}, path)
        cfg = short_cfg("impala", lake)
        cfg["paths"]["warmstart"] = str(path)
        with self.assertRaisesRegex(ValueError, "does not match"):
            tr.LeagueTrainer(cfg)
        torch.save({"model": ag.Net(16, 16).state_dict()}, path)
        t = tr.LeagueTrainer(cfg)
        self.assertTrue(t.agent.warmstarted)
        t.close()

    def test_none_means_off_and_dreamer_rejects_sources(self):
        lake = tempfile.mkdtemp()
        for kind in ("impala", "dreamer"):
            t = tr.LeagueTrainer(short_cfg(kind, lake))
            self.assertFalse(t.agent.warmstarted)
            self.assertIsNone(getattr(t.agent, "expert", None))
            t.close()
        cfg = short_cfg("dreamer", lake)
        cfg["paths"].update(warmstart="/nonexistent/w.pt", expert="/nonexistent/e.parquet")
        with self.assertRaises(ValueError):
            tr.LeagueTrainer(cfg)

    def test_dreamer_default_paths_are_off(self):
        captured = {}
        old = tr.LeagueTrainer, tr.ray_init, tr.free_cpu_error
        tr.ray_init, tr.free_cpu_error = (lambda cfg: None), (lambda cfg: None)
        tr.LeagueTrainer = lambda cfg: captured.setdefault("cfg", cfg) and (_ for _ in ()).throw(Stop())
        try:
            with self.assertRaises(Stop), contextlib.redirect_stdout(io.StringIO()):
                tr.run({"agent": "dreamer", "lake": tempfile.mkdtemp()})
        finally:
            tr.LeagueTrainer, tr.ray_init, tr.free_cpu_error = old
        self.assertEqual(captured["cfg"]["paths"], {"warmstart": None, "expert": None})

    def test_agent_rejects_missing_files_directly(self):
        spec = tr.agent_spec(small_cfg("impala"))
        for key in ("warmstart", "expert"):
            with self.assertRaises(FileNotFoundError):
                ag.make_agent({**spec, key: "/nonexistent/x"}, learner=True)


class Workflow(Patched):
    def test_gen_expert_then_bc_then_train_roundtrip(self):
        lake = tempfile.mkdtemp()
        store = GameLake(lake)
        write_expert(store, 3)
        small = {"impala": copy.deepcopy(small_cfg("impala")["impala"])}
        with contextlib.redirect_stdout(io.StringIO()):
            tr.run({"mode": "bc", "lake": lake, "preset": "T", "paths": {"expert": "lake:expert", "warmstart": None}, "bc": {"epochs": 2, "batch_size": 16}, **small})
        bc = [m for m in store.metas() if m["kind"] == "bc"]
        self.assertEqual(len(bc), 1)
        self.assertTrue(bc[0]["run_id"].startswith("bc_T_"))
        self.assertEqual((bc[0]["dataset"]["name"], bc[0]["dataset"]["version"]), ("expert", 1))
        self.assertTrue(store.lake.run_dir(bc[0]["run_id"]).joinpath("checkpoints", "final.pt").exists())
        self.assertEqual(len(store.lake.read(bc[0]["run_id"], "metrics")), 2)
        cfg = short_cfg("impala", lake)
        cfg["paths"].update(warmstart="lake:bc", expert="lake:expert@1")
        with contextlib.redirect_stdout(io.StringIO()):
            t = tr.LeagueTrainer(cfg)
        self.assertTrue(t.agent.warmstarted)
        self.assertEqual(len(t.agent.expert.window_slices) > 0 and t.agent.expert.num_episodes, 3)
        self.assertEqual(t.run.meta["sources"]["warmstart"]["run_id"], bc[0]["run_id"])
        self.assertEqual(t.run.meta["sources"]["expert"]["version"], 1)
        self.assertIn("warmstart", t.league.members)
        t.close()


if __name__ == "__main__":
    unittest.main()
