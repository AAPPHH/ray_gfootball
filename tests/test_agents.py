import contextlib
import copy
import io
import math
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import agents as ag
import football_env as fe
import league as lg
import train as tr

SMALL = {
    "learner": {"num_workers": 1, "rollout_len": 200, "batch_size": 1, "goal_traj_len": 8},
    "impala": {"d_model": 16, "lstm_hidden": 16, "golden_capacity": 8},
    "dreamer": {"batch": 2, "T": 8, "replay_cap": 2000, "prefill": 0, "train_ratio": 1, "max_updates": 2,
                "model": {"deter": 16, "hidden": 16, "units": 16, "stoch": 4, "discrete": 4, "horizon": 3, "bins": 31, "warmup": 1}},
    "paths": {"warmstart": None, "expert_parquet": None},
}
EASY = ("11_vs_11_easy_stochastic", 1, 0)
KINDS = ("impala", "dreamer")


def small_cfg(kind):
    return tr.deep_merge(copy.deepcopy(tr.CONFIG), {**copy.deepcopy(SMALL), "agent": kind})


def small_spec(kind, rollout_len=200):
    spec = tr.agent_spec(small_cfg(kind))
    spec["rollout_len"] = rollout_len
    return spec


def learner_for(kind):
    agent = ag.make_agent(tr.agent_spec(small_cfg(kind)), learner=True)
    agent.attach(lg.PureLeague(Path(tempfile.mkdtemp()), tr.agent_spec(small_cfg(kind))))
    return agent


class Interface(unittest.TestCase):
    def test_both_agents_satisfy_the_interface(self):
        obs, feat = np.random.randn(fe.OBS_DIM).astype(np.float32), np.random.randn(fe.FEATURE_DIM).astype(np.float32)
        for kind in KINDS:
            with self.subTest(kind):
                agent = ag.make_agent(small_spec(kind))
                self.assertIsInstance(agent, ag.Agent)
                self.assertEqual(agent.kind, kind)
                a, lp, v, state = agent.act(obs, feat, fe.NO_ACTION, True, None)
                self.assertTrue(0 <= a < fe.NUM_ACTIONS and isinstance(a, int))
                self.assertTrue(math.isfinite(lp) and math.isfinite(v))
                a2, _, _, _ = agent.act(obs, feat, a, False, state, deterministic=True)
                self.assertTrue(0 <= a2 < fe.NUM_ACTIONS)
                fp = agent.fingerprint(np.random.randn(5, fe.OBS_DIM).astype(np.float32), np.random.randn(5, fe.FEATURE_DIM).astype(np.float32))
                self.assertEqual(fp.shape, (fe.NUM_ACTIONS,))
                self.assertAlmostEqual(float(fp.sum()), 1.0, places=4)
                other = ag.make_agent(small_spec(kind))
                other.set_weights(agent.weights())
                for k, v in agent.weights().items():
                    self.assertTrue(np.array_equal(v, other.weights()[k]))
                path = Path(tempfile.mkdtemp()) / "s.pt"
                agent.save_snapshot(path)
                self.assertEqual(set(torch.load(path)["model"]), set(agent.weights()))
                for name in ("observe", "update", "fmt", "save", "load", "attach"):
                    self.assertTrue(callable(getattr(agent, name)))

    def test_unknown_agent_is_rejected(self):
        with self.assertRaises(ValueError):
            ag.make_agent({"agent": "nope"})

    def test_checkpoint_roundtrip(self):
        for kind in KINDS:
            with self.subTest(kind):
                a, b = learner_for(kind), learner_for(kind)
                path = Path(tempfile.mkdtemp()) / "c.pt"
                a.save(path, {"updates": 3})
                self.assertEqual(b.load(path)["updates"], 3)
                for k, v in a.model.state_dict().items():
                    self.assertTrue(torch.equal(v, b.model.state_dict()[k]))

    def test_checkpoint_and_league_of_another_agent_are_refused(self):
        a, b = learner_for("impala"), learner_for("dreamer")
        path = Path(tempfile.mkdtemp()) / "c.pt"
        a.save(path, {"updates": 3})
        with self.assertRaisesRegex(ValueError, "impala.*dreamer"):
            b.load(path)
        a.league.save()
        b.league.checkpoint_dir = a.league.checkpoint_dir
        with self.assertRaisesRegex(ValueError, "impala.*dreamer"):
            b.league.load()
        self.assertTrue(a.league.load())

    def test_trainer_dir_is_per_agent(self):
        old = tr.ray, lg.ray, tr.SelfPlayWorker
        tr.ray = lg.ray = FakeRay()
        tr.SelfPlayWorker = FakeWorkerClass()
        try:
            base = tempfile.mkdtemp()
            dirs = []
            for kind in KINDS:
                cfg = small_cfg(kind)
                cfg["paths"]["checkpoint_dir"] = base
                cfg.update(eval={**cfg["eval"], "every": 0})
                t = tr.LeagueTrainer(cfg)
                dirs.append(t.checkpoint_dir)
                t._save_checkpoint()
                t.close()
            self.assertEqual(dirs, [Path(base) / k for k in KINDS])
        finally:
            tr.ray, lg.ray, tr.SelfPlayWorker = old


class TryConfig(unittest.TestCase):
    def test_cpu_try_replay_is_small_and_needs_at_most_eight_cpus(self):
        cfg = tr.deep_merge(copy.deepcopy(tr.CONFIG), copy.deepcopy(tr.CPU_TRY))
        self.assertLessEqual(cfg["learner"]["num_workers"] * cfg["ray"]["worker"]["num_cpus"] + cfg["learner"]["driver_threads"], 8)
        self.assertEqual(cfg["ray"]["worker"]["num_gpus"], 0)
        self.assertTrue(cfg["paths"]["checkpoint_dir"].startswith(tr.GF_RUNS))
        agent = ag.make_agent(tr.agent_spec(cfg), learner=True)
        self.assertLessEqual(sum(a.nbytes for a in agent.replay.data.values()), 200 * 1024 ** 2)

    def test_gpu_configs_resolve_and_every_actor_needs_gfootball(self):
        for name, kind, workers in (("GPU_E0", "impala", 16), ("GPU_D0", "dreamer", 12)):
            with self.subTest(name):
                cfg = tr.deep_merge(copy.deepcopy(tr.CONFIG), copy.deepcopy(tr.RUNS[name]))
                self.assertEqual((cfg["agent"], cfg["learner"]["num_workers"], cfg[kind]["device"]), (kind, workers, "cuda"))
                self.assertIsNone(cfg["paths"]["warmstart"])
                self.assertIsNone(cfg["paths"]["expert_parquet"])
                self.assertEqual((cfg["eval"]["every"], cfg["eval"]["games"], cfg["checkpoint_every"]), (50, 30, 50))
                self.assertEqual(cfg["env"]["rewards"], "scoring,checkpoints")
                for opts in (cfg["ray"]["worker"], cfg["eval"]["actor"], cfg["expert_gen"]["ray_worker"]):
                    self.assertEqual(opts["resources"], {"gfootball": 1})
                    self.assertEqual(opts["num_gpus"], 0)
                self.assertEqual(cfg["ray"]["env_vars"]["OMP_NUM_THREADS"], "1")
                tr.agent_spec(cfg)

    def test_free_check_reads_gfootball_resource(self):
        class Pool:
            def available_resources(self):
                return {"CPU": 1000, "gfootball": 18}
        old = tr.ray
        try:
            tr.ray = Pool()
            cfg = tr.deep_merge(copy.deepcopy(tr.CONFIG), copy.deepcopy(tr.GPU_E0))
            self.assertIn("gfootball", tr.free_cpu_error(cfg))
            tr.ray = type("P", (), {"available_resources": lambda s: {"CPU": 0, "gfootball": 19}})()
            self.assertIsNone(tr.free_cpu_error(cfg))
        finally:
            tr.ray = old

    def test_free_cpu_check_blocks_a_start_on_a_busy_cluster(self):
        class Busy:
            def __init__(self, cpus):
                self.cpus = cpus

            def available_resources(self):
                return {"CPU": self.cpus}
        old = tr.ray
        try:
            cfg = small_cfg("dreamer")
            cfg["min_free_cpus"] = 8
            tr.ray = Busy(7)
            self.assertIn("7 available, 8 needed", tr.free_cpu_error(cfg))
            tr.ray = Busy(8)
            self.assertIsNone(tr.free_cpu_error(cfg))
        finally:
            tr.ray = old


class Rollouts(unittest.TestCase):
    def test_worker_rollout_with_each_agent(self):
        for kind in KINDS:
            with self.subTest(kind):
                core = lg.SelfPlayCore(0, small_spec(kind, 12), 12, 4)
                r = core.collect(EASY, "bot_easy")
                core.close()
                self.assertEqual(r["obs"].shape, (12, fe.OBS_DIM))
                self.assertEqual(r["feat"].shape, (12, fe.FEATURE_DIM))
                self.assertEqual({"first", "trunc", "next_obs", "next_feat"} <= set(r), kind == "dreamer")
                self.assertEqual(ag.make_agent(small_spec(kind, 12)).rollout_fields == (), kind == "impala")
                if kind == "dreamer":
                    self.assertEqual(r["next_obs"].shape, (12, fe.OBS_DIM))
                    self.assertEqual(r["first"][0], 1.0)
                    self.assertEqual(r["trunc"].sum(), 0.0)
                self.assertTrue(((r["act"] >= 0) & (r["act"] < fe.NUM_ACTIONS)).all())
                self.assertEqual(r["wid"], 0)

    def test_worker_plays_a_league_snapshot_of_each_agent(self):
        for kind in KINDS:
            with self.subTest(kind):
                spec = small_spec(kind, 6)
                core = lg.SelfPlayCore(0, spec, 6, 4)
                weights = ag.make_agent(spec).weights()
                r = core.collect((fe.SELFPLAY_ENV, 1, 1), "snapshot", weights)
                core.close()
                self.assertEqual(r["opponent_type"], "snapshot")
                self.assertEqual(len(r["act"]), 6)

    def test_league_snapshot_fingerprint_and_weights_for_each_agent(self):
        for kind in KINDS:
            with self.subTest(kind):
                spec = small_spec(kind)
                league = lg.PureLeague(Path(tempfile.mkdtemp()), spec)
                agent = ag.make_agent(spec)
                path = league.checkpoint_dir / "snap.pt"
                agent.save_snapshot(path)
                league.add_snapshot("snap", str(path), agent.weights(), lg.TS_ENV.create_rating())
                self.assertEqual(league.policy_fingerprints["snap"].shape, (fe.NUM_ACTIONS,))
                self.assertEqual(set(league.get_member_weights("snap")), set(agent.weights()))

    def test_evaluator_runs_a_short_game_with_each_agent(self):
        orig = fe.FootballEnv.step
        def short(self, actions):
            *views, rew, done = orig(self, actions)
            self.n = getattr(self, "n", 0) + 1
            return (*views, rew, done or self.n % 5 == 0)
        fe.FootballEnv.step = short
        try:
            for kind in KINDS:
                with self.subTest(kind):
                    spec = small_spec(kind, 20)
                    core = lg.EvalCore(spec, 20, 1)
                    episodes = core.evaluate(ag.make_agent(spec).weights(), "bot_easy", 2)
                    core.close()
                    self.assertEqual(len(episodes), 2)
                    self.assertEqual(episodes[0]["length"], 5)
        finally:
            fe.FootballEnv.step = orig


class EvalGreedy(unittest.TestCase):
    def test_eval_core_passes_the_greedy_flag_to_each_agent(self):
        self.assertIn("greedy", tr.CONFIG["eval"])
        for kind in KINDS:
            for greedy in (False, True):
                with self.subTest(kind=kind, greedy=greedy):
                    spec = small_spec(kind, 4)
                    core = lg.EvalCore(spec, 4, 1, greedy)
                    seen = []
                    act = core.agent.act
                    core.agent.act = lambda *a, **k: (seen.append(a[5] if len(a) > 5 else None), act(*a, **k))[1]
                    core.set_weights(ag.make_agent(spec).weights())
                    core.collect(EASY, "bot_easy")
                    core.close()
                    self.assertEqual(set(seen) - {None}, {greedy})


class Updates(unittest.TestCase):
    def rollout(self, kind):
        core = lg.SelfPlayCore(0, small_spec(kind), 200, 8)
        r = core.collect(EASY, "bot_easy")
        core.close()
        return r

    def test_impala_one_update_is_finite(self):
        learner = learner_for("impala")
        r = self.rollout("impala")
        learner.observe(r, 25.0)
        stats = learner.update([r])
        self.assertTrue(stats and all(math.isfinite(float(v)) for v in stats.values()))
        self.assertIn("VT", learner.fmt(stats))

    def test_dreamer_200_step_rollout_and_two_updates_are_finite(self):
        learner = learner_for("dreamer")
        r = self.rollout("dreamer")
        self.assertEqual(len(r["act"]), 200)
        learner.observe(r, 25.0)
        stats = learner.update([r])
        self.assertEqual(stats["model_updates"], 2.0)
        self.assertGreater(len(stats), 3)
        self.assertTrue(all(math.isfinite(float(v)) for v in stats.values()))
        self.assertIn("DR", learner.fmt(stats))

    def test_dreamer_default_max_updates_reaches_the_configured_train_ratio(self):
        self.assertIsNone(tr.CONFIG["dreamer"]["max_updates"])
        cfg = small_cfg("dreamer")
        cfg["dreamer"].update(max_updates=None, train_ratio=4)
        spec = tr.agent_spec(cfg)
        spec["rollout_len"] = 200
        learner = ag.make_agent(spec, learner=True)
        learner.attach(lg.PureLeague(Path(tempfile.mkdtemp()), spec))
        r = self.rollout("dreamer")
        learner.observe(r, 25.0)
        stats = learner.update([r])
        self.assertEqual(stats["model_updates"], 200 * 4 / (2 * 8))
        self.assertAlmostEqual(stats["train_ratio_eff"], 4.0)
        self.assertIn("ratio:4/4", learner.fmt(stats))

    def test_default_dreamer_log_keys_are_real_metrics(self):
        learner = learner_for("dreamer")
        r = self.rollout("dreamer")
        learner.observe(r, 25.0)
        stats = learner.update([r])
        self.assertEqual([k for k in tr.CONFIG["dreamer"]["log_keys"] if k not in stats], [])

    def test_dreamer_game_end_at_the_step_limit_is_truncation_and_a_real_end_is_terminal(self):
        for trunc, term in ((1.0, 0.0), (0.0, 1.0)):
            learner = learner_for("dreamer")
            n = 4
            r = {"wid": 0, "obs": np.ones((n, fe.OBS_DIM), np.float32), "feat": np.ones((n, fe.FEATURE_DIM), np.float32), "act": np.arange(n),
                 "next_obs": np.ones((n, fe.OBS_DIM), np.float32), "next_feat": np.ones((n, fe.FEATURE_DIM), np.float32),
                 "rew": np.zeros(n, np.float32), "done": np.array([0, 0, 0, 1], np.float32), "trunc": np.array([0, 0, 0, trunc], np.float32), "first": np.array([1, 0, 0, 0], np.float32)}
            learner.observe(r)
            learner.observe({**r, "done": np.zeros(n, np.float32), "trunc": np.zeros(n, np.float32)})
            d = learner.replay.data
            self.assertEqual(d["term"][0, :n + 1, 0].tolist(), [0] * n + [term])
            self.assertEqual(d["first"][0, n + 1, 0], 1.0)

    def test_core_marks_a_done_at_the_step_limit_as_truncated(self):
        for ep_len, trunc in ((fe.GAME_STEPS - 1, 1.0), (10, 0.0)):
            core = lg.SelfPlayCore(0, small_spec("dreamer", 3), 3, 4)
            core.collect(EASY, "bot_easy")
            core.ep_len = ep_len
            core.env.step = lambda a, f=core.env.step: (*f(a)[:5], True)
            r = core.collect(EASY, "bot_easy")
            core.close()
            self.assertEqual((r["done"].tolist(), r["trunc"].tolist()), ([1.0], [trunc]))

    def test_dreamer_replay_holds_terminal_records_and_carries_reward(self):
        learner = learner_for("dreamer")
        n = 6
        r = {"wid": 0, "obs": np.ones((n, fe.OBS_DIM), np.float32), "feat": np.ones((n, fe.FEATURE_DIM), np.float32), "act": np.arange(n),
             "next_obs": np.full((n, fe.OBS_DIM), 2, np.float32), "next_feat": np.full((n, fe.FEATURE_DIM), 2, np.float32),
             "rew": np.array([0, 0, 1, 0, 0, 1], np.float32), "done": np.array([0, 0, 0, 0, 0, 1], np.float32), "first": np.array([1, 0, 0, 0, 0, 0], np.float32), "trunc": np.zeros(n, np.float32)}
        learner.observe(r)
        d = learner.replay.data
        self.assertEqual(int(learner.replay.n[0]), n + 1)
        self.assertEqual(d["rew"][0, :n + 1, 0].tolist(), [0, 0, 0, 1, 0, 0, 1])
        self.assertEqual(d["term"][0, :n + 1, 0].tolist(), [0] * n + [1])
        self.assertEqual(d["first"][0, :n + 1, 0].tolist(), [1] + [0] * n)
        self.assertEqual(float(d["act"][0, n].sum()), 0.0)
        self.assertEqual(int(d["act"][0, 3].argmax()), 3)


class Switching(unittest.TestCase):
    def test_only_the_agent_key_changes_between_agents(self):
        for kind in KINDS:
            with self.subTest(kind):
                cfg = small_cfg(kind)
                spec = tr.agent_spec(cfg)
                self.assertEqual(spec["agent"], kind)
                learner = ag.make_agent(spec, learner=True)
                learner.attach(lg.PureLeague(Path(tempfile.mkdtemp()), spec))
                core = lg.SelfPlayCore(0, spec, 200, 8)
                core.set_weights(learner.weights())
                r = core.collect(EASY, "bot_easy")
                core.close()
                learner.observe(r, 25.0)
                self.assertTrue(all(math.isfinite(float(v)) for v in learner.update([r]).values()))
        self.assertEqual(tr.CONFIG["agent"], "impala")
        self.assertEqual(set(tr.agent_spec(small_cfg("impala"))) & {"model"}, set())


class FakeRef:
    def __init__(self, fn):
        self.fn = fn


class FakeMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *a, **k):
        return FakeRef(lambda: self.fn(*a, **k))


class FakeWorker:
    def __init__(self, core):
        self.collect, self.set_weights, self.close = FakeMethod(core.collect), FakeMethod(core.set_weights), FakeMethod(core.close)


class FakeWorkerClass:
    def options(self, **kw):
        return self

    def remote(self, wid, spec, rollout_len, goal_traj_len):
        return FakeWorker(lg.SelfPlayCore(wid, spec, rollout_len, goal_traj_len))


class FakeRay:
    def get(self, ref, timeout=None):
        return [r.fn() for r in ref] if isinstance(ref, list) else ref.fn()

    def wait(self, refs, num_returns=1, timeout=None):
        return refs[:num_returns], refs[num_returns:]

    def cancel(self, ref):
        pass

    def kill(self, actor):
        pass


class Stop(Exception):
    pass


class TrainerLoop(unittest.TestCase):
    def test_progress_line_shows_the_windowed_goal_difference_for_each_agent(self):
        old = tr.ray, lg.ray, tr.SelfPlayWorker
        tr.ray = lg.ray = FakeRay()
        tr.SelfPlayWorker = FakeWorkerClass()
        try:
            for kind in KINDS:
                cfg = small_cfg(kind)
                cfg["paths"]["checkpoint_dir"] = tempfile.mkdtemp()
                cfg.update(eval={**cfg["eval"], "every": 0})
                t = tr.LeagueTrainer(cfg)
                t.start = 0.0
                t.goal_diffs.extend([2, -1, 0, 1])
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    t._log_progress({"model_updates": 1.0, "train_ratio_eff": 1.0, **{k: 0.0 for k in ("loss", "pi", "v", "ent", "rho", "rho_max", "ev", "grad", "grad_clip")}})
                t.close()
                self.assertIn("GD:+0.50", buf.getvalue())
        finally:
            tr.ray, lg.ray, tr.SelfPlayWorker = old

    def test_trainer_runs_updates_checkpoints_and_resumes_with_each_agent(self):
        old = tr.ray, lg.ray, tr.SelfPlayWorker
        tr.ray = lg.ray = FakeRay()
        tr.SelfPlayWorker = FakeWorkerClass()
        try:
            for kind in KINDS:
                with self.subTest(kind):
                    cfg = small_cfg(kind)
                    cfg["paths"]["checkpoint_dir"] = tempfile.mkdtemp()
                    cfg.update(eval={**cfg["eval"], "every": 0}, checkpoint_every=2)
                    cfg["learner"].update(rollout_len=40, num_workers=2, batch_size=2)

                    class Short(tr.LeagueTrainer):
                        def _update(self, batch):
                            stats = super()._update(batch)
                            if self.updates >= 2:
                                raise Stop
                            return stats

                    trainer = Short(cfg)
                    with self.assertRaises(Stop):
                        trainer.train(max_time=600)
                    self.assertEqual(trainer.updates, 2)
                    self.assertGreater(trainer.total_steps, 0)
                    trainer.updates += 1
                    trainer._save_checkpoint()
                    again = Short({**cfg, "resume": "latest"})
                    self.assertEqual((again.updates, again.total_steps), (trainer.updates, trainer.total_steps))
                    trainer.close()
                    again.close()
        finally:
            tr.ray, lg.ray, tr.SelfPlayWorker = old


class ExpertGeneration(unittest.TestCase):
    def test_short_game_loads_through_bc_and_expert_buffer(self):
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d) / "expert.parquet")
            report = tr.run({"mode": "gen_expert", "expert_gen": {"games": 1, "keep": "all", "max_steps": 150, "workers": 0, "out": out}})
            self.assertEqual((report["kept_games"], report["kept_rows"], report["inferred_action_fraction"]), (1, 150, 0.0))
            self.assertAlmostEqual(sum(report["action_hist"].values()), 1.0, places=3)
            table = pq.read_table(out)
            self.assertEqual(set(table.column_names), set(fe.ExpertGenerator.COLUMNS))
            self.assertEqual(len(np.frombuffer(table["obs"][0].as_py(), dtype=np.float32)), fe.OBS_DIM)
            self.assertTrue(0 <= min(table["action"].to_pylist()) and max(table["action"].to_pylist()) < fe.NUM_ACTIONS)
            buf = ag.ExpertBufferGPU(out, torch.device("cpu"), 64)
            self.assertEqual((buf.num_episodes, len(buf.obs)), (1, 150))
            self.assertEqual(buf.sample(2)["obs"].shape[-1], fe.OBS_DIM)
            save = str(Path(d) / "bc.pt")
            with contextlib.redirect_stdout(io.StringIO()):
                model, _ = ag.BCWarmstart(small_spec("impala"), out, save, epochs=1, batch_size=32).run()
            self.assertTrue(Path(save).exists())


if __name__ == "__main__":
    unittest.main()
