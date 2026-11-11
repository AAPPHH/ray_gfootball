import copy
import math
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
import numpy as np
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


class Rollouts(unittest.TestCase):
    def test_worker_rollout_with_each_agent(self):
        for kind in KINDS:
            with self.subTest(kind):
                core = lg.SelfPlayCore(0, small_spec(kind, 12), 12, 4)
                r = core.collect(EASY, "bot_easy")
                core.close()
                self.assertEqual(r["obs"].shape, (12, fe.OBS_DIM))
                self.assertEqual(r["feat"].shape, (12, fe.FEATURE_DIM))
                self.assertEqual(r["next_obs"].shape, (12, fe.OBS_DIM))
                self.assertEqual(r["first"][0], 1.0)
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

    def test_dreamer_replay_holds_terminal_records_and_carries_reward(self):
        learner = learner_for("dreamer")
        n = 6
        r = {"wid": 0, "obs": np.ones((n, fe.OBS_DIM), np.float32), "feat": np.ones((n, fe.FEATURE_DIM), np.float32), "act": np.arange(n),
             "next_obs": np.full((n, fe.OBS_DIM), 2, np.float32), "next_feat": np.full((n, fe.FEATURE_DIM), 2, np.float32),
             "rew": np.array([0, 0, 1, 0, 0, 1], np.float32), "done": np.array([0, 0, 0, 0, 0, 1], np.float32), "first": np.array([1, 0, 0, 0, 0, 0], np.float32)}
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


if __name__ == "__main__":
    unittest.main()
