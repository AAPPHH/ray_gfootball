import os
import sys
import json
import tempfile
import unittest
import torch
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import agents as ag
import league as lg
import train as tr


SPEC = {"agent": "impala", "d_model": 8, "lstm_hidden": 8}


def make_league():
    return lg.PureLeague(Path(tempfile.mkdtemp()))


class StickyOpponent(unittest.TestCase):
    def test_opponent_kept_until_game_over(self):
        d = lg.StickyDispatcher(make_league(), ["bot_hard"], eval_period=50)
        first = d.assign(0, True, lg.TS_ENV.create_rating())
        for _ in range(20):
            again = d.assign(0, False, lg.TS_ENV.create_rating())
            self.assertEqual(again[:3], first[:3])
            self.assertIsNone(again[3])
        self.assertEqual(d.cycle, 1)

    def test_new_assignment_only_after_game_over(self):
        d = lg.StickyDispatcher(make_league(), ["random", "lazy", "bot_easy"], eval_period=50)
        seen = [d.assign(0, True, lg.TS_ENV.create_rating())[0] for _ in range(60)]
        self.assertEqual(d.cycle, 60)
        self.assertEqual(seen[:2], ["lazy", "bot_easy"])

    def test_workers_independent(self):
        d = lg.StickyDispatcher(make_league(), [], eval_period=50)
        a = d.assign(0, True, lg.TS_ENV.create_rating())
        d.assign(1, True, lg.TS_ENV.create_rating())
        self.assertEqual(d.assign(0, False, lg.TS_ENV.create_rating())[:3], a[:3])

    def test_forced_eval_is_whole_game(self):
        d = lg.StickyDispatcher(make_league(), ["bot_hard"], eval_period=1)
        r = d.assign(0, True, lg.TS_ENV.create_rating())
        self.assertEqual(r[0], "bot_hard")
        self.assertEqual(r[1][0], "11_vs_11_hard_stochastic")
        for _ in range(5):
            self.assertEqual(d.assign(0, False, lg.TS_ENV.create_rating())[0], "bot_hard")


class ReportAttribution(unittest.TestCase):
    def test_game_credited_to_assigned_opponent(self):
        league = make_league()
        d = lg.StickyDispatcher(league, ["bot_medium"], eval_period=1)
        opponent = d.assign(0, True, lg.TS_ENV.create_rating())[0]
        d.assign(0, False, lg.TS_ENV.create_rating())
        d.assign(0, False, lg.TS_ENV.create_rating())
        league.report_game(d.assigned[0][0], lg.TS_ENV.create_rating(), won=True)
        self.assertEqual(opponent, "bot_medium")
        self.assertEqual(league.members["bot_medium"].games_played, 1)
        self.assertEqual(sum(m.games_played for m in league.members.values()), 1)


class SnapshotInUse(unittest.TestCase):
    def league_with_snapshots(self, n):
        league = make_league()
        for i in range(n):
            league.members[f"s{i}"] = lg.LeagueMember(name=f"s{i}", member_type="snapshot", rating=lg.TS_ENV.create_rating(mu=10.0 + i * 0.1), rating_update_factor=1.0, created_idx=i)
        return league

    def test_assigned_snapshot_survives_pruning_and_spacing(self):
        league = self.league_with_snapshots(20)
        d = lg.StickyDispatcher(league, [], eval_period=50)
        d.assigned[0] = ("s0", None, "snapshot")
        league.max_snapshots = 5
        league._maybe_prune_snapshots()
        self.assertIn("s0", league.members)
        self.assertLess(len(league.members), 5 + 20)
        league = self.league_with_snapshots(10)
        d = lg.StickyDispatcher(league, [], eval_period=50)
        d.assigned[0] = ("s0", None, "snapshot")
        league._ensure_skill_spacing()
        self.assertIn("s0", league.members)
        self.assertNotIn("s1", league.members)

    def test_unassigned_snapshot_is_pruned_after_game_moves_on(self):
        league = self.league_with_snapshots(10)
        d = lg.StickyDispatcher(league, [], eval_period=50)
        d.assigned[0] = ("s0", None, "snapshot")
        d.assigned[0] = ("bot_easy", None, "bot_easy")
        league._ensure_skill_spacing()
        self.assertNotIn("s0", league.members)

    def test_game_for_removed_opponent_is_counted_as_dropped(self):
        league = make_league()
        league.report_game("gone", lg.TS_ENV.create_rating(), won=True)
        self.assertEqual(league.dropped_games, 1)
        self.assertEqual(league.total_games, 0)


class ExpertGate(unittest.TestCase):
    def learner(self, hard_results, min_games=20, threshold=0.7):
        league = make_league()
        for won in hard_results:
            league.report_game("bot_hard", lg.TS_ENV.create_rating(), won=won)
        gate = ag.ImpalaAgent.__new__(ag.ImpalaAgent)
        gate.league, gate.expert_min_games, gate.expert_threshold = league, min_games, threshold
        return gate

    def test_empty_history_is_not_a_win_rate(self):
        league = make_league()
        stats = league.get_stats()["bots"]["bot_hard"]
        self.assertEqual((stats["games"], stats["wr"], stats["dr"], stats["lr"]), (0, 0.0, 0.0, 0.0))
        self.assertEqual(tr.LeagueTrainer._opp_col("H", stats), "H:-(0)")

    def test_gate_stays_open_below_min_games(self):
        self.assertFalse(self.learner([True] * 19).expert_gate_closed())

    def test_gate_closes_after_min_games_with_high_win_rate(self):
        self.assertTrue(self.learner([True] * 20).expert_gate_closed())

    def test_gate_stays_open_with_low_win_rate(self):
        self.assertFalse(self.learner([False] * 30).expert_gate_closed())


class GameCounters(unittest.TestCase):
    def test_finished_games_per_opponent(self):
        league = make_league()
        for name in ["bot_hard", "bot_hard", "bot_easy"]:
            league.report_game(name, lg.TS_ENV.create_rating(), won=True)
        stats = league.get_stats()
        self.assertEqual(stats["games_by_opp"]["bot_hard"], 2)
        self.assertEqual(stats["games_by_opp"]["bot_easy"], 1)
        self.assertEqual(stats["games_by_opp"]["random"], 0)
        self.assertEqual(stats["snap_games"], 0)


class ExplainedVariance(unittest.TestCase):
    def test_perfect_and_constant_predictions(self):
        t = torch.randn(4, 50)
        self.assertAlmostEqual(ag.explained_variance(t.clone(), t), 1.0, places=6)
        self.assertAlmostEqual(ag.explained_variance(torch.full_like(t, t.mean().item()), t), 0.0, places=5)

    def test_known_value_and_mask(self):
        t = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        v = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        self.assertAlmostEqual(ag.explained_variance(v, t, mask), 1.0, places=6)
        v2 = t + torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        self.assertAlmostEqual(ag.explained_variance(v2, t), 1.0 - 1.0 / 1.25, places=6)

    def test_zero_variance_targets(self):
        self.assertEqual(ag.explained_variance(torch.zeros(3, 3), torch.ones(3, 3)), 0.0)


class WilsonCI(unittest.TestCase):
    def test_known_values(self):
        lo, hi = lg.wilson_ci(10, 20)
        self.assertAlmostEqual(lo, 0.2993, places=3)
        self.assertAlmostEqual(hi, 0.7007, places=3)
        self.assertEqual(lg.wilson_ci(0, 0), (0.0, 1.0))
        lo, hi = lg.wilson_ci(50, 50)
        self.assertLessEqual(hi, 1.0)
        self.assertGreater(lo, 0.9)
        self.assertEqual(lg.wilson_ci(0, 50)[0], 0.0)

    def test_summary(self):
        eps = [{"won": True, "drawn": False, "goals_scored": 2, "goals_conceded": 0}, {"won": False, "drawn": True, "goals_scored": 1, "goals_conceded": 1},
               {"won": False, "drawn": False, "goals_scored": 0, "goals_conceded": 3}]
        s = lg.summarize_games(eps)
        self.assertEqual((s["n"], s["win"], s["draw"], s["loss"]), (3, 1, 1, 1))
        self.assertAlmostEqual(s["goal_diff"], -1.0 / 3)
        self.assertEqual(lg.summarize_games([])["n"], 0)


class FakeRay:
    def __init__(self):
        self.ready, self.killed = set(), []

    def put(self, x):
        return x

    def wait(self, refs, num_returns, timeout):
        return [r for r in refs if r.key in self.ready], [r for r in refs if r.key not in self.ready]

    def get(self, ref):
        if ref.episodes is None:
            raise RuntimeError("actor died")
        return ref.episodes

    def kill(self, actor):
        self.killed.append(actor.key)


class FakeRef:
    def __init__(self, key, episodes):
        self.key, self.episodes = key, episodes


class FakeActor:
    def __init__(self, key):
        self.key, self.broken = key, set()
        self.evaluate = self

    def remote(self, weights, bot, games):
        if self.key in self.broken:
            return FakeRef(self.key, None)
        return FakeRef(self.key, [{"won": True, "drawn": False, "goals_scored": 1, "goals_conceded": 0}] * games)


class EvalRunnerFlow(unittest.TestCase):
    def test_non_blocking_round_written_to_jsonl(self):
        fake, old = FakeRay(), lg.ray
        lg.ray = fake
        try:
            out = Path(tempfile.mkdtemp()) / "eval.jsonl"
            r = lg.EvalRunner.__new__(lg.EvalRunner)
            r.cfg, r.out_path = {"every": 50, "games": 4, "bots": ["bot_easy", "bot_hard"]}, out
            r.actors = {"bot_easy": FakeActor("a"), "bot_hard": FakeActor("b")}
            r.refs, r.results, r.errors, r.meta, r.last = {}, {}, {}, None, {}
            self.assertFalse(r.due(49))
            self.assertTrue(r.due(50))
            r.start(50, 1000, {})
            self.assertFalse(r.due(100))
            self.assertIsNone(r.poll())
            fake.ready.add("a")
            self.assertIsNone(r.poll())
            self.assertFalse(out.exists())
            fake.ready.add("b")
            rec = r.poll()
            self.assertEqual(rec["update"], 50)
            self.assertEqual(rec["bots"]["bot_hard"]["n"], 4)
            self.assertEqual(json.loads(out.read_text())["bots"]["bot_easy"]["win"], 4)
            self.assertTrue(r.due(100))
            self.assertEqual(rec["errors"], {})
        finally:
            lg.ray = old

    def test_failing_bot_keeps_other_results_and_actor_is_recreated(self):
        fake, old = FakeRay(), lg.ray
        lg.ray = fake
        try:
            out = Path(tempfile.mkdtemp()) / "eval.jsonl"
            r = lg.EvalRunner.__new__(lg.EvalRunner)
            r.cfg, r.out_path = {"every": 50, "games": 4, "bots": ["bot_easy", "bot_hard"]}, out
            bad, spare = FakeActor("b"), FakeActor("c")
            bad.broken.add("b")
            r.actors = {"bot_easy": FakeActor("a"), "bot_hard": bad}
            r.make = lambda: spare
            r.refs, r.results, r.errors, r.meta, r.last = {}, {}, {}, None, {}
            r.start(50, 1000, {})
            fake.ready.update(["a", "b"])
            rec = r.poll()
            self.assertEqual(rec["errors"], {"bot_hard": "RuntimeError"})
            self.assertEqual(rec["n_errors"], 1)
            self.assertEqual(rec["bots"]["bot_easy"]["n"], 4)
            self.assertNotIn("bot_hard", rec["bots"])
            self.assertIs(r.actors["bot_hard"], spare)
            self.assertEqual(fake.killed, ["b"])
            self.assertEqual(json.loads(out.read_text())["n_errors"], 1)
            self.assertFalse(r.refs)
            self.assertTrue(r.due(100))
        finally:
            lg.ray = old


class EvalBotsValidation(unittest.TestCase):
    def test_only_builtin_bots_allowed(self):
        for bad in ("random", "lazy", "nope"):
            cfg = {"every": 0, "games": 1, "bots": ["bot_easy", bad], "file": "e", "torch_threads": 1, "actor": {}}
            with self.assertRaises(ValueError):
                lg.EvalRunner(cfg, SPEC, 8, Path(tempfile.mkdtemp()) / "e.jsonl")

    def test_default_bots_accepted(self):
        cfg = {**tr.CONFIG["eval"], "every": 0}
        self.assertEqual(lg.EvalRunner(cfg, SPEC, 8, Path(tempfile.mkdtemp()) / "e.jsonl").actors, {})


if __name__ == "__main__":
    unittest.main()
