import os
import sys
import tempfile
import unittest
import torch
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "main"))
import full_league as fl


def make_league():
    return fl.PureLeague(Path(tempfile.mkdtemp()))


class StickyOpponent(unittest.TestCase):
    def test_opponent_kept_until_game_over(self):
        d = fl.StickyDispatcher(make_league(), ["bot_hard"], eval_period=50)
        first = d.assign(0, True, fl.TS_ENV.create_rating())
        for _ in range(20):
            again = d.assign(0, False, fl.TS_ENV.create_rating())
            self.assertEqual(again[:3], first[:3])
            self.assertIsNone(again[3])
        self.assertEqual(d.cycle, 1)

    def test_new_assignment_only_after_game_over(self):
        d = fl.StickyDispatcher(make_league(), ["random", "lazy", "bot_easy"], eval_period=50)
        seen = [d.assign(0, True, fl.TS_ENV.create_rating())[0] for _ in range(60)]
        self.assertEqual(d.cycle, 60)
        self.assertEqual(seen[:2], ["lazy", "bot_easy"])

    def test_workers_independent(self):
        d = fl.StickyDispatcher(make_league(), [], eval_period=50)
        a = d.assign(0, True, fl.TS_ENV.create_rating())
        d.assign(1, True, fl.TS_ENV.create_rating())
        self.assertEqual(d.assign(0, False, fl.TS_ENV.create_rating())[:3], a[:3])

    def test_forced_eval_is_whole_game(self):
        d = fl.StickyDispatcher(make_league(), ["bot_hard"], eval_period=1)
        r = d.assign(0, True, fl.TS_ENV.create_rating())
        self.assertEqual(r[0], "bot_hard")
        self.assertEqual(r[1][0], "11_vs_11_hard_stochastic")
        for _ in range(5):
            self.assertEqual(d.assign(0, False, fl.TS_ENV.create_rating())[0], "bot_hard")


class ReportAttribution(unittest.TestCase):
    def test_game_credited_to_assigned_opponent(self):
        league = make_league()
        d = fl.StickyDispatcher(league, ["bot_medium"], eval_period=1)
        opponent = d.assign(0, True, fl.TS_ENV.create_rating())[0]
        d.assign(0, False, fl.TS_ENV.create_rating())
        d.assign(0, False, fl.TS_ENV.create_rating())
        league.report_game(d.assigned[0][0], fl.TS_ENV.create_rating(), won=True)
        self.assertEqual(opponent, "bot_medium")
        self.assertEqual(league.members["bot_medium"].games_played, 1)
        self.assertEqual(sum(m.games_played for m in league.members.values()), 1)


class ExpertGate(unittest.TestCase):
    def learner(self, hard_results, min_games=20, threshold=0.7):
        league = make_league()
        for won in hard_results:
            league.report_game("bot_hard", fl.TS_ENV.create_rating(), won=won)
        gate = fl.LeagueLearner.__new__(fl.LeagueLearner)
        gate.league, gate.expert_min_games, gate.expert_threshold = league, min_games, threshold
        return gate

    def test_empty_history_is_not_a_win_rate(self):
        league = make_league()
        stats = league.get_stats()["bots"]["bot_hard"]
        self.assertEqual((stats["games"], stats["wr"], stats["dr"], stats["lr"]), (0, 0.0, 0.0, 0.0))
        self.assertEqual(fl.LeagueLearner._opp_col("H", stats), "H:-(0)")

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
            league.report_game(name, fl.TS_ENV.create_rating(), won=True)
        stats = league.get_stats()
        self.assertEqual(stats["games_by_opp"]["bot_hard"], 2)
        self.assertEqual(stats["games_by_opp"]["bot_easy"], 1)
        self.assertEqual(stats["games_by_opp"]["random"], 0)
        self.assertEqual(stats["snap_games"], 0)


class ExplainedVariance(unittest.TestCase):
    def test_perfect_and_constant_predictions(self):
        t = torch.randn(4, 50)
        self.assertAlmostEqual(fl.explained_variance(t.clone(), t), 1.0, places=6)
        self.assertAlmostEqual(fl.explained_variance(torch.full_like(t, t.mean().item()), t), 0.0, places=5)

    def test_known_value_and_mask(self):
        t = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        v = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        self.assertAlmostEqual(fl.explained_variance(v, t, mask), 1.0, places=6)
        v2 = t + torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        self.assertAlmostEqual(fl.explained_variance(v2, t), 1.0 - 1.0 / 1.25, places=6)

    def test_zero_variance_targets(self):
        self.assertEqual(fl.explained_variance(torch.zeros(3, 3), torch.ones(3, 3)), 0.0)


if __name__ == "__main__":
    unittest.main()
