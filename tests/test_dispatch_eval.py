import os
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
