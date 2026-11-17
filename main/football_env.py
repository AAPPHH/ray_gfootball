from collections import Counter
import numpy as np
import pyarrow as pa
import ray
import gfootball.env as football_env

BOT_CONFIGS = [
    ("random", 5.0, 2.0, "11_vs_11_stochastic", False),
    ("lazy", 8.0, 2.0, "11_vs_11_stochastic", False),
    ("bot_easy", 18.0, 2.0, "11_vs_11_easy_stochastic", True),
    ("bot_medium", 28.0, 2.0, "11_vs_11_stochastic", True),
    ("bot_hard", 38.0, 2.0, "11_vs_11_hard_stochastic", True),
]
BOT_ENVS = {name: env_name for name, _, _, env_name, _ in BOT_CONFIGS}
BUILTIN_BOTS = ("bot_easy", "bot_medium", "bot_hard")
SELFPLAY_ENV = "11_vs_11_stochastic"
FEATURE_DIM, OBS_DIM, NUM_ACTIONS = 93, 115, 19
NO_ACTION = NUM_ACTIONS
GAME_STEPS = 3000

def goal_event(rew: float) -> int:
    return 1 if rew > 0.5 else -1 if rew < -0.5 else 0

def raw_obs(x) -> np.ndarray:
    return np.array(x).flatten()[:OBS_DIM].astype(np.float32)

class FeatureEngineer:
    GOAL = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL = np.array([-1.0, 0.0], dtype=np.float32)

    @staticmethod
    def extract(obs: np.ndarray, active_idx: int = None) -> np.ndarray:
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.reshape(1, -1)
        B = obs.shape[0]
        obs = obs[:, :115] if obs.shape[1] >= 115 else np.pad(obs, ((0, 0), (0, 115 - obs.shape[1])))
        feat = np.zeros((B, FEATURE_DIM), dtype=np.float32)
        left_pos = obs[:, 0:22].reshape(B, 11, 2)
        left_dir = obs[:, 22:44].reshape(B, 11, 2)
        right_pos = obs[:, 44:66].reshape(B, 11, 2)
        right_dir = obs[:, 66:88].reshape(B, 11, 2)
        ball_pos, ball_z, ball_dir = obs[:, 88:90], obs[:, 90:91], obs[:, 91:94]
        ball_owned_team = np.argmax(obs[:, 94:97], axis=1) - 1
        game_mode, sticky = obs[:, 98:105], obs[:, 105:115]
        if active_idx is None:
            active_idx = np.argmax(obs[:, 97:108], axis=1)
        elif isinstance(active_idx, int):
            active_idx = np.full(B, active_idx)
        bi = np.arange(B)
        active_pos = left_pos[bi, active_idx]
        feat[:, 0:2] = ball_pos
        feat[:, 2] = np.clip(ball_z[:, 0], 0, 1)
        feat[:, 3] = np.clip(np.linalg.norm(ball_dir[:, :2], axis=1), 0, 2)
        feat[:, 4:6] = ball_dir[:, :2]
        feat[:, 6] = ball_owned_team / 2.0
        rel_ball = ball_pos - active_pos
        feat[:, 7:9] = rel_ball
        feat[:, 9] = np.clip(np.linalg.norm(rel_ball, axis=1), 0, 2)
        feat[:, 10] = np.arctan2(rel_ball[:, 1], rel_ball[:, 0]) / np.pi
        goal_vec = FeatureEngineer.GOAL - ball_pos
        dist_goal = np.linalg.norm(goal_vec, axis=1)
        feat[:, 11] = np.clip(dist_goal, 0, 2)
        feat[:, 12] = np.abs(np.arctan2(goal_vec[:, 1], goal_vec[:, 0])) / np.pi
        feat[:, 13] = np.clip(0.088 / (dist_goal + 0.01), 0, 1)
        feat[:, 14] = (dist_goal < 0.35).astype(np.float32)
        feat[:, 15] = np.clip(np.linalg.norm(FeatureEngineer.OWN_GOAL - ball_pos, axis=1), 0, 2)
        right_x = right_pos[:, :, 0]
        keeper_idx = np.argmax(right_x, axis=1)
        keeper_pos = right_pos[bi, keeper_idx]
        keeper_dist = np.linalg.norm(ball_pos - keeper_pos, axis=1)
        feat[:, 16] = np.clip(keeper_dist, 0, 2)
        feat[:, 17] = np.arctan2((ball_pos - keeper_pos)[:, 1], (ball_pos - keeper_pos)[:, 0]) / np.pi
        left_active = np.abs(left_pos[:, :, 0]) > 0.01
        tm_dist = np.linalg.norm(left_pos - active_pos[:, None, :], axis=2)
        tm_dist[bi, active_idx] = 999.0
        tm_dist = np.where(left_active, tm_dist, 999.0)
        tm_idx = np.argsort(tm_dist, axis=1)
        for i in range(5):
            idx = tm_idx[:, i]
            valid = tm_dist[bi, idx] < 100
            feat[:, 18+i*4:20+i*4] = np.where(valid[:, None], left_pos[bi, idx] - active_pos, 0)
            feat[:, 20+i*4:22+i*4] = np.where(valid[:, None], left_dir[bi, idx], 0)
        right_active = np.abs(right_pos[:, :, 0]) > 0.01
        op_dist = np.where(right_active, np.linalg.norm(right_pos - active_pos[:, None, :], axis=2), 999.0)
        op_idx = np.argsort(op_dist, axis=1)
        for i in range(5):
            idx = op_idx[:, i]
            valid = op_dist[bi, idx] < 100
            feat[:, 38+i*4:40+i*4] = np.where(valid[:, None], right_pos[bi, idx] - active_pos, 0)
            feat[:, 40+i*4:42+i*4] = np.where(valid[:, None], right_dir[bi, idx], 0)
        ball_x = ball_pos[:, 0]
        left_x = left_pos[:, :, 0]
        feat[:, 58] = np.sum((left_x > ball_x[:, None]) & left_active, axis=1) / 11.0
        feat[:, 59] = np.sum((right_x > ball_x[:, None]) & right_active, axis=1) / 11.0
        feat[:, 60] = np.clip((feat[:, 58] - feat[:, 59]) * 2, -1, 1)
        opp_dist_active = np.where(right_active, np.linalg.norm(right_pos - active_pos[:, None, :], axis=2), 10.0)
        feat[:, 63] = np.clip(np.min(opp_dist_active, axis=1), 0, 1)
        offside_line = np.maximum(ball_x, np.sort(right_x, axis=1)[:, 1])
        feat[:, 67] = offside_line
        feat[:, 68] = ((active_pos[:, 0] > offside_line) & (ball_owned_team == 0)).astype(np.float32)
        feat[:, 69:72] = np.column_stack([(ball_x > 0.33), ((ball_x >= -0.33) & (ball_x <= 0.33)), (ball_x < -0.33)])
        feat[:, 72:74] = np.column_stack([(ball_pos[:, 1] > 0.2), (ball_pos[:, 1] < -0.2)])
        feat[:, 74:76] = sticky[:, 8:10]
        sticky_dir = sticky[:, :8]
        sticky_active = np.any(sticky_dir > 0, axis=1)
        angle = np.argmax(sticky_dir, axis=1) * (2 * np.pi / 8)
        feat[:, 76:78] = np.column_stack([np.where(sticky_active, np.cos(angle), 0), np.where(sticky_active, np.sin(angle), 0)])
        feat[:, 78:85] = game_mode
        feat[:, 85:93] = 0
        return feat[0] if squeeze else feat

class FootballEnv:
    def __init__(self, env_name: str, left: int, right: int, rewards: str = "scoring"):
        self.key = (env_name, left, right, rewards)
        self.right = right
        self.env = football_env.create_environment(
            env_name=env_name, representation="simple115v2",
            number_of_left_players_agent_controls=left,
            number_of_right_players_agent_controls=right,
            rewards=rewards, render=False
        )

    def _views(self, raw):
        if self.right > 0:
            if isinstance(raw, list) and len(raw) == 2:
                own, opp = raw_obs(raw[0]), raw_obs(raw[1])
            else:
                own = raw_obs(raw)
                opp = own.copy()
            return own, FeatureEngineer.extract(own), opp, FeatureEngineer.extract(opp)
        own = raw_obs(raw)
        return own, FeatureEngineer.extract(own), None, None

    def reset(self):
        return self._views(self.env.reset())

    def step(self, actions):
        raw, rew, done, _ = self.env.step(actions)
        rew = float(rew[0]) if isinstance(rew, (list, np.ndarray)) else float(rew)
        done = bool(done[0] if isinstance(done, (list, np.ndarray)) else done)
        return (*self._views(raw), rew, done)

    def close(self):
        self.env.close()

class ExpertTeacher:
    DIRS = (5, 4, 3, 2, 1, 8, 7, 6)
    SHOT, SHORT_PASS, SPRINT, RELEASE_SPRINT, SLIDE = 12, 11, 13, 15, 16

    def __init__(self, cfg: dict):
        self.cfg, self.sprinting, self.cooldown = cfg, False, 0

    def reset(self):
        self.sprinting, self.cooldown = False, 0

    def direction(self, delta) -> int:
        if np.abs(delta).max() < self.cfg["deadzone"]:
            return 0
        return self.DIRS[int(np.round(np.arctan2(-delta[1], delta[0]) / (np.pi / 4))) % 8]

    def act(self, obs: np.ndarray) -> int:
        c = self.cfg
        self.cooldown = max(self.cooldown - 1, 0)
        me = int(np.argmax(obs[97:108]))
        left, right, ball = obs[0:22].reshape(11, 2), obs[44:66].reshape(11, 2), obs[88:90]
        owner, pos = int(np.argmax(obs[94:97])) - 1, left[me]
        d_ball, d_goal = np.linalg.norm(ball - pos), np.linalg.norm(np.array([1.0, 0.0]) - ball)
        if owner == 0 and d_ball < c["own_ball_dist"]:
            if d_goal < c["shot_dist"] or (ball[0] > c["shot_x"] and abs(ball[1]) < c["shot_y"]):
                return self.SHOT
            pressed = np.linalg.norm(right - pos, axis=1).min() < c["press_dist"]
            mates = [i for i in range(11) if i != me and left[i][0] > pos[0] and np.linalg.norm(left[i] - pos) < c["pass_dist"]]
            if pressed and mates and d_goal > c["shot_dist"]:
                return self.SHORT_PASS
            return self.direction(np.array([1.0, 0.0]) - ball)
        if owner == 1 and d_ball < c["slide_dist"] and not self.cooldown:
            self.cooldown = c["slide_cooldown"]
            return self.SLIDE
        far = d_ball > c["sprint_dist"]
        if far != self.sprinting:
            self.sprinting = far
            return self.SPRINT if far else self.RELEASE_SPRINT
        return self.direction(ball - pos)

class ExpertGenerator:
    COLUMNS = ("obs", "action", "reward", "active", "episode_id", "step", "score")

    def __init__(self, cfg: dict):
        unknown = [o for o in cfg["opponents"] if o not in BOT_ENVS or o in ("random", "lazy")]
        if unknown:
            raise ValueError(f"unsupported opponents {unknown}")
        self.cfg, self.teacher, self.envs = cfg, ExpertTeacher(cfg["teacher"]), {}

    def play(self, episode_id: int, opponent: str) -> dict:
        if opponent not in self.envs:
            self.envs[opponent] = FootballEnv(BOT_ENVS[opponent], 1, 0)
        env = self.envs[opponent]
        self.teacher.reset()
        obs = env.reset()[0]
        rows, goals, done = [], [0, 0], False
        while not done and len(rows) < self.cfg["max_steps"]:
            action = self.teacher.act(obs)
            nxt, _, _, _, rew, done = env.step([action])
            event = goal_event(rew)
            goals[event < 0] += abs(event)
            rows.append((obs.tobytes(), action, float(event > 0), int(np.argmax(obs[97:108])), episode_id, len(rows)))
            obs = nxt
        return {"opponent": opponent, "episode_id": episode_id, "goals_for": goals[0], "goals_against": goals[1], "rows": rows}

    @staticmethod
    def keep(cfg: dict, game: dict) -> bool:
        return cfg["keep"] == "all" or game["goals_for"] > game["goals_against"]

    def games(self, ids) -> list:
        opponents = self.cfg["opponents"]
        return [self.play(i, opponents[i % len(opponents)]) for i in ids]

    def close(self):
        for env in self.envs.values():
            env.close()
        self.envs = {}

    @staticmethod
    def table(games: list) -> pa.Table:
        cols = {k: [] for k in ExpertGenerator.COLUMNS}
        for g in games:
            for row in g["rows"]:
                for k, v in zip(ExpertGenerator.COLUMNS, (*row, g["goals_for"])):
                    cols[k].append(v)
        types = {"obs": pa.binary(), "action": pa.int64(), "reward": pa.float32(), "active": pa.int64(), "episode_id": pa.int64(), "step": pa.int64(), "score": pa.int64()}
        return pa.table({k: pa.array(v, type=types[k]) for k, v in cols.items()})

    @staticmethod
    def report(games: list, kept: list, inferred_fraction: float = 0.0) -> dict:
        hist = Counter(r[1] for g in kept for r in g["rows"])
        total = max(sum(hist.values()), 1)
        return {
            "games": [(g["episode_id"], g["opponent"], g["goals_for"], g["goals_against"], len(g["rows"])) for g in games],
            "kept_games": len(kept), "kept_rows": total if hist else 0,
            "win_rate": sum(g["goals_for"] > g["goals_against"] for g in games) / max(len(games), 1),
            "goal_diff_mean": float(np.mean([g["goals_for"] - g["goals_against"] for g in games])) if games else 0.0,
            "action_hist": {a: round(hist[a] / total, 4) for a in range(NUM_ACTIONS)},
            "inferred_action_fraction": inferred_fraction,
        }

ExpertGenWorker = ray.remote(num_cpus=1, memory=2 * 1024 ** 3)(ExpertGenerator)

def generate_expert(cfg: dict) -> tuple:
    ids = list(range(cfg["games"]))
    if cfg["workers"] > 0:
        workers = [ExpertGenWorker.options(**cfg["ray_worker"]).remote(cfg) for _ in range(cfg["workers"])]
        try:
            games = [g for part in ray.get([w.games.remote(ids[i::cfg["workers"]]) for i, w in enumerate(workers)]) for g in part]
            ray.get([w.close.remote() for w in workers])
        finally:
            for w in workers:
                ray.kill(w)
    else:
        gen = ExpertGenerator(cfg)
        try:
            games = gen.games(ids)
        finally:
            gen.close()
    kept = [g for g in games if ExpertGenerator.keep(cfg, g)]
    if not kept:
        raise ValueError("no games kept; refusing to write an empty dataset")
    return ExpertGenerator.table(kept), ExpertGenerator.report(games, kept)
