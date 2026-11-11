import json
import time
import math
import pickle
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import trueskill
import ray
from agents import make_agent
from football_env import BOT_CONFIGS, BOT_ENVS, BUILTIN_BOTS, FEATURE_DIM, NO_ACTION, NUM_ACTIONS, OBS_DIM, SELFPLAY_ENV, FootballEnv, goal_event

trueskill.setup(backend='mpmath')
TS_ENV = trueskill.TrueSkill(mu=25.0, sigma=8.333, beta=4.166, tau=0.083, draw_probability=0.05)
IMPALA_SPEC = {"agent": "impala", "d_model": 512, "lstm_hidden": 512}

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def summarize_games(episodes: List[dict]) -> dict:
    n = len(episodes)
    wins = sum(e['won'] for e in episodes)
    draws = sum(e['drawn'] for e in episodes)
    lo, hi = wilson_ci(wins, n)
    return {'n': n, 'win': wins, 'draw': draws, 'loss': n - wins - draws, 'win_rate': wins / n if n else 0.0, 'ci_lo': lo, 'ci_hi': hi,
            'goal_diff': float(np.mean([e['goals_scored'] - e['goals_conceded'] for e in episodes])) if n else 0.0}


@dataclass
class LeagueMember:
    name: str
    member_type: str
    rating: trueskill.Rating
    weights_path: Optional[str] = None
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    env_name: Optional[str] = None
    controls_right: bool = False
    recent_wins: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_draws: deque = field(default_factory=lambda: deque(maxlen=100))
    rating_history: deque = field(default_factory=lambda: deque(maxlen=500))
    rating_update_factor: float = 1.0
    created_idx: int = 0
    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games_played)
    @property
    def recent_win_rate(self) -> float:
        return np.mean(list(self.recent_wins)) if self.recent_wins else 0.0
    @property
    def recent_draw_rate(self) -> float:
        return np.mean(list(self.recent_draws)) if self.recent_draws else 0.0
    @property
    def recent_loss_rate(self) -> float:
        return 1.0 - self.recent_win_rate - self.recent_draw_rate
    @property
    def agent_win_rate(self) -> float:
        return 1.0 - self.recent_win_rate - self.recent_draw_rate if self.recent_wins else 0.0
    @property
    def conservative_skill(self) -> float:
        return self.rating.mu - 3 * self.rating.sigma
    def record_game(self, won: bool, drawn: bool = False):
        self.games_played += 1
        if drawn:
            self.draws += 1
            self.recent_wins.append(0.0)
            self.recent_draws.append(1.0)
        elif won:
            self.wins += 1
            self.recent_wins.append(1.0)
            self.recent_draws.append(0.0)
        else:
            self.losses += 1
            self.recent_wins.append(0.0)
            self.recent_draws.append(0.0)
        self.rating_history.append(self.rating.mu)

class PureLeague:
    def __init__(self, checkpoint_dir: Path, spec: dict = None, max_snapshots: int = 15, snapshot_on_rating_gain: float = 2.0, snapshot_on_champion_wins: int = 5, bot_floor: float = 0.25, latest_snapshot_prob: float = 0.5, min_skill_spacing: float = 3.0, min_policy_distance: float = 0.1):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.spec = spec or IMPALA_SPEC
        self.max_snapshots = max_snapshots
        self.snapshot_on_rating_gain = snapshot_on_rating_gain
        self.snapshot_on_champion_wins = snapshot_on_champion_wins
        self.bot_floor = bot_floor
        self.latest_snapshot_prob = latest_snapshot_prob
        self.snapshot_counter = 0
        self.min_skill_spacing = min_skill_spacing
        self.min_policy_distance = min_policy_distance
        self.members: Dict[str, LeagueMember] = {}
        self.champion: Optional[str] = None
        self.last_snapshot_rating = 25.0
        self.wins_vs_champion = 0
        self.total_games = 0
        self.policy_fingerprints: Dict[str, np.ndarray] = {}
        self.protected = lambda: set()
        self.dropped_games = 0
        self._init_bots()

    def _init_bots(self):
        for name, mu, sigma, env_name, controls_right in BOT_CONFIGS:
            self.members[name] = LeagueMember(
                name=name, member_type="bot", rating=TS_ENV.create_rating(mu=mu, sigma=sigma),
                env_name=env_name, controls_right=controls_right, rating_update_factor=0.0
            )
        self._update_champion()

    def _update_champion(self):
        best_name, best_skill = None, -np.inf
        for name, member in self.members.items():
            if member.conservative_skill > best_skill:
                best_skill, best_name = member.conservative_skill, name
        if best_name != self.champion:
            old = self.champion
            self.champion, self.wins_vs_champion = best_name, 0
            if old:
                print(f"  👑 Champion: {best_name} (μ={self.members[best_name].rating.mu:.1f}) < {old}")

    def win_probability(self, player: trueskill.Rating, opponent: trueskill.Rating) -> float:
        delta = player.mu - opponent.mu
        denom = math.sqrt(2 * (TS_ENV.beta ** 2) + player.sigma ** 2 + opponent.sigma ** 2)
        return 0.5 * (1 + math.erf(delta / (denom * math.sqrt(2))))

    def select_opponent(self, current_rating: trueskill.Rating, force_bot: str = None) -> Tuple[str, str]:
        if force_bot and force_bot in self.members:
            return force_bot, "eval"
        snapshots = [(n, m) for n, m in self.members.items() if m.member_type == "snapshot"]
        r = np.random.random()
        if r < self.bot_floor or not snapshots:
            bots = [n for n, m in self.members.items() if m.member_type == "bot"]
            weights = []
            for name in bots:
                member = self.members[name]
                if len(member.recent_wins) >= 5:
                    agent_score = 1 - member.recent_win_rate - member.recent_draw_rate * 0.5
                    weight = agent_score * (1 - agent_score) + 0.1
                else:
                    weight = 0.5
                weights.append(weight)
            return np.random.choice(bots, p=np.array(weights) / sum(weights)), "bot"
        if r < self.bot_floor + self.latest_snapshot_prob:
            newest = max(snapshots, key=lambda x: x[1].created_idx)
            return newest[0], "latest"
        candidates, weights = list(self.members.keys()), []
        for name in candidates:
            member = self.members[name]
            win_prob = self.win_probability(current_rating, member.rating)
            if len(member.recent_wins) >= 5:
                agent_score = 1 - member.recent_win_rate - member.recent_draw_rate * 0.5
                p = win_prob * 0.5 + agent_score * 0.5
            else:
                p = win_prob
            weight = p * (1 - p) + 0.05
            if member.games_played < 50:
                weight *= 1.5
            weights.append(weight)
        return np.random.choice(candidates, p=np.array(weights) / sum(weights)), "pfsp"

    def report_game(self, opponent_name: str, current_rating: trueskill.Rating, won: bool, drawn: bool = False) -> trueskill.Rating:
        if opponent_name not in self.members:
            self.dropped_games += 1
            print(f"  game vs removed opponent {opponent_name} not counted ({self.dropped_games} dropped)")
            return current_rating
        member = self.members[opponent_name]
        self.total_games += 1
        if drawn:
            new_current, new_opponent = TS_ENV.rate_1vs1(current_rating, member.rating, drawn=True)
        elif won:
            new_current, new_opponent = TS_ENV.rate_1vs1(current_rating, member.rating)
        else:
            new_opponent, new_current = TS_ENV.rate_1vs1(member.rating, current_rating)
        if member.member_type == "snapshot" and member.rating_update_factor > 0:
            if member.rating_update_factor < 1.0:
                new_mu = member.rating.mu + (new_opponent.mu - member.rating.mu) * member.rating_update_factor
                new_sigma = member.rating.sigma + (new_opponent.sigma - member.rating.sigma) * member.rating_update_factor
                member.rating = TS_ENV.create_rating(mu=new_mu, sigma=new_sigma)
            else:
                member.rating = new_opponent
        member.record_game(won=not won, drawn=drawn)
        if opponent_name == self.champion and won:
            self.wins_vs_champion += 1
        self._update_champion()
        return new_current

    def compute_policy_fingerprint(self, weights: dict, num_samples: int = 100) -> np.ndarray:
        rng = np.random.RandomState(42)
        test_obs = rng.randn(num_samples, OBS_DIM).astype(np.float32) * 0.5
        test_feat = rng.randn(num_samples, FEATURE_DIM).astype(np.float32) * 0.5
        agent = make_agent(self.spec)
        agent.set_weights(weights)
        return agent.fingerprint(test_obs, test_feat)

    def policy_distance(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        fp1, fp2 = np.clip(fp1, 1e-8, 1.0), np.clip(fp2, 1e-8, 1.0)
        m = 0.5 * (fp1 + fp2)
        return float(np.sqrt(0.5 * (np.sum(fp1 * np.log(fp1 / m)) + np.sum(fp2 * np.log(fp2 / m)))))

    def should_snapshot(self, current_rating: trueskill.Rating, current_weights: dict = None) -> bool:
        rating_ok = current_rating.mu - self.last_snapshot_rating >= self.snapshot_on_rating_gain
        champ_ok = self.wins_vs_champion >= self.snapshot_on_champion_wins
        if not rating_ok and not champ_ok:
            return False
        if current_weights is None or not self.policy_fingerprints:
            return True
        current_fp = self.compute_policy_fingerprint(current_weights)
        min_dist = min((self.policy_distance(current_fp, fp) for fp in self.policy_fingerprints.values()), default=float('inf'))
        if min_dist < self.min_policy_distance:
            print(f"  ⏭️ Snapshot skipped: Policy too similar (dist={min_dist:.3f} < {self.min_policy_distance})")
            return False
        return True

    def _ensure_skill_spacing(self):
        snapshots = [(n, m) for n, m in self.members.items() if m.member_type == "snapshot"]
        if len(snapshots) < 3:
            return
        snapshots.sort(key=lambda x: x[1].conservative_skill, reverse=True)
        keep = set(self.protected())
        if self.champion and self.members[self.champion].member_type == "snapshot":
            keep.add(self.champion)
        last_kept_skill = float('inf')
        for name, member in snapshots:
            if last_kept_skill - member.conservative_skill >= self.min_skill_spacing or name in keep:
                keep.add(name)
                last_kept_skill = member.conservative_skill
        for name, _ in sorted(snapshots, key=lambda x: x[1].created_idx, reverse=True)[:2]:
            keep.add(name)
        removed = []
        for name, member in snapshots:
            if name not in keep:
                if member.weights_path:
                    Path(member.weights_path).unlink(missing_ok=True)
                if name in self.policy_fingerprints:
                    del self.policy_fingerprints[name]
                del self.members[name]
                removed.append(name)
        if removed:
            print(f"  🧹 Skill-Spacing: Removed {len(removed)} snapshots")

    def add_snapshot(self, name: str, weights_path: str, weights: dict, rating: trueskill.Rating) -> bool:
        self._maybe_prune_snapshots()
        self._ensure_skill_spacing()
        self.policy_fingerprints[name] = self.compute_policy_fingerprint(weights)
        self.snapshot_counter += 1
        self.members[name] = LeagueMember(
            name=name, member_type="snapshot",
            rating=TS_ENV.create_rating(mu=rating.mu, sigma=rating.sigma),
            weights_path=weights_path, rating_update_factor=1.0,
            created_idx=self.snapshot_counter
        )
        self.last_snapshot_rating = rating.mu
        self.wins_vs_champion = 0
        self._update_champion()
        print(f"  📸 Snapshot: {name} (μ={rating.mu:.1f})")
        return True

    def _maybe_prune_snapshots(self):
        snapshots = [m for m in self.members.values() if m.member_type == "snapshot"]
        if len(snapshots) < self.max_snapshots:
            return
        keep = set(self.protected())
        if self.champion and self.members[self.champion].member_type == "snapshot":
            keep.add(self.champion)
        for m in sorted(snapshots, key=lambda m: m.conservative_skill, reverse=True)[:5]:
            keep.add(m.name)
        for m in sorted(snapshots, key=lambda m: m.created_idx, reverse=True)[:3]:
            keep.add(m.name)
        for m in snapshots:
            if m.name not in keep:
                if m.weights_path:
                    Path(m.weights_path).unlink(missing_ok=True)
                if m.name in self.policy_fingerprints:
                    del self.policy_fingerprints[m.name]
                del self.members[m.name]

    def get_member_weights(self, name: str) -> Optional[Dict]:
        member = self.members.get(name)
        if member is None or member.member_type != "snapshot":
            return None
        if member.weights_path and Path(member.weights_path).exists():
            ckpt = torch.load(member.weights_path, map_location='cpu', weights_only=False)
            return {k: v.numpy() for k, v in ckpt['model'].items()}
        return None

    def get_env_config(self, name: str) -> Tuple[str, int, int]:
        member = self.members.get(name)
        if member is None:
            return SELFPLAY_ENV, 1, 0
        if member.member_type == "bot" and member.controls_right:
            return member.env_name, 1, 0
        return SELFPLAY_ENV, 1, 1

    def get_stats(self) -> Dict:
        bots = {n: m for n, m in self.members.items() if m.member_type == "bot"}
        snaps = {n: m for n, m in self.members.items() if m.member_type == "snapshot"}
        total = sum(m.games_played for m in self.members.values())
        if total > 0:
            probs = [m.games_played / total for m in self.members.values() if m.games_played > 0]
            entropy = -sum(p * np.log(p + 1e-8) for p in probs)
            max_entropy = np.log(len([m for m in self.members.values() if m.games_played > 0]) + 1e-8)
            diversity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            diversity = 0
        return {
            'total_games': self.total_games, 'champion': self.champion,
            'champion_mu': self.members[self.champion].rating.mu if self.champion else 0,
            'num_snapshots': len(snaps), 'games_by_opp': {n: m.games_played for n, m in bots.items()}, 'snap_games': sum(m.games_played for m in snaps.values()), 'wins_vs_champion': self.wins_vs_champion, 'diversity': diversity,
            'bots': {n: {'mu': m.rating.mu, 'games': m.games_played, 'wr': m.agent_win_rate,
                         'dr': m.recent_draw_rate, 'lr': m.recent_win_rate} for n, m in bots.items()}
        }

    def get_ranking(self) -> List[Tuple[str, float, str]]:
        ranked = sorted(self.members.values(), key=lambda m: m.conservative_skill, reverse=True)
        return [(m.name, m.conservative_skill, m.member_type) for m in ranked]

    def print_ranking(self, top_n: int = 10):
        print(f"\n{'='*70}\n LIGA RANKING (Top {top_n})\n{'='*70}")
        for i, (name, skill, mtype) in enumerate(self.get_ranking()[:top_n], 1):
            m = self.members[name]
            champ = "👑" if name == self.champion else "  "
            icon = "🤖" if mtype == "bot" else "📸"
            print(f"{champ}{i:2d}. {icon} {name:25s} μ={m.rating.mu:5.1f} σ={m.rating.sigma:4.2f} skill={skill:5.1f} games={m.games_played:4d} agent_wr={(1 - m.recent_win_rate - m.recent_draw_rate)*100:4.0f}%")
        print("="*70)

    def save(self, path: Path = None):
        path = path or (self.checkpoint_dir / "league.pkl")
        state = {
            'members': {name: {
                'name': m.name, 'member_type': m.member_type, 'rating_mu': m.rating.mu, 'rating_sigma': m.rating.sigma,
                'weights_path': m.weights_path, 'games_played': m.games_played, 'wins': m.wins, 'losses': m.losses,
                'draws': m.draws, 'env_name': m.env_name, 'controls_right': m.controls_right,
                'rating_update_factor': m.rating_update_factor, 'recent_wins': list(m.recent_wins),
                'recent_draws': list(m.recent_draws), 'created_idx': m.created_idx
            } for name, m in self.members.items()},
            'champion': self.champion, 'last_snapshot_rating': self.last_snapshot_rating,
            'wins_vs_champion': self.wins_vs_champion, 'total_games': self.total_games,
            'snapshot_counter': self.snapshot_counter,
            'policy_fingerprints': {k: v.tolist() for k, v in self.policy_fingerprints.items()}
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: Path = None) -> bool:
        path = path or (self.checkpoint_dir / "league.pkl")
        if not path.exists():
            return False
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.members = {}
        for name, d in state['members'].items():
            m = LeagueMember(
                name=d['name'], member_type=d['member_type'],
                rating=TS_ENV.create_rating(d['rating_mu'], d['rating_sigma']),
                weights_path=d.get('weights_path'), games_played=d.get('games_played', 0),
                wins=d.get('wins', 0), losses=d.get('losses', 0), draws=d.get('draws', 0),
                env_name=d.get('env_name'), controls_right=d.get('controls_right', False),
                rating_update_factor=d.get('rating_update_factor', 1.0),
                created_idx=d.get('created_idx', 0)
            )
            m.recent_wins = deque(d.get('recent_wins', []), maxlen=100)
            m.recent_draws = deque(d.get('recent_draws', []), maxlen=100)
            self.members[name] = m
        self.champion = state.get('champion')
        self.last_snapshot_rating = state.get('last_snapshot_rating', 25.0)
        self.wins_vs_champion = state.get('wins_vs_champion', 0)
        self.total_games = state.get('total_games', 0)
        self.snapshot_counter = state.get('snapshot_counter', max((m.created_idx for m in self.members.values()), default=0))
        self.policy_fingerprints = {k: np.array(v) for k, v in state.get('policy_fingerprints', {}).items()}
        print(f"  📂 League loaded: {len(self.members)} members, champion={self.champion}")
        return True

class SelfPlayCore:
    def __init__(self, wid: int, spec: dict, rollout_len: int = 512, goal_traj_len: int = 128):
        self.wid = wid
        self.spec = spec
        self.rollout_len = rollout_len
        self.goal_traj_len = goal_traj_len
        self.agent = make_agent(spec)
        self.opponent_agent = None
        self.opponent_type = None
        self.env = None
        self.opp_obs = self.opp_feat = self.opp_state = None
        self.opp_prev_act, self.opp_first = NO_ACTION, True
        self.obs = self.feat = self.state = None
        self.prev_act, self.first = NO_ACTION, True
        self.ep_ret = 0.0
        self.ep_len = 0
        self.ep_score = [0, 0]
        self.rolling = {k: deque(maxlen=goal_traj_len) for k in ['obs', 'feat', 'prev_act', 'act', 'lp']}

    def set_weights(self, weights: dict):
        self.agent.set_weights(weights)

    def set_opponent(self, opponent_type: str, weights: dict = None):
        prev_type = self.opponent_type
        self.opponent_type = opponent_type
        if weights is not None:
            if self.opponent_agent is None:
                self.opponent_agent = make_agent(self.spec)
            self.opponent_agent.set_weights(weights)
            if prev_type != "snapshot":
                self.opp_prev_act, self.opp_first, self.opp_state = NO_ACTION, True, None

    def _create_env(self, env_name: str, left: int, right: int, rewards: str = "scoring") -> bool:
        env_key = (env_name, left, right, rewards)
        if self.env is not None and self.env.key == env_key:
            return False
        if self.env is not None:
            self.env.close()
        self.env = FootballEnv(env_name, left, right, rewards)
        return True

    def _reset(self):
        self.obs, self.feat, self.opp_obs, self.opp_feat = self.env.reset()
        if self.env.right > 0:
            self.opp_prev_act, self.opp_first, self.opp_state = NO_ACTION, True, None
        self.prev_act, self.first, self.state = NO_ACTION, True, None
        self.ep_ret, self.ep_len = 0.0, 0
        self.ep_score = [0, 0]
        for buf in self.rolling.values():
            buf.clear()

    def _get_opponent_action(self) -> Optional[int]:
        if self.opponent_type == "random":
            return np.random.randint(0, NUM_ACTIONS)
        elif self.opponent_type == "lazy":
            return 0
        elif self.opponent_type == "snapshot" and self.opponent_agent is not None:
            act, _, _, self.opp_state = self.opponent_agent.act(self.opp_obs, self.opp_feat, self.opp_prev_act, self.opp_first, self.opp_state)
            self.opp_prev_act, self.opp_first = act, False
            return act
        return None

    def collect(self, env_config: Tuple[str, int, int], opponent_type: str, opponent_weights: dict = None, rewards: str = "scoring") -> dict:
        env_name, left, right = env_config
        self.set_opponent(opponent_type, opponent_weights)
        env_changed = self._create_env(env_name, left, right, rewards)
        if env_changed or self.obs is None:
            self._reset()
        is_selfplay = right > 0
        keys = ['obs', 'feat', 'prev_act', 'act', 'lp', 'first', 'next_obs', 'next_feat', 'rew', 'done']
        data = {k: [] for k in keys}
        episodes = []
        goal_trajectories = []
        game_over = False
        for _ in range(self.rollout_len):
            act, lp, _, self.state = self.agent.act(self.obs, self.feat, self.prev_act, self.first, self.state)
            step_rec = {'obs': self.obs.copy(), 'feat': self.feat.copy(), 'prev_act': self.prev_act, 'act': act, 'lp': lp}
            for k, v in step_rec.items():
                data[k].append(v)
                self.rolling[k].append(v)
            data['first'].append(float(self.first))
            self.first = False
            self.prev_act = act
            env_action = [act, self._get_opponent_action()] if is_selfplay else [act]
            self.obs, self.feat, self.opp_obs, self.opp_feat, rew, done = self.env.step(env_action)
            data['next_obs'].append(self.obs.copy())
            data['next_feat'].append(self.feat.copy())
            self.ep_ret += rew
            self.ep_len += 1
            event = goal_event(rew)
            if event > 0:
                self.ep_score[0] += 1
                if len(self.rolling['obs']) >= self.goal_traj_len // 4:
                    traj_len = len(self.rolling['obs'])
                    traj_rew = np.zeros(traj_len, dtype=np.float32)
                    traj_done = np.zeros(traj_len, dtype=np.float32)
                    traj_rew[-1] = 1.0
                    traj_done[-1] = 1.0
                    goal_trajectories.append({
                        'obs': np.array(list(self.rolling['obs']), dtype=np.float32),
                        'feat': np.array(list(self.rolling['feat']), dtype=np.float32),
                        'prev_act': np.array(list(self.rolling['prev_act']), dtype=np.int64),
                        'act': np.array(list(self.rolling['act']), dtype=np.int64),
                        'lp': np.array(list(self.rolling['lp']), dtype=np.float32),
                        'rew': traj_rew, 'done': traj_done, 'bootstrap': 0.0
                    })
                for buf in self.rolling.values():
                    buf.clear()
            elif event < 0:
                self.ep_score[1] += 1
                for buf in self.rolling.values():
                    buf.clear()
            data['rew'].append(rew)
            data['done'].append(float(done))
            if done:
                game_over = True
                episodes.append({
                    'return': self.ep_ret, 'won': self.ep_score[0] > self.ep_score[1],
                    'drawn': self.ep_score[0] == self.ep_score[1], 'length': self.ep_len,
                    'opponent': opponent_type, 'goals_scored': self.ep_score[0], 'goals_conceded': self.ep_score[1]
                })
                self._reset()
                break
        _, _, bootstrap, _ = self.agent.act(self.obs, self.feat, self.prev_act, self.first, self.state)
        return {
            'obs': np.array(data['obs'], dtype=np.float32),
            'feat': np.array(data['feat'], dtype=np.float32),
            'prev_act': np.array(data['prev_act'], dtype=np.int64),
            'act': np.array(data['act'], dtype=np.int64),
            'lp': np.array(data['lp'], dtype=np.float32),
            'rew': np.array(data['rew'], dtype=np.float32),
            'done': np.array(data['done'], dtype=np.float32),
            'first': np.array(data['first'], dtype=np.float32),
            'next_obs': np.array(data['next_obs'], dtype=np.float32),
            'next_feat': np.array(data['next_feat'], dtype=np.float32),
            'wid': self.wid,
            'bootstrap': bootstrap,
            'episodes': episodes,
            'game_over': game_over,
            'opponent_type': opponent_type,
            'goal_trajectories': goal_trajectories
        }

    def close(self):
        if self.env is not None:
            self.env.close()

SelfPlayWorker = ray.remote(num_cpus=1)(SelfPlayCore)

class EvalCore(SelfPlayCore):
    def __init__(self, spec: dict, rollout_len: int, threads: int):
        torch.set_num_threads(threads)
        super().__init__(-1, spec, rollout_len)

    def evaluate(self, weights: dict, bot: str, games: int) -> List[dict]:
        self.set_weights(weights)
        episodes = []
        while len(episodes) < games:
            episodes += self.collect((BOT_ENVS[bot], 1, 0), bot)['episodes']
        return episodes

Evaluator = ray.remote(EvalCore)

class EvalRunner:
    def __init__(self, cfg: dict, spec: dict, rollout_len: int, out_path: Path):
        invalid = [b for b in cfg["bots"] if b not in BUILTIN_BOTS]
        if invalid:
            raise ValueError(f"eval.bots {invalid} not allowed, only built-in bots {list(BUILTIN_BOTS)}")
        self.cfg, self.out_path = cfg, Path(out_path)
        self.make = lambda: Evaluator.options(**cfg["actor"]).remote(spec, rollout_len, cfg["torch_threads"])
        self.actors = {bot: self.make() for bot in cfg["bots"]} if cfg["every"] > 0 else {}
        self.refs, self.results, self.errors, self.meta = {}, {}, {}, None
        self.last = {}

    def due(self, updates: int) -> bool:
        return bool(self.actors) and updates % self.cfg["every"] == 0 and not self.refs

    def start(self, updates: int, steps: int, weights: dict):
        shared = ray.put(weights)
        self.refs = {actor.evaluate.remote(shared, bot, self.cfg["games"]): bot for bot, actor in self.actors.items()}
        self.results, self.errors, self.meta = {}, {}, {"update": updates, "steps": steps, "start": time.time()}

    def poll(self):
        if not self.refs:
            return None
        ready, _ = ray.wait(list(self.refs), num_returns=len(self.refs), timeout=0)
        for ref in ready:
            bot = self.refs.pop(ref)
            try:
                self.results[bot] = summarize_games(ray.get(ref))
            except Exception as e:
                self.errors[bot] = type(e).__name__
                print(f"  eval {bot} failed: {type(e).__name__}")
                self.replace(bot)
        if self.refs:
            return None
        meta = dict(self.meta)
        wall = round(time.time() - meta.pop("start"), 1)
        record = {**meta, "wall_s": wall, "bots": self.results, "errors": self.errors, "n_errors": len(self.errors)}
        with open(self.out_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.last = record
        print("  EVAL u{update} ".format(**record) + " ".join(f"{b.replace('bot_', '')[0].upper()}:{r['win']}/{r['draw']}/{r['loss']} wr={r['win_rate']*100:.0f}% [{r['ci_lo']*100:.0f},{r['ci_hi']*100:.0f}] gd={r['goal_diff']:+.2f}" for b, r in record["bots"].items()))
        return record

    def replace(self, bot: str):
        try:
            ray.kill(self.actors[bot])
        except Exception:
            pass
        self.actors[bot] = self.make()

    def close(self):
        for a in self.actors.values():
            try:
                ray.kill(a)
            except Exception:
                pass

class StickyDispatcher:
    def __init__(self, league: PureLeague, eval_bots: List[str], eval_period: int = 50):
        self.league, self.eval_bots, self.eval_period = league, eval_bots, eval_period
        self.cycle = 0
        self.assigned = {}
        league.protected = self.in_use

    def in_use(self) -> set:
        return {a[0] for a in self.assigned.values()}

    def assign(self, wid: int, game_over: bool, rating: trueskill.Rating):
        current = self.assigned.get(wid)
        if current is not None and not game_over:
            return (*current, None)
        self.cycle += 1
        slot = self.cycle % self.eval_period
        opponent, _ = self.league.select_opponent(rating, force_bot=self.eval_bots[slot] if slot < len(self.eval_bots) else None)
        weights = self.league.get_member_weights(opponent)
        self.assigned[wid] = (opponent, self.league.get_env_config(opponent), "snapshot" if weights else opponent)
        return (*self.assigned[wid], weights)
