import sys
import json
import copy
import time
from pathlib import Path
from collections import deque
import numpy as np
import torch
import ray
import actor_critic
from agents import VTRACE, BCWarmstart, make_agent
from football_env import generate_expert
from league import TS_ENV, EvalRunner, PureLeague, SelfPlayWorker, StickyDispatcher

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GIB = 1024 ** 3
GF_DATA = "/tmp/claude-1005/-home-john-tinywhoop-bc-racing/a3574b03-9a09-436a-8c5e-cf180ed5b823/scratchpad/gf_data"
GF_RUNS = "/tmp/claude-1005/-home-john-tinywhoop-bc-racing/a3574b03-9a09-436a-8c5e-cf180ed5b823/scratchpad/gf_runs"
CONFIG = {
    "agent": "impala",
    "mode": "train",
    "paths": {"warmstart": str(ROOT / "bc_warmstart_v2.pt"), "expert_parquet": str(HERE / "expert.parquet"), "checkpoint_dir": "./checkpoints_league_v77"},
    "min_free_cpus": 8,
    "ray": {"address": "auto", "namespace": None, "worker": {"num_cpus": 1, "num_gpus": 0, "memory": 2 * GIB}, "env_vars": {"OMP_NUM_THREADS": "1"}},
    "learner": dict(
        num_workers=28, rollout_len=512, batch_size=64, goal_traj_len=128, driver_threads=4,
        max_snapshots=15, snapshot_on_rating_gain=2.0, snapshot_on_champion_wins=5,
        bot_floor=0.25, latest_snapshot_prob=0.5, min_skill_spacing=3.0, min_policy_distance=0.02,
    ),
    "impala": dict(
        lr=0.0001, gamma=0.999, entropy_coeff=0.01, value_coeff=0.2, sil_coeff=0.5,
        d_model=512, lstm_hidden=512, expert_threshold=0.7, expert_min_games=20,
        golden_capacity=2048, golden_max_uses=5, vtrace=dict(VTRACE),
    ),
    "dreamer": dict(
        batch=16, T=64, replay_cap=1_000_000, prefill=20_000, train_ratio=16, max_updates=None, seed=0, device="auto",
        log_keys=["dyn", "rep", "recon", "rew", "con", "policy", "value", "repval", "actor_entropy", "ret_mean", "v_ev", "adv_std"],
        model={
            "deter": 512, "hidden": 512, "units": 512, "stoch": 16, "discrete": 16, "layers": 2, "head_layers": 1, "unimix": 0.01,
            "bins": 255, "bin_lim": 20.0, "min_std": 0.1, "max_std": 1.0, "log_sigma_lim": (-5.0, 2.0), "repval_detach": True,
            "free": 1.0, "horizon": 15, "lam": 0.95, "disc_horizon": 333, "ent": 3e-4, "act_unimix": 0.01,
            "lr": 1e-4, "agc": 0.3, "pmin": 1e-3, "eps": 1e-20, "b1": 0.9, "b2": 0.999, "warmup": 1000, "slow_frac": 0.02,
            "ret_decay": 0.99, "ret_lo": 0.05, "ret_hi": 0.95, "max_nonfinite": 1000,
            "scales": {"dyn": 0.5, "rep": 0.1, "recon": 1.0, "rew": 1.0, "con": 1.0, "priv": 1.0, "policy": 1.0, "value": 1.0, "repval": 0.3},
            "critic_priv": "none", "priv_stop_grad": False, "head_action": False, "priv_norm": "symlog", "fast": False, "graph": False, "graph_ac": False,
            "graph_strict": False, "amp": None, "actor_in": "latent", "gru_hidden": 128, "bc_into_wm": False, "bc_wm_scale": 1.0, "priv_aux": False,
            "priv_aux_scale": 5.0, "action_repeat": 1, "reward_ensemble": 0, "dyn_ensemble": 0, "pess_coef": 1.0, "dyn_pess_coef": 1.0, "trunc_pct": 0.0,
            "trunc_decay": 0.99, "ens_boot": 0.8, "ens_graph": False,
        },
    ),
    "eval": {"every": 50, "greedy": False, "games": 50, "bots": ["bot_easy", "bot_medium", "bot_hard"], "file": "eval.jsonl", "torch_threads": 1,
             "actor": {"num_cpus": 1, "num_gpus": 0, "memory": 2 * GIB}},
    "expert_gen": {
        "games": 80, "opponents": ["bot_easy"], "keep": "win", "max_steps": 3000, "workers": 0, "out": GF_DATA + "/expert.parquet",
        "ray_worker": {"num_cpus": 1, "num_gpus": 0, "memory": 2 * GIB},
        "teacher": {"deadzone": 0.01, "own_ball_dist": 0.03, "shot_dist": 0.3, "shot_x": 0.7, "shot_y": 0.2, "press_dist": 0.1, "pass_dist": 0.4, "slide_dist": 0.05, "slide_cooldown": 60, "sprint_dist": 0.25},
    },
    "bc": {"epochs": 20, "batch_size": 512, "lr": 1e-3},
    "train_seconds": 24 * 3600,
    "log_every": 10,
    "checkpoint_every": 100,
    "ranking_every": 500,
    "resume": None,
}


CPU_TRY = {
    "agent": "dreamer", "train_seconds": 1800, "min_free_cpus": 8,
    "paths": {"checkpoint_dir": GF_RUNS, "warmstart": None, "expert_parquet": None},
    "ray": {"worker": {"num_cpus": 1, "num_gpus": 0, "memory": 2 * GIB}, "env_vars": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}},
    "learner": {"num_workers": 6, "rollout_len": 512, "batch_size": 6, "goal_traj_len": 64, "driver_threads": 2},
    "dreamer": {"device": "cpu", "replay_cap": 150_000, "prefill": 6000, "train_ratio": 16, "max_updates": None,
                "model": {"deter": 128, "hidden": 128, "units": 128}},
    "eval": {"every": 0}, "log_every": 5, "checkpoint_every": 25,
}


def deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        base[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def agent_spec(cfg: dict) -> dict:
    kind, paths, l = cfg["agent"], cfg["paths"], cfg["learner"]
    spec = {"agent": kind, **copy.deepcopy(cfg[kind]), **{k: l[k] for k in ("rollout_len", "batch_size", "goal_traj_len", "num_workers")}}
    if kind == "impala":
        spec.update(warmstart=paths["warmstart"], expert_parquet=paths["expert_parquet"])
    return spec


def ray_init(cfg: dict):
    r = cfg["ray"]
    ray.init(address=r["address"], namespace=r["namespace"], ignore_reinit_error=True, logging_level="WARNING",
             runtime_env={"py_modules": [str(Path(actor_critic.__file__).resolve().parent)], "working_dir": str(HERE),
                          "excludes": ["__pycache__", "*.pt", "*.parquet"], "env_vars": r["env_vars"]})


def free_cpu_error(cfg: dict):
    need = max(cfg["min_free_cpus"], cfg["learner"]["num_workers"] * cfg["ray"]["worker"]["num_cpus"])
    free = ray.available_resources().get("CPU", 0)
    return f"insufficient CPU on the shared Ray cluster: {free} available, {need} needed; not started" if free < need else None


class LeagueTrainer:
    def __init__(self, cfg: dict):
        l = cfg["learner"]
        self.cfg, self.num_workers, self.rollout_len, self.batch_size = cfg, l["num_workers"], l["rollout_len"], l["batch_size"]
        self.spec = agent_spec(cfg)
        self.checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"]) / cfg["agent"]
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.agent = make_agent(self.spec, learner=True)
        self.current_rating = TS_ENV.create_rating(mu=25.0, sigma=8.333)
        self.league = PureLeague(
            checkpoint_dir=self.checkpoint_dir, spec=self.spec,
            **{k: l[k] for k in ("max_snapshots", "snapshot_on_rating_gain", "snapshot_on_champion_wins", "bot_floor", "latest_snapshot_prob", "min_skill_spacing", "min_policy_distance")}
        )
        self.agent.attach(self.league)
        if (self.checkpoint_dir / "league.pkl").exists():
            self.league.load()
        if self.agent.warmstarted:
            snap_path = self.checkpoint_dir / "snapshot_warmstart.pt"
            if not snap_path.exists():
                self.agent.save_snapshot(snap_path)
            if "warmstart" not in self.league.members:
                self.league.add_snapshot("warmstart", str(snap_path), self.agent.weights(), TS_ENV.create_rating(mu=25.0, sigma=8.0))
        self.total_steps = self.total_matches = self.updates = 0
        self._resume(cfg["resume"])
        self.workers = [SelfPlayWorker.options(**cfg["ray"]["worker"]).remote(i, self.spec, self.rollout_len, l["goal_traj_len"]) for i in range(self.num_workers)]
        self.start = None
        self.ep_wins, self.ep_draws, self.ep_losses, self.returns, self.lengths, self.returns_vs_hard, self.goal_diffs = (deque(maxlen=100) for _ in range(7))
        self.pending, self.queue = {}, []
        self._weights_at = (None, None)
        self.evaluator = EvalRunner(cfg["eval"], self.spec, self.rollout_len, self.checkpoint_dir / cfg["eval"]["file"])
        self.dispatcher = StickyDispatcher(self.league, ["random", "lazy", "bot_easy", "bot_medium", "bot_hard"])
        print(f"{cfg['agent']} + PFSP League | {self.num_workers}W | batch {self.batch_size} x {self.rollout_len} | mu={self.current_rating.mu:.1f}")

    def _resume(self, resume):
        path = max(self.checkpoint_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime, default=None) if resume == "latest" else resume
        if not path:
            return
        ck = self.agent.load(path)
        self.updates, self.total_steps, self.total_matches = ck['updates'], ck['total_steps'], ck['total_matches']
        self.current_rating = TS_ENV.create_rating(ck['rating_mu'], ck['rating_sigma'])
        print(f"Resumed from {path} at update {self.updates}")

    def _weights(self) -> dict:
        if self._weights_at[0] != self.updates:
            self._weights_at = (self.updates, self.agent.weights())
        return self._weights_at[1]

    def _dispatch(self, w, game_over: bool = True):
        opponent, env_config, opp_type, opp_weights = self.dispatcher.assign(self.workers.index(w), game_over, self.current_rating)
        self.pending[w.collect.remote(env_config, opp_type, opp_weights)] = (w, opponent)

    def _update(self, batch: list) -> dict:
        return self.agent.update(batch)

    def _maybe_snapshot(self):
        weights = self._weights()
        if self.league.should_snapshot(self.current_rating, weights):
            name = f"snap_u{self.updates}_r{self.current_rating.mu:.0f}"
            path = self.checkpoint_dir / f"{name}.pt"
            self.agent.save_snapshot(path)
            self.league.add_snapshot(name, str(path), weights, self.current_rating)

    @staticmethod
    def _opp_col(tag: str, s: dict) -> str:
        if not s.get('games'):
            return f"{tag}:-(0)"
        return f"{tag}:{s['wr']*100:.0f}/{s['dr']*100:.0f}/{s['lr']*100:.0f}({s['games']})"

    def _log_progress(self, stats: dict):
        elapsed = time.time() - self.start
        sps = self.total_steps / elapsed
        mpm = self.total_matches / (elapsed / 60) if elapsed > 0 else 0
        n_eps = max(len(self.ep_wins), 1)
        win_rate, draw_rate, loss_rate = (sum(d) / n_eps * 100 for d in (self.ep_wins, self.ep_draws, self.ep_losses))
        ret = np.mean(list(self.returns)) if self.returns else 0
        gd = np.mean(list(self.goal_diffs)) if self.goal_diffs else 0.0
        ret_max = np.max(list(self.returns)) if self.returns else 0
        avg_vs_hard = np.mean(list(self.returns_vs_hard)) if len(self.returns_vs_hard) >= 10 else 0
        league_stats = self.league.get_stats()
        bot_stats = league_stats['bots']
        games_str = "/".join(f"{n.replace('bot_', '')[0]}{g}" for n, g in league_stats['games_by_opp'].items()) + f"/s{league_stats['snap_games']}"
        opp_str = " ".join(self._opp_col(tag, bot_stats.get(name, {})) for tag, name in (("E", "bot_easy"), ("M", "bot_medium"), ("H", "bot_hard")))
        champ = league_stats['champion'] or "none"
        print(f"[{self.updates:4d}] {self.total_steps/1e6:.1f}M {sps/1e3:.0f}k/s {mpm:.0f}m/min {elapsed/60:.0f}m | W/D/L:{win_rate:.0f}/{draw_rate:.0f}/{loss_rate:.0f}% R:{ret:+.1f}({ret_max:+.0f}) GD:{gd:+.2f} | mu={self.current_rating.mu:.1f} sigma={self.current_rating.sigma:.2f} | {opp_str} N:{games_str} RvH:{avg_vs_hard:+.1f} | {self.agent.fmt(stats)} | {league_stats['num_snapshots']}snap champ:{champ}")

    def _ingest(self, ref):
        w, opponent_name = self.pending.pop(ref)
        rollout = ray.get(ref)
        self.queue.append(rollout)
        self.total_steps += rollout['obs'].shape[0]
        for ep in rollout['episodes']:
            self.total_matches += 1
            self.returns.append(ep['return'])
            self.lengths.append(ep['length'])
            self.goal_diffs.append(ep['goals_scored'] - ep['goals_conceded'])
            for d, hit in ((self.ep_wins, ep['won']), (self.ep_draws, ep['drawn'] and not ep['won']), (self.ep_losses, not ep['won'] and not ep['drawn'])):
                d.append(int(hit))
            if opponent_name == "bot_hard":
                self.returns_vs_hard.append(ep['return'])
            self.current_rating = self.league.report_game(opponent_name, self.current_rating, won=ep['won'], drawn=ep['drawn'])
        opp_mu = self.league.members[opponent_name].rating.mu if opponent_name in self.league.members else 25.0
        self.agent.observe(rollout, opp_mu)
        w.set_weights.remote(self._weights())
        self._dispatch(w, rollout['game_over'])

    def train(self, max_time: int = 3600):
        c = self.cfg
        print(f"Training for {max_time}s...")
        self.start = time.time()
        ray.get([w.set_weights.remote(self._weights()) for w in self.workers])
        for w in self.workers:
            self._dispatch(w)
        while time.time() - self.start < max_time:
            while len(self.queue) < self.batch_size:
                done_refs, _ = ray.wait(list(self.pending.keys()), num_returns=1)
                for ref in done_refs:
                    self._ingest(ref)
            batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            stats = self._update(batch)
            self.updates += 1
            self._maybe_snapshot()
            self.evaluator.poll()
            if self.evaluator.due(self.updates):
                self.evaluator.start(self.updates, self.total_steps, self._weights())
            if self.updates % c["log_every"] == 0:
                self._log_progress(stats)
            if self.updates % c["checkpoint_every"] == 0:
                self._save_checkpoint()
            if self.updates % c["ranking_every"] == 0:
                self.league.print_ranking()
        print("\nTime limit reached.")
        self._save_checkpoint("final")
        self.league.print_ranking()

    def _save_checkpoint(self, name: str = None):
        path = self.checkpoint_dir / f"ckpt_{name or f'u{self.updates}'}.pt"
        self.agent.save(path, {'updates': self.updates, 'total_steps': self.total_steps, 'total_matches': self.total_matches,
                               'rating_mu': self.current_rating.mu, 'rating_sigma': self.current_rating.sigma})
        self.league.save()
        print(f"  Saved {path}")

    def close(self):
        for ref in self.pending:
            try:
                ray.cancel(ref)
            except Exception:
                pass
        try:
            self.evaluator.close()
        except Exception as e:
            print(f"  evaluator close failed: {type(e).__name__}")
        for w in self.workers:
            try:
                ray.get(w.close.remote(), timeout=2)
            except Exception:
                pass
            ray.kill(w)


def run(overrides: dict = None):
    cfg = deep_merge(copy.deepcopy(CONFIG), overrides or {})
    torch.set_num_threads(cfg["learner"]["driver_threads"])
    if cfg["mode"] == "bc":
        BCWarmstart(agent_spec({**cfg, "agent": "impala"}), cfg["paths"]["expert_parquet"], cfg["paths"]["warmstart"], **cfg["bc"]).run()
        return
    if cfg["mode"] == "gen_expert":
        gen = cfg["expert_gen"]
        if gen["workers"] > 0:
            ray_init(cfg)
        report = generate_expert(gen)
        print(json.dumps(report))
        return report
    ray_init(cfg)
    error = free_cpu_error(cfg)
    if error:
        ray.shutdown()
        raise SystemExit(error)
    trainer = LeagueTrainer(cfg)
    try:
        trainer.train(max_time=cfg["train_seconds"])
    except KeyboardInterrupt:
        print("\nStopped!")
    finally:
        trainer.close()


if __name__ == "__main__":
    run(json.loads(sys.argv[1]) if len(sys.argv) > 1 else None)
