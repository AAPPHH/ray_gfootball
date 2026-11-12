import math
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from actor_critic import dreamer as DR
from actor_critic.dreamer_core import Replay
from actor_critic.impala import PopArtAdapter, impala_loss, impala_targets
from actor_critic.sil import GoldenMemory, sil_clipped_step, sil_mc_returns
from football_env import FEATURE_DIM, NO_ACTION, NUM_ACTIONS, OBS_DIM, FeatureEngineer

VTRACE = {"rho_bar": 1.0, "c_bar": 1.0, "log_rho_clip": 20.0, "adv_sigma_norm": False, "value_loss_form": "mse"}
DREAMER_POLICY_KEYS = ("enc", "rssm", "actor")

def prefixed(stats: Optional[dict], prefix: str) -> dict:
    return {prefix + k: v for k, v in stats.items()} if stats else {}

def explained_variance(values: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> float:
    values, targets = values.reshape(-1), targets.reshape(-1)
    if mask is not None:
        values, targets = values[mask.reshape(-1).bool()], targets[mask.reshape(-1).bool()]
    var = targets.double().var(unbiased=False)
    return 1.0 - ((targets - values).double().var(unbiased=False) / var).item() if var > 1e-12 else 0.0

class PopArtValueHead(nn.Module):
    def __init__(self, input_dim: int, beta: float = 1e-3):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(input_dim, 1)
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        self.register_buffer('nu', torch.ones(1))
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def denormalize(self, normalized):
        return normalized * self.sigma + self.mu

    def normalize_target(self, targets):
        return (targets - self.mu) / self.sigma

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor):
        old_mu, old_sigma = self.mu.clone(), self.sigma.clone()
        self.mu.mul_(1 - self.beta).add_(self.beta * targets.mean())
        self.nu.mul_(1 - self.beta).add_(self.beta * (targets ** 2).mean())
        self.sigma.copy_(torch.sqrt(torch.clamp(self.nu - self.mu ** 2, min=1e-4)))
        self.linear.weight.data.mul_(old_sigma / self.sigma)
        self.linear.bias.data.copy_((self.linear.bias.data * old_sigma + old_mu - self.mu) / self.sigma)

class Net(nn.Module):
    def __init__(self, d_model: int = 512, lstm_hidden: int = 512):
        super().__init__()
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.action_emb = nn.Embedding(NUM_ACTIONS + 1, 16)
        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM + FEATURE_DIM + 16, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=1, batch_first=True)
        self.policy = nn.Sequential(
            nn.Linear(lstm_hidden, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(lstm_hidden, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.value = PopArtValueHead(128)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.lstm_hidden, device=device),
                torch.zeros(1, batch_size, self.lstm_hidden, device=device))

    def forward(self, obs, feat, prev_actions=None, hidden=None):
        squeeze = obs.dim() == 2
        if squeeze:
            obs, feat = obs.unsqueeze(1), feat.unsqueeze(1)
        B, L, _ = obs.shape
        if prev_actions is None:
            prev_actions = torch.full((B,), NUM_ACTIONS, dtype=torch.long, device=obs.device)
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(1).expand(-1, L)
        x = self.encoder(torch.cat([obs, feat, self.action_emb(prev_actions)], dim=-1))
        if hidden is None:
            hidden = self.init_hidden(B, obs.device)
        x, hidden = self.lstm(x, hidden)
        logits = self.policy(x)
        values_norm = self.value(self.value_mlp(x))
        if squeeze:
            logits, values_norm = logits.squeeze(1), values_norm.squeeze(1)
        return logits, values_norm, hidden

    def get_action(self, obs, feat, prev_actions, hidden=None, deterministic=False):
        logits, values_norm, hidden = self.forward(obs, feat, prev_actions, hidden)
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), self.value.denormalize(values_norm), hidden

class ExpertBufferGPU:
    def __init__(self, parquet_path: str, device: torch.device, window_len: int = 512):
        self.device = device
        self.window_len = window_len
        self.num_episodes = 0
        self.episode_slices = []
        self.window_slices = []
        self.obs = self.feat = self.act = self.rew = self.done = self.prev_act = None
        if parquet_path and Path(parquet_path).exists():
            self._load_parquet(parquet_path)

    def _load_parquet(self, parquet_path):
        print(f"Loading expert data to GPU from {parquet_path}...")
        df = pq.read_table(parquet_path).to_pandas()
        obs_all = np.array([np.frombuffer(b, dtype=np.float32) for b in df['obs']])
        actions = df['action'].values.astype(np.int64)
        rewards = df['reward'].values.astype(np.float32)
        active = df['active'].values.astype(np.int64)
        episode_ids = df['episode_id'].values
        print(f"Computing features for {len(obs_all)} samples...")
        feat_all = np.array([FeatureEngineer.extract(o, a) for o, a in zip(obs_all, active)])
        unique_eps = np.unique(episode_ids)
        print(f"Processing {len(unique_eps)} episodes...")
        all_obs, all_feat, all_act, all_rew, all_done, all_prev_act = [], [], [], [], [], []
        current_idx = 0
        for ep_id in unique_eps:
            mask = episode_ids == ep_id
            ep_obs, ep_feat, ep_act, ep_rew = obs_all[mask], feat_all[mask], actions[mask], rewards[mask]
            ep_len = len(ep_obs)
            prev_acts = np.zeros(ep_len, dtype=np.int64)
            prev_acts[0] = NUM_ACTIONS
            prev_acts[1:] = ep_act[:-1]
            ep_done = np.zeros(ep_len, dtype=np.float32)
            ep_done[-1] = 1.0
            all_obs.append(ep_obs)
            all_feat.append(ep_feat)
            all_act.append(ep_act)
            all_rew.append(ep_rew)
            all_done.append(ep_done)
            all_prev_act.append(prev_acts)
            self.episode_slices.append((current_idx, current_idx + ep_len))
            current_idx += ep_len
        self.obs = torch.from_numpy(np.concatenate(all_obs)).float().to(self.device)
        self.feat = torch.from_numpy(np.concatenate(all_feat)).float().to(self.device)
        self.act = torch.from_numpy(np.concatenate(all_act)).long().to(self.device)
        self.rew = torch.from_numpy(np.concatenate(all_rew)).float().to(self.device)
        self.done = torch.from_numpy(np.concatenate(all_done)).float().to(self.device)
        self.prev_act = torch.from_numpy(np.concatenate(all_prev_act)).long().to(self.device)
        self.num_episodes = len(unique_eps)
        for start, end in self.episode_slices:
            s = start
            while s < end:
                self.window_slices.append((s, min(s + self.window_len, end)))
                s += self.window_len
        vram_mb = (self.obs.numel()*4 + self.feat.numel()*4 + self.act.numel()*8 + self.rew.numel()*4 + self.done.numel()*4 + self.prev_act.numel()*8) / 1e6
        print(f"Expert Buffer on GPU: {self.num_episodes} episodes, {len(self.window_slices)} windows (<= {self.window_len} steps), {len(self.obs)} steps, {vram_mb:.1f}MB VRAM")

    def sample(self, n: int) -> dict:
        if not self.window_slices:
            return None
        n = min(n, len(self.window_slices))
        idx = np.random.choice(len(self.window_slices), size=n, replace=False)
        max_len = max(self.window_slices[i][1] - self.window_slices[i][0] for i in idx)
        obs = torch.zeros(n, max_len, OBS_DIM, device=self.device)
        feat = torch.zeros(n, max_len, FEATURE_DIM, device=self.device)
        act = torch.zeros(n, max_len, dtype=torch.long, device=self.device)
        rew = torch.zeros(n, max_len, device=self.device)
        done = torch.ones(n, max_len, device=self.device)
        prev_act = torch.zeros(n, max_len, dtype=torch.long, device=self.device)
        mask = torch.zeros(n, max_len, device=self.device)
        lp = torch.zeros(n, max_len, device=self.device)
        for i, win_idx in enumerate(idx):
            start, end = self.window_slices[win_idx]
            length = end - start
            obs[i, :length] = self.obs[start:end]
            feat[i, :length] = self.feat[start:end]
            act[i, :length] = self.act[start:end]
            rew[i, :length] = self.rew[start:end]
            done[i, :length] = self.done[start:end]
            prev_act[i, :length] = self.prev_act[start:end]
            mask[i, :length] = 1.0
        return {
            'obs': obs, 'feat': feat, 'act': act, 'rew': rew, 'done': done,
            'prev_act': prev_act, 'mask': mask, 'lp': lp,
            'bootstrap': torch.zeros(n, device=self.device), 'max_len': max_len
        }

    def __len__(self):
        return len(self.window_slices)

class GoldenMemoryGPU(GoldenMemory):
    KEYS = ('obs', 'feat', 'act', 'prev_act', 'rew', 'done', 'mask', 'lp')

    def __init__(self, capacity: int, max_uses: int, traj_len: int, device: torch.device):
        super().__init__(capacity, max_uses)
        self.traj_len = traj_len
        self.device = device
        self.obs = torch.zeros(capacity, traj_len, OBS_DIM, device=device)
        self.feat = torch.zeros(capacity, traj_len, FEATURE_DIM, device=device)
        self.act = torch.zeros(capacity, traj_len, dtype=torch.long, device=device)
        self.prev_act = torch.zeros(capacity, traj_len, dtype=torch.long, device=device)
        self.rew = torch.zeros(capacity, traj_len, device=device)
        self.done = torch.ones(capacity, traj_len, device=device)
        self.mask = torch.zeros(capacity, traj_len, device=device)
        self.lp = torch.zeros(capacity, traj_len, device=device)

    opp_ratings = property(lambda self: self.score)
    lengths = property(lambda self: self.ep_lens)

    def length(self, trajectory):
        return min(trajectory['obs'].shape[0], self.traj_len)

    def weights(self, score):
        return 1.0 + score / 50.0

    def _recycle(self, used):
        return int(used[0])

    def write(self, idx, trajectory):
        n = self.length(trajectory)
        for k in self.KEYS:
            getattr(self, k)[idx].fill_(1.0 if k == 'done' else 0.0)
        for k in ('obs', 'feat', 'act', 'prev_act', 'rew', 'done'):
            getattr(self, k)[idx, :n] = torch.from_numpy(trajectory[k][:n].copy()).to(self.device)
        self.mask[idx, :n] = 1.0

    def add(self, trajectory: dict, opp_rating: float = 25.0):
        return super().add(trajectory, 0.0, opp_rating)

    def sample(self, n: int) -> dict:
        chosen = self.draw(n)
        if len(chosen) == 0:
            return None
        idx = torch.as_tensor(chosen)
        return {
            'obs': self.obs[idx], 'feat': self.feat[idx], 'act': self.act[idx],
            'prev_act': self.prev_act[idx], 'rew': self.rew[idx], 'done': self.done[idx],
            'mask': self.mask[idx], 'lp': self.lp[idx],
            'bootstrap': torch.zeros(len(idx), device=self.device), 'max_len': self.traj_len
        }

    def stats(self):
        if self.size == 0:
            return {'size': 0, 'fresh': 0, 'avg_opp': 0}
        fresh = int(np.count_nonzero(self.uses[:self.size] < self.max_uses))
        return {'size': self.size, 'fresh': fresh, 'avg_opp': torch.from_numpy(self.score[:self.size]).mean().item()}

    def __len__(self):
        return self.size

class Agent:
    kind = None
    rollout_fields = ()

    def attach(self, league):
        self.league = league

    def policy_state(self) -> dict:
        return self.model.state_dict()

    def weights(self) -> dict:
        return {k: v.detach().cpu().numpy() for k, v in self.policy_state().items()}

    def set_weights(self, weights: dict):
        self.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=True)

    def save_snapshot(self, path):
        torch.save({'model': self.policy_state(), 'agent': self.kind}, path)

    def act(self, obs, feat, prev_act, first, state, deterministic=False):
        raise NotImplementedError

    def fingerprint(self, obs: np.ndarray, feat: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def observe(self, rollout: dict, opp_mu: float = 25.0):
        raise NotImplementedError

    def update(self, rollouts: list) -> dict:
        raise NotImplementedError

    def fmt(self, stats: dict) -> str:
        raise NotImplementedError

    def save(self, path, extra: dict):
        torch.save({'model': self.model.state_dict(), 'optimizer': self.opt.state_dict(), 'agent': self.kind, **extra}, path)

    def load(self, path) -> dict:
        ckpt = self.read_checkpoint(path)
        self.model.load_state_dict(ckpt['model'])
        self.opt.load_state_dict(ckpt['optimizer'])
        return ckpt

    def read_checkpoint(self, path) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        found = ckpt.get('agent', 'impala')
        if found != self.kind:
            raise ValueError(f"checkpoint {path} was written by agent {found!r}, current agent is {self.kind!r}; use a separate checkpoint_dir per agent")
        return ckpt

class ImpalaAgent(Agent):
    kind = "impala"
    vtrace = dict(VTRACE)

    def __init__(self, spec: dict, learner: bool = False):
        self.spec, self.learner, self.warmstarted = spec, learner, False
        self.model = Net(spec["d_model"], spec["lstm_hidden"])
        if not learner:
            self.device = torch.device("cpu")
            self.model.eval()
            return
        self.vtrace = {**VTRACE, **spec.get("vtrace", {})}
        self.batch_size, self.gamma = spec["batch_size"], spec["gamma"]
        self.entropy_coeff, self.value_coeff, self.sil_coeff = spec["entropy_coeff"], spec["value_coeff"], spec["sil_coeff"]
        self.expert_threshold, self.expert_min_games, self.expert_disabled = spec["expert_threshold"], spec["expert_min_games"], False
        self.device = torch.device(spec["device"])
        self.model.to(self.device)
        warm = spec.get("warmstart")
        if warm and Path(warm).exists():
            print(f"Loading warmstart from {warm}...")
            self.model.load_state_dict(torch.load(warm, map_location=self.device, weights_only=False)['model'], strict=False)
            self.warmstarted = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=spec["lr"], eps=1e-5)
        self.scaler = GradScaler('cuda')
        parquet = spec.get("expert_parquet")
        self.expert = ExpertBufferGPU(parquet, self.device, spec["rollout_len"]) if parquet else None
        self.golden = GoldenMemoryGPU(spec["golden_capacity"], spec["golden_max_uses"], spec["goal_traj_len"], self.device)

    def act(self, obs, feat, prev_act, first, state, deterministic=False):
        with torch.no_grad():
            action, lp, value, state = self.model.get_action(
                torch.from_numpy(obs).float().unsqueeze(0), torch.from_numpy(feat).float().unsqueeze(0),
                torch.tensor([prev_act], dtype=torch.long), state, deterministic)
        return action.item(), lp.item(), value.item(), state

    def fingerprint(self, obs, feat):
        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model.forward(torch.from_numpy(obs), torch.from_numpy(feat), torch.full((len(obs),), NO_ACTION, dtype=torch.long))
            return F.softmax(logits, dim=-1).numpy().mean(axis=0)

    def observe(self, rollout, opp_mu=25.0):
        for traj in rollout.get('goal_trajectories', []):
            self.golden.add(traj, opp_mu)

    def _prepare_batch(self, rollouts):
        max_len = max(r['obs'].shape[0] for r in rollouts)
        B = len(rollouts)
        def pad(arr, ml, pv=0):
            if arr.shape[0] == ml:
                return arr
            if arr.ndim == 1:
                return np.pad(arr, (0, ml - arr.shape[0]), constant_values=pv)
            return np.pad(arr, ((0, ml - arr.shape[0]), (0, 0)), constant_values=pv)
        obs = torch.from_numpy(np.stack([pad(r['obs'], max_len) for r in rollouts])).float().to(self.device)
        feat = torch.from_numpy(np.stack([pad(r['feat'], max_len) for r in rollouts])).float().to(self.device)
        prev_act = torch.from_numpy(np.stack([pad(r['prev_act'], max_len) for r in rollouts])).long().to(self.device)
        act = torch.from_numpy(np.stack([pad(r['act'], max_len) for r in rollouts])).long().to(self.device)
        beh_lp = torch.from_numpy(np.stack([pad(r['lp'], max_len) for r in rollouts])).float().to(self.device)
        rew = torch.from_numpy(np.stack([pad(r['rew'], max_len) for r in rollouts])).float().to(self.device)
        done = torch.from_numpy(np.stack([pad(r['done'], max_len, 1.0) for r in rollouts])).float().to(self.device)
        bootstrap = torch.tensor([r['bootstrap'] for r in rollouts], dtype=torch.float32, device=self.device)
        mask = torch.from_numpy(np.stack([
            np.concatenate([np.ones(r['obs'].shape[0]), np.zeros(max_len - r['obs'].shape[0])])
            for r in rollouts
        ])).float().to(self.device)
        return {'obs': obs, 'feat': feat, 'prev_act': prev_act, 'act': act, 'lp': beh_lp,
                'rew': rew, 'done': done, 'bootstrap': bootstrap, 'mask': mask, 'max_len': max_len}

    def _update(self, rollouts) -> dict:
        batch = self._prepare_batch(rollouts)
        obs, feat, prev_act, act, beh_lp = batch['obs'], batch['feat'], batch['prev_act'], batch['act'], batch['lp']
        rew, done, bootstrap, mask, T = batch['rew'], batch['done'], batch['bootstrap'], batch['mask'], batch['max_len']
        with autocast('cuda'):
            logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
            dist = Categorical(logits=logits)
            target_lp = dist.log_prob(act)
            entropy = dist.entropy()
            values = self.model.value.denormalize(values_norm)
        m = mask.bool()
        vcfg = {**self.vtrace, "gamma": self.gamma, "alpha": self.entropy_coeff, "value_coef": self.value_coeff}
        targets = impala_targets(beh_lp, target_lp.detach(), rew, values.detach(), done, bootstrap.float(), PopArtAdapter(self.model.value), vcfg, mask=m)
        with torch.no_grad():
            mean_rho = targets["avg_rho"]
            max_rho = targets["diag"]["is_ratio"].clamp(max=vcfg["rho_bar"])[m].max().item()
            ev = explained_variance(values.detach().float(), targets["vtargets"], m)
        with autocast('cuda'):
            losses = impala_loss(targets, target_lp[m], entropy[m], values_norm[m], vcfg, mask=m)
        loss, policy_loss, value_loss = losses["loss"], losses["policy"], losses["value"]
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = sum(p.grad.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
        grad_clipped = nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {
            'loss': loss.item(), 'pi': policy_loss.item(), 'v': value_loss.item(),
            'ent': (entropy * mask).sum().item() / mask.sum().item(),
            'rho': mean_rho, 'rho_max': max_rho, 'ev': ev, 'grad': grad_norm,
            'grad_clip': grad_clipped.item() if torch.is_tensor(grad_clipped) else grad_clipped
        }

    def _update_sil_gpu(self, batch: dict) -> Optional[dict]:
        if batch is None:
            return None
        obs, feat, prev_act, act = batch['obs'], batch['feat'], batch['prev_act'], batch['act']
        rew, done, bootstrap, mask, T = batch['rew'], batch['done'], batch['bootstrap'], batch['mask'], batch['max_len']
        with autocast('cuda'):
            logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
            target_lp = Categorical(logits=logits).log_prob(act)
            values = self.model.value.denormalize(values_norm)
        with torch.no_grad():
            mc_returns = sil_mc_returns(rew, done, self.gamma, bootstrap)
            sil_adv = torch.clamp(mc_returns - values.detach(), min=0)
        sil_mask = (sil_adv > 0) & (mask > 0)
        if not sil_mask.any():
            return {'sil_loss': 0.0, 'sil_pi': 0.0, 'sil_v': 0.0, 'sil_adv': 0.0, 'sil_frac': 0.0}
        with autocast('cuda'):
            sil_policy_loss = -(target_lp[sil_mask] * sil_adv[sil_mask]).mean()
            sil_value_loss = 0.5 * (sil_adv[sil_mask] ** 2).mean()
            sil_loss = self.sil_coeff * (sil_policy_loss + sil_value_loss)
        sil_clipped_step(self.optimizer, self.scaler, sil_loss, self.model.parameters(), 40.0)
        return {
            'sil_loss': sil_loss.item(), 'sil_pi': sil_policy_loss.item(), 'sil_v': sil_value_loss.item(),
            'sil_adv': sil_adv[sil_mask].mean().item(), 'sil_frac': sil_mask.float().sum().item() / mask.sum().item()
        }

    def expert_gate_closed(self) -> bool:
        hard = self.league.members['bot_hard']
        return hard.games_played >= self.expert_min_games and hard.agent_win_rate >= self.expert_threshold

    def update(self, rollouts: list) -> dict:
        stats = self._update(rollouts)
        stats_exp = stats_gold = None
        if self.expert and len(self.expert) >= self.batch_size // 2 and not self.expert_disabled:
            if self.expert_gate_closed():
                self.expert_disabled = True
                hard = self.league.members['bot_hard']
                print(f"  Expert disabled (H:{hard.agent_win_rate*100:.0f}% n={hard.games_played})")
            else:
                stats_exp = self._update_sil_gpu(self.expert.sample(self.batch_size // 2))
        if len(self.golden) >= self.batch_size // 4:
            golden_batch = self.golden.sample(min(self.batch_size // 2, len(self.golden)))
            if golden_batch:
                stats_gold = self._update_sil_gpu(golden_batch)
        return {**stats, **prefixed(stats_exp, "exp_"), **prefixed(stats_gold, "gold_")}

    def fmt(self, stats: dict) -> str:
        if "exp_sil_loss" in stats:
            exp_str = f"EXP L:{stats['exp_sil_loss']:.2f} p:{stats['exp_sil_pi']:.2f} v:{stats['exp_sil_v']:.2f} adv:{stats['exp_sil_adv']:.2f}({stats['exp_sil_frac']:.0%})"
        else:
            exp_str = "EXP:OFF" if self.expert_disabled else "EXP:--"
        sil_str = f"SIL L:{stats['gold_sil_loss']:.2f} adv:{stats['gold_sil_adv']:.2f}({stats['gold_sil_frac']:.0%})" if "gold_sil_loss" in stats else "SIL:--"
        gm = self.golden.stats()
        return f"VT L:{stats['loss']:.2f} p:{stats['pi']:+.3f} v:{stats['v']:.2f} H:{stats['ent']:.2f} rho:{stats['rho']:.1f}/{stats['rho_max']:.1f} EV:{stats['ev']:.2f} | {exp_str} | {sil_str} | GM:{gm['size']}({gm['fresh']}) | grad:{stats['grad']:.1f}->{stats['grad_clip']:.1f}"

    opt = property(lambda self: self.optimizer)

class DreamerAgent(Agent):
    kind = "dreamer"
    rollout_fields = ("first", "trunc", "next_obs", "next_feat")

    def __init__(self, spec: dict, learner: bool = False):
        self.spec, self.learner, self.warmstarted = spec, learner, False
        self.A, self.O = NUM_ACTIONS, OBS_DIM + FEATURE_DIM
        auto = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = (auto if spec["device"] == "auto" else torch.device(spec["device"])) if learner else torch.device("cpu")
        cfg = {**spec["model"], "action_space": "discrete", "batch": spec["batch"], "T": spec["T"], "seed": spec["seed"], "cpus": 1, "device": str(self.device), "task": "football"}
        self.model = DR.Dreamer(cfg, self.O, self.A, 1).to(self.device)
        if not learner:
            self.model.eval()
            return
        workers = spec["num_workers"]
        self.replay = Replay(workers, max(spec["replay_cap"] // workers, spec["T"] + 1), {"obs": self.O, "priv": 1, "act": self.A, "rew": 1, "first": 1, "term": 1})
        self.rng = np.random.default_rng(spec["seed"])
        self.carry, self.env_steps, self.credit, self.updates, self.last, self.credited_steps = np.zeros(workers), 0, 0.0, 0, {}, 0

    def policy_state(self):
        return {k: v for k, v in self.model.state_dict().items() if k.split(".")[0] in DREAMER_POLICY_KEYS}

    def set_weights(self, weights):
        missing, unexpected = self.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()}, strict=False)
        if unexpected or any(k.split(".")[0] in DREAMER_POLICY_KEYS for k in missing):
            raise KeyError(f"dreamer weights mismatch: unexpected={unexpected} missing={missing}")

    def _one_hot(self, idx: int) -> torch.Tensor:
        return torch.zeros(1, self.A) if idx == NO_ACTION else F.one_hot(torch.tensor([idx]), self.A).float()

    def act(self, obs, feat, prev_act, first, state, deterministic=False):
        x = torch.from_numpy(np.concatenate([obs, feat])).float().unsqueeze(0)
        action, state = self.model.act(x, torch.tensor([float(first)]), self._one_hot(prev_act), state, greedy=deterministic)
        return int(action.argmax(-1).item()), 0.0, 0.0, state

    def fingerprint(self, obs, feat):
        with torch.no_grad():
            x = torch.from_numpy(np.concatenate([obs, feat], axis=1)).float()
            n = len(x)
            deter, stoch, _, _ = self.model.rssm.obs_step(*self.model.rssm.initial(n), torch.zeros(n, self.A), torch.ones(n), self.model.enc(x))
            return self.model.actor_dist(self.model.feat(deter, stoch)).probs.mean(0).numpy()

    def observe(self, r, opp_mu=25.0):
        e, n = r['wid'], len(r['act'])
        x = np.concatenate([r['obs'], r['feat']], axis=1)
        nx = np.concatenate([r['next_obs'], r['next_feat']], axis=1)
        acts = np.eye(self.A, dtype=np.float32)[r['act']]
        zero_priv, zero_act = np.zeros(1, np.float32), np.zeros(self.A, np.float32)
        rew_in = 0.0 if r['first'][0] else self.carry[e]
        for i in range(n):
            self.replay.add(e, {"obs": x[i], "priv": zero_priv, "act": acts[i], "rew": rew_in, "first": r['first'][i], "term": 0.0})
            rew_in = r['rew'][i]
            if r['done'][i]:
                self.replay.add(e, {"obs": nx[i], "priv": zero_priv, "act": zero_act, "rew": rew_in, "first": 0.0, "term": 1.0 - r['trunc'][i]})
                rew_in = 0.0
        self.carry[e] = rew_in
        self.env_steps += n
        if self.env_steps >= self.spec["prefill"]:
            self.credit += n * self.spec["train_ratio"]
            self.credited_steps += n

    def update(self, rollouts: list) -> dict:
        bt = self.spec["batch"] * self.spec["T"]
        cap = self.spec["max_updates"] or 2 * math.ceil(sum(len(r['act']) for r in rollouts) * self.spec["train_ratio"] / bt)
        done_updates = 0
        while self.credit >= bt and self.replay.ready(self.spec["T"]) and done_updates < cap:
            self.credit -= bt
            batch = {k: torch.as_tensor(v, device=self.device) for k, v in self.replay.sample(self.spec["batch"], self.spec["T"], self.rng).items()}
            m = self.model.update(batch)
            done_updates += 1
            if m is not None:
                self.last = {k: float(v) for k, v in m.items() if v.ndim == 0}
        self.credit = min(self.credit, bt * cap)
        self.updates += done_updates
        return {**self.last, "model_updates": float(done_updates), "train_ratio_eff": self.updates * bt / max(self.credited_steps, 1)}

    def fmt(self, stats: dict) -> str:
        return "DR " + " ".join(f"{k}:{stats[k]:.3g}" for k in self.spec["log_keys"] if k in stats) + f" upd:{int(stats['model_updates'])} ratio:{stats['train_ratio_eff']:.3g}/{self.spec['train_ratio']} replay:{int(self.replay.n.sum())}"

    opt = property(lambda self: self.model.opt)

class ExpertDataset(Dataset):
    def __init__(self, parquet_path):
        df = pq.read_table(parquet_path).to_pandas()
        self.obs = np.array([np.frombuffer(b, dtype=np.float32) for b in df['obs']])
        self.actions = df['action'].values.astype(np.int64)
        self.active = df['active'].values.astype(np.int64)
        self.feat = np.array([FeatureEngineer.extract(o, a) for o, a in zip(self.obs, self.active)])

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {'obs': torch.from_numpy(self.obs[idx]).float(), 'feat': torch.from_numpy(self.feat[idx]).float(), 'action': torch.tensor(self.actions[idx]).long()}

class BCWarmstart:
    def __init__(self, spec: dict, parquet_path: str, save_path: str, epochs: int = 20, batch_size: int = 512, lr: float = 1e-3):
        self.spec, self.parquet_path, self.save_path = spec, parquet_path, save_path
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.device = torch.device(spec["device"])

    def _epoch(self, model, loader, optimizer=None):
        model.train(optimizer is not None)
        loss_sum, correct, total = 0.0, 0, 0
        for batch in loader:
            obs, feat, actions = batch['obs'].to(self.device), batch['feat'].to(self.device), batch['action'].to(self.device)
            with torch.set_grad_enabled(optimizer is not None):
                logits, _, _ = model(obs, feat)
                loss = F.cross_entropy(logits, actions)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            loss_sum += loss.item() * len(obs)
            correct += (logits.argmax(dim=-1) == actions).sum().item()
            total += len(obs)
        return loss_sum / total, correct / total * 100

    def run(self):
        dataset = ExpertDataset(self.parquet_path)
        n_train = int(len(dataset) * 0.9)
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        model = ImpalaAgent(self.spec).model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        best = -1.0
        for epoch in range(self.epochs):
            tr_loss, tr_acc = self._epoch(model, train_loader, optimizer)
            with torch.no_grad():
                val_loss, val_acc = self._epoch(model, val_loader)
            print(f"{epoch+1:3d} train {tr_loss:.4f}/{tr_acc:.1f}% val {val_loss:.4f}/{val_acc:.1f}%")
            if val_acc > best:
                best = val_acc
                torch.save({'model': model.state_dict(), 'val_acc': val_acc, 'epoch': epoch + 1}, self.save_path)
        print(f"best val acc {best:.1f}% saved to {self.save_path}")
        return model, val_acc

AGENTS = {"impala": ImpalaAgent, "dreamer": DreamerAgent}

def make_agent(spec: dict, learner: bool = False) -> Agent:
    if spec["agent"] not in AGENTS:
        raise ValueError(f"agent must be one of {sorted(AGENTS)}, got {spec['agent']!r}")
    return AGENTS[spec["agent"]](spec, learner)
