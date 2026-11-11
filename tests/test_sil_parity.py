import sys
import unittest
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.distributions import Categorical

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "main"))
import agents as AG
import football_env as FE
ATOL = 1e-6
TRAJ = 12


class FrozenGoldenMemoryGPU:
    def __init__(self, capacity, max_uses, traj_len, device):
        self.capacity = capacity
        self.max_uses = max_uses
        self.traj_len = traj_len
        self.device = device
        self.size = 0
        self.obs = torch.zeros(capacity, traj_len, FE.OBS_DIM, device=device)
        self.feat = torch.zeros(capacity, traj_len, FE.FEATURE_DIM, device=device)
        self.act = torch.zeros(capacity, traj_len, dtype=torch.long, device=device)
        self.prev_act = torch.zeros(capacity, traj_len, dtype=torch.long, device=device)
        self.rew = torch.zeros(capacity, traj_len, device=device)
        self.done = torch.ones(capacity, traj_len, device=device)
        self.mask = torch.zeros(capacity, traj_len, device=device)
        self.lp = torch.zeros(capacity, traj_len, device=device)
        self.uses = torch.zeros(capacity, dtype=torch.long, device='cpu')
        self.opp_ratings = torch.zeros(capacity, device='cpu')
        self.lengths = torch.zeros(capacity, dtype=torch.long, device='cpu')

    def add(self, trajectory, opp_rating=25.0):
        traj_len = min(trajectory['obs'].shape[0], self.traj_len)
        if self.size >= self.capacity:
            expired = (self.uses >= self.max_uses).nonzero(as_tuple=True)[0]
            if len(expired) > 0:
                idx = expired[0].item()
            else:
                idx = self.opp_ratings[:self.size].argmin().item()
                if opp_rating <= self.opp_ratings[idx].item():
                    return False
        else:
            idx = self.size
            self.size += 1
        self.obs[idx].zero_()
        self.feat[idx].zero_()
        self.act[idx].zero_()
        self.prev_act[idx].zero_()
        self.rew[idx].zero_()
        self.done[idx].fill_(1.0)
        self.mask[idx].zero_()
        self.lp[idx].zero_()
        self.obs[idx, :traj_len] = torch.from_numpy(trajectory['obs'][:traj_len].copy()).to(self.device)
        self.feat[idx, :traj_len] = torch.from_numpy(trajectory['feat'][:traj_len].copy()).to(self.device)
        self.act[idx, :traj_len] = torch.from_numpy(trajectory['act'][:traj_len].copy()).to(self.device)
        self.prev_act[idx, :traj_len] = torch.from_numpy(trajectory['prev_act'][:traj_len].copy()).to(self.device)
        self.rew[idx, :traj_len] = torch.from_numpy(trajectory['rew'][:traj_len].copy()).to(self.device)
        self.done[idx, :traj_len] = torch.from_numpy(trajectory['done'][:traj_len].copy()).to(self.device)
        self.mask[idx, :traj_len] = 1.0
        self.uses[idx] = 0
        self.opp_ratings[idx] = opp_rating
        self.lengths[idx] = traj_len
        return True

    def sample(self, n):
        if self.size == 0:
            return None
        valid_idx = (self.uses[:self.size] < self.max_uses).nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            return None
        n = min(n, len(valid_idx))
        weights = 1.0 + self.opp_ratings[valid_idx] / 50.0
        probs = (weights / weights.sum()).numpy()
        chosen = np.random.choice(len(valid_idx), size=n, replace=False, p=probs)
        idx = valid_idx[chosen]
        for i in idx:
            self.uses[i] += 1
        return {
            'obs': self.obs[idx], 'feat': self.feat[idx], 'act': self.act[idx],
            'prev_act': self.prev_act[idx], 'rew': self.rew[idx], 'done': self.done[idx],
            'mask': self.mask[idx], 'lp': self.lp[idx],
            'bootstrap': torch.zeros(n, device=self.device), 'max_len': self.traj_len
        }

    def stats(self):
        if self.size == 0:
            return {'size': 0, 'fresh': 0, 'avg_opp': 0}
        fresh = (self.uses[:self.size] < self.max_uses).sum().item()
        return {'size': self.size, 'fresh': fresh, 'avg_opp': self.opp_ratings[:self.size].mean().item()}

    def __len__(self):
        return self.size


def frozen_update_sil_gpu(self, batch):
    if batch is None:
        return None
    obs, feat, prev_act, act = batch['obs'], batch['feat'], batch['prev_act'], batch['act']
    rew, done, bootstrap, mask, T = batch['rew'], batch['done'], batch['bootstrap'], batch['mask'], batch['max_len']
    with autocast('cuda'):
        logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
        target_lp = Categorical(logits=logits).log_prob(act)
        values = self.model.value.denormalize(values_norm)
    with torch.no_grad():
        mc_returns = torch.zeros_like(rew)
        running = bootstrap.clone()
        for t in reversed(range(T)):
            running = torch.where(done[:, t].bool(), rew[:, t], rew[:, t] + self.gamma * running)
            mc_returns[:, t] = running
        sil_adv = torch.clamp(mc_returns - values.detach(), min=0)
    sil_mask = (sil_adv > 0) & (mask > 0)
    if not sil_mask.any():
        return {'sil_loss': 0.0, 'sil_pi': 0.0, 'sil_v': 0.0, 'sil_adv': 0.0, 'sil_frac': 0.0}
    with autocast('cuda'):
        sil_policy_loss = -(target_lp[sil_mask] * sil_adv[sil_mask]).mean()
        sil_value_loss = 0.5 * (sil_adv[sil_mask] ** 2).mean()
        sil_loss = self.sil_coeff * (sil_policy_loss + sil_value_loss)
    self.optimizer.zero_grad()
    self.scaler.scale(sil_loss).backward()
    self.scaler.unscale_(self.optimizer)
    nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return {
        'sil_loss': sil_loss.item(), 'sil_pi': sil_policy_loss.item(), 'sil_v': sil_value_loss.item(),
        'sil_adv': sil_adv[sil_mask].mean().item(), 'sil_frac': sil_mask.float().sum().item() / mask.sum().item()
    }


def traj(rng, n):
    rew = np.zeros(n, dtype=np.float32)
    rew[-1] = 1.0
    done = np.zeros(n, dtype=np.float32)
    done[-1] = 1.0
    return {'obs': rng.normal(size=(n, FE.OBS_DIM)).astype(np.float32), 'feat': rng.normal(size=(n, FE.FEATURE_DIM)).astype(np.float32),
            'prev_act': rng.integers(0, FE.NUM_ACTIONS, n), 'act': rng.integers(0, FE.NUM_ACTIONS, n), 'rew': rew, 'done': done}


def make_learner(model):
    learner = object.__new__(AG.ImpalaAgent)
    learner.model = model
    learner.device = torch.device("cpu")
    learner.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    learner.scaler = GradScaler('cuda')
    learner.gamma = 0.99
    learner.sil_coeff = 0.5
    return learner


def batches_equal(a, b):
    if a is None or b is None:
        return a is b
    return all(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


class SilParity(unittest.TestCase):
    def test_golden_memory_matches_frozen(self):
        for cap, uses, seed in ((4, 2, 0), (6, 3, 1), (3, 1, 2)):
            rng = np.random.default_rng(seed)
            old, new = FrozenGoldenMemoryGPU(cap, uses, TRAJ, torch.device("cpu")), AG.GoldenMemoryGPU(cap, uses, TRAJ, torch.device("cpu"))
            for step in range(40):
                t, opp = traj(rng, int(rng.integers(3, TRAJ + 4))), float(rng.integers(5, 45))
                self.assertEqual(old.add(t, opp), new.add(t, opp), (seed, step))
                if step % 2 == 1:
                    n = int(rng.integers(1, 5))
                    np.random.seed(step)
                    a = old.sample(n)
                    np.random.seed(step)
                    b = new.sample(n)
                    self.assertTrue(batches_equal(a, b), (seed, step))
                self.assertEqual((len(old), old.stats()), (len(new), new.stats()))
                for k in ("uses", "opp_ratings", "lengths"):
                    self.assertTrue(torch.equal(getattr(old, k), torch.as_tensor(getattr(new, k))), (k, seed, step))
                for k in ("obs", "feat", "act", "prev_act", "rew", "done", "mask"):
                    self.assertTrue(torch.equal(getattr(old, k), getattr(new, k)), (k, seed, step))

    def test_update_sil_matches_frozen(self):
        torch.manual_seed(3)
        m_old = AG.Net(32, 32)
        m_new = AG.Net(32, 32)
        m_new.load_state_dict(m_old.state_dict())
        lo, ln = make_learner(m_old), make_learner(m_new)
        rng = np.random.default_rng(7)
        gm = AG.GoldenMemoryGPU(6, 4, TRAJ, torch.device("cpu"))
        for i in range(5):
            gm.add(traj(rng, 8 + i), 20.0 + i)
        for step in range(3):
            np.random.seed(step)
            batch = gm.sample(3)
            so, sn = frozen_update_sil_gpu(lo, batch), AG.ImpalaAgent._update_sil_gpu(ln, batch)
            self.assertEqual(so.keys(), sn.keys())
            for k in so:
                self.assertAlmostEqual(so[k], sn[k], delta=ATOL * max(1.0, abs(so[k])), msg=f"{k} step {step}")
            for (n, po), pn in zip(m_old.named_parameters(), m_new.parameters()):
                self.assertTrue(torch.allclose(po, pn, atol=ATOL), f"param {n} step {step}")
        self.assertTrue(batches_equal(None, AG.ImpalaAgent._update_sil_gpu(ln, None)))


if __name__ == "__main__":
    unittest.main()
