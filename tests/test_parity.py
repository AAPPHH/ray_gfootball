import copy
import importlib.util
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
spec = importlib.util.spec_from_file_location("full_league", ROOT / "main" / "full_league.py")
FL = importlib.util.module_from_spec(spec)
sys.modules["full_league"] = FL
spec.loader.exec_module(FL)
from actor_critic.impala import compute_vtrace, impala_targets

ATOL = 1e-6
CFG = {"gamma": 0.999, "rho_bar": 1.0, "c_bar": 1.0, "log_rho_clip": 20.0, "adv_sigma_norm": False, "value_loss_form": "mse"}


def frozen_vtrace(behavior_lp, target_lp, rewards, values, bootstrap, dones, gamma=0.99, rho_bar=1.0, c_bar=1.0):
    B, T = rewards.shape
    rhos = torch.exp((target_lp - behavior_lp).clamp(-20, 20))
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)
    not_done = 1.0 - dones
    values_tp1 = torch.cat([values[:, 1:], bootstrap.unsqueeze(1)], dim=1)
    deltas = clipped_rhos * (rewards + gamma * values_tp1 * not_done - values)
    vs_minus_v = torch.zeros_like(rewards)
    acc = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        acc = deltas[:, t] + gamma * cs[:, t] * acc * not_done[:, t]
        vs_minus_v[:, t] = acc
    vs = values + vs_minus_v
    vs_tp1 = torch.cat([vs[:, 1:], bootstrap.unsqueeze(1)], dim=1)
    advantages = clipped_rhos * (rewards + gamma * vs_tp1 * not_done - values)
    return vs, advantages, clipped_rhos


class FrozenPopArtValueHead(nn.Module):
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


def frozen_update(self, rollouts) -> dict:
    batch = self._prepare_batch(rollouts)
    obs, feat, prev_act, act, beh_lp = batch['obs'], batch['feat'], batch['prev_act'], batch['act'], batch['lp']
    rew, done, bootstrap, mask, T = batch['rew'], batch['done'], batch['bootstrap'], batch['mask'], batch['max_len']
    with autocast('cuda'):
        logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
        dist = Categorical(logits=logits)
        target_lp = dist.log_prob(act)
        entropy = dist.entropy()
        values = self.model.value.denormalize(values_norm)
    with torch.no_grad():
        vs, adv, rhos = frozen_vtrace(beh_lp, target_lp.float().detach(), rew, values.float().detach(), bootstrap.float(), done, self.gamma)
        self.model.value.update_stats(vs[mask.bool()])
        vs_norm = self.model.value.normalize_target(vs)
        mean_rho = rhos[mask.bool()].mean().item()
        max_rho = rhos[mask.bool()].max().item()
    with autocast('cuda'):
        policy_loss = -((target_lp * adv.detach()) * mask).sum() / mask.sum()
        value_loss = ((values_norm - vs_norm.detach()) ** 2 * mask).sum() / mask.sum()
        ent_loss = -(entropy * mask).sum() / mask.sum()
        loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * ent_loss
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
        'rho': mean_rho, 'rho_max': max_rho, 'grad': grad_norm,
        'grad_clip': grad_clipped.item() if torch.is_tensor(grad_clipped) else grad_clipped
    }


def random_batch(B, T, gen, lengths=None, scale=1.0):
    lengths = lengths or [T] * B
    lp_b = -torch.rand(B, T, generator=gen) * 3
    lp_t = lp_b + (torch.randn(B, T, generator=gen) * scale)
    rew = torch.randn(B, T, generator=gen) * 0.3
    rew[torch.rand(B, T, generator=gen) < 0.8] = 0.0
    done = (torch.rand(B, T, generator=gen) < 0.05).float()
    values = torch.randn(B, T, generator=gen) * 2
    bootstrap = torch.randn(B, generator=gen)
    mask = torch.zeros(B, T)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1.0
        done[i, n:] = 1.0
        rew[i, n:] = 0.0
        lp_b[i, n:] = 0.0
    return lp_b, lp_t, rew, values, bootstrap, done, mask


def make_rollouts(lengths, seed):
    rng = np.random.default_rng(seed)
    out = []
    for n in lengths:
        rew = np.where(rng.random(n) < 0.2, rng.normal(size=n), 0.0)
        out.append({
            'obs': rng.normal(size=(n, FL.OBS_DIM)).astype(np.float32),
            'feat': rng.normal(size=(n, FL.FEATURE_DIM)).astype(np.float32),
            'prev_act': rng.integers(0, FL.NUM_ACTIONS, n),
            'act': rng.integers(0, FL.NUM_ACTIONS, n),
            'lp': (-rng.random(n) * 3).astype(np.float32),
            'rew': rew.astype(np.float32),
            'done': (rng.random(n) < 0.03).astype(np.float32),
            'bootstrap': float(rng.normal()),
        })
    return out


def make_learner(model, seed_lr=1e-3):
    learner = object.__new__(FL.LeagueLearner)
    learner.model = model
    learner.device = torch.device("cpu")
    learner.optimizer = torch.optim.Adam(model.parameters(), lr=seed_lr, eps=1e-5)
    learner.scaler = GradScaler('cuda')
    learner.gamma = 0.999
    learner.value_coeff = 0.2
    learner.entropy_coeff = 0.01
    return learner


def twin_models(seed):
    torch.manual_seed(seed)
    new = FL.Net(32, 32)
    old = copy.deepcopy(new)
    head = FrozenPopArtValueHead(128)
    head.load_state_dict(old.value.state_dict())
    old.value = head
    return old, new


class ParityTests(unittest.TestCase):
    def test_vtrace_matches_package(self):
        gen = torch.Generator().manual_seed(0)
        for B, T, lengths in [(4, 16, None), (5, 32, [32, 20, 7, 32, 1]), (1, 8, None), (3, 64, [64, 64, 33])]:
            for scale in (0.2, 1.0, 30.0):
                lp_b, lp_t, rew, values, bootstrap, done, mask = random_batch(B, T, gen, lengths, scale)
                vs, adv, rhos = frozen_vtrace(lp_b, lp_t, rew, values, bootstrap, done, 0.999)
                pv, pa, avg_rho, diag = compute_vtrace(lp_b, lp_t, rew, values, done, bootstrap, 0.999, 1.0, 1.0, 20.0)
                m = mask.bool()
                self.assertTrue(torch.allclose(vs.reshape(-1), pv, atol=ATOL))
                self.assertTrue(torch.allclose(adv.reshape(-1), pa, atol=ATOL))
                self.assertTrue(torch.allclose(rhos[m].mean(), torch.tensor(float(diag["is_ratio"].clamp(max=1.0)[m].mean())), atol=ATOL))

    def test_popart_matches_adapter(self):
        gen = torch.Generator().manual_seed(1)
        old = FrozenPopArtValueHead(128)
        new = FL.PopArtValueHead(128)
        new.load_state_dict(old.state_dict())
        self.assertEqual(list(old.state_dict().keys()), list(new.state_dict().keys()))
        for i in range(30):
            lp_b, lp_t, rew, values, bootstrap, done, mask = random_batch(4, 24, gen, [24, 24, 10, 17], 0.5)
            vs, _, _ = frozen_vtrace(lp_b, lp_t, rew, values, bootstrap, done, 0.999)
            old.update_stats(vs[mask.bool()])
            tg = impala_targets(lp_b, lp_t, rew, values, done, bootstrap, new, CFG, mask=mask)
            for k in ("mu", "sigma", "nu"):
                self.assertTrue(torch.allclose(getattr(old, k), getattr(new, k), atol=ATOL), k)
            self.assertTrue(torch.allclose(old.linear.weight, new.linear.weight, atol=ATOL))
            self.assertTrue(torch.allclose(old.linear.bias, new.linear.bias, atol=ATOL))
            self.assertTrue(torch.allclose(old.normalize_target(vs).reshape(-1), tg["normalized_targets"], atol=ATOL))

    def test_update_matches_frozen(self):
        for lengths in ([24, 24, 24], [24, 24, 9, 17], [16, 3]):
            old, new = twin_models(3)
            lo, ln = make_learner(old), make_learner(new)
            for step in range(3):
                rollouts = make_rollouts(lengths, 10 + step)
                so = frozen_update(lo, rollouts)
                sn = FL.LeagueLearner._update(ln, rollouts)
                for k in so:
                    self.assertAlmostEqual(so[k], sn[k], delta=ATOL * max(1.0, abs(so[k])), msg=f"{k} step {step} {lengths}")
                for (n, po), pn in zip(old.named_parameters(), new.parameters()):
                    self.assertTrue(torch.allclose(po.grad, pn.grad, atol=ATOL), f"grad {n}")
                    self.assertTrue(torch.allclose(po, pn, atol=ATOL), f"param {n}")
                for k in ("mu", "sigma", "nu"):
                    self.assertTrue(torch.allclose(getattr(old.value, k), getattr(new.value, k), atol=ATOL), k)

    def test_update_changes_state(self):
        _, new = twin_models(4)
        before = [p.detach().clone() for p in new.parameters()]
        FL.LeagueLearner._update(make_learner(new), make_rollouts([12, 12], 0))
        self.assertTrue(any(not torch.equal(b, p) for b, p in zip(before, new.parameters())))
        self.assertNotEqual(float(new.value.sigma), 1.0)


if __name__ == "__main__":
    unittest.main()
