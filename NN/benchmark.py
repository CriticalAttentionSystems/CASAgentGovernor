"""Toy bump-hunt benchmark: is the in-network regulator more efficient?

THE PROBLEM IT MODELS

A bump hunt looks for a Gaussian signal sitting on a smoothly falling background
mass spectrum. A classifier that uses mass-correlated ("sharp") features can
raise its raw signal/background separation, but a cut on those features *sculpts*
the background mass shape -- it carves a fake bump, inflating or faking the very
excess the analysis is trying to measure. The job of decorrelation is to buy
separation without sculpting.

WHAT WE COMPARE (same data, same two-pathway score s = robust + (1-g)*sharp)

  none         no decorrelation (g -> 0): maximal separation, maximal sculpting.
  fixed-lambda constant g and a constant penalty weight: the standard baseline.
  cost-blind   the regulator, per-event g, but k=None so it never backs off.
  governed     the full regulator: per-event g * cost-aware backoff (k=3).

METRICS (a working point = keep the top `eps_b` fraction of background by score)

  Z         Asimov significance of the signal in a mass window at the working point.
  sculpt    W1 between the surviving-background mass shape and the inclusive
            background mass shape (mass units). Lower = less sculpting.

"Efficient" = more Z per unit sculpt. A good regulator sits on a better frontier.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regulator import GovernedScoreHead, soft_template, wasserstein1_grid

# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #
M_LO, M_HI = 100.0, 180.0
M_SIG, W_SIG = 140.0, 4.0
ROBUST_DIM, SHARP_DIM = 3, 4


def _sample(n: int, signal: bool, gen: torch.Generator) -> dict[str, torch.Tensor]:
    if signal:
        mass = M_SIG + W_SIG * torch.randn(n, generator=gen)
        mass = mass.clamp(M_LO, M_HI)
    else:
        # falling spectrum: truncated exponential on [M_LO, M_HI]
        u = torch.rand(n, generator=gen)
        scale = 30.0
        mass = M_LO - scale * torch.log(1 - u * (1 - math.exp(-(M_HI - M_LO) / scale)))

    cls = 1.0 if signal else 0.0
    mnorm = (mass - M_SIG) / (M_HI - M_LO)

    # robust: mass-blind, weakly separating
    x_robust = 0.6 * cls + torch.randn(n, ROBUST_DIM, generator=gen)

    # sharp: strongly separating BUT mass-correlated (sculpting risk).
    # each sharp dim leans on mass with a different slope; signal is shifted.
    slopes = torch.tensor([2.5, 1.5, -1.8, 0.9])
    x_sharp = (slopes[None, :] * mnorm[:, None]
               + 1.3 * cls
               + torch.randn(n, SHARP_DIM, generator=gen))
    return {"mass": mass, "x_robust": x_robust, "x_sharp": x_sharp,
            "label": torch.full((n,), cls)}


def make_dataset(n_sig: int, n_bkg: int, seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    s, b = _sample(n_sig, True, gen), _sample(n_bkg, False, gen)
    out = {k: torch.cat([s[k], b[k]], 0) for k in s}
    perm = torch.randperm(out["label"].numel(), generator=gen)
    return {k: v[perm] for k, v in out.items()}


# --------------------------------------------------------------------------- #
# mass-template machinery (shared working point)
# --------------------------------------------------------------------------- #
N_BINS = 40
CENTRES = torch.linspace(M_LO, M_HI, N_BINS)
BIN_W = float(CENTRES[1] - CENTRES[0])
BANDWIDTH = BIN_W


def hard_template(mass: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Normalised histogram of `mass` for the kept (boolean/weight) events."""
    k = torch.exp(-0.5 * ((mass[:, None] - CENTRES[None, :]) / BANDWIDTH) ** 2)
    t = (k * keep[:, None]).sum(0)
    return t / t.sum().clamp(min=1e-8)


# --------------------------------------------------------------------------- #
# baseline (fixed-g) model
# --------------------------------------------------------------------------- #
class PlainTwoPathway(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.robust = nn.Sequential(nn.Linear(ROBUST_DIM, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.sharp = nn.Sequential(nn.Linear(SHARP_DIM, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x_robust, x_sharp, g: float):
        return self.robust(x_robust).squeeze(-1) + (1.0 - g) * self.sharp(x_sharp).squeeze(-1)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
CONFIGS = {
    # no decorrelation: full sharp pathway, maximal sculpting
    "none":           dict(kind="plain",     g=0.0, lambda_max=0.0),
    # blunt global suppression: halve the sharp pathway, no penalty
    "fixed-suppress": dict(kind="plain",     g=0.5, lambda_max=0.0),
    # FAIR baseline: standard penalty-method decorrelation (g=0, direct eta penalty)
    "fixed-penalty":  dict(kind="plain",     g=0.0, lambda_max=50.0),
    # the regulator, per-event g, no cost-aware backoff
    "cost-blind":     dict(kind="regulator", k=None, lambda_max=50.0),
    # the full regulator: per-event g * cost-aware backoff
    "governed":       dict(kind="regulator", k=3.0,  lambda_max=50.0),
}


def train(cfg: dict, data: dict, ref_template: torch.Tensor, sigma_mass: float,
          steps: int = 400, batch: int = 4096, seed: int = 0):
    torch.manual_seed(seed)
    n = data["label"].numel()
    gen = torch.Generator().manual_seed(seed)

    if cfg["kind"] == "plain":
        model = PlainTwoPathway()
    else:
        model = GovernedScoreHead(ROBUST_DIM, SHARP_DIM, d_global=1,
                                  event_dim=ROBUST_DIM, k=cfg["k"], eta_tol=0.05)
        model.cell.reset()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=gen)
        mass, xr, xs, y = (data["mass"][idx], data["x_robust"][idx],
                           data["x_sharp"][idx], data["label"][idx])
        bkgw = (y < 0.5).float()
        # tight soft working point: threshold at the batch's 90th score percentile
        with torch.no_grad():
            pre = (model(xr, xs, cfg["g"]) if cfg["kind"] == "plain"
                   else model(xr, xs, torch.zeros(1), xr)[0])
            thr = torch.quantile(pre, 0.90)

        if cfg["kind"] == "plain":
            score = model(xr, xs, cfg["g"])
            reg_g_mean = torch.tensor(cfg["g"])
            eta_hat = None
        else:
            score, reg = model(xr, xs, torch.zeros(1), xr)
            reg_g_mean = reg["g"].mean()
            eta_hat = reg["eta_hat"]

        bce = F.binary_cross_entropy_with_logits(score, y)
        tmpl = soft_template(mass, score, bkgw, CENTRES, thr, BANDWIDTH)
        eta = wasserstein1_grid(tmpl, ref_template, BIN_W) / sigma_mass
        # Sculpting is penalised DIRECTLY (not gated by g). g is the control that
        # reduces eta by suppressing the mass-correlated sharp pathway; BCE pulls
        # g down, this term pulls it up, and g settles at the trade-off. Gating
        # the penalty by g (as the raw port did) lets a learned g escape to 0.
        loss = bce + cfg["lambda_max"] * eta ** 2
        if eta_hat is not None:
            loss = loss + (eta_hat - eta.detach()).abs()

        opt.zero_grad(); loss.backward(); opt.step()

        if cfg["kind"] == "regulator":
            model.cell.observe(eta, reg["g"])

    return model


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(model, cfg: dict, data: dict, ref_template: torch.Tensor,
             sigma_mass: float, eps_b: float = 0.02) -> dict:
    with torch.no_grad():
        xr, xs, mass, y = data["x_robust"], data["x_sharp"], data["mass"], data["label"]
        if cfg["kind"] == "plain":
            score = model(xr, xs, cfg["g"])
            g_mean = float(cfg["g"])
        else:
            score, reg = model(xr, xs, torch.zeros(1), xr)
            g_mean = float(reg["g"].mean())

        is_bkg, is_sig = y < 0.5, y > 0.5
        # working point: keep the top eps_b of BACKGROUND by score
        thr = torch.quantile(score[is_bkg], 1 - eps_b)
        keep = score > thr

        # sculpting: surviving-background mass shape vs inclusive background shape
        surv_bkg = keep & is_bkg
        sculpt = float(wasserstein1_grid(
            hard_template(mass[surv_bkg], torch.ones(surv_bkg.sum())),
            ref_template, BIN_W)) if surv_bkg.sum() > 1 else float("nan")

        # significance in the signal window
        win = (mass > M_SIG - 2 * W_SIG) & (mass < M_SIG + 2 * W_SIG)
        S = float((keep & is_sig & win).sum())
        B = float((keep & is_bkg & win).sum())
        Z = math.sqrt(2 * ((S + B) * math.log(1 + S / B) - S)) if B > 0 and S > 0 else 0.0

        sig_eff = float((keep & is_sig).sum()) / max(1, int(is_sig.sum()))
        return {"Z": Z, "sculpt": sculpt, "S": S, "B": B,
                "sig_eff": sig_eff, "g_mean": g_mean}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    train_data = make_dataset(n_sig=20000, n_bkg=20000, seed=1)   # balanced for training
    eval_data = make_dataset(n_sig=4000, n_bkg=80000, seed=2)     # realistic small-signal

    # static decorrelation anchor: inclusive background mass shape (mass-blind target)
    tr_bkg = train_data["label"] < 0.5
    ref_template = hard_template(train_data["mass"][tr_bkg],
                                 torch.ones(int(tr_bkg.sum())))
    sigma_mass = float(train_data["mass"][tr_bkg].std())

    print(f"{'config':<14}{'Z':>8}{'sculpt':>9}{'sig_eff':>9}{'g_mean':>8}"
          f"{'S':>8}{'B':>8}")
    print("-" * 64)
    rows = {}
    for name, cfg in CONFIGS.items():
        model = train(cfg, train_data, ref_template, sigma_mass, seed=0)
        m = evaluate(model, cfg, eval_data, ref_template, sigma_mass)
        rows[name] = m
        print(f"{name:<14}{m['Z']:>8.2f}{m['sculpt']:>9.3f}{m['sig_eff']:>9.3f}"
              f"{m['g_mean']:>8.3f}{m['S']:>8.0f}{m['B']:>8.0f}")

    print("-" * 64)
    # efficiency frontier: Z per unit sculpting (higher is better)
    print("\nZ per unit sculpt (higher = more efficient decorrelation):")
    for name, m in rows.items():
        eff = m["Z"] / m["sculpt"] if m["sculpt"] and m["sculpt"] == m["sculpt"] else float("nan")
        print(f"  {name:<14}{eff:>8.2f}")


if __name__ == "__main__":
    main()
