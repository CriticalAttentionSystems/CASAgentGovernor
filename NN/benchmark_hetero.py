"""Heterogeneous bump-hunt benchmark: the case where per-event g SHOULD win.

RESULT (2026-07-18): it does not. The differentiable gate stays inert. Trained
jointly with the network by SGD, g collapses to ~0 (BCE pulls it down to use the
sharp pathway) and the sculpting penalty is satisfied by adjusting the feature
MLPs' weights instead of by raising g. This holds across lambda in [50, 3000];
where strong decorrelation does occur it is done GLOBALLY by the weights (Z
collapses), never by the per-event gate (g_safe == g_risky at every setting).
Even forcing g to be the *only* decorrelation lever (detaching the penalty
gradient from the feature nets) did not help: g trained to seed-dependent,
useless values and produced no localized decorrelation. The optimizer routes
around the gate. The governor appears to work as an EXTERNAL imposed schedule
(its native form in the paper, where g is set by a state machine) but does not
survive being made a co-trained layer competing with the network it regulates.
The harness below is kept as the test that established this.

WHY THE HOMOGENEOUS TOY WAS A NULL RESULT

In ``benchmark.py`` a single global decorrelation weight already solved the
problem: the sharp network learned a mass-independent function by itself, so the
per-event gate had nothing to do (g collapsed to ~0) and the regulator did not
beat the plain penalty method. For the regulator's phase-space conditioning to
matter, TWO things must hold that did not hold there:

  1. **Heterogeneous sculpting risk.** The sharp features are mass-correlated
     (sculpt if cut) in a "risky" kinematic region and mass-blind (free to use)
     in a "safe" region. A single global decorrelation strength must compromise
     across the two; a per-event g can decorrelate hard in the risky region and
     stand down in the safe one.
  2. **Entanglement.** The region is visible to the gate (it is encoded in the
     mass-blind robust features) but NOT recoverable from the sharp features
     alone: every sharp dim is the same entangled scalar
     ``shift*label + slope*mass*risky`` plus independent noise, so no function of
     x_sharp can separate the class signal from the mass term. The sharp network
     therefore *cannot* self-decorrelate region-by-region -- only a gate that
     sees the region can.

Under those conditions the plain penalty method has to suppress the sharp
pathway everywhere (it cannot tell safe from risky), sacrificing the free
signal in the safe region. The regulator keeps it. That gap is the efficiency
win, and the per-region g diagnostic below shows the gate actually learned it.

Configs, metrics, and the two-pathway score are identical to ``benchmark.py``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .regulator import GovernedScoreHead, soft_template, wasserstein1_grid

M_LO, M_HI = 100.0, 180.0
M_SIG, W_SIG = 140.0, 4.0
ROBUST_DIM, SHARP_DIM = 3, 4
SHIFT, SLOPE = 1.6, 4.0            # class shift in sharp; mass slope in risky region


def _sample(n: int, signal: bool, gen: torch.Generator) -> dict[str, torch.Tensor]:
    cls = 1.0 if signal else 0.0
    region = (torch.rand(n, generator=gen) < 0.5).float()      # 1 = risky, 0 = safe
    if signal:
        mass = (M_SIG + W_SIG * torch.randn(n, generator=gen)).clamp(M_LO, M_HI)
    else:
        u = torch.rand(n, generator=gen)
        scale = 30.0
        mass = M_LO - scale * torch.log(1 - u * (1 - math.exp(-(M_HI - M_LO) / scale)))
    mnorm = (mass - M_SIG) / (M_HI - M_LO)

    # robust: mass-blind. [0] encodes region (visible to the gate); [1] weak class; [2] filler
    x_robust = torch.stack([
        3.0 * (2 * region - 1) + torch.randn(n, generator=gen),
        0.6 * cls + torch.randn(n, generator=gen),
        torch.randn(n, generator=gen),
    ], dim=1)

    # sharp: entangled. Same scalar in every dim (equal slope) + independent noise,
    # so the sharp net cannot separate `cls` from `mnorm*region` on its own.
    base = SHIFT * cls + SLOPE * mnorm * region                # [n]
    x_sharp = base[:, None] + torch.randn(n, SHARP_DIM, generator=gen)
    return {"mass": mass, "x_robust": x_robust, "x_sharp": x_sharp,
            "label": torch.full((n,), cls), "region": region}


def make_dataset(n_sig: int, n_bkg: int, seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    s, b = _sample(n_sig, True, gen), _sample(n_bkg, False, gen)
    out = {k: torch.cat([s[k], b[k]], 0) for k in s}
    perm = torch.randperm(out["label"].numel(), generator=gen)
    return {k: v[perm] for k, v in out.items()}


# ------------------------- mass templates (shared) ------------------------- #
N_BINS = 40
CENTRES = torch.linspace(M_LO, M_HI, N_BINS)
BIN_W = float(CENTRES[1] - CENTRES[0])
BANDWIDTH = BIN_W


def hard_template(mass: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    k = torch.exp(-0.5 * ((mass[:, None] - CENTRES[None, :]) / BANDWIDTH) ** 2)
    t = (k * keep[:, None]).sum(0)
    return t / t.sum().clamp(min=1e-8)


class PlainTwoPathway(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.robust = nn.Sequential(nn.Linear(ROBUST_DIM, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.sharp = nn.Sequential(nn.Linear(SHARP_DIM, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x_robust, x_sharp, g: float):
        return self.robust(x_robust).squeeze(-1) + (1.0 - g) * self.sharp(x_sharp).squeeze(-1)


CONFIGS = {
    "none":           dict(kind="plain",     g=0.0, lambda_max=0.0),
    "fixed-suppress": dict(kind="plain",     g=0.5, lambda_max=0.0),
    "fixed-penalty":  dict(kind="plain",     g=0.0, lambda_max=50.0),
    "cost-blind":     dict(kind="regulator", k=None, lambda_max=50.0),
    "governed":       dict(kind="regulator", k=3.0,  lambda_max=50.0),
}


def train(cfg, data, ref_template, sigma_mass, steps=800, batch=4096, seed=0):
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

    for step in range(steps):
        # penalty warm-up (same schedule for every config): let the sharp network
        # learn to discriminate before decorrelation ramps in, so the gate can see
        # WHERE the sharp pathway is worth keeping instead of starving it globally.
        lam_t = cfg["lambda_max"] * min(1.0, 2.0 * step / steps)
        idx = torch.randint(0, n, (batch,), generator=gen)
        mass, xr, xs, y = (data["mass"][idx], data["x_robust"][idx],
                           data["x_sharp"][idx], data["label"][idx])
        bkgw = (y < 0.5).float()
        with torch.no_grad():
            pre = (model(xr, xs, cfg["g"]) if cfg["kind"] == "plain"
                   else model(xr, xs, torch.zeros(1), xr)[0])
            thr = torch.quantile(pre, 0.90)

        if cfg["kind"] == "plain":
            score = model(xr, xs, cfg["g"]); reg = None
        else:
            score, reg = model(xr, xs, torch.zeros(1), xr)

        bce = F.binary_cross_entropy_with_logits(score, y)
        tmpl = soft_template(mass, score, bkgw, CENTRES, thr, BANDWIDTH)
        eta = wasserstein1_grid(tmpl, ref_template, BIN_W) / sigma_mass
        loss = bce + lam_t * eta ** 2                           # direct penalty (not gated by g), warmed up
        if reg is not None:
            loss = loss + (reg["eta_hat"] - eta.detach()).abs()

        opt.zero_grad(); loss.backward(); opt.step()
        if cfg["kind"] == "regulator":
            model.cell.observe(eta, reg["g"])
    return model


def evaluate(model, cfg, data, ref_template, sigma_mass, eps_b=0.02):
    with torch.no_grad():
        xr, xs, mass, y, region = (data["x_robust"], data["x_sharp"], data["mass"],
                                   data["label"], data["region"])
        if cfg["kind"] == "plain":
            score = model(xr, xs, cfg["g"]); g_all = torch.full_like(score, cfg["g"])
        else:
            score, reg = model(xr, xs, torch.zeros(1), xr); g_all = reg["g"]

        is_bkg, is_sig = y < 0.5, y > 0.5
        thr = torch.quantile(score[is_bkg], 1 - eps_b)
        keep = score > thr

        surv_bkg = keep & is_bkg
        sculpt = float(wasserstein1_grid(
            hard_template(mass[surv_bkg], torch.ones(int(surv_bkg.sum()))),
            ref_template, BIN_W)) if surv_bkg.sum() > 1 else float("nan")

        win = (mass > M_SIG - 2 * W_SIG) & (mass < M_SIG + 2 * W_SIG)
        S = float((keep & is_sig & win).sum())
        B = float((keep & is_bkg & win).sum())
        Z = math.sqrt(2 * ((S + B) * math.log(1 + S / B) - S)) if B > 0 and S > 0 else 0.0

        safe, risky = region < 0.5, region > 0.5
        return {"Z": Z, "sculpt": sculpt, "S": S, "B": B,
                "g_safe": float(g_all[safe].mean()), "g_risky": float(g_all[risky].mean())}


def main():
    torch.manual_seed(0)
    train_data = make_dataset(n_sig=20000, n_bkg=20000, seed=1)
    eval_data = make_dataset(n_sig=4000, n_bkg=80000, seed=2)

    tr_bkg = train_data["label"] < 0.5
    ref_template = hard_template(train_data["mass"][tr_bkg], torch.ones(int(tr_bkg.sum())))
    sigma_mass = float(train_data["mass"][tr_bkg].std())

    print(f"{'config':<15}{'Z':>7}{'sculpt':>9}{'g_safe':>8}{'g_risky':>9}"
          f"{'S':>8}{'B':>7}")
    print("-" * 63)
    rows = {}
    for name, cfg in CONFIGS.items():
        model = train(cfg, train_data, ref_template, sigma_mass, seed=0)
        m = evaluate(model, cfg, eval_data, ref_template, sigma_mass)
        rows[name] = m
        print(f"{name:<15}{m['Z']:>7.2f}{m['sculpt']:>9.3f}{m['g_safe']:>8.3f}"
              f"{m['g_risky']:>9.3f}{m['S']:>8.0f}{m['B']:>7.0f}")
    print("-" * 63)
    print("\nZ per unit sculpt (higher = more efficient decorrelation):")
    for name, m in rows.items():
        eff = m["Z"] / m["sculpt"] if m["sculpt"] and m["sculpt"] == m["sculpt"] else float("nan")
        print(f"  {name:<15}{eff:>8.2f}")


if __name__ == "__main__":
    main()
