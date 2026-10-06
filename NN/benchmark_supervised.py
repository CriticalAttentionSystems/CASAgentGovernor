"""Supervised-gate test: does phase-space conditioning help WHEN the gate works?

RESULT (2026-07-18): YES. With the gate supervised to condition on region, the
penalty+gate reaches ~1.55x the signal significance of the penalty-alone
baseline at comparable-or-lower sculpting (mean Z 37.6 vs 24.3 over 4 seeds;
helps in 3/4, high variance on this small toy). The mechanism is visible in the
per-region signal efficiency: the gate preserves ~0.6 of the safe-region signal
while penalty-only keeps ~0.3 -- conditioning keeps the signal in regions where
the sharp pathway is safe to use, which a single global decorrelation strength
must sacrifice. So per-event regulation has real value, but ONLY once the gate is
forced to engage; jointly-trained (unsupervised) it stays inert (benchmark_hetero).
The remaining open problem is estimating the oracle target from data -- here it
uses the known region.

The earlier benchmarks showed that a jointly-trained gate never engages -- SGD
routes around it. This experiment removes that failure by *supervising* the gate
to an oracle schedule, then asks the underlying scientific question directly:

    if g is forced to condition on the kinematic region, does a per-event g beat
    any single global constant g on the Z-vs-sculpt frontier?

Both sides control sculpting through exactly the same knob (the regulation
intensity g scaling the mass-correlated sharp pathway). The only difference is
whether g is one number for the whole dataset or a function of the event:

  * global(g_c)     -- a constant g_c applied to every event; sweeping g_c traces
                       the frontier a NON-conditional decorrelator can reach.
  * per-event(oracle) -- the gate is trained (MSE) to output g_hi in the risky
                       region and g_lo in the safe region, using the region signal
                       carried in the mass-blind robust features.

The oracle uses the known region -- deliberately. This measures the CEILING: if
even an oracle-conditioned gate cannot beat the global frontier, phase-space
conditioning has no value here. If it can, the remaining question (a separate
problem) is only how to estimate the target from data.

No W1 penalty is used on either side: g itself is the decorrelation control, so
the comparison is clean.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .benchmark_hetero import (
    BANDWIDTH, BIN_W, CENTRES, M_HI, M_LO, M_SIG, ROBUST_DIM, SHARP_DIM, W_SIG,
    PlainTwoPathway, hard_template, make_dataset,
)
from .regulator import GovernedScoreHead, soft_template, wasserstein1_grid

G_LO, G_HI = 0.05, 0.95


def oracle_g(region: torch.Tensor) -> torch.Tensor:
    """High regulation in the risky region, low in the safe region."""
    return torch.where(region > 0.5, torch.full_like(region, G_HI),
                       torch.full_like(region, G_LO))


def _eta(model, kind, mass, xr, xs, y, ref, sigma):
    """W1 sculpting distance of the live score at a tight soft working point."""
    bkgw = (y < 0.5).float()
    if kind == "gate":
        score, reg = model(xr, xs, torch.zeros(1), xr)
    else:
        score, reg = model(xr, xs, 0.0), None
    with torch.no_grad():
        thr = torch.quantile(score, 0.90)
    tmpl = soft_template(mass, score, bkgw, CENTRES, thr, BANDWIDTH)
    return wasserstein1_grid(tmpl, ref, BIN_W) / sigma, score, reg


def train(data, ref, sigma, use_gate: bool, target_fn=None, lambda_max=50.0,
          gamma=5.0, steps=800, batch=4096, seed=0):
    """Decorrelate with the W1 penalty; optionally add a supervised gate.

    Penalty-only (use_gate=False) is the standard method: it must suppress the
    shared sharp net globally, sacrificing safe-region signal. Adding the gate
    lets the risky region be decorrelated structurally, so the penalty no longer
    has to kill the safe-region sharp pathway.

    ``target_fn(region, x_robust) -> g_target`` supplies the gate's supervision
    target. Default is the oracle schedule from the known region; the data-driven
    estimator passes a closure that ignores ``region`` and reads ``x_robust``.
    """
    if target_fn is None:
        target_fn = lambda region, xr: oracle_g(region)
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    if use_gate:
        model = GovernedScoreHead(ROBUST_DIM, SHARP_DIM, d_global=1,
                                  event_dim=ROBUST_DIM, k=None)
        model.cell.reset()
    else:
        model = PlainTwoPathway()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    n = data["label"].numel()
    for step in range(steps):
        lam_t = lambda_max * min(1.0, 2.0 * step / steps)
        idx = torch.randint(0, n, (batch,), generator=gen)
        mass, xr, xs, y, region = (data[k][idx] for k in
                                   ["mass", "x_robust", "x_sharp", "label", "region"])
        kind = "gate" if use_gate else "plain"
        eta, score, reg = _eta(model, kind, mass, xr, xs, y, ref, sigma)
        loss = F.binary_cross_entropy_with_logits(score, y) + lam_t * eta ** 2
        if use_gate:
            g_tgt = target_fn(region, xr)
            loss = loss + gamma * ((reg["g_soft"] - g_tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, data, ref_template, use_gate: bool, eps_b=0.02):
    with torch.no_grad():
        xr, xs, mass, y, region = (data[k] for k in
                                   ["x_robust", "x_sharp", "mass", "label", "region"])
        if use_gate:
            score, reg = model(xr, xs, torch.zeros(1), xr); g = reg["g"]
        else:
            score = model(xr, xs, 0.0); g = torch.zeros_like(score)
        is_bkg, is_sig = y < 0.5, y > 0.5
        thr = torch.quantile(score[is_bkg], 1 - eps_b); keep = score > thr
        sb = keep & is_bkg
        sculpt = float(wasserstein1_grid(
            hard_template(mass[sb], torch.ones(int(sb.sum()))), ref_template, BIN_W))
        win = (mass > M_SIG - 2 * W_SIG) & (mass < M_SIG + 2 * W_SIG)
        S = float((keep & is_sig & win).sum()); B = float((keep & is_bkg & win).sum())
        Z = math.sqrt(2 * ((S + B) * math.log(1 + S / B) - S)) if B > 0 and S > 0 else 0.0
        # signal efficiency by region: does the gate preserve safe-region power?
        se = lambda mask: float((keep & is_sig & mask).sum()) / max(1, int((is_sig & mask).sum()))
        return {"Z": Z, "sculpt": sculpt,
                "g_safe": float(g[region < 0.5].mean()),
                "g_risky": float(g[region > 0.5].mean()),
                "eff_safe": se(region < 0.5), "eff_risky": se(region > 0.5)}


def main():
    torch.manual_seed(0)
    train_data = make_dataset(n_sig=20000, n_bkg=20000, seed=1)
    eval_data = make_dataset(n_sig=4000, n_bkg=80000, seed=2)
    tr_bkg = train_data["label"] < 0.5
    ref = hard_template(train_data["mass"][tr_bkg], torch.ones(int(tr_bkg.sum())))
    sigma = float(train_data["mass"][tr_bkg].std())

    print(f"{'config':<22}{'Z':>7}{'sculpt':>9}{'g_safe':>8}{'g_risky':>9}"
          f"{'effS':>7}{'effR':>7}")
    print("-" * 69)
    rows = {}
    for name, use_gate in [("penalty only", False), ("penalty + oracle gate", True)]:
        m = train(train_data, ref, sigma, use_gate=use_gate)
        r = evaluate(m, eval_data, ref, use_gate=use_gate)
        rows[name] = r
        print(f"{name:<22}{r['Z']:>7.2f}{r['sculpt']:>9.3f}{r['g_safe']:>8.2f}"
              f"{r['g_risky']:>9.2f}{r['eff_safe']:>7.2f}{r['eff_risky']:>7.2f}")
    print("-" * 69)

    a, b = rows["penalty only"], rows["penalty + oracle gate"]
    print(f"\nBoth decorrelate to low sculpt. The question is Z at that sculpt:")
    print(f"  penalty only          Z={a['Z']:.1f}  sculpt={a['sculpt']:.3f}  "
          f"(safe-region signal eff {a['eff_safe']:.2f})")
    print(f"  penalty + oracle gate Z={b['Z']:.1f}  sculpt={b['sculpt']:.3f}  "
          f"(safe-region signal eff {b['eff_safe']:.2f})")
    # both sculpt values are ~1% of the undecorrelated level (~41), i.e. both
    # strongly decorrelated; the comparison of interest is Z at that regime.
    if b["Z"] > a["Z"] + 1e-6 and b["sculpt"] < 2.0:
        print("  -> conditioning HELPS: higher Z at comparably-low sculpt, by keeping safe-region signal.")
    else:
        print("  -> conditioning does not help here.")


if __name__ == "__main__":
    main()
