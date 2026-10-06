"""Toy Monte Carlo for Omega- -> Lambda(p,pi-) K-, to validate the run + CV.

Generates a realistic-shaped candidate sample -- an Omega- peak at 1.6725 GeV on
a smooth falling combinatorial background -- with the same sculpting structure the
method is meant to handle: some kinematic ("sharp") features are mass-correlated
in a "risky" region (cutting on them carves the M(Lambda K-) spectrum) and safe
elsewhere, while the topology/PID ("robust") features are mass-blind.

It then runs the full pipeline from ``omega_run`` -- data-driven risk target, gate
vs penalty-only, and k-fold cross-validation -- so you can confirm that:

  * the gate's advantage survives HONEST out-of-fold evaluation (not just the
    overfit single-split number), and
  * the significance is a real Omega- peak effect, cross-checkable against the
    injected truth.

This is a validation harness; it uses no real data. Feature/column layout matches
``omega_run.COLUMNS`` so the same code path exercises both.

WHAT THE VALIDATION SHOWS (2026-07-18)

  * The k-fold wrapper works: Z_naive > Z_oof, i.e. it removes the overfitting a
    single train/eval split hides -- trust Z_oof.
  * The data-driven risk estimator recovers the sculpting region WITHOUT truth
    (g_safe ~= 0.08, g_risky ~= 0.91 every seed).
  * Whether the gate improves out-of-fold significance is REGIME-DEPENDENT, and
    the two label modes behave differently:
      - MC-truth mode: the gate sits at ~parity with penalty-only on this toy
        (mean ratio ~0.9-0.95x over seeds) and stays there even as
        ``sculpt_strength`` is raised -- the W1 penalty alone already controls
        sculpting, so the gate's extra machinery mostly adds variance. At small
        Omega- yields the significance spread swamps any small edge.
      - CWoLa (data-only) mode: this is where the gate tends to help, because the
        peak-vs-sideband classifier can CHEAT by reading the mass through
        mass-correlated features (faking a peak); the gate suppresses exactly that
        pathway where it is dangerous. (The fake ntuple in the repo shows the gate
        winning out-of-fold in CWoLa mode.)
    So: run YOUR data through ``omega_run --kfold --repeats N`` in BOTH modes and
    read the Z_oof spread. This toy shows the tool honestly reports when the gate
    helps -- not that it always does.

Use ``sculpt_strength`` to vary the sculpting; use ``mode`` to switch label schemes.
"""
from __future__ import annotations

import torch

from . import omega_run as O
from .omega_run import OMEGA_MASS, FIT_RANGE, _standardize, assemble, metrics_from_scores


def make_toy_omega(n_sig: int, n_bkg: int, seed: int = 0, resolution: float = 0.008,
                   sculpt_strength: float = 4.0, mode: str = "cwola") -> dict:
    """``sculpt_strength`` = slope of the sharp-feature mass-correlation in the
    risky region. 0 = no sculpting (gate has nothing to do); larger = stronger
    sculpting tension (where the gate's phase-space conditioning pays off)."""
    gen = torch.Generator().manual_seed(seed)
    lo, hi = FIT_RANGE
    span = hi - lo

    def block(n, sig):
        region = (torch.rand(n, generator=gen) < 0.5).float()      # risky / safe
        if sig:
            mass = OMEGA_MASS + resolution * torch.randn(n, generator=gen)
        else:
            mass = lo + span * torch.rand(n, generator=gen) ** 1.3  # smooth falling
        mass = mass.clamp(lo, hi)
        cls = 1.0 if sig else 0.0
        mnorm = (mass - OMEGA_MASS) / span
        # sharp: the DOMINANT discriminator, entangled and mass-correlated ONLY in
        # the risky region -> using it there sculpts the M(Lambda K-) spectrum.
        base = 1.8 * cls + sculpt_strength * mnorm * region
        sharp = base[:, None] + torch.randn(n, 4, generator=gen)
        # robust: mass-blind topology/PID, deliberately WEAK, so decorrelation that
        # kills the sharp pathway genuinely costs signal (the tension the gate eases).
        # [0] carries the region (visible to the gate); [1] weak class; [2..3] filler
        # -- like Lambda decay length / pointing angle / dm_lambda.
        rob = torch.stack([
            3.0 * (2 * region - 1) + torch.randn(n, generator=gen),
            0.3 * cls + torch.randn(n, generator=gen),
            torch.randn(n, generator=gen),
            torch.randn(n, generator=gen),
        ], dim=1)
        return mass, sharp, rob, torch.full((n,), cls)

    ms, ss, rs, ts = block(n_sig, True)
    mb, sb, rb, tb = block(n_bkg, False)
    mass = torch.cat([ms, mb]); sharp = torch.cat([ss, sb])
    rob = torch.cat([rs, rb]); truth = torch.cat([ts, tb])
    perm = torch.randperm(mass.numel(), generator=gen)
    mass, sharp, rob, truth = mass[perm], sharp[perm], rob[perm], truth[perm]
    return assemble(mass, _standardize(sharp), _standardize(rob), truth, mode=mode)


def main():
    torch.manual_seed(0)
    # realistic-ish small yield: a few thousand candidates, signal a minority.
    # MC-truth mode is the clean validation (CWoLa is available in omega_run for
    # data-only running, but its interaction with mass-correlated features is subtler).
    data = make_toy_omega(n_sig=1500, n_bkg=30000, seed=0, mode="mc")
    n, npk, nsb = (data["mass"].numel(), int(data["in_peak"].sum()),
                   int(data["is_sb"].sum()))
    true_sig_in_peak = int((data["truth"] > 0.5).logical_and(data["in_peak"]).sum())
    print(f"toy MC: {n} candidates  (peak {npk}: {true_sig_in_peak} true Omega + "
          f"{npk - true_sig_in_peak} bkg; sideband {nsb})  mode=mc")

    # validate the risk estimator recovers the sculpting region without truth
    tfn, risk = O.estimate_risk_target(data, data["x_sharp"].shape[1], seed=0)
    with torch.no_grad():
        gt = tfn(data["x_robust"]); risky = data["x_robust"][:, 0] > 0  # region proxy for check
    print(f"data-driven risk target: g(safe)={float(gt[~risky].mean()):.2f}  "
          f"g(risky)={float(gt[risky].mean()):.2f}   "
          f"(cluster |corr(sharp,mass)| in [{risk.min():.2f}, {risk.max():.2f}])")

    # multi-seed CV with the peak-fit significance -> report spread, not one number
    O.run_kfold_repeats(data, K=5, seeds=range(4), sig_mode="fit", steps=400)

    print("\nValidation: read the mean +/- spread, not a single seed. On this toy in "
          "MC mode the gate is ~parity; its edge shows in CWoLa mode (mode='cwola'), "
          "where mass-leakage is the threat it is built to stop.")


if __name__ == "__main__":
    main()
