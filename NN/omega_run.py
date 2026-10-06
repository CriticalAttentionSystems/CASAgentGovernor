"""Run the governed decorrelation gate on real Omega- -> Lambda(p,pi-) K- data.

WHAT THIS TESTS

A classifier that separates true Omega- from combinatorial Lambda-K- background
can sharpen the peak in M(Lambda K-), but if it leans on mass-correlated
kinematics it SCULPTS the combinatorial spectrum -- faking or distorting the very
1.6725 GeV peak you are measuring. This script compares:

  penalty only          standard decorrelation (a global penalty on sculpting)
  penalty + gate        the governor's per-event gate, supervised to a
                        data-driven sculpting-risk target (no truth region needed)

and reports Omega- peak significance vs. background sculpting for each.

HOW TO USE

  1. Produce a FLAT ntuple (ROOT or CSV), one row per Omega- candidate, from your
     existing HIPO selection -- do the Lambda vertex fit / combinatorics in your
     analysis code, then just dump features + M(Lambda K-). Reading raw HIPO
     banks here is not recommended.
  2. Fill in COLUMNS below with your branch names and the robust/sharp split.
  3. Pick LABEL_MODE ("mc" if you have a truth flag, else "cwola" for data-only).
  4. python -m NN.omega_run --table path/to/ntuple.root   (or .csv)

Dependencies: torch (installed). ROOT input needs `uproot`; CSV needs `pandas`.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .benchmark_datadriven import _kmeans          # general Lloyd k-means
from .regulator import GovernedScoreHead, soft_template, wasserstein1_grid

# =========================================================================== #
# ADAPTER -- edit this block for your ntuple
# =========================================================================== #
COLUMNS = dict(
    mass="om_mass",                 # M(Lambda K-) invariant mass, in GeV
    # mass-correlated, sculpting-prone: kinematics that feed the invariant mass
    sharp=["p_lambda", "p_km", "cos_open_lam_km", "p_omega"],
    # mass-blind topology / quality: should NOT encode the Omega mass
    robust=["lambda_decaylen", "cos_pointing", "dm_lambda", "vtx_chi2",
            "chi2pid_p", "chi2pid_pim", "chi2pid_km"],
    truth="is_signal",              # MC truth flag (LABEL_MODE="mc"); ignored for cwola
)

LABEL_MODE = "cwola"                # "mc" (truth flag) or "cwola" (peak vs sidebands)

OMEGA_MASS = 1.6725                 # PDG Omega- mass, GeV
FIT_RANGE = (1.55, 1.85)            # mass window analysed (GeV)
PEAK_HALFWIDTH = 0.012             # +/- around OMEGA_MASS = signal region (~3 sigma); tune to resolution
SIDEBAND = (0.020, 0.060)          # |m - OMEGA_MASS| in this ring = background (sidebands)
N_BINS = 60                         # mass-template bins across FIT_RANGE
SIG_RESOLUTION = 0.006             # Omega- peak sigma (GeV) for the peak-fit significance; set to your fitted width
BKG_ORDER = 2                       # polynomial-background order for the peak fit


# =========================================================================== #
# data loading
# =========================================================================== #
def _read_table(path: str) -> dict[str, "torch.Tensor"]:
    cols = [COLUMNS["mass"], *COLUMNS["sharp"], *COLUMNS["robust"]]
    if LABEL_MODE == "mc":
        cols.append(COLUMNS["truth"])
    if path.endswith(".root"):
        import uproot
        with uproot.open(path) as f:
            tree = f[f.keys()[0].split(";")[0]] if hasattr(f, "keys") else f
            arr = tree.arrays(cols, library="np")
    else:
        import pandas as pd
        arr = pd.read_csv(path)[cols].to_dict("series")
    return {c: torch.as_tensor(arr[c], dtype=torch.float32).reshape(-1) for c in cols}


def _standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(0)) / x.std(0).clamp(min=1e-6)


def assemble(mass, x_sharp, x_robust, truth=None, mode=LABEL_MODE) -> dict:
    """Attach labels, training weights, and the peak/sideband masks.

    Shared by the real-data loader and the toy MC so both go through exactly the
    same windowing and (for CWoLa) weak-supervision setup.
    """
    dm = (mass - OMEGA_MASS).abs()
    in_peak = dm < PEAK_HALFWIDTH
    is_sb = (dm >= SIDEBAND[0]) & (dm < SIDEBAND[1])
    if mode == "mc":
        if truth is None:
            raise ValueError("LABEL_MODE='mc' needs a truth column")
        label = truth.float(); train_w = torch.ones_like(mass)
    elif mode == "cwola":
        # weak supervision: peak-region (signal+bkg) vs sideband (bkg). Only
        # events in one of the two windows carry a training label; others weight 0.
        label = in_peak.float(); train_w = (in_peak | is_sb).float()
    else:
        raise ValueError(mode)
    return dict(mass=mass, x_sharp=x_sharp, x_robust=x_robust, label=label,
                train_w=train_w, is_sb=is_sb, in_peak=in_peak,
                truth=(truth.float() if truth is not None else torch.full_like(mass, -1.0)))


def _subset(data: dict, mask: torch.Tensor) -> dict:
    return {k: v[mask] for k, v in data.items()}


def load_omega(path: str) -> dict:
    raw = _read_table(path)
    mass = raw[COLUMNS["mass"]]
    keep = (mass > FIT_RANGE[0]) & (mass < FIT_RANGE[1])
    mass = mass[keep]
    x_sharp = _standardize(torch.stack([raw[c][keep] for c in COLUMNS["sharp"]], 1))
    x_robust = _standardize(torch.stack([raw[c][keep] for c in COLUMNS["robust"]], 1))
    truth = raw[COLUMNS["truth"]][keep] if LABEL_MODE == "mc" else None
    return assemble(mass, x_sharp, x_robust, truth, LABEL_MODE)


# =========================================================================== #
# mass templates on the real grid
# =========================================================================== #
CENTRES = torch.linspace(*FIT_RANGE, N_BINS)
BIN_W = float(CENTRES[1] - CENTRES[0])
BANDWIDTH = BIN_W


def hard_template(mass, keep):
    k = torch.exp(-0.5 * ((mass[:, None] - CENTRES[None, :]) / BANDWIDTH) ** 2)
    t = (k * keep[:, None]).sum(0)
    return t / t.sum().clamp(min=1e-8)


# =========================================================================== #
# models
# =========================================================================== #
class Plain(nn.Module):
    def __init__(self, rdim, sdim, hidden=64):
        super().__init__()
        self.robust = nn.Sequential(nn.Linear(rdim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.sharp = nn.Sequential(nn.Linear(sdim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, xr, xs, g=0.0):
        return self.robust(xr).squeeze(-1) + (1.0 - g) * self.sharp(xs).squeeze(-1)


def estimate_risk_target(data, sdim, k=12, corr_mid=0.3, corr_width=0.08, seed=0):
    """Per-event sculpting-risk target from data: local |corr(sharp, mass)| among
    SIDEBAND (background) events, clustered in mass-blind robust space. No truth used."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(sdim, 64), nn.GELU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    y_sb = data["in_peak"].float()                 # proxy target just to get a sharp summary
    w = (data["in_peak"] | data["is_sb"]).float()
    gen = torch.Generator().manual_seed(seed); n = data["mass"].numel()
    for _ in range(400):
        idx = torch.randint(0, n, (4096,), generator=gen)
        s = net(data["x_sharp"][idx]).squeeze(-1)
        loss = (F.binary_cross_entropy_with_logits(s, y_sb[idx], reduction="none")
                * w[idx]).sum() / w[idx].sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        s_sharp = net(data["x_sharp"]).squeeze(-1)
        xr = data["x_robust"]; rmean, rstd = xr.mean(0), xr.std(0).clamp(min=1e-6)
        xr_std = (xr - rmean) / rstd
        centroids = _kmeans(xr_std, k=k, seed=seed)
        assign = torch.cdist(xr_std, centroids).argmin(1)
        risk = torch.zeros(k)
        sb = data["is_sb"]
        for c in range(k):
            m = sb & (assign == c)
            if m.sum() > 20:
                risk[c] = torch.corrcoef(torch.stack([s_sharp[m], data["mass"][m]]))[0, 1].abs()
        gtarget = 0.05 + 0.90 * torch.sigmoid((risk - corr_mid) / corr_width)

    def target_fn(xr_batch):
        a = torch.cdist((xr_batch - rmean) / rstd, centroids).argmin(1)
        return gtarget[a]
    return target_fn, risk


def train(data, ref, sigma, use_gate, target_fn=None, lam_max=50.0, gamma=5.0,
          steps=800, batch=4096, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    rdim, sdim = data["x_robust"].shape[1], data["x_sharp"].shape[1]
    model = (GovernedScoreHead(rdim, sdim, d_global=1, event_dim=rdim, k=None)
             if use_gate else Plain(rdim, sdim))
    if use_gate:
        model.cell.reset()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    n = data["mass"].numel()
    for step in range(steps):
        lam_t = lam_max * min(1.0, 2.0 * step / steps)
        idx = torch.randint(0, n, (batch,), generator=gen)
        mass, xr, xs = data["mass"][idx], data["x_robust"][idx], data["x_sharp"][idx]
        y, w = data["label"][idx], data["train_w"][idx]
        if use_gate:
            score, reg = model(xr, xs, torch.zeros(1), xr)
        else:
            score, reg = model(xr, xs, 0.0), None
        # background for the sculpting template = sidebands (pure combinatorial)
        bkgw = data["is_sb"][idx].float()
        with torch.no_grad():
            thr = torch.quantile(score, 0.90)
        eta = wasserstein1_grid(soft_template(mass, score, bkgw, CENTRES, thr, BANDWIDTH),
                                ref, BIN_W) / sigma
        bce = (F.binary_cross_entropy_with_logits(score, y, reduction="none") * w
               ).sum() / w.sum().clamp(min=1)
        loss = bce + lam_t * eta ** 2
        if use_gate:
            loss = loss + gamma * ((reg["g_soft"] - target_fn(xr)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def peak_fit_significance(mass_kept, resolution=SIG_RESOLUTION, bkg_order=BKG_ORDER):
    """Signal yield and its significance from a peak+background fit of the kept
    events' M(Lambda K-) spectrum.

    With the Omega- mass and resolution fixed, the model
        counts(x) ~ S * gauss(x; OMEGA_MASS, resolution) + sum_k c_k * x^k
    is LINEAR in {S, c_k}, so the weighted least-squares fit (Poisson bin errors)
    has a closed form -- no iterative fitter or scipy needed. Returns
    (Z = S/sigma_S, S, sigma_S, B_under_peak).
    """
    n_par = 1 + bkg_order + 1
    if mass_kept.numel() < n_par + 3:
        return 0.0, 0.0, float("nan"), 0.0
    lo, hi = FIT_RANGE
    counts = torch.histc(mass_kept, bins=N_BINS, min=lo, max=hi)
    x = CENTRES
    xs = 2 * (x - lo) / (hi - lo) - 1.0                         # conditioned poly basis on [-1,1]
    g = torch.exp(-0.5 * ((x - OMEGA_MASS) / resolution) ** 2); g = g / g.sum()
    M = torch.stack([g] + [xs ** k for k in range(bkg_order + 1)], dim=1)
    w = 1.0 / counts.clamp(min=1.0)                            # Poisson weights
    MtW = M.t() * w
    A = MtW @ M
    beta = torch.linalg.solve(A, MtW @ counts)
    cov = torch.linalg.inv(A)
    S = float(beta[0]); sigma_S = float(cov[0, 0].clamp(min=0).sqrt())
    peak = (x - OMEGA_MASS).abs() < PEAK_HALFWIDTH
    B = float((M[:, 1:] @ beta[1:])[peak].clamp(min=0).sum())
    Z = S / sigma_S if sigma_S > 0 and S > 0 else 0.0
    return Z, S, sigma_S, B


def metrics_from_scores(data, score, ref, eps_b=0.05, sig_mode="count"):
    """Omega- significance vs sculpting from a vector of per-event scores.

    sig_mode="count": S+B in the peak window vs sidebands scaled to the window
    (fast, robust). sig_mode="fit": peak+polynomial fit of the kept spectrum
    (a real bump-hunt yield). Both report the same sculpting metric.
    """
    with torch.no_grad():
        mass, is_sb, in_peak = data["mass"], data["is_sb"], data["in_peak"]
        # working point: keep the top eps_b of SIDEBAND (background) by score
        thr = torch.quantile(score[is_sb], 1 - eps_b); keep = score > thr
        sb_keep = keep & is_sb
        sculpt = float(wasserstein1_grid(
            hard_template(mass[sb_keep], torch.ones(int(sb_keep.sum()))), ref, BIN_W)) \
            if sb_keep.sum() > 1 else float("nan")
        if sig_mode == "fit":
            Z, S, sigma_S, B = peak_fit_significance(mass[keep])
            return {"Z": Z, "sculpt": sculpt, "S": S, "B": B, "sigma_S": sigma_S}
        # count mode
        Npk = float((keep & in_peak).sum())
        Nsb = float((keep & is_sb).sum())
        B = Nsb * (2 * PEAK_HALFWIDTH) / (2 * (SIDEBAND[1] - SIDEBAND[0]))
        S = max(0.0, Npk - B)
        Z = math.sqrt(2 * ((S + B) * math.log(1 + S / B) - S)) if B > 0 and S > 0 else 0.0
        return {"Z": Z, "sculpt": sculpt, "S": S, "B": B, "kept_peak": Npk}


def _score(model, data, use_gate):
    with torch.no_grad():
        xr, xs = data["x_robust"], data["x_sharp"]
        return model(xr, xs, torch.zeros(1), xr)[0] if use_gate else model(xr, xs, 0.0)


def evaluate(model, data, ref, use_gate, eps_b=0.05):
    return metrics_from_scores(data, _score(model, data, use_gate), ref, eps_b)


def _ref_sigma(data):
    sb = data["is_sb"]
    ref = hard_template(data["mass"][sb], torch.ones(int(sb.sum())))
    return ref, float(data["mass"][sb].std())


def kfold_scores(data, K, use_gate, seed=0, **train_kw):
    """Out-of-fold scores: each event is scored by a model trained on other folds.

    The sculpting-risk target (for the gate) and the sideband reference are both
    re-fit on each fold's TRAINING events only, so nothing leaks into the held-out
    scores -- the honest significance the toy MC below validates against.
    """
    n = data["mass"].numel()
    g = torch.Generator().manual_seed(seed)
    fold = torch.randperm(n, generator=g) % K
    oof = torch.zeros(n)
    for i in range(K):
        te = fold == i
        tr = _subset(data, ~te)
        ref, sigma = _ref_sigma(tr)
        target_fn = None
        if use_gate:
            target_fn, _ = estimate_risk_target(tr, tr["x_sharp"].shape[1], seed=seed)
        model = train(tr, ref, sigma, use_gate=use_gate, target_fn=target_fn,
                      seed=seed, **train_kw)
        oof[te] = _score(model, _subset(data, te), use_gate)
    return oof


def run_kfold(data, K=5, seed=0, sig_mode="count"):
    ref, _ = _ref_sigma(data)
    print(f"\n{K}-fold cross-validated (out-of-fold scores, significance='{sig_mode}'):")
    print(f"{'config':<22}{'Z_naive':>9}{'Z_oof':>8}{'sculpt':>9}{'S':>8}{'B':>8}")
    print("-" * 64)
    for name, use_gate in [("penalty only", False), ("penalty + gate", True)]:
        # naive: one model trained and scored on all events (overfit-prone)
        ref_all, sigma_all = _ref_sigma(data)
        tfn = estimate_risk_target(data, data["x_sharp"].shape[1], seed=seed)[0] if use_gate else None
        naive_model = train(data, ref_all, sigma_all, use_gate=use_gate, target_fn=tfn, seed=seed)
        zna = evaluate(naive_model, data, ref_all, use_gate)["Z"] if sig_mode == "count" \
            else metrics_from_scores(data, _score(naive_model, data, use_gate), ref_all, sig_mode="fit")["Z"]
        # honest: out-of-fold
        oof = kfold_scores(data, K, use_gate, seed=seed)
        m = metrics_from_scores(data, oof, ref, sig_mode=sig_mode)
        print(f"{name:<22}{zna:>9.2f}{m['Z']:>8.2f}{m['sculpt']:>9.3f}"
              f"{m['S']:>8.0f}{m['B']:>8.0f}")
    print("-" * 64)
    print("Trust Z_oof, not Z_naive. Z_naive > Z_oof is the overfitting the CV removes.")


def run_kfold_repeats(data, K=5, seeds=range(5), sig_mode="count", **train_kw):
    """Repeat K-fold CV over several seeds; report Z_oof spread, not one number.

    Each seed reshuffles the fold assignment and re-inits the models, so the
    spread is the honest uncertainty on the significance -- the thing to read on a
    small Omega- sample where a single number is not trustworthy.
    """
    import statistics as st
    ref, _ = _ref_sigma(data)
    seeds = list(seeds)
    print(f"\n{K}-fold CV over {len(seeds)} seeds (significance='{sig_mode}'):")
    print(f"{'config':<22}{'Z_oof mean':>12}{'std':>7}{'min':>7}{'max':>7}"
          f"{'sculpt':>9}")
    print("-" * 64)
    out = {}
    for name, use_gate in [("penalty only", False), ("penalty + gate", True)]:
        zs, scs = [], []
        for s in seeds:
            oof = kfold_scores(data, K, use_gate, seed=s, **train_kw)
            m = metrics_from_scores(data, oof, ref, sig_mode=sig_mode)
            zs.append(m["Z"]); scs.append(m["sculpt"])
        out[name] = zs
        sd = st.pstdev(zs) if len(zs) > 1 else 0.0
        print(f"{name:<22}{st.mean(zs):>12.2f}{sd:>7.2f}{min(zs):>7.2f}"
              f"{max(zs):>7.2f}{st.mean(scs):>9.3f}")
    print("-" * 64)
    a, b = out["penalty only"], out["penalty + gate"]
    wins = sum(gi > pi for gi, pi in zip(b, a))
    print(f"gate/penalty mean Z_oof ratio: {st.mean(b) / st.mean(a):.2f}x   "
          f"(gate higher in {wins}/{len(seeds)} seeds)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True, help="flat ntuple .root or .csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kfold", type=int, default=0,
                    help="K for cross-validation (0 = single train/eval, overfit-prone)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="number of CV seeds; >1 reports Z_oof mean +/- spread")
    ap.add_argument("--sig", choices=["count", "fit"], default="count",
                    help="significance: 'count' (window/sideband) or 'fit' (peak+bkg fit)")
    args = ap.parse_args()

    data = load_omega(args.table)
    if args.kfold:
        print(f"loaded {data['mass'].numel()} candidates  "
              f"(peak {int(data['in_peak'].sum())}, sideband {int(data['is_sb'].sum())}); "
              f"label_mode={LABEL_MODE}")
        if args.repeats > 1:
            run_kfold_repeats(data, K=args.kfold, seeds=range(args.repeats), sig_mode=args.sig)
        else:
            run_kfold(data, K=args.kfold, seed=args.seed, sig_mode=args.sig)
        return
    ref = hard_template(data["mass"][data["is_sb"]],
                        torch.ones(int(data["is_sb"].sum())))     # inclusive sideband shape
    sigma = float(data["mass"][data["is_sb"]].std())
    print(f"loaded {data['mass'].numel()} candidates  "
          f"(peak {int(data['in_peak'].sum())}, sideband {int(data['is_sb'].sum())}); "
          f"label_mode={LABEL_MODE}")

    target_fn, risk = estimate_risk_target(data, data["x_sharp"].shape[1], seed=args.seed)
    print(f"per-cluster sculpting risk |corr(sharp,mass)|: "
          f"min={risk.min():.2f} max={risk.max():.2f}")

    print(f"\n{'config':<22}{'Z':>8}{'sculpt':>9}{'S':>9}{'B':>9}")
    print("-" * 57)
    for name, kw in [("penalty only", dict(use_gate=False)),
                     ("penalty + gate", dict(use_gate=True, target_fn=target_fn))]:
        m = train(data, ref, sigma, seed=args.seed, **kw)
        r = evaluate(m, data, ref, use_gate=kw["use_gate"])
        print(f"{name:<22}{r['Z']:>8.2f}{r['sculpt']:>9.3f}{r['S']:>9.0f}{r['B']:>9.0f}")
    print("-" * 57)
    print("\nHigher Z at comparable-or-lower sculpt => the gate helps on your data.")


if __name__ == "__main__":
    main()
