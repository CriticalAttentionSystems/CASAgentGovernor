# `NN/` — The governor as a differentiable decorrelation gate

This package ports the CAS finite-state **governor** into a neural network as a
differentiable layer, and tests whether it makes a **physics bump-hunt analysis
more efficient** — specifically, whether an in-network regulator can raise signal
significance *without sculpting the invariant-mass spectrum*.

It contains the layer itself, a graded series of synthetic benchmarks that
establish when the idea works (and when it does not), and a runner + toy Monte
Carlo for a real Ω⁻ → Λ(pπ⁻)K⁻ bump hunt.

**One-line summary of the result:** the regulator's *per-event, phase-space-aware*
decorrelation is genuinely valuable, but only when the gate is **supervised to
engage** (a jointly-trained gate is out-competed by the network it regulates),
and its benefit is **regime-dependent** — largest in a data-driven (CWoLa) search
where mass-leakage is the real threat.

---

## Contents

- [Background: the two ideas](#background-the-two-ideas)
- [Install](#install)
- [Package layout](#package-layout)
- [Core API (`regulator.py`)](#core-api-regulatorpy)
- [The synthetic benchmarks](#the-synthetic-benchmarks)
- [Results: the full arc](#results-the-full-arc)
- [Running on real Ω⁻ data (`omega_run.py`)](#running-on-real-ω-data-omega_runpy)
- [The Ω⁻ toy Monte Carlo (`omega_toymc.py`)](#the-ω-toy-monte-carlo-omega_toymcpy)
- [Findings and caveats](#findings-and-caveats)
- [Reproducing every result](#reproducing-every-result)

---

## Background: the two ideas

### The governor, as a layer

In the CAS paper the governor sits *outside* an agent and its regulation
intensity `g ∈ [0, 0.99]` is a schedule set by a five-state machine
(`NORMAL → DISTURBED → RECOVERING → STUCK → RELEASED`, intensities
`0.00, 0.75, 0.85, 0.95, 0.99`). Built into a network, the same object becomes an
**activation**: the five-state ladder is a softmax over five regimes and `g` is
its expectation — an input-conditioned scalar the network computes about itself
and backpropagates through.

### The bump-hunt sculpting problem

A classifier that separates signal from background on a falling mass spectrum can
sharpen a peak, but if it leans on **mass-correlated features** a cut on them
*sculpts* the background mass shape — carving a fake or distorted bump. The job of
**decorrelation** is to buy separation without sculpting.

The governor's proposed contribution is **phase-space conditioning**: compute `g`
*per event* so the network leans on mass-correlated information where sculpting is
cheap and stands down where it is expensive — something a single global
decorrelation weight cannot do.

### The two-pathway score

Every model here scores an event as

```
s(x) = s_robust(x_blind)  +  (1 − g(x)) · s_sharp(x_all)
```

- `s_robust` sees only **mass-blind** features (safe to use freely).
- `s_sharp` may use **mass-correlated** features (powerful, but sculpts).
- `g(x)` is the regulation intensity; `g → 1` suppresses the sculpting pathway.

---

## Install

Requires Python 3.11+ and PyTorch.

```bash
pip install torch          # required for everything
pip install pandas         # only for reading CSV ntuples (real data)
pip install uproot         # only for reading ROOT ntuples (real data)
```

All modules are run as package modules from the repository root
(`CASAgentGovernor/`):

```bash
python -m NN.benchmark
```

---

## Package layout

| File | Role |
|---|---|
| `intensity.py` | The five-state regulatory ladder (`RegulatoryState`, `INTENSITY`), shared with the Java governor. |
| `regulator.py` | **The core.** The governor as a differentiable layer: `RegulatoryCell`, `GovernedScoreHead`, and the differentiable sculpting cost. |
| `benchmark.py` | Homogeneous toy — the null case: the regulator gives no benefit. |
| `benchmark_hetero.py` | Heterogeneous toy — a jointly-trained gate stays **inert**. |
| `benchmark_supervised.py` | Oracle-supervised gate — phase-space conditioning **does** help (~1.55×). |
| `benchmark_datadriven.py` | Estimates the gate target from data — recovers the oracle benefit with no truth. |
| `omega_run.py` | **Real-data runner:** load an ntuple, k-fold CV, peak-fit significance, gate vs baseline. |
| `omega_toymc.py` | A realistic Ω⁻ toy MC that validates the runner + CV. |

The four `benchmark_*` files form a deliberate progression — each answers a
question raised by the previous one. Read them in order.

---

## Core API (`regulator.py`)

### `RegulatoryCell(d_global, d_event=0, hidden=64, k=3.0, lam=0.3, eta_tol=0.05, hard=False)`

A recurrent five-state regulator producing a **per-event** intensity times a
**per-step** cost-aware backoff. The two axes are deliberately separate:

- **Per event (kinematics):** `g_soft(x)`, from a softmax over the ladder
  conditioned on `d_event` per-event features — the phase-space conditioning.
- **Per step (history):** the backoff `p_apply = 1 − min(1, k·r)`, a single scalar
  evolving over training steps via the EWMA induced-cost rate `r`. Global on
  purpose: "how much has my recent tightening been sculpting?" is a property of
  the whole batch, not one event.

```
g_applied(x) = g_soft(x) · p_apply
```

- `forward(global_context, event_feats) → dict` with keys `g` (per-event applied
  intensity), `g_soft`, `state_probs`, `eta_hat`, `p_apply`, `gate`.
- `observe(eta_obs, g_applied)` — closes the loop after the loss: one-sided
  induced-cost gate (tightening is charged, loosening is free) + EWMA update.
- `k=None` → cost-blind (no backoff). `hard=True` → straight-through Bernoulli
  backoff (keeps the stochastic basin structure). `reset()` at episode boundaries.

Training uses **truncated (one-step) BPTT** — the recurrent state is detached each
step, matching the online `forward → loss → observe` loop.

### `GovernedScoreHead(robust_dim, sharp_dim, d_global, event_dim=0, hidden=64, **cell_kwargs)`

The two-pathway score with an internal `RegulatoryCell` gating the sharp pathway.
`forward(x_robust, x_sharp, global_context, event_feats) → (score, reg)`.

### Differentiable sculpting cost

- `soft_template(mass, score, bkg_weight, centres, thr, bandwidth, tau=0.05)` —
  a differentiable background-only mass template of the *surviving* events. The
  veto is a sigmoid and binning is a Gaussian kernel, so it has a gradient.
- `wasserstein1_grid(a, b, bin_width)` — exact W1 between two densities on a shared
  grid, `∫|CDF_a − CDF_b|`, differentiable in closed form (no sorting/OT solver).
- `governed_loss(...)` — `BCE + mean(g)·λ·W1² + β·|η̂ − η|`. **Note:** this gated
  form is degenerate for a *learned* `g` (see [Results](#results-the-full-arc));
  the benchmarks penalize sculpting **directly** instead.

### `intensity.py`

`RegulatoryState` (enum), `INTENSITY` (dict → the ladder), and
`clamp_from_intensity(g) → Clamp(penalty_scale=g, sharp_scale=1−g)` — the two
complementary arms an intensity drives.

---

## The synthetic benchmarks

Each prints a table and a verdict. Run them from the repo root.

```bash
python -m NN.benchmark              # homogeneous
python -m NN.benchmark_hetero       # heterogeneous, jointly trained
python -m NN.benchmark_supervised   # oracle-supervised gate
python -m NN.benchmark_datadriven   # data-driven target
```

**Common setup.** A signal bump on a falling mass spectrum. "Sharp" features are
mass-correlated (cutting on them sculpts); "robust" features are mass-blind. The
metric pair is always **Z** (Asimov significance in the signal window at a fixed
background working point) and **sculpt** (W1 between the surviving-background mass
shape and the inclusive shape — lower is less sculpting).

**Configs.**

- `none` — no decorrelation (full sharp pathway; maximal sculpting).
- `fixed-suppress` — blunt: halve the sharp pathway, no penalty.
- `fixed-penalty` — standard penalty-method decorrelation (the fair baseline).
- `cost-blind` — the regulator, per-event `g`, no cost-aware backoff.
- `governed` — the full regulator: per-event `g` × cost-aware backoff (`k=3`).

---

## Results: the full arc

### 1. Homogeneous (`benchmark.py`) — no benefit

| config | Z | sculpt | Z / sculpt |
|---|---|---|---|
| none | 96.5 | 19.3 | 5.0 |
| fixed-suppress | 96.7 | 19.2 | 5.0 |
| **fixed-penalty** | **50.0** | **0.31** | **160** |
| cost-blind | 47.2 | 0.71 | 67 |
| governed | 32.8 | 1.0 | 32 |

The plain penalty method wins; the regulator is no better. With nothing to
condition on, `g` collapses to ≈0 and the sharp network decorrelates itself.

> **A bug found and fixed here.** The original `mean(g)·λ·η²` loss gates the
> sculpting penalty *by g itself*. A learned `g` then zeroes the whole penalty by
> setting `g=0` (which also turns the sharp pathway fully on) — the penalty
> gradient actively drives `g→0`. This is fine when `g` is an imposed schedule but
> degenerate when learned. The benchmarks penalize sculpting **directly** instead.

### 2. Heterogeneous, jointly trained (`benchmark_hetero.py`) — the gate is inert

The data now has **heterogeneous sculpting risk**: sharp features are
mass-correlated in a "risky" region and mass-blind in a "safe" region, with the
region visible to the gate but **entangled** in the sharp features so the sharp
network cannot self-decorrelate. A per-event gate *should* win here.

It does not. Trained jointly by SGD, `g` starts at ≈0.71 (uniform over the ladder)
and is driven to **0** — `g_safe = g_risky` at every setting, across `λ` from 50
to 3000. Where strong decorrelation does happen (`λ=800`) it is done *globally* by
the weights and Z collapses to 8. **The optimizer routes around the gate:** BCE
pulls `g` down, and the sculpting penalty is cheaper to satisfy by adjusting the
feature-network weights than by raising `g`.

### 3. Oracle-supervised gate (`benchmark_supervised.py`) — conditioning helps

Force the gate to engage by *supervising* it to an oracle schedule (high `g` in
the risky region, low in the safe region), then compare the standard penalty
against penalty + gate.

| config | Z (mean ± sd, 4 seeds) | safe-region signal eff |
|---|---|---|
| penalty only | 24.3 ± 7.4 | 0.30 |
| **penalty + oracle gate** | **37.6 ± 15.0** | **0.60** |

**~1.55× the significance** at comparable sculpting (helps in 3/4 seeds). The
mechanism is visible in `eff_safe`: the gate keeps ~2× the safe-region signal that
the global decorrelator must sacrifice. So phase-space conditioning has real
value — once the gate is made to work.

### 4. Data-driven target (`benchmark_datadriven.py`) — deployable

Remove the oracle: estimate each event's sculpting risk from data as the local
`|corr(sharp_score, mass)|` among background, clustered in mass-blind feature
space, and supervise the gate to that.

| config | Z (mean, 4 seeds) | gain |
|---|---|---|
| penalty only | 24.3 | 1.00× |
| penalty + oracle gate | 37.6 | 1.55× |
| **penalty + data-driven gate** | **42.5** | **1.75×** |

The estimator recovers the true region at **99.7–99.8% agreement using no region
label**, and matches (within noise) the oracle it approximates.

### Bottom line of the arc

1. The regulator's phase-space conditioning is genuinely more efficient — up to
   ~1.75× significance at matched decorrelation on the toy.
2. **It must be supervised to engage.** A co-trained differentiable gate is
   out-competed by the network it regulates and collapses to `g=0`.
3. The useful mechanism is the **per-event conditioning**, *not* the cost-aware
   backoff. In decorrelation, tightening *reduces* the disturbance (sculpting),
   whereas the governor's backoff assumes tightening *causes* it — so the backoff
   points the wrong way and the `governed` config trails `cost-blind`.

---

## Running on real Ω⁻ data (`omega_run.py`)

Maps a real Ω⁻ → Λ(pπ⁻)K⁻ analysis onto the method: **`mass` = M(ΛK⁻)** is the
protected bump-hunt observable, the signal is true Ω⁻, the background is
combinatorial ΛK⁻.

### Step 1 — produce a flat ntuple

From your existing HIPO selection, emit **one row per Ω⁻ candidate** (ROOT or CSV)
with `M(ΛK⁻)` and your features as columns. Do the Λ vertex fit and combinatorics
in your analysis code; **do not parse raw HIPO banks in Python.**

### Step 2 — fill in the adapter block

Edit the top of `omega_run.py`:

```python
COLUMNS = dict(
    mass="om_mass",                          # M(Lambda K-), GeV
    sharp=["p_lambda", "p_km", ...],         # mass-correlated kinematics (sculpting-prone)
    robust=["lambda_decaylen", "cos_pointing", "dm_lambda", "vtx_chi2",
            "chi2pid_p", "chi2pid_pim", "chi2pid_km"],   # mass-blind topology/PID
    truth="is_signal",                       # MC truth flag (only for LABEL_MODE="mc")
)
LABEL_MODE = "cwola"        # "mc" or "cwola"
OMEGA_MASS, FIT_RANGE = 1.6725, (1.55, 1.85)
PEAK_HALFWIDTH, SIDEBAND = 0.012, (0.020, 0.060)
SIG_RESOLUTION, BKG_ORDER = 0.006, 2
```

**Feature split guidance.** `sharp` = anything that feeds the invariant mass
(daughter momenta, opening angle, candidate momentum). `robust` = topology/PID
that should not encode the Ω mass (Λ decay length, pointing angle, Λ-mass residual
`M(pπ⁻) − m_Λ`, vertex χ²/DOCA, PID χ²). When unsure, put it in `sharp` — the
data-driven risk estimator measures each cluster's correlation with mass and lets
the gate protect what actually sculpts. **Never put `M(ΛK⁻)` (or anything nearly
collinear with it) in the features** — it is the protected variable.

### Step 3 — labels

- **`LABEL_MODE = "mc"`** — use a truth flag from your Monte Carlo. Cleanest.
- **`LABEL_MODE = "cwola"`** — data-only weak supervision: train to separate the
  Ω peak window from the sidebands. This is exactly where decorrelation is
  *essential* — without it the classifier learns "am I near 1.6725 GeV?" through
  mass-correlated features and manufactures a peak; the gate stops that.

The sidebands double as the **background sample** for the risk estimator — no
truth needed.

### Step 4 — run

```bash
# single split (fast, overfit-prone — for a first look only)
python -m NN.omega_run --table your_ntuple.root --kfold 5

# honest: multi-seed CV with a real peak fit (recommended)
python -m NN.omega_run --table your_ntuple.root --kfold 5 --repeats 5 --sig fit
```

**CLI flags**

| flag | meaning |
|---|---|
| `--table PATH` | flat `.root` (needs `uproot`) or `.csv` (needs `pandas`) |
| `--kfold K` | K-fold cross-validation; `0` = single train/eval |
| `--repeats N` | number of CV seeds; `>1` reports Z_oof mean ± spread |
| `--sig {count,fit}` | significance: sideband-count Asimov, or a peak+background fit |
| `--seed S` | RNG seed for the single-split path |

### What it reports

- **`Z_naive` vs `Z_oof`** — single-split vs out-of-fold significance. **Trust
  `Z_oof`.** `Z_naive > Z_oof` is the overfitting the CV removes. This matters:
  Ω⁻ is rare, and a NN will happily inflate significance on a small sample.
- **`sculpt`** — residual background sculpting at the working point.
- With `--repeats`, the **mean, std, min, max** of `Z_oof` and the gate/penalty
  ratio, plus how many seeds the gate won. On a small sample, read the *spread*,
  not one number.

### Significance modes

- `count` — S+B in the peak window vs sidebands scaled to the window width. Fast,
  robust.
- `fit` — `peak_fit_significance()` fits the kept M(ΛK⁻) spectrum with a Gaussian
  (mass and `SIG_RESOLUTION` fixed) plus an order-`BKG_ORDER` polynomial. Because
  the shape is fixed the fit is **linear** and solved in closed form — no scipy.
  Returns `Z = S/σ_S`, the fitted yield `S`, and `σ_S`. Validated: injected S=300
  → fitted **311 ± 26 (Z=12)**; background-only → **Z=1.2** (no fake peak). For a
  publication number, swap in your collaboration's fitter here.

### The data contract

`load_omega()` / `assemble()` return a dict consumed by everything downstream:

| key | meaning |
|---|---|
| `mass` | M(ΛK⁻), GeV, filtered to `FIT_RANGE` |
| `x_sharp`, `x_robust` | standardized feature tensors `[N, d]` |
| `label` | training target (truth in `mc`; in-peak flag in `cwola`) |
| `train_w` | per-event training weight (1 in `mc`; peak∪sideband in `cwola`) |
| `is_sb`, `in_peak` | sideband / peak-window boolean masks |
| `truth` | MC truth if present, else `−1` |

Build this dict any way you like (a different reader, an in-memory generator) and
the whole pipeline runs.

---

## The Ω⁻ toy Monte Carlo (`omega_toymc.py`)

A realistic-shaped validation sample — an Ω⁻ peak at 1.6725 GeV on a smooth
falling combinatorial background — with the same sculpting structure the method is
meant to handle. Uses no real data; the column layout matches `omega_run.COLUMNS`
so both exercise the same code path.

```bash
python -m NN.omega_toymc
```

`make_toy_omega(n_sig, n_bkg, seed, resolution=0.008, sculpt_strength=4.0, mode="cwola")`
— `sculpt_strength` is the mass-correlation slope of the sharp features (0 = no
sculpting; larger = stronger sculpting tension); `mode` switches label schemes.

### Validation results

Three things checked out cleanly:

1. **The k-fold wrapper works** — `Z_naive > Z_oof`, i.e. it removes the
   overfitting a single split hides. Trust `Z_oof`.
2. **The data-driven risk estimator recovers the sculpting region without truth**
   — `g_safe ≈ 0.08`, `g_risky ≈ 0.91` every seed.
3. **The peak fit is accurate** — injected 300 → 311 ± 26 (Z=12); null → Z=1.2.

And the headline, **honest** finding — whether the gate improves out-of-fold
significance is **regime-dependent**, and the two label modes differ:

| setting | penalty-only Z_oof | gate Z_oof | ratio |
|---|---|---|---|
| MC-truth, 4 seeds | 6.09 ± 1.29 | 5.76 ± 0.85 | **0.95×** |
| MC-truth, `sculpt_strength` 1 → 5 → 12 | — | — | 0.65× → 0.89× → 0.95× (never > 1) |
| **CWoLa (data-only)**, fake ntuple | 11 | **17** | **~1.5×** |

**Interpretation.** With MC truth the W1 penalty already controls sculpting, so
the gate mostly adds variance and sits at parity — even as sculpting is cranked
up. The gate's edge appears in the **CWoLa / data-only** regime, where the
peak-vs-sideband classifier can cheat by reading the mass through mass-correlated
features, and the gate suppresses exactly that pathway. This is the physically
sensible place for it.

---

## Findings and caveats

- **The regulator concept is sound but needs supervision.** As a co-trained layer
  it is inert; supervise the gate to a (data-driven) sculpting-risk target and it
  works.
- **Use the per-event conditioning, not the backoff.** The cost-aware backoff
  (`k`, EWMA) points the wrong way for decorrelation; run the gate with `k=None`.
- **The benefit is conditional on your data actually sculpting.** If the plain
  penalty already gives low sculpt at good Z, the gate won't add much — and the
  tool correctly reports that.
- **Statistics dominate on rare signals.** Ω⁻ yields are small; a single Z is not
  trustworthy. Always use `--kfold` (out-of-fold) and `--repeats` (spread).
- **Keep the mass out of the features.** In CWoLa especially, mass leakage lets
  the classifier fake a peak and defeats the whole test.
- **The `fit` significance is a linear least-squares peak fit** with Poisson bin
  weights; for a publication result substitute your collaboration's fitter.

---

## Reproducing every result

From the repository root (`CASAgentGovernor/`):

```bash
# The arc (each prints its table and verdict)
python -m NN.benchmark              # homogeneous: regulator ~ plain penalty
python -m NN.benchmark_hetero       # heterogeneous: gate inert (g -> 0)
python -m NN.benchmark_supervised   # oracle gate: ~1.55x
python -m NN.benchmark_datadriven   # data-driven target: ~1.75x, 99.7% region recovery

# The Omega toy MC: CV + peak fit + multi-seed spread
python -m NN.omega_toymc

# Real data (after filling in COLUMNS / LABEL_MODE)
python -m NN.omega_run --table your_ntuple.root --kfold 5 --repeats 5 --sig fit
```

Every module also imports cleanly for use as a library
(`from NN.regulator import GovernedScoreHead`, etc.).
