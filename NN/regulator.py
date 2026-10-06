"""The governor as a LAYER, not a wrapper.

In the paper the governor sits outside the agent and its intensity g is a
hyperparameter schedule discovered by a state machine. Built into a network, the
same object becomes an *activation*: the five-state ladder is a softmax over five
regimes, and g is its expectation under that softmax -- an input-conditioned
scalar the network computes about itself, and backpropagates through.

WHY THIS IS NOT JUST "ADAPTIVE LAMBDA"

Adaptive-weight schemes (GradNorm, adversarial balancing, uncertainty weighting)
adapt a penalty weight to balance *losses*. The regulatory cell adapts it to a
measured **side-effect**: how much did my own last tightening move the
background template? That is a feedback signal on the intervention's collateral
damage, not on the objective. Two consequences follow that fixed-lambda
decorrelation cannot reproduce:

  1. **One-sidedness.** Eq. (2) gates cost on positive tightening only:
     loosening is free. That asymmetry is an inductive bias with real content and
     it is what makes the loop stable rather than oscillatory.
  2. **Phase-space conditioning.** g is computed *per event* from the event's
     kinematics, so one network leans on mass-correlated information where
     sculpting is cheap and stands down where it is expensive. A scalar lambda is
     one number for the whole dataset and cannot do this. This is the concrete
     fix for the failure mode where an external governor's only move is to
     abstain globally.

TWO AXES, DELIBERATELY SEPARATED

The regulator varies along two independent axes and the code keeps them apart:

  * **Per event (kinematics).** The base intensity ``g_soft(x)`` is a function of
    each event's features, so it is a vector over the batch. This is the
    phase-space conditioning above.
  * **Per step (history).** The cost-aware backoff ``p_apply`` is a single scalar
    that evolves over training steps via the EWMA induced-cost rate ``r``. It is
    global on purpose: "how much has my recent tightening been sculpting the
    background" is a distributional property of the whole batch, not of one
    event. The induced cost eta is likewise a batch-level quantity (a distance
    between mass templates), so the feedback loop that consumes it is global.

    g_applied(x) = g_soft(x) * p_apply

  A recurrent (GRU) cell carries the per-step regime; a feed-forward head mixes
  that regime with per-event kinematics to produce ``g_soft(x)``.

ARCHITECTURE

Two pathways into the score:

    s(x) = s_robust(x_blind)  +  (1 - g(x)) * s_sharp(x_all)

``s_sharp`` may use mass-correlated features -- it is powerful and it sculpts.
``s_robust`` sees only mass-blind features. g(x) is the per-event regulatory
intensity, so tightening (g -> 1) suppresses the sculpting pathway for that
event. The same g scales the decorrelation penalty (mean(g) * lambda_max),
exactly as the clamp's two arms in ``intensity.clamp_from_intensity``.

Backoff scales g itself: ``g_applied = g_soft * p_apply``, with
``p_apply = 1 - min(1, k*r)`` and r the EWMA induced-cost rate. When the cell's
own tightening has been sculpting, r rises, p_apply falls, and BOTH arms relax
together -- preserving the property that makes backoff work: a backed-off step is
genuinely hands-off, not merely softened.

A CHOICE THAT MATTERS

``hard=False`` uses the expectation of the Bernoulli backoff, which is smooth and
easy to train. But the transition band in the paper is a *stochastic* phenomenon
-- seeds splitting between a regulated and a suppressed basin. A deterministic
soft gate has no basins to split between, so soft mode may well destroy the band
while keeping the suppression. ``hard=True`` keeps the Bernoulli via a
straight-through estimator. Which one to use is an empirical question this
formalism actually predicts, and worth its own ablation.

TRUNCATED BPTT

The per-step hidden state is detached each step (``self.h = h.detach()``), so the
recurrent regime is trained one step at a time. That matches the online loop
below (forward -> loss -> observe, once per batch). If you want the regime to
learn multi-step policy, keep the graph over a window and detach only at the
window boundary.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .intensity import INTENSITY, RegulatoryState

_LADDER = torch.tensor([INTENSITY[s] for s in RegulatoryState], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# differentiable induced cost
# --------------------------------------------------------------------------- #
def soft_template(
    mass: torch.Tensor,
    score: torch.Tensor,
    bkg_weight: torch.Tensor,
    centres: torch.Tensor,
    thr: torch.Tensor,
    bandwidth: float,
    tau: float = 0.05,
) -> torch.Tensor:
    """Differentiable background-only mass template of the SURVIVING events.

    The veto is softened to sigmoid((s - thr)/tau) so the template is a function
    of the score with a gradient. Binning is a Gaussian kernel on ``centres``, so
    there is no non-differentiable histogram.
    """
    keep = torch.sigmoid((score - thr) / tau) * bkg_weight
    k = torch.exp(-0.5 * ((mass[:, None] - centres[None, :]) / bandwidth) ** 2)
    t = (k * keep[:, None]).sum(0)
    return t / t.sum().clamp(min=1e-8)


def wasserstein1_grid(a: torch.Tensor, b: torch.Tensor, bin_width: float) -> torch.Tensor:
    """Exact W1 between two normalised densities on a shared 1-D grid.

    On a grid, W1 = integral |CDF_a - CDF_b|, which is differentiable in closed
    form -- no sorting, no optimal-transport solver.
    """
    return (a.cumsum(0) - b.cumsum(0)).abs().sum() * bin_width


# --------------------------------------------------------------------------- #
# the cell
# --------------------------------------------------------------------------- #
class RegulatoryCell(nn.Module):
    """Recurrent five-state regulator, differentiable, with cost-aware backoff.

    Produces a *per-event* intensity g_soft(x) (phase-space conditioning) times a
    *per-step* scalar backoff p_apply (the induced-cost feedback loop).

    Parameters
    ----------
    d_global
        Width of the per-step context vector (a pooled batch/fit summary). The
        GRU also consumes the current r and previous mean-g, so its input is
        ``d_global + 2``.
    d_event
        Width of the per-event feature vector that conditions g_soft(x). Pass 0
        to recover a single global g (the pre-refactor behaviour).
    k
        Backoff sensitivity. ``None`` == cost-blind baseline (ablation arm).
    lam
        EWMA rate, Eq. (4).
    eta_tol
        Induced-failure threshold, in units of the channel's mass sigma.
        MUST be calibrated on an alpha=0 / mass-blind null -- shot noise alone
        gives nonzero eta, and an uncalibrated tolerance makes the cell back off
        for free.
    hard
        Straight-through Bernoulli backoff (preserves stochasticity) vs the
        smooth expectation.
    """

    def __init__(
        self,
        d_global: int,
        d_event: int = 0,
        hidden: int = 64,
        k: float | None = 3.0,
        lam: float = 0.3,
        eta_tol: float = 0.05,
        hard: bool = False,
    ):
        super().__init__()
        self.k, self.lam, self.eta_tol, self.hard = k, lam, eta_tol, hard
        self.d_event = d_event
        self.cell = nn.GRUCell(d_global + 2, hidden)      # +2: current r, previous mean g
        self.to_states = nn.Linear(hidden + d_event, len(RegulatoryState))
        self.to_eta = nn.Linear(hidden, 1)                # predicts its own induced cost
        self.register_buffer("ladder", _LADDER.clone())
        self.register_buffer("h", torch.zeros(1, hidden))
        self.register_buffer("r", torch.zeros(()))        # EWMA induced-cost rate
        self.register_buffer("g_prev", torch.zeros(()))   # previous mean applied g

    def reset(self) -> None:
        self.h.zero_(); self.r.zero_(); self.g_prev.zero_()

    @property
    def cost_blind(self) -> bool:
        return self.k is None

    def p_apply(self) -> torch.Tensor:
        """Eq. (5). Scalar per step."""
        if self.cost_blind:
            return torch.ones((), device=self.r.device)
        return 1.0 - torch.clamp(self.k * self.r, max=1.0)

    def forward(
        self,
        global_context: torch.Tensor,
        event_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One regulatory step.

        ``global_context`` : shape ``[d_global]`` or ``[1, d_global]``.
        ``event_feats``     : shape ``[N, d_event]`` -- per-event kinematics. If
                              ``d_event == 0`` this may be ``None`` and g is a
                              single scalar broadcast over the batch.
        """
        x = torch.cat([global_context.view(1, -1),
                       self.r.view(1, 1),
                       self.g_prev.view(1, 1)], dim=-1)
        h = self.cell(x, self.h)                          # [1, hidden] global regime
        self.h = h.detach()                               # truncated BPTT (see module docstring)

        if self.d_event > 0:
            if event_feats is None:
                raise ValueError("event_feats required when d_event > 0")
            n = event_feats.shape[0]
            z = torch.cat([h.expand(n, -1), event_feats], dim=-1)   # [N, hidden + d_event]
        else:
            z = h                                                    # [1, hidden]

        probs = self.to_states(z).softmax(-1)             # [N, 5] or [1, 5]
        g_soft = (probs * self.ladder).sum(-1)            # [N] or [1]  per-event intensity
        eta_hat = F.softplus(self.to_eta(h)).squeeze()    # scalar: predicted batch sculpting

        p = self.p_apply()                                # scalar backoff
        if self.cost_blind:
            gate = torch.ones_like(p)
        elif self.hard:
            # straight-through Bernoulli: keeps the stochastic basin structure
            # that the transition band depends on, while staying differentiable.
            draw = torch.bernoulli(p.detach())
            gate = draw + p - p.detach()
        else:
            gate = p

        g_applied = g_soft * gate                         # [N]  per-event, scaled by scalar backoff
        return {"g": g_applied, "g_soft": g_soft, "state_probs": probs,
                "eta_hat": eta_hat, "p_apply": p, "gate": gate}

    def observe(self, eta_obs: torch.Tensor, g_applied: torch.Tensor) -> torch.Tensor:
        """Close the loop. Eq. (2) gate + Eq. (4) EWMA. Call once per step, after loss.

        ``eta_obs`` is the batch-level induced cost (a scalar). ``g_applied`` is
        the per-event intensity from the forward pass; the loop tracks its batch
        mean, because the induced cost it is compared against is itself a
        batch-level quantity. Returns the one-sided induced cost for the step:
        zero unless the cell actually tightened on average. Loosening is free --
        that asymmetry is the stabiliser, not an implementation detail.
        """
        g_mean = g_applied.detach().mean()
        dg_plus = torch.clamp(g_mean - self.g_prev, min=0.0)
        cost = eta_obs.detach() * (dg_plus > 0).float()
        if not self.cost_blind:
            fired = (cost > self.eta_tol).float()
            self.r.mul_(1 - self.lam).add_(self.lam * fired)
        self.g_prev.copy_(g_mean)
        return cost


# --------------------------------------------------------------------------- #
# the head
# --------------------------------------------------------------------------- #
class GovernedScoreHead(nn.Module):
    """Two-pathway score with an internal regulator gating the sculpting pathway.

    ``robust_dim`` features are mass-blind and always available. ``sharp_dim``
    features may be mass-correlated: powerful, and the thing that sculpts. The
    regulator conditions its per-event intensity on ``event_dim`` features -- by
    default the mass-blind robust features, so the gate decides *where* to trust
    the sharp pathway without keying directly on the protected mass.
    """

    def __init__(self, robust_dim: int, sharp_dim: int, d_global: int,
                 event_dim: int = 0, hidden: int = 64, **cell_kwargs):
        super().__init__()
        self.robust = nn.Sequential(nn.Linear(robust_dim, hidden), nn.GELU(),
                                    nn.Linear(hidden, 1))
        self.sharp = nn.Sequential(nn.Linear(sharp_dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, 1))
        self.cell = RegulatoryCell(d_global, d_event=event_dim, hidden=hidden, **cell_kwargs)

    def forward(self, x_robust, x_sharp, global_context, event_feats=None):
        reg = self.cell(global_context, event_feats)
        s = self.robust(x_robust).squeeze(-1) + (1.0 - reg["g"]) * self.sharp(x_sharp).squeeze(-1)
        return s, reg


def governed_loss(
    score, label, mass, bkg_weight, reg, centres, ref_template,
    thr, bandwidth, bin_width, sigma_mass,
    lambda_max: float = 50.0, beta_eta: float = 1.0,
):
    """L = BCE + mean(g)*lambda_max*W1(template, ref)^2 + beta*|eta_hat - eta_obs|.

    ``ref_template`` is an EMA of the background template at earlier steps: the
    comparison is temporal, so the penalty charges the *change* the cell caused,
    which is what makes it induced cost rather than a static decorrelation term.
    (For absolute mass-decorrelation you would also anchor W1 to a fixed
    mass-blind template; this toy charges only the change, matching the paper's
    induced-cost definition.)

    ``reg["g"]`` is per-event; the batch penalty is scaled by its mean, since the
    sculpting distance eta is itself a batch-level scalar.
    """
    bce = F.binary_cross_entropy_with_logits(score, label)
    tmpl = soft_template(mass, score, bkg_weight, centres, thr, bandwidth)
    eta = wasserstein1_grid(tmpl, ref_template, bin_width) / sigma_mass

    loss = bce + reg["g"].mean() * lambda_max * eta ** 2
    loss = loss + beta_eta * (reg["eta_hat"] - eta.detach()).abs()
    return loss, eta, tmpl
