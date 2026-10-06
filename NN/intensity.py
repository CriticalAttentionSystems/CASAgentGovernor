"""The five-state regulatory ladder, shared by the governor and its NN port.

These are the same intensities the Java governor applies
(``InducedCostStochasticToolAgent``): a monotone ladder from an un-tightened
NORMAL state to a nearly-fully-suppressed RELEASED state. ``g`` is the
regulation intensity; the complementary quantity ``1 - g`` is how much of the
un-regulated action survives (the Java ``retryMultiplier`` arm).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RegulatoryState(Enum):
    """Ordered so that iteration yields the ladder low-to-high intensity."""

    NORMAL = "NORMAL"
    DISTURBED = "DISTURBED"
    RECOVERING = "RECOVERING"
    STUCK = "STUCK"
    RELEASED = "RELEASED"


#: Regulation intensity g(state). Mirrors the Java ladder exactly.
INTENSITY: dict[RegulatoryState, float] = {
    RegulatoryState.NORMAL: 0.00,
    RegulatoryState.DISTURBED: 0.75,
    RegulatoryState.RECOVERING: 0.85,
    RegulatoryState.STUCK: 0.95,
    RegulatoryState.RELEASED: 0.99,
}


@dataclass(frozen=True)
class Clamp:
    """The two arms an intensity drives.

    ``penalty_scale`` (= g) scales the decorrelation penalty; ``sharp_scale``
    (= 1 - g) is the fraction of the mass-correlated ("sharp") pathway that
    survives. They are complementary by construction, which is what keeps the
    two arms of the regulator moving together.
    """

    penalty_scale: float
    sharp_scale: float


def clamp_from_intensity(g: float) -> Clamp:
    """Map a regulation intensity to its two complementary arms."""
    g = min(1.0, max(0.0, g))
    return Clamp(penalty_scale=g, sharp_scale=1.0 - g)
