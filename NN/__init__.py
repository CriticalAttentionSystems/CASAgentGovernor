"""Governor-as-a-layer: a differentiable port of the CAS regulator for NNs."""
from __future__ import annotations

from .intensity import INTENSITY, Clamp, RegulatoryState, clamp_from_intensity

__all__ = ["INTENSITY", "Clamp", "RegulatoryState", "clamp_from_intensity"]
