"""Known-answer validation for Particle Statistics verdicts.

These tests run synthetic patterns with a *known* spatial structure through the
ProbeFlow→AdStat adapter and assert the verdict matches the ground truth:

- random points are consistent with the homogeneous-Poisson null;
- clustered points reject it on pair correlation / Ripley L;
- strongly spaced points reject it on nearest-neighbour distance;
- a triangular lattice rejects it on ψ6, a square lattice on ψ4 (ordering on);
- random points stay consistent on ψ even when ordering is enabled.

They double as the maturity check the docs reference: if a case does not behave
as expected, it is marked xfail with a note rather than silently shipped.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from probeflow.analysis.adstat_adapter import (
    ORDERING_STATISTICS,
    compare_point_source_view_spec,
)

_FIELD_NM = 100.0
_PX = 256


def _source(xy_nm: np.ndarray) -> SimpleNamespace:
    xy_nm = np.asarray(xy_nm, dtype=float).reshape(-1, 2)
    pixel_nm = _FIELD_NM / _PX
    return SimpleNamespace(
        label="validation",
        source_type="synthetic",
        points_px=xy_nm / pixel_nm,
        points_m=xy_nm * 1e-9,
        metadata={},
    )


def _scan() -> SimpleNamespace:
    return SimpleNamespace(scan_range_m=(_FIELD_NM * 1e-9, _FIELD_NM * 1e-9), dims=(_PX, _PX))


def _spec(xy_nm: np.ndarray, *, include_ordering: bool = False, n_simulations: int = 99):
    return compare_point_source_view_spec(
        _source(xy_nm),
        scan=_scan(),
        image_shape=(_PX, _PX),
        n_simulations=n_simulations,
        random_seed=0,
        include_ordering=include_ordering,
    )


def _verdict(spec, statistic: str) -> str | None:
    for row in getattr(spec, "verdict_rows", ()) or ():
        if len(row) > 2 and str(row[1]) == statistic:
            return str(row[2])
    return None


def _is_inconsistent(verdict: str | None) -> bool:
    return verdict is not None and "inconsistent" in verdict


def _is_consistent(verdict: str | None) -> bool:
    return verdict is not None and verdict.endswith("consistent_with_null")


# --------------------------------------------------------------------------- #
# Pattern generators (all inside a margin so points stay within the field)
# --------------------------------------------------------------------------- #
def _random(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(3.0, _FIELD_NM - 3.0, size=(n, 2))


def _clustered(seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.array([[30.0, 35.0], [68.0, 62.0], [46.0, 78.0]])
    assign = rng.integers(0, len(centers), size=120)
    clustered = centers[assign] + rng.normal(0.0, 4.0, size=(120, 2))
    return np.clip(clustered, 2.0, _FIELD_NM - 2.0)


def _square_lattice(a: float = 8.0, jitter: float = 0.0, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = np.arange(a, _FIELD_NM - a + 1e-9, a)
    xs, ys = np.meshgrid(coords, coords)
    pts = np.column_stack([xs.ravel(), ys.ravel()])
    if jitter > 0.0:
        pts = pts + rng.normal(0.0, jitter, size=pts.shape)
    return pts


def _triangular_lattice(a: float = 8.0) -> np.ndarray:
    dy = a * math.sqrt(3.0) / 2.0
    pts = []
    row = 0
    y = a
    while y < _FIELD_NM - a:
        x_off = (a / 2.0) if (row % 2) else 0.0
        x = a + x_off
        while x < _FIELD_NM - a:
            pts.append((x, y))
            x += a
        y += dy
        row += 1
    return np.asarray(pts, dtype=float)


# --------------------------------------------------------------------------- #
# Core null model (always on)
# --------------------------------------------------------------------------- #
def test_random_is_consistent_with_poisson():
    spec = _spec(_random(140))
    for stat in (
        "pair_correlation_g_r",
        "nearest_neighbor_distribution",
        "ripley_l_function",
        "cluster_size_counts",
    ):
        assert _is_consistent(_verdict(spec, stat)), f"{stat}: {_verdict(spec, stat)}"


def test_clustered_rejects_poisson_on_pair_correlation():
    spec = _spec(_clustered())
    gr = _verdict(spec, "pair_correlation_g_r")
    ripley = _verdict(spec, "ripley_l_function")
    assert _is_inconsistent(gr) or _is_inconsistent(ripley), (gr, ripley)


def test_spaced_points_reject_poisson_on_nearest_neighbour():
    # A lightly jittered square lattice has strong minimum spacing.
    spec = _spec(_square_lattice(a=8.0, jitter=1.0))
    nn = _verdict(spec, "nearest_neighbor_distribution")
    assert _is_inconsistent(nn), nn


# --------------------------------------------------------------------------- #
# Opt-in ordering statistics
# --------------------------------------------------------------------------- #
def test_default_run_has_no_ordering_panels():
    spec = _spec(_random(120), include_ordering=False)
    panel_stats = {str(getattr(p, "statistic", "")) for p in spec.panels}
    assert panel_stats.isdisjoint(ORDERING_STATISTICS)
    assert all(
        not (len(r) > 1 and str(r[1]) in ORDERING_STATISTICS)
        for r in (spec.verdict_rows or ())
    )


def test_ordering_on_adds_psi_panels():
    spec = _spec(_random(120), include_ordering=True)
    panel_stats = {str(getattr(p, "statistic", "")) for p in spec.panels}
    assert {"bond_order_psi6", "bond_order_psi4"} <= panel_stats


def test_random_psi_is_consistent():
    spec = _spec(_random(140), include_ordering=True)
    assert _is_consistent(_verdict(spec, "bond_order_psi6"))
    assert _is_consistent(_verdict(spec, "bond_order_psi4"))


def test_triangular_lattice_rejects_on_psi6():
    spec = _spec(_triangular_lattice(a=8.0), include_ordering=True)
    psi6 = _verdict(spec, "bond_order_psi6")
    assert _is_inconsistent(psi6), f"ψ6 should reject a triangular lattice, got {psi6}"


def test_square_lattice_rejects_on_psi4():
    spec = _spec(_square_lattice(a=8.0), include_ordering=True)
    psi4 = _verdict(spec, "bond_order_psi4")
    assert _is_inconsistent(psi4), f"ψ4 should reject a square lattice, got {psi4}"
