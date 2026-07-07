"""Scale-invariant crop pipeline + bank scale-gating (the scale-blind-bank fix).

The bank stored a CLIP fingerprint computed from a fixed 48-*pixel* crop, whose
physical field of view varied per scan, so the same molecule embedded
differently at different resolutions and cross-scan matching was silently wrong.
These tests pin the fix: physical-FOV crops are scale-invariant, and the bank
read path excludes entries computed by a different pipeline/scale.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from probeflow.analysis import feature_bank
from probeflow.analysis.features import (
    DEFAULT_CROP_FOV_NM,
    DEFAULT_OUT_PX,
    EMBED_VERSION,
    _crop_particle,
    _crop_particle_physical,
    classify_particles,
    segment_particles,
)


def _disk_scene(shape, cx, cy, r, height=5.0):
    Ny, Nx = shape
    Y, X = np.mgrid[:Ny, :Nx]
    a = np.zeros(shape, dtype=np.float64)
    a[(X - cx) ** 2 + (Y - cy) ** 2 <= r * r] = height
    return a


def _one_particle(arr, px_m):
    parts = segment_particles(arr, pixel_size_m=px_m, min_area_nm2=0.1,
                              size_sigma_clip=None)
    assert parts, "expected one segmented particle"
    return max(parts, key=lambda p: p.area_nm2)


def _cos(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ── The core proof: same molecule, two resolutions → same crop ────────────────

def test_physical_crop_is_scale_invariant():
    """A 4 nm disk sampled at 0.1 and 0.4 nm/px yields near-identical physical
    crops; the legacy fixed-pixel crop does not."""
    # High-res: 4 nm disk = 40 px radius → 400 px image; pixel = 0.1 nm.
    hi = _disk_scene((400, 400), 200, 200, r=40)
    p_hi = _one_particle(hi, 0.1e-9)
    # Low-res: same 4 nm disk = 10 px radius → 100 px image; pixel = 0.4 nm.
    lo = _disk_scene((100, 100), 50, 50, r=10)
    p_lo = _one_particle(lo, 0.4e-9)

    kw = dict(fov_nm=DEFAULT_CROP_FOV_NM, out_px=DEFAULT_OUT_PX, apply_mask=False)
    c_hi = _crop_particle_physical(hi, p_hi, pixel_size_x_m=0.1e-9,
                                   pixel_size_y_m=0.1e-9, **kw)
    c_lo = _crop_particle_physical(lo, p_lo, pixel_size_x_m=0.4e-9,
                                   pixel_size_y_m=0.4e-9, **kw)
    assert c_hi.shape == c_lo.shape == (DEFAULT_OUT_PX, DEFAULT_OUT_PX)
    # Physical crops of the same molecule at 4× different pixel size are nearly
    # identical (residual difference is edge aliasing of a hard-edged disk).
    phys_cos = _cos(c_hi, c_lo)
    assert phys_cos > 0.95, f"physical crops must match across resolutions ({phys_cos:.3f})"

    # Contrast: the legacy fixed-48px crop sees very different FOVs → diverges
    # substantially, proving the physical path is what buys scale-invariance.
    l_hi = _crop_particle(hi, p_hi, 48)
    l_lo = _crop_particle(lo, p_lo, 48)
    assert _cos(l_hi, l_lo) < phys_cos - 0.05


def test_crop_too_coarse_raises():
    """A FOV that spans fewer than MIN_CROP_PX pixels must refuse, not upsample
    noise into a fake match."""
    arr = _disk_scene((64, 64), 32, 32, r=6)
    p = _one_particle(arr, 2.0e-9)  # 2 nm/px → 15 nm FOV ≈ 7.5 px < MIN_CROP_PX
    with pytest.raises(ValueError, match="too coarse"):
        _crop_particle_physical(arr, p, fov_nm=DEFAULT_CROP_FOV_NM,
                                pixel_size_x_m=2.0e-9, pixel_size_y_m=2.0e-9)


# ── Bank read path excludes incompatible entries ──────────────────────────────

def _entry(name, *, embed_version, fov_nm=DEFAULT_CROP_FOV_NM,
           out_px=DEFAULT_OUT_PX, dim=512, stale=False):
    e = feature_bank.make_entry(
        list(np.ones(dim)), name,
        source_path=f"/s/{name}.dat", particle_index=0,
        embed_version=embed_version, fov_nm=fov_nm, out_px=out_px,
        pixel_size_nm=0.3, area_nm2=10.0,
    )
    if stale:
        e["stale"] = True
    return e


def test_select_bank_samples_keeps_only_compatible():
    bank = {"entries": [
        _entry("good", embed_version=EMBED_VERSION),
        _entry("legacy", embed_version=None),                    # schema-1
        _entry("other_pipeline", embed_version="old-vX"),
        _entry("wrong_fov", embed_version=EMBED_VERSION, fov_nm=40.0),
        _entry("stale", embed_version=EMBED_VERSION, stale=True),
    ]}
    sel = feature_bank.select_bank_samples(
        bank, embed_version=EMBED_VERSION,
        fov_nm=DEFAULT_CROP_FOV_NM, out_px=DEFAULT_OUT_PX,
    )
    assert sel["names"] == ["good"]
    assert sel["kept"] == 1
    assert sel["skipped"] == 4
    assert sel["reasons"] == {
        "legacy": 1, "pipeline": 1, "fov": 1, "stale": 1, "malformed": 0}


def test_classify_requires_pixel_sizes_for_bank():
    arr = _disk_scene((64, 64), 32, 32, r=6)
    parts = segment_particles(arr, pixel_size_m=1e-9, min_area_nm2=0.1,
                              size_sigma_clip=None)
    with pytest.raises(ValueError, match="physical-FOV crop path"):
        classify_particles(
            arr, parts, samples=[], encoder="clip",
            bank_samples=[("x", list(np.ones(512)))],
        )


def test_area_gate_excludes_size_mismatched_bank_sample():
    """The area gate blanks a bank candidate whose physical size disagrees.

    A particle re-classified against its own CLIP embedding matches itself; a
    100× area mismatch with area_gate_ratio=2 must instead force "other".
    """
    pytest.importorskip("torch")
    pytest.importorskip("clip")
    from probeflow.analysis.features import embed_particles_clip
    px = 0.3e-9  # fine enough that a 15 nm FOV spans ≥ MIN_CROP_PX pixels
    arr = _disk_scene((96, 96), 48, 48, r=10)
    parts = segment_particles(arr, pixel_size_m=px, min_area_nm2=0.1,
                              size_sigma_clip=None)
    p = max(parts, key=lambda q: q.area_nm2)
    emb = embed_particles_clip(arr, [p], pixel_size_x_m=px, pixel_size_y_m=px)
    common = dict(
        arr=arr, particles=[p], samples=[], encoder="clip",
        pixel_size_x_m=px, pixel_size_y_m=px,
        threshold_method="manual", manual_threshold=0.5,
        bank_samples=[("disk", emb[0].tolist())],
        bank_areas=[p.area_nm2 * 100.0],
    )
    gated = classify_particles(**common, area_gate_ratio=2.0)
    assert gated[0].class_name == "other"   # size gate rejected it
    ungated = classify_particles(**common, area_gate_ratio=None)
    assert ungated[0].class_name == "disk"  # same vector matches itself
