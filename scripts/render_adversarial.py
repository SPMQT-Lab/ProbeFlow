"""
render_adversarial.py — generate side-by-side before/after PNG comparison images
for each adversarial (fixture, operation) pair.

Output: scripts/adversarial_renders/<fixture>_<op>.png

Usage:
    python scripts/render_adversarial.py

Each image shows:
    Left  — input array
    Right — output array after the operation
    Title — fixture name | operation | key metric

Colourmap: afmhot.  vmin/vmax set to the 2nd/98th percentile of finite input
pixels (robust to outliers).

The fwd_bwd fixture is handled separately: shows fwd, bwd, and blend.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure probeflow and the tests/ directory are importable when run from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt

from probeflow.processing.image import (
    align_rows,
    subtract_background,
    stm_line_background,
    fourier_filter,
    fft_soft_border,
    tv_denoise,
    gaussian_smooth,
    gaussian_high_pass,
    remove_bad_lines,
    detect_grains,
    facet_level,
    blend_forward_backward,
    rotate_arbitrary,
)

from adversarial_fixtures import (
    flat_with_outlier,
    tilted_plane_with_step,
    lattice_with_scanline_glitch,
    nan_horizontal_stripe,
    real_islands_mimic_artefact,
    anisotropic_pixels,
    non_square_image,
    tiny_3x3,
    tiny_2x2,
    constant_nonzero,
    negative_heights,
    fwd_bwd_asymmetric,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_RENDER_DIR = _REPO_ROOT / "scripts" / "adversarial_renders"
_RENDER_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_percentiles(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    """Return (vmin, vmax) from finite pixels in arr."""
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return (0.0, 1.0)
    vmin = float(np.percentile(vals, lo))
    vmax = float(np.percentile(vals, hi))
    if vmin == vmax:
        vmax = vmin + 1e-30
    return vmin, vmax


def _metric(arr_in: np.ndarray, arr_out: np.ndarray) -> str:
    """Compute a concise key metric string for the panel title."""
    in_mean = float(np.nanmean(arr_in))
    out_mean = float(np.nanmean(arr_out))
    nan_in = int(np.sum(~np.isfinite(arr_in)))
    nan_out = int(np.sum(~np.isfinite(arr_out)))
    return (
        f"mean: {in_mean:.2e} → {out_mean:.2e}  "
        f"NaN: {nan_in} → {nan_out}"
    )


def _save_pair(
    fixture_name: str,
    op_name: str,
    arr_in: np.ndarray,
    arr_out: np.ndarray,
) -> Path:
    """Save a two-panel (before/after) PNG and return its path."""
    vmin, vmax = _finite_percentiles(arr_in)
    metric = _metric(arr_in, arr_out)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"{fixture_name}  |  {op_name}\n{metric}",
        fontsize=9,
        y=1.01,
    )

    for ax, data, label in zip(axes, [arr_in, arr_out], ["Input", "Output"]):
        im = ax.imshow(
            data,
            cmap="afmhot",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="upper",
        )
        ax.set_title(label, fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = _RENDER_DIR / f"{fixture_name}_{op_name}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_triple(
    fixture_name: str,
    op_name: str,
    arrays: list[tuple[str, np.ndarray]],
    ref_arr: np.ndarray,
) -> Path:
    """Save a multi-panel PNG (for fwd/bwd/blend). ref_arr sets vmin/vmax."""
    vmin, vmax = _finite_percentiles(ref_arr)

    ncols = len(arrays)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.suptitle(f"{fixture_name}  |  {op_name}", fontsize=9, y=1.01)

    for ax, (label, data) in zip(axes, arrays):
        im = ax.imshow(
            data,
            cmap="afmhot",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="upper",
        )
        ax.set_title(label, fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = _RENDER_DIR / f"{fixture_name}_{op_name}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _run_op_safe(name: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs); return (result, error_str)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Render plan: list of (fixture_factory, fixture_name, op_name, op_fn, op_kwargs)
# ---------------------------------------------------------------------------

def _build_plan():
    """Return list of (fixture_name, op_name, arr_in, arr_out_fn) tuples."""
    plan = []

    def add(fx_name, op_name, arr, op_fn, **kw):
        plan.append((fx_name, op_name, arr, op_fn, kw))

    # ── flat_with_outlier ────────────────────────────────────────────────────
    fx = flat_with_outlier()
    arr = fx["arr"]
    add("flat_with_outlier", "align_rows_median", arr, align_rows, method="median")
    add("flat_with_outlier", "remove_bad_lines", arr, remove_bad_lines,
        method="step", threshold_mad=5.0)

    # ── tilted_plane_with_step ───────────────────────────────────────────────
    fx = tilted_plane_with_step()
    arr = fx["arr"]
    add("tilted_plane_with_step", "subtract_bg_order1_step_tol", arr,
        subtract_background, order=1, step_tolerance=True)
    add("tilted_plane_with_step", "subtract_bg_order4_no_step_tol", arr,
        subtract_background, order=4, step_tolerance=False)
    add("tilted_plane_with_step", "stm_line_background", arr, stm_line_background)
    add("tilted_plane_with_step", "rotate_arbitrary_45", arr, rotate_arbitrary, angle_degrees=45.0)

    # ── lattice_with_scanline_glitch ─────────────────────────────────────────
    fx = lattice_with_scanline_glitch()
    arr = fx["arr"]
    add("lattice_with_scanline_glitch", "remove_bad_lines", arr,
        remove_bad_lines, threshold_mad=3.0)

    # ── nan_horizontal_stripe ────────────────────────────────────────────────
    fx = nan_horizontal_stripe()
    arr = fx["arr"]
    add("nan_horizontal_stripe", "align_rows", arr, align_rows)
    add("nan_horizontal_stripe", "subtract_background", arr,
        subtract_background, order=1)
    add("nan_horizontal_stripe", "stm_line_background", arr, stm_line_background)
    add("nan_horizontal_stripe", "fourier_filter_hp", arr,
        fourier_filter, mode="high_pass", cutoff=0.2)
    add("nan_horizontal_stripe", "fft_soft_border_hp", arr,
        fft_soft_border, mode="high_pass", cutoff=0.2)
    add("nan_horizontal_stripe", "tv_denoise", arr, tv_denoise)
    add("nan_horizontal_stripe", "gaussian_smooth", arr, gaussian_smooth)
    add("nan_horizontal_stripe", "gaussian_high_pass", arr, gaussian_high_pass)

    # ── real_islands_mimic_artefact ──────────────────────────────────────────
    fx = real_islands_mimic_artefact()
    arr = fx["arr"]
    add("real_islands_mimic_artefact", "remove_bad_lines", arr,
        remove_bad_lines, threshold_mad=3.0)

    # ── anisotropic_pixels ───────────────────────────────────────────────────
    fx = anisotropic_pixels()
    arr = fx["arr"]
    add("anisotropic_pixels", "facet_level",
        arr, facet_level,
        pixel_size_x_m=fx["pixel_size_x_m"],
        pixel_size_y_m=fx["pixel_size_y_m"])

    # ── non_square_image ─────────────────────────────────────────────────────
    fx = non_square_image()
    arr = fx["arr"]
    add("non_square_image", "align_rows", arr, align_rows)
    add("non_square_image", "subtract_background", arr, subtract_background, order=1)
    add("non_square_image", "fourier_filter_lp", arr,
        fourier_filter, mode="low_pass", cutoff=0.3)

    # ── tiny_3x3 ─────────────────────────────────────────────────────────────
    fx = tiny_3x3()
    arr = fx["arr"]
    add("tiny_3x3", "align_rows", arr, align_rows)
    add("tiny_3x3", "subtract_background", arr, subtract_background, order=1)
    add("tiny_3x3", "gaussian_smooth", arr, gaussian_smooth)

    # ── tiny_2x2 ─────────────────────────────────────────────────────────────
    fx = tiny_2x2()
    arr = fx["arr"]
    add("tiny_2x2", "align_rows", arr, align_rows)
    add("tiny_2x2", "gaussian_smooth", arr, gaussian_smooth)

    # ── constant_nonzero ─────────────────────────────────────────────────────
    fx = constant_nonzero()
    arr = fx["arr"]
    add("constant_nonzero", "fourier_filter_hp", arr,
        fourier_filter, mode="high_pass", cutoff=0.2)
    add("constant_nonzero", "fft_soft_border_hp", arr,
        fft_soft_border, mode="high_pass", cutoff=0.2)

    # ── negative_heights ─────────────────────────────────────────────────────
    fx = negative_heights()
    arr = fx["arr"]
    add("negative_heights", "subtract_background", arr,
        subtract_background, order=1)
    add("negative_heights", "fourier_filter_hp", arr,
        fourier_filter, mode="high_pass", cutoff=0.2)

    return plan


def _render_fwd_bwd():
    """Special-case rendering for fwd_bwd_asymmetric."""
    fx = fwd_bwd_asymmetric()
    fwd = fx["fwd"]
    bwd = fx["bwd"]
    blend_05 = blend_forward_backward(fwd, bwd, weight=0.5)
    blend_10 = blend_forward_backward(fwd, bwd, weight=1.0)
    blend_00 = blend_forward_backward(fwd, bwd, weight=0.0)

    out_path = _save_triple(
        "fwd_bwd_asymmetric",
        "blend_w05",
        [("fwd", fwd), ("bwd_flipped", np.fliplr(bwd)), ("blend_w=0.5", blend_05)],
        ref_arr=fwd,
    )
    print(f"  saved: {out_path}")

    out_path = _save_triple(
        "fwd_bwd_asymmetric",
        "blend_weights",
        [("w=1.0 (fwd only)", blend_10),
         ("w=0.5 (equal)", blend_05),
         ("w=0.0 (bwd only)", blend_00)],
        ref_arr=fwd,
    )
    print(f"  saved: {out_path}")


def _render_detect_grains():
    """Special render for detect_grains: show label_map instead of float output."""
    fx = negative_heights()
    arr = fx["arr"]
    label_map, n_grains, stats = detect_grains(arr, threshold_pct=50, above=False)

    vmin, vmax = _finite_percentiles(arr)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"negative_heights  |  detect_grains(above=False)\n"
        f"found {n_grains} grain(s)",
        fontsize=9,
        y=1.01,
    )
    im0 = axes[0].imshow(arr, cmap="afmhot", vmin=vmin, vmax=vmax,
                         aspect="auto", origin="upper")
    axes[0].set_title("Input", fontsize=8)
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(label_map, cmap="tab20", aspect="auto", origin="upper")
    axes[1].set_title(f"Label map ({n_grains} grains)", fontsize=8)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = _RENDER_DIR / "negative_heights_detect_grains.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    plan = _build_plan()
    total = len(plan)
    errors = []

    print(f"Rendering {total} (fixture, op) pairs into {_RENDER_DIR}/\n")

    for i, (fx_name, op_name, arr_in, op_fn, kw) in enumerate(plan, 1):
        arr_out, err = _run_op_safe(op_name, op_fn, arr_in, **kw)
        if err:
            print(f"  [{i:3d}/{total}] SKIP  {fx_name} / {op_name}: {err}")
            errors.append((fx_name, op_name, err))
            continue
        path = _save_pair(fx_name, op_name, arr_in, arr_out)
        print(f"  [{i:3d}/{total}] OK    {path.name}")

    # Special-case renderers
    print("\nRendering fwd_bwd fixtures...")
    _render_fwd_bwd()

    print("\nRendering detect_grains on negative_heights...")
    _render_detect_grains()

    print(f"\nDone. {total - len(errors)} images saved to {_RENDER_DIR}")
    if errors:
        print(f"\n{len(errors)} operation(s) skipped due to errors:")
        for fx_name, op_name, err in errors:
            print(f"  {fx_name} / {op_name}: {err}")
    return len(errors)


if __name__ == "__main__":
    sys.exit(main())
