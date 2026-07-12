"""Total-variation denoising (Chambolle–Pock primal-dual, ported from AiSurf)."""

from __future__ import annotations

import math

import numpy as np


# Features-tab placement note:
# ``tv_denoise`` is kept here as a GUI-free numerical kernel because it is also
# useful from the CLI and tests. The GUI wraps it from ``probeflow.gui.tv`` as
# an optional workspace, not as a Browse thumbnail correction and not as a
# normal Viewer quick-processing control. The intent is to keep
# experimental/optional add-ons isolated from routine browsing, conversion,
# and basic image manipulation dependencies.

def _nabla_apply(x: np.ndarray, Ny: int, Nx: int, comp: str) -> np.ndarray:
    """Forward gradient (periodic-edge) for TV methods.

    Returns a flattened (2*N,) vector with the x- and y-gradient stacked.
    """
    img = x.reshape(Ny, Nx)
    if comp in ("both", "x") and Nx > 1:
        gx = np.zeros_like(img)
        gx[:, :-1] = img[:, 1:] - img[:, :-1]
    else:
        gx = np.zeros_like(img)
    if comp in ("both", "y") and Ny > 1:
        gy = np.zeros_like(img)
        gy[:-1, :] = img[1:, :] - img[:-1, :]
    else:
        gy = np.zeros_like(img)
    return np.concatenate([gx.ravel(), gy.ravel()])


def _nabla_T_apply(p: np.ndarray, Ny: int, Nx: int, comp: str) -> np.ndarray:
    """Adjoint of the forward gradient (negative divergence)."""
    N = Ny * Nx
    px = p[:N].reshape(Ny, Nx)
    py = p[N:].reshape(Ny, Nx)

    div = np.zeros((Ny, Nx))
    if comp in ("both", "x") and Nx > 1:
        # x-component of -div
        d = np.zeros_like(px)
        d[:, 0] = px[:, 0]
        d[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
        d[:, -1] = -px[:, -2]
        div -= d
    if comp in ("both", "y") and Ny > 1:
        d = np.zeros_like(py)
        d[0, :] = py[0, :]
        d[1:-1, :] = py[1:-1, :] - py[:-2, :]
        d[-1, :] = -py[-2, :]
        div -= d
    return div.ravel()


def tv_denoise(
    arr: np.ndarray,
    *,
    method: str = "huber_rof",
    lam: float = 0.05,
    alpha: float = 0.05,
    tau: float = 0.25,
    max_iter: int = 500,
    tol: float = 5e-6,
    nabla_comp: str = "both",
) -> np.ndarray:
    """Edge-preserving total-variation denoising.

    Two variants are available, ported from AiSurf:

    * ``"huber_rof"``  — Huber-ROF (smooth TV). Good general-purpose default;
      preserves terraces without staircasing.
    * ``"tv_l1"``      — Isotropic TV-L1. More aggressive on impulsive noise,
      but staircases on gently curved terraces.

    Parameters
    ----------
    arr
        2-D float input (any range — no prior normalisation required).
    method
        ``"huber_rof"`` | ``"tv_l1"``.
    lam
        Data-fidelity weight λ. Larger values stay closer to the input.
    alpha
        Huber smoothing parameter (ignored for ``tv_l1``). Typical 0.01–0.1.
    tau
        Primal step size. Default 0.25 satisfies the Chambolle-Pock
        convergence condition ``τ·σ·L² ≤ 1`` (L = √8 here).
    max_iter
        Hard cap on iterations.
    tol
        RMSE convergence threshold between primal iterates (checked every 50).
    nabla_comp
        ``"both"`` (isotropic, default), ``"x"`` (removes vertical scratches),
        or ``"y"`` (removes horizontal scratches).

    Returns
    -------
    ndarray
        The denoised image, same shape and dtype as ``arr``.
    """
    if arr.ndim != 2:
        raise ValueError("tv_denoise expects a 2-D array")
    if nabla_comp not in ("both", "x", "y"):
        raise ValueError(f"nabla_comp must be 'both', 'x', or 'y', got {nabla_comp!r}")

    Ny, Nx = arr.shape
    f = arr.astype(np.float64, copy=True).ravel()
    nan_mask_1d = ~np.isfinite(f)
    if nan_mask_1d.any():
        finite_vals = f[~nan_mask_1d]
        fill_val = float(finite_vals.mean()) if finite_vals.size > 0 else 0.0
        f[nan_mask_1d] = fill_val
    u = f.copy()
    p = np.zeros(2 * Ny * Nx)

    L = math.sqrt(8.0)
    sigma = 1.0 / (tau * L * L)

    if method not in ("huber_rof", "tv_l1"):
        raise ValueError(f"Unknown method {method!r}")

    # Review numerical #2 (fixed 2026-05-28): the loop previously used
    # range(max_iter + 1), which ran one extra iteration past the cap,
    # and only checked convergence at it % 50, so callers using
    # max_iter < 50 (e.g. preview pipelines) never saw their tol
    # honoured.  Now: exactly max_iter iterations, with the RMSE-vs-tol
    # check every 10 iterations after a brief warmup.
    _CONVERGENCE_CHECK_EVERY = 10
    _CONVERGENCE_WARMUP = 5
    for it in range(max_iter):
        u_old = u.copy()
        u = u - tau * _nabla_T_apply(p, Ny, Nx, nabla_comp)

        if method == "tv_l1":
            diff = u - f
            u = f + np.maximum(0.0, np.abs(diff) - tau * lam) * np.sign(diff)
            eff_alpha = 0.0
        else:  # huber_rof
            u = (u + tau * lam * f) / (1.0 + tau * lam)
            eff_alpha = alpha

        u_bar = 2.0 * u - u_old
        p = (p + sigma * _nabla_apply(u_bar, Ny, Nx, nabla_comp)) / (1.0 + sigma * eff_alpha)

        # Proximal projection onto the unit ball (isotropic TV).
        p2 = p.reshape(2, -1)
        norm = np.sqrt(p2[0] ** 2 + p2[1] ** 2)
        denom = np.maximum(1.0, norm)
        p = (p2 / denom[np.newaxis, :]).ravel()

        if it >= _CONVERGENCE_WARMUP and it % _CONVERGENCE_CHECK_EVERY == 0:
            rmse = float(np.sqrt(np.mean((u - u_old) ** 2)))
            if rmse < tol:
                break

    result = u.reshape(Ny, Nx)
    if nan_mask_1d.any():
        result[nan_mask_1d.reshape(Ny, Nx)] = np.nan
    return result.astype(arr.dtype, copy=False)
