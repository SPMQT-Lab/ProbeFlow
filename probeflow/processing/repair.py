"""Defect repair: interpolate image data under a mask ("remove spots").

The everyday SPM fix for a tip change, a dirt speck, or a glitch: replace the
pixels inside a marked region with a smooth membrane interpolated from the
surrounding data (the same Laplace interpolation Gwyddion's "remove data under
mask" uses). The repaired region has no extrema of its own — every interior
value lies between its neighbours — so it can never fabricate features.
"""

from __future__ import annotations

import numpy as np


def interpolate_masked(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace the pixels under *mask* by Laplace interpolation.

    Solves the discrete Laplace equation on the masked region with the
    surrounding finite pixels as the boundary condition (a "soap film"
    stretched across the hole). NaN pixels inside the mask are filled like
    any other masked pixel; NaN pixels *outside* the mask are left untouched
    and do not act as boundary values.

    Degenerate inputs degrade gracefully: an empty mask returns the array
    unchanged; masked pixels with no finite boundary anywhere in their
    connected region (e.g. a fully-masked image) keep their original values.
    """
    arr = np.asarray(arr, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != arr.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {arr.shape[:2]}"
        )
    if not mask.any():
        return arr

    ny, nx = arr.shape[:2]
    known = ~mask & np.isfinite(arr)
    if not known.any():
        return arr  # no boundary information anywhere (e.g. fully masked)

    # Only solve mask components that touch at least one known pixel — a
    # component ringed entirely by NaN/border has a singular (pure-Neumann)
    # Laplacian whose "solution" would be arbitrary; leave those unchanged.
    from scipy import ndimage

    labels, n_labels = ndimage.label(mask)
    if n_labels:
        # A component is solvable iff dilating it reaches a known pixel.
        boundary_touch = np.zeros(n_labels + 1, dtype=bool)
        for lbl in range(1, n_labels + 1):
            comp_ring = ndimage.binary_dilation(labels == lbl) & known
            boundary_touch[lbl] = bool(comp_ring.any())
        solvable_mask = mask & boundary_touch[labels]
        if not solvable_mask.any():
            return arr
        mask = solvable_mask

    # Unknown-pixel indexing: solve A·x = b over the masked pixels only.
    unknown_idx = -np.ones((ny, nx), dtype=np.int64)
    ys, xs = np.nonzero(mask)
    n = len(ys)
    unknown_idx[ys, xs] = np.arange(n)

    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.zeros(n, dtype=np.float64)
    diag = np.zeros(n, dtype=np.float64)

    for k in range(n):
        y, x = int(ys[k]), int(xs[k])
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if not (0 <= yy < ny and 0 <= xx < nx):
                continue  # image border: natural (Neumann) boundary
            j = unknown_idx[yy, xx]
            if j >= 0:
                rows.append(k)
                cols.append(int(j))
                vals.append(-1.0)
                diag[k] += 1.0
            elif known[yy, xx]:
                b[k] += arr[yy, xx]
                diag[k] += 1.0
            # NaN outside the mask: no information — skip (Neumann).

    # Pixels with no neighbours contributing information (fully NaN-ringed or
    # a fully-masked image) are pinned by an identity row so the system stays
    # non-singular; their original values (possibly NaN) are restored after
    # the solve so one hopeless pixel cannot poison the repair of the rest.
    pinned = diag <= 0
    diag[pinned] = 1.0
    b[pinned] = 0.0

    rows.extend(range(n))
    cols.extend(range(n))
    vals.extend(diag.tolist())
    a_mat = sparse.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=(n, n)
    )
    try:
        solution = spsolve(a_mat, b)
    except Exception:
        # Singular system despite the pinning: leave the region unrepaired
        # rather than writing garbage.
        return arr
    solution = np.atleast_1d(np.asarray(solution, dtype=np.float64))
    originals = arr[ys, xs]
    solution[pinned] = originals[pinned]
    # A connected sub-region whose entire boundary is NaN solves to a
    # singular/non-finite block: keep those originals too.
    bad = ~np.isfinite(solution) & ~pinned
    solution[bad] = originals[bad]
    arr[ys, xs] = solution
    return arr
