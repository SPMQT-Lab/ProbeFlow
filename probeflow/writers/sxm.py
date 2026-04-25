"""Write a :class:`probeflow.scan.Scan` to a Nanonis ``.sxm`` file.

Two code paths depending on where the Scan came from:

* ``source_format == "sxm"`` — fast path that reuses the source file's header
  and binary layout verbatim via :func:`probeflow.sxm_io.write_sxm_with_planes`.
  Only the float payload is rewritten with the Scan's (possibly processed)
  planes.

* ``source_format == "dat"`` — full reconstruction path.  We build a Nanonis
  header from the original Createc metadata via
  :func:`probeflow.dat_sxm.construct_hdr` and emit a brand-new ``.sxm`` binary
  via :func:`probeflow.dat_sxm.reconstruct_from_hdr_imgs`.  This is what
  ``probeflow dat2sxm`` did before, but it now takes the *processed* planes
  from the Scan instead of re-decoding the raw ``.dat`` file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from probeflow.common import check_overwrite
from probeflow.scan import Scan
from probeflow.sxm_io import write_sxm_with_planes


def _build_comment(scan: Scan) -> str:
    """Format scan.processing_history into a human-readable COMMENT string."""
    source_name = scan.source_path.name if scan.source_path else "unknown"
    lines = [f"Source: {source_name}"]
    if scan.processing_history:
        lines.append("Operations:")
        for i, entry in enumerate(scan.processing_history, 1):
            params_str = " ".join(f"{k}={v}" for k, v in entry["params"].items())
            op_line = f"  {i}. {entry['op']}"
            if params_str:
                op_line += f" {params_str}"
            lines.append(op_line)
    return "\n".join(lines)


def write_sxm(scan: Scan, out_path) -> None:
    out_path = Path(out_path)
    if scan.source_path is not None:
        check_overwrite(scan.source_path, out_path)
    if scan.source_format == "sxm":
        _write_from_sxm(scan, out_path)
    elif scan.source_format == "dat":
        _write_from_dat(scan, out_path)
    else:
        raise ValueError(
            f"Cannot write .sxm from source_format={scan.source_format!r}"
        )


# ─── SXM-sourced fast path ──────────────────────────────────────────────────

def _write_from_sxm(scan: Scan, out_path: Path) -> None:
    write_sxm_with_planes(
        scan.source_path, out_path, scan.planes,
        comment_override=_build_comment(scan),
    )


# ─── DAT-sourced reconstruction path ────────────────────────────────────────

def _write_from_dat(scan: Scan, out_path: Path) -> None:
    # Lazy-import to avoid pulling in the full dat_sxm machinery on every
    # probeflow.scan import — dat_sxm has heavy top-level imports.
    from probeflow.dat_sxm import (
        DEFAULT_CUSHION_DIR,
        construct_hdr,
        load_layout_and_format,
        reconstruct_from_hdr_imgs,
        to_f32,
    )

    hdr = scan.header

    # Nanonis .sxm stores backward planes right-to-left; fliplr converts
    # display-oriented backward planes into that native storage order so that
    # orient_plane restores them correctly on the next read.
    def _undo_orient(arr: np.ndarray, is_backward: bool) -> np.ndarray:
        out = np.fliplr(arr) if is_backward else arr
        return np.ascontiguousarray(out, dtype=np.float32)

    if scan.n_planes < 4:
        raise ValueError(
            f"Expected 4 planes in dat-sourced Scan, got {scan.n_planes}"
        )

    FT = _undo_orient(scan.planes[0], is_backward=False)
    BT = _undo_orient(scan.planes[1], is_backward=True)
    FC = _undo_orient(scan.planes[2], is_backward=False)
    BC = _undo_orient(scan.planes[3], is_backward=True)

    # The .sxm always stores four direction-resolved planes, even when the
    # original .dat had only two channels (backward planes are synthesised).
    num_chan_for_header = 4

    sxm_hdr = construct_hdr(
        hdr, scan.source_path, num_chan_for_header,
        clip_low=1.0, clip_high=99.0,
    )
    sxm_hdr["COMMENT"] = _build_comment(scan)

    Ny2, Nx2 = FT.shape
    sxm_hdr["SCAN_PIXELS"] = f"{Nx2}{' ' * 7}{Ny2}"

    imgs = [
        ("Z",       "m", "forward",  to_f32(FT)),
        ("Z",       "m", "backward", to_f32(BT)),
        ("Current", "A", "forward",  to_f32(FC)),
        ("Current", "A", "backward", to_f32(BC)),
    ]

    layout, header_format = load_layout_and_format(DEFAULT_CUSHION_DIR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reconstruct_from_hdr_imgs(
        hdr=sxm_hdr,
        imgs=imgs,
        header_format=header_format,
        post_end_bytes=layout["post_end_bytes"],
        pre_payload_bytes=layout["pre_payload_bytes"],
        out_path=out_path,
        tail_bytes=layout["tail_bytes"],
        force_data_offset=layout["data_offset"],
        filler_char=b" ",
    )
