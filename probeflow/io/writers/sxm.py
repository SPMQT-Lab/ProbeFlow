"""Write a :class:`probeflow.core.scan_model.Scan` to a Nanonis ``.sxm`` file.

Two code paths depending on where the Scan came from:

* ``source_format == "sxm"`` — fast path that reuses the source file's header
  and binary layout verbatim via :func:`probeflow.io.sxm_io.write_sxm_with_planes`.
  Only the float payload is rewritten with the Scan's (possibly processed)
  planes.

* ``source_format == "dat"`` — full reconstruction path.  We build a Nanonis
  header from the original Createc metadata via
  :func:`probeflow.io.converters.createc_dat_to_sxm.construct_hdr` and emit a brand-new ``.sxm`` binary
  via :func:`probeflow.io.converters.createc_dat_to_sxm.reconstruct_from_hdr_imgs`.  This is what
  ``probeflow dat2sxm`` did before, but it now takes the *processed* planes
  from the Scan instead of re-decoding the raw ``.dat`` file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from probeflow.io.common import check_output_available, check_overwrite
from probeflow.provenance.export import (
    ExportProvenance,
    build_scan_export_provenance,
    check_provenance_sidecar_collisions,
    human_summary_from_provenance,
    write_provenance_sidecars,
)
from probeflow.core.scan_model import PLANE_CANON_NAMES, PLANE_CANON_UNITS, Scan
from probeflow.io.sxm_io import write_sxm_with_planes


def _build_comment(scan: Scan, prov: ExportProvenance) -> str:
    """Format provenance fields into a human-readable COMMENT string."""
    source_name = scan.source_path.name if scan.source_path else "unknown"
    lines = [f"Source: {source_name}"]
    if prov.source_id:
        lines.append(f"SourceId: {prov.source_id}")
    if prov.artifact_id:
        lines.append(f"ArtifactId: {prov.artifact_id}")
    lines.append(f"ProcessingStateHash: {prov.processing_state_hash}")
    summary = human_summary_from_provenance(prov)
    if summary:
        lines.append("")
        lines.append(summary)
    if scan.processing_history:
        lines.append("Operations:")
        for i, entry in enumerate(scan.processing_history, 1):
            params_str = " ".join(f"{k}={v}" for k, v in entry["params"].items())
            op_line = f"  {i}. {entry['op']}"
            if params_str:
                op_line += f" {params_str}"
            lines.append(op_line)
    return "\n".join(lines)


def write_sxm(
    scan: Scan,
    out_path,
    *,
    cushion_dir=None,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    overwrite: bool = False,
    overwrite_sidecars: bool = False,
) -> None:
    out_path = Path(out_path)
    if scan.source_path is not None:
        check_overwrite(scan.source_path, out_path)
    prov = build_scan_export_provenance(
        scan,
        channel_index=None,
        display_state={
            "clip_low": float(clip_low),
            "clip_high": float(clip_high),
        },
        export_kind="sxm",
        output_path=out_path,
    )
    check_provenance_sidecar_collisions(
        out_path,
        legacy=False,
        probeflow=True,
        overwrite=overwrite_sidecars,
    )
    check_output_available(out_path, overwrite=overwrite)
    if scan.source_format == "sxm":
        _write_from_sxm(scan, out_path, prov)
    elif scan.source_format == "dat":
        _write_from_dat(
            scan,
            out_path,
            prov,
            cushion_dir=cushion_dir,
            clip_low=clip_low,
            clip_high=clip_high,
        )
    else:
        raise ValueError(
            f"Cannot write .sxm from source_format={scan.source_format!r}"
        )
    write_provenance_sidecars(
        out_path,
        prov,
        legacy=False,
        probeflow=True,
        export_format="sxm",
        overwrite=overwrite_sidecars,
    )


# ─── SXM-sourced fast path ──────────────────────────────────────────────────

def _write_from_sxm(scan: Scan, out_path: Path, prov: ExportProvenance) -> None:
    # write_sxm already ran check_overwrite() against scan.source_path and
    # check_output_available() against out_path, so pass overwrite=True
    # to the underlying writer to suppress its own existence check.  The
    # source-collision guard inside write_sxm_with_planes still runs.
    write_sxm_with_planes(
        scan.source_path, out_path, scan.planes,
        comment_override=_build_comment(scan, prov),
        overwrite=True,
    )


# ─── DAT-sourced reconstruction path ────────────────────────────────────────

def _write_from_dat(
    scan: Scan,
    out_path: Path,
    prov: ExportProvenance,
    *,
    cushion_dir=None,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
) -> None:
    # Lazy-import to avoid pulling in the full dat_sxm machinery on every
    # probeflow.core.scan_loader import — dat_sxm has heavy top-level imports.
    from probeflow.io.converters.createc_dat_to_sxm import (
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

    if not _is_canonical_dat_sxm_layout(scan):
        raise ValueError(
            "DAT-to-SXM export currently supports only canonical STM "
            "[Z forward, Z backward, Current forward, Current backward] "
            f"scans; got {scan.n_planes} plane(s): {scan.plane_names}"
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
        clip_low=clip_low, clip_high=clip_high,
    )
    sxm_hdr["COMMENT"] = _build_comment(scan, prov)

    Ny2, Nx2 = FT.shape
    sxm_hdr["SCAN_PIXELS"] = f"{Nx2}{' ' * 7}{Ny2}"

    imgs = [
        ("Z",       "m", "forward",  to_f32(FT)),
        ("Z",       "m", "backward", to_f32(BT)),
        ("Current", "A", "forward",  to_f32(FC)),
        ("Current", "A", "backward", to_f32(BC)),
    ]

    layout_dir = Path(cushion_dir) if cushion_dir is not None else DEFAULT_CUSHION_DIR
    layout, header_format = load_layout_and_format(layout_dir)
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


def _is_canonical_dat_sxm_layout(scan: Scan) -> bool:
    return (
        scan.n_planes == 4
        and tuple(scan.plane_names) == PLANE_CANON_NAMES
        and tuple(scan.plane_units) == PLANE_CANON_UNITS
    )
