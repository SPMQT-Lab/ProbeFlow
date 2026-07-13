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
from typing import Optional

import numpy as np

from probeflow.io.common import check_output_available, check_overwrite
from probeflow.provenance.export import (
    ExportProvenance,
    build_scan_export_provenance,
    check_provenance_sidecar_collisions,
    human_summary_from_provenance,
    write_provenance_sidecars,
)
from probeflow.core.scan_model import Scan
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
    if prov.channel_index is not None and scan.processing_state.steps:
        channel = prov.channel_name or f"plane {prov.channel_index}"
        lines.append(
            f"ProcessedPlane: {prov.channel_index} ({channel}); "
            "other planes preserved without this processing"
        )
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
    processed_plane_idx: Optional[int] = None,
    include_provenance: bool = True,
    overwrite: bool = False,
    overwrite_sidecars: bool = False,
) -> None:
    out_path = Path(out_path)
    if scan.source_path is not None:
        check_overwrite(scan.source_path, out_path)
    if processed_plane_idx is not None and not (
        0 <= int(processed_plane_idx) < scan.n_planes
    ):
        raise ValueError(
            f"processed_plane_idx={processed_plane_idx} out of range for Scan "
            f"with {scan.n_planes} plane(s)"
        )
    prov = None
    if include_provenance:
        channel_idx = (
            int(processed_plane_idx) if processed_plane_idx is not None else None
        )
        channel_name = (
            scan.plane_names[channel_idx] if channel_idx is not None else None
        )
        prov = build_scan_export_provenance(
            scan,
            channel_index=channel_idx,
            channel_name=channel_name,
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
    if prov is not None:
        write_provenance_sidecars(
            out_path,
            prov,
            legacy=False,
            probeflow=True,
            export_format="sxm",
            overwrite=overwrite_sidecars,
        )


# ─── SXM-sourced fast path ──────────────────────────────────────────────────

def _write_from_sxm(
    scan: Scan,
    out_path: Path,
    prov: Optional[ExportProvenance],
) -> None:
    # write_sxm already ran check_overwrite() against scan.source_path and
    # check_output_available() against out_path, so pass overwrite=True
    # to the underlying writer to suppress its own existence check.  The
    # source-collision guard inside write_sxm_with_planes still runs.
    write_sxm_with_planes(
        scan.source_path, out_path, scan.planes,
        comment_override=_build_comment(scan, prov) if prov is not None else None,
        overwrite=True,
    )


# ─── DAT-sourced reconstruction path ────────────────────────────────────────

def _write_from_dat(
    scan: Scan,
    out_path: Path,
    prov: Optional[ExportProvenance],
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

    sxm_hdr = construct_hdr(
        hdr, scan.source_path, scan.n_planes,
        clip_low=clip_low, clip_high=clip_high,
    )
    if prov is not None:
        sxm_hdr["COMMENT"] = _build_comment(scan, prov)

    if scan.n_planes == 0:
        raise ValueError("Cannot write .sxm from a Scan with no planes")
    shapes = {tuple(np.asarray(plane).shape) for plane in scan.planes}
    if len(shapes) != 1:
        raise ValueError(
            "SXM export requires every plane to have the same dimensions; "
            "shape-changing processing can only be exported as PNG, PDF, CSV, "
            "or GWY."
        )
    Ny2, Nx2 = scan.planes[0].shape
    sxm_hdr["SCAN_PIXELS"] = f"{Nx2}{' ' * 7}{Ny2}"

    data_info_lines = ["Channel\tName\tUnit\tDirection\tCalibration\tOffset"]
    imgs = []
    for idx, plane in enumerate(scan.planes):
        public_name = (
            scan.plane_names[idx]
            if idx < len(scan.plane_names)
            else f"Channel {idx}"
        )
        unit = scan.plane_units[idx] if idx < len(scan.plane_units) else ""
        base_name, direction = _split_plane_name(public_name, idx)
        storage_plane = np.fliplr(plane) if direction == "backward" else plane
        sxm_name = "_".join(base_name.split()) or f"Channel_{idx}"
        sxm_unit = "_".join(str(unit).split()) or "a.u."
        data_info_lines.append(
            f"\t{idx}\t{sxm_name}\t{sxm_unit}\t{direction}"
            "\t1.000E+0\t0.000E+0"
        )
        imgs.append((sxm_name, sxm_unit, direction, to_f32(storage_plane)))
    sxm_hdr["DATA_INFO"] = "\n".join(data_info_lines)

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


def _split_plane_name(public_name: str, idx: int) -> tuple[str, str]:
    """Return an SXM channel name and explicit scan direction."""

    name = str(public_name).strip() or f"Channel {idx}"
    lowered = name.lower()
    for suffix, direction in (
        (" backward", "backward"),
        (" forward", "forward"),
        (" back", "backward"),
        (" fwd", "forward"),
    ):
        if lowered.endswith(suffix):
            return name[: -len(suffix)].strip() or f"Channel {idx}", direction
    return name, "forward"
