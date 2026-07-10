"""
ProbeFlow unified command-line interface — backward-compatibility shim.

All implementations have been moved to their canonical submodules:
  - cli/parser.py            _build_parser, main
  - cli/processing_ops.py    _Op, _record_op, shared helpers, _op_* factories
  - cli/commands/analysis.py
  - cli/commands/conversion.py
  - cli/commands/gui.py
  - cli/commands/processing.py
  - cli/commands/scan.py
  - cli/commands/spectroscopy.py

This module re-exports every public name so that
``from probeflow.cli import _legacy as _impl`` (used by cli/__init__.py)
continues to work without changes.
"""

from __future__ import annotations

# ── Entry point ──────────────────────────────────────────────────────────────
from probeflow.cli.parser import _build_parser, main

# ── Shared processing-op infrastructure ──────────────────────────────────────
from probeflow.cli.processing_ops import (
    _Op,
    _record_op,
    _add_common_io,
    _derive_output,
    _default_output_suffix,
    _png_output_suffix,
    _ensure_output_available,
    _apply_to_plane,
    _write_output,
    _cli_png_provenance,
    _op_plane_bg,
    _op_align_rows,
    _op_remove_bad_lines,
    _op_facet_level,
    _op_smooth,
    _op_edge,
    _op_fft,
    _op_flip_horizontal,
    _op_flip_vertical,
    _op_rotate_90_cw,
    _op_rotate_180,
    _op_rotate_270_cw,
    _op_rotate_arbitrary,
    _parse_processing_steps,
    _processing_state_from_ops,
    _cmd_single_op,
    _load_plane_for_analysis,
    _pixel_size_m_from_scan,
    _pixel_sizes_m_from_scan,
    _load_named_roi,
    _resolve_inline_roi,
)

# ── Command runners ───────────────────────────────────────────────────────────
from probeflow.cli.commands.analysis import (
    _cmd_grains,
    _cmd_autoclip,
    _cmd_periodicity,
    _cmd_tv_denoise,
    _cmd_lattice,
    _cmd_profile,
    _cmd_histogram,
    _cmd_fft_spectrum,
    _cmd_unit_cell,
)
from probeflow.cli.commands.conversion import (
    _cmd_dat2sxm,
    _cmd_dat2png,
)
from probeflow.cli.commands.gui import _cmd_gui
from probeflow.cli.commands.processing import (
    _cmd_pipeline,
    _cmd_prepare_png,
    _cmd_plane_bg,
)
from probeflow.cli.commands.scan import (
    _cmd_sxm2png,
    _cmd_info,
    _cmd_convert,
    _cmd_diag_z,
)
from probeflow.cli.commands.spectroscopy import (
    _cmd_spec_info,
    _cmd_spec_plot,
    _cmd_spec_overlay,
    _cmd_spec_positions,
)

__all__ = [
    # entry point
    "_build_parser", "main",
    # infrastructure
    "_Op", "_record_op",
    "_add_common_io", "_derive_output", "_default_output_suffix",
    "_png_output_suffix", "_ensure_output_available",
    "_apply_to_plane", "_write_output", "_cli_png_provenance",
    "_op_plane_bg", "_op_align_rows", "_op_remove_bad_lines",
    "_op_facet_level", "_op_smooth", "_op_edge", "_op_fft",
    "_op_flip_horizontal", "_op_flip_vertical",
    "_op_rotate_90_cw", "_op_rotate_180", "_op_rotate_270_cw",
    "_op_rotate_arbitrary",
    "_parse_processing_steps", "_processing_state_from_ops",
    "_cmd_single_op", "_load_plane_for_analysis",
    "_pixel_size_m_from_scan", "_pixel_sizes_m_from_scan",
    "_load_named_roi", "_resolve_inline_roi",
    # commands
    "_cmd_grains", "_cmd_autoclip", "_cmd_periodicity",
    "_cmd_tv_denoise",
    "_cmd_lattice", "_cmd_profile",
    "_cmd_histogram", "_cmd_fft_spectrum", "_cmd_unit_cell",
    "_cmd_dat2sxm", "_cmd_dat2png",
    "_cmd_gui",
    "_cmd_pipeline", "_cmd_prepare_png", "_cmd_plane_bg",
    "_cmd_sxm2png", "_cmd_info", "_cmd_convert", "_cmd_diag_z",
    "_cmd_spec_info", "_cmd_spec_plot", "_cmd_spec_overlay", "_cmd_spec_positions",
]
