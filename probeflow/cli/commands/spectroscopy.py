"""Spectroscopy CLI commands: spec-info, spec-plot, spec-overlay, spec-positions."""

from __future__ import annotations

import logging
from pathlib import Path

from probeflow.io.common import setup_logging

log = logging.getLogger(__name__)


def _cmd_spec_info(args) -> int:
    setup_logging(args.verbose)
    from probeflow.io.spectroscopy import read_spec_file, spec_channel_to_dict
    import json
    spec = read_spec_file(args.input)
    channels = list(spec.channel_order) if spec.channel_order else list(spec.channels.keys())
    if args.json:
        out = {
            "file": str(args.input),
            "sweep_type": spec.metadata["sweep_type"],
            "measurement_family": spec.metadata.get("measurement_family"),
            "feedback_mode": spec.metadata.get("feedback_mode"),
            "derivative_label": spec.metadata.get("derivative_label"),
            "measurement_confidence": spec.metadata.get("measurement_confidence"),
            "measurement_evidence": spec.metadata.get("measurement_evidence"),
            "n_points": spec.metadata["n_points"],
            "channels": channels,
            "channel_info": [
                spec_channel_to_dict(spec.channel_info[ch])
                for ch in channels
                if ch in spec.channel_info
            ],
            "x_label": spec.x_label,
            "x_unit": spec.x_unit,
            "position_m": list(spec.position),
            "metadata": spec.metadata,
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"file        : {args.input}")
        print(f"sweep type  : {spec.metadata['sweep_type']}")
        if spec.metadata.get("measurement_family"):
            print(f"measurement : {spec.metadata['measurement_family']}")
        if spec.metadata.get("feedback_mode"):
            print(f"feedback    : {spec.metadata['feedback_mode']}")
        if spec.metadata.get("derivative_label"):
            print(f"derivative  : {spec.metadata['derivative_label']}")
        print(f"n_points    : {spec.metadata['n_points']}")
        print(f"channels    : {', '.join(channels)}")
        print(f"x_axis      : {spec.x_label}")
        x = spec.x_array
        print(f"x_range     : {x.min():.4g} to {x.max():.4g} {spec.x_unit}")
        px, py = spec.position
        print(f"position    : ({px*1e9:.3f}, {py*1e9:.3f}) nm")
        for key in ("bias_mv", "spec_freq_hz", "gain_pre_exp", "fb_log", "title"):
            if key in spec.metadata:
                print(f"{key:12s}: {spec.metadata[key]}")
    return 0


def _cmd_spec_plot(args) -> int:
    setup_logging(args.verbose)
    import matplotlib
    matplotlib.use("Agg" if args.output else "TkAgg", force=False)
    import matplotlib.pyplot as plt
    from probeflow.io.spectroscopy import read_spec_file
    from probeflow.analysis.spec_plot import plot_spectrum

    spec = read_spec_file(args.input)
    fig, ax = plt.subplots()
    plot_spectrum(spec, channel=args.channel, ax=ax)
    ax.set_title(Path(args.input).stem)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        log.info("[OK] %s → %s", args.input.name, args.output)
    else:
        plt.show()
    return 0


def _cmd_spec_overlay(args) -> int:
    setup_logging(args.verbose)
    import matplotlib
    matplotlib.use("Agg" if args.output else "TkAgg", force=False)
    import matplotlib.pyplot as plt
    from probeflow.io.spectroscopy import read_spec_file
    from probeflow.analysis.spec_plot import plot_spectra
    from probeflow.processing.spectroscopy import average_spectra

    specs = [read_spec_file(p) for p in args.inputs]
    fig, ax = plt.subplots()
    plot_spectra(specs, channel=args.channel, offset=args.offset, ax=ax)

    if args.average:
        ch_data = [s.channels[args.channel] for s in specs]
        avg = average_spectra(ch_data)
        ax.plot(specs[0].x_array, avg, "k--", linewidth=2, label="average")

    ax.legend(fontsize=7)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        log.info("[OK] overlay → %s", args.output)
    else:
        plt.show()
    return 0


def _cmd_spec_positions(args) -> int:
    setup_logging(args.verbose)
    import matplotlib
    matplotlib.use("Agg" if args.output else "TkAgg", force=False)
    import matplotlib.pyplot as plt
    from probeflow.io.spectroscopy import read_spec_file
    from probeflow.analysis.spec_plot import plot_spec_positions

    specs = [read_spec_file(p) for p in args.inputs]
    fig, ax = plt.subplots()
    plot_spec_positions(str(args.image), specs, ax=ax)
    ax.set_title(Path(args.image).stem)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        log.info("[OK] positions → %s", args.output)
    else:
        plt.show()
    return 0


__all__ = [
    "_cmd_spec_info",
    "_cmd_spec_overlay",
    "_cmd_spec_plot",
    "_cmd_spec_positions",
]
