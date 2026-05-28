"""Interpret Createc acquisition metadata without changing decoded data.

This is an IO-side parser helper. It may enrich metadata on core scan objects,
but it should not define ``Scan``/``Spectrum`` models, processing kernels, GUI
presentation, or provenance records.
"""

from __future__ import annotations

from typing import Any

from probeflow.io.common import _f, _i


def createc_dat_experiment_metadata(hdr: dict[str, Any]) -> dict[str, Any]:
    """Return conservative experiment metadata for a Createc scan header."""

    pll_enabled = _truthy(hdr.get("PLLOn"))
    memo = str(hdr.get("MEMO_STMAFM", "") or "").strip()
    memo_l = memo.lower()
    feedback_channel = _i(hdr.get("FBChannel"), None)
    evidence: list[str] = []

    if pll_enabled:
        evidence.append("PLLOn=1")
    if memo:
        evidence.append("MEMO_STMAFM is non-empty")
    if any(token in memo_l for token in ("afm", "qplus", "q-plus", "stmafm")):
        evidence.append("MEMO_STMAFM mentions AFM/qPlus")
    if feedback_channel is not None:
        evidence.append(f"FBChannel={feedback_channel}")

    if pll_enabled and (memo or _memo_mentions_afm(memo_l)):
        acquisition_mode = "afm"
        feedback_mode = "constant_frequency_shift"
        topography_role = "afm_topography"
        confidence = "high"
    elif pll_enabled:
        acquisition_mode = "afm"
        feedback_mode = "constant_frequency_shift"
        topography_role = "afm_topography"
        confidence = "medium"
    elif hdr.get("PLLOn") is not None and not memo:
        acquisition_mode = "stm"
        feedback_mode = "current"
        topography_role = "stm_topography"
        confidence = "high"
        evidence.append("PLLOn=0")
    else:
        acquisition_mode = "unknown"
        feedback_mode = "unknown"
        topography_role = "topography"
        confidence = "low"

    return {
        "acquisition_mode": acquisition_mode,
        "feedback_mode": feedback_mode,
        "pll_enabled": bool(pll_enabled),
        "feedback_channel": feedback_channel,
        "topography_role": topography_role,
        "confidence": confidence,
        "evidence": evidence,
    }


def createc_vert_measurement_metadata(
    report,
    *,
    measurement_mode: str | None = None,
) -> dict[str, Any]:
    """Return spectroscopy measurement semantics for a Createc VERT report."""

    mode = normalize_measurement_mode(measurement_mode)
    if mode != "auto":
        return _override_measurement(mode)

    hdr = report.header
    channel_names = set(report.column_names)
    feedback_on = not _truthy(hdr.get("FBOff"))
    vert_feedback_mode = _i(hdr.get("VertFBMode"), None)
    has_lockin = any(name in channel_names for name in ("dI/dV", "di_q", "di2_q"))
    has_z_i = {"Z", "I"}.issubset(channel_names)
    has_raw_feedback_height = "Raw column 9" in channel_names
    z_points_static = _z_points_static(hdr)

    evidence: list[str] = []
    if feedback_on:
        evidence.append("FBOff=0")
    if vert_feedback_mode is not None:
        evidence.append(f"VertFBMode={vert_feedback_mode}")
    if has_z_i:
        evidence.append("Z and I columns present")
    if has_lockin:
        evidence.append("lock-in derivative channel present")
    if z_points_static:
        evidence.append("Z command program is static")
    if has_raw_feedback_height:
        evidence.append("Raw column 9 carries feedback height counts")

    if feedback_on and has_z_i and has_lockin and z_points_static:
        return {
            "measurement_family": "iz",
            "feedback_mode": "on",
            "derivative_label": "dI/dz",
            "height_channel": "Z feedback" if has_raw_feedback_height else "Z",
            "height_source_channel": "Raw column 9" if has_raw_feedback_height else "Z",
            "z_command_channel": "Z command" if has_raw_feedback_height else None,
            "confidence": "medium",
            "evidence": evidence,
        }

    return {
        "measurement_family": "sts",
        "feedback_mode": "on" if feedback_on else "off",
        "derivative_label": "dI/dV" if "dI/dV" in channel_names else None,
        "height_channel": None,
        "height_source_channel": None,
        "z_command_channel": None,
        "confidence": "low" if has_lockin else "medium",
        "evidence": evidence,
    }


def normalize_measurement_mode(measurement_mode: str | None) -> str:
    """Normalise public measurement-mode overrides."""

    if measurement_mode is None:
        return "auto"
    mode = str(measurement_mode).strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "auto": "auto",
        "sts": "sts",
        "iv": "sts",
        "i_v": "sts",
        "iz": "iz",
        "i_z": "iz",
        "didz": "iz",
        "di_dz": "iz",
    }
    if mode not in aliases:
        raise ValueError(
            "measurement_mode must be one of None, 'auto', 'sts', or 'iz'"
        )
    return aliases[mode]


def scan_mode_label(experiment_metadata: dict[str, Any]) -> str | None:
    """Return a compact GUI label for scan experiment metadata."""

    if experiment_metadata.get("topography_role") == "afm_topography":
        return "AFM df topography"
    if experiment_metadata.get("acquisition_mode") == "stm":
        return "STM topography"
    return None


def spec_measurement_label(metadata: dict[str, Any]) -> str | None:
    """Return a compact GUI label for spectroscopy metadata."""

    if metadata.get("measurement_family") == "iz":
        return "I(z) / dI/dz"
    if metadata.get("measurement_family") == "sts":
        return "STS"
    return None


def _override_measurement(mode: str) -> dict[str, Any]:
    if mode == "iz":
        return {
            "measurement_family": "iz",
            "feedback_mode": "on",
            "derivative_label": "dI/dz",
            "height_channel": None,
            "height_source_channel": None,
            "z_command_channel": None,
            "confidence": "override",
            "evidence": ["measurement_mode override: iz"],
        }
    return {
        "measurement_family": "sts",
        "feedback_mode": "unknown",
        "derivative_label": "dI/dV",
        "height_channel": None,
        "height_source_channel": None,
        "z_command_channel": None,
        "confidence": "override",
        "evidence": ["measurement_mode override: sts"],
    }


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _memo_mentions_afm(memo_l: str) -> bool:
    return any(token in memo_l for token in ("afm", "qplus", "q-plus", "stmafm"))


def _z_points_static(hdr: dict[str, Any]) -> bool:
    values = [_f(hdr.get(f"Zpoint{i}.z"), 0.0) for i in range(8)]
    times = [_f(hdr.get(f"Zpoint{i}.t"), 0.0) for i in range(8)]
    return all((v or 0.0) == 0.0 for v in values) and all(
        (t or 0.0) == 0.0 for t in times
    )
