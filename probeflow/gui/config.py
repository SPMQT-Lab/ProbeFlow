"""GUI configuration: paths, font settings, load/save helpers."""

from __future__ import annotations

from pathlib import Path

from probeflow.core.resources import FILE_CUSHIONS_DIR, asset_path

CONFIG_PATH     = Path.home() / ".probeflow_config.json"
DEFAULT_CUSHION = FILE_CUSHIONS_DIR
LOGO_PATH       = asset_path("logo.png")
LOGO_GIF_PATH   = asset_path("logo.gif")
LOGO_NAV_PATH   = asset_path("logo_nav.png")
GITHUB_URL      = "https://github.com/SPMQT-Lab/ProbeFlow"

GUI_FONT_SIZES   = {"Small": 9, "Medium": 12, "Large": 14}
GUI_FONT_DEFAULT = "Medium"

DEFAULT_CLIP_LOW  = 1.0
DEFAULT_CLIP_HIGH = 99.0


def normalise_gui_font_size(label: str | None) -> str:
    return label if label in GUI_FONT_SIZES else GUI_FONT_DEFAULT


def load_config() -> dict:
    import warnings
    from probeflow.gui.rendering import DEFAULT_CMAP_LABEL
    defaults = {
        "dark_mode":       True,
        "input_dir":       "",
        "output_dir":      "",
        "custom_output":   False,
        "do_png":          False,
        "do_sxm":          True,
        "clip_low":        DEFAULT_CLIP_LOW,
        "clip_high":       DEFAULT_CLIP_HIGH,
        "colormap":        DEFAULT_CMAP_LABEL,
        "browse_filter":   "all",
        "gui_font_size":   GUI_FONT_DEFAULT,
        "thumbnail_size":  "large",
    }
    try:
        import json
        if CONFIG_PATH.exists():
            defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except Exception as exc:
        warnings.warn(f"ProbeFlow: failed to load config — {exc}", RuntimeWarning, stacklevel=2)
    defaults["gui_font_size"] = normalise_gui_font_size(defaults.get("gui_font_size"))
    return defaults


def save_config(cfg: dict) -> None:
    import warnings
    try:
        import json
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as exc:
        warnings.warn(f"ProbeFlow: failed to save config — {exc}", RuntimeWarning, stacklevel=2)


__all__ = [
    "CONFIG_PATH", "DEFAULT_CUSHION",
    "LOGO_PATH", "LOGO_GIF_PATH", "LOGO_NAV_PATH",
    "GITHUB_URL",
    "GUI_FONT_SIZES", "GUI_FONT_DEFAULT",
    "DEFAULT_CLIP_LOW", "DEFAULT_CLIP_HIGH",
    "normalise_gui_font_size", "load_config", "save_config",
]
