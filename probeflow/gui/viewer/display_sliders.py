"""Display-range slider controller extracted from ImageViewerDialog."""

from __future__ import annotations

from typing import Callable

import numpy as np


class DisplaySliderController:
    """Syncs the histogram-panel sliders with the DisplayRangeController.

    Owns the five slider-interaction methods that were previously inline on
    ImageViewerDialog.  The controller holds references to ``_drs`` and
    ``_hist_panel``; ``_display_arr`` is accessed via a callable so the
    controller always sees the current array rather than the one that existed
    at construction time.
    """

    def __init__(
        self,
        drs,
        hist_panel,
        get_display_arr: Callable[[], "np.ndarray | None"],
        channel_unit_fn: Callable[[], tuple[float, str, str]],
    ) -> None:
        # *drs* may be a DisplayRangeController or a zero-arg callable returning
        # one.  The callable form lets the viewer retarget the sliders at the
        # active per-ROI display range without the controller being rebuilt.
        self._drs_source = drs
        self._hist_panel = hist_panel
        self._get_display_arr = get_display_arr
        self._channel_unit_fn = channel_unit_fn

    @property
    def _drs(self):
        src = self._drs_source
        return src() if callable(src) else src

    # ── Sync ──────────────────────────────────────────────────────────────────

    def update(self) -> None:
        """Sync all four sliders to the current drs / display_arr state."""
        display_arr = self._get_display_arr()
        if display_arr is None or self._hist_panel.data_min_si is None:
            return
        vmin_si, vmax_si = self._drs.resolve(display_arr)
        if vmin_si is None:
            return
        data_min = self._hist_panel.data_min_si
        data_max = self._hist_panel.data_max_si or 0.0
        data_range = data_max - data_min
        center = (vmin_si + vmax_si) / 2.0
        width = vmax_si - vmin_si
        center_clamped = max(data_min, min(data_max, center))
        if data_range > 0:
            width_frac = max(0.001, min(1.0, width / data_range))
            contrast_sl = max(0, min(1000, round((1.0 - width_frac) * 1000)))
        else:
            contrast_sl = 0

        self._hist_panel.set_slider_positions(
            self._hist_panel.si_to_sl(vmin_si),
            self._hist_panel.si_to_sl(vmax_si),
            self._hist_panel.si_to_sl(center_clamped),
            contrast_sl,
        )

        scale, unit, _ = self._channel_unit_fn()
        if scale:
            self._hist_panel.set_slider_labels(
                f"{vmin_si * scale:.3g} {unit}",
                f"{vmax_si * scale:.3g} {unit}",
                f"{center * scale:.3g} {unit}",
                f"{width * scale:.3g} {unit}",
            )

    # ── Slider handlers ───────────────────────────────────────────────────────

    def on_min_changed(self, v: int) -> None:
        display_arr = self._get_display_arr()
        if display_arr is None or self._hist_panel.data_min_si is None:
            return
        vmin_si = self._hist_panel.sl_to_si(v)
        _, vmax_si = self._drs.resolve(display_arr)
        if vmax_si is None:
            return
        vmin_si = min(vmin_si, vmax_si - abs(vmax_si) * 1e-9 - 1e-30)
        self._drs.set_manual(vmin_si, vmax_si)

    def on_max_changed(self, v: int) -> None:
        display_arr = self._get_display_arr()
        if display_arr is None or self._hist_panel.data_min_si is None:
            return
        vmax_si = self._hist_panel.sl_to_si(v)
        vmin_si, _ = self._drs.resolve(display_arr)
        if vmin_si is None:
            return
        vmax_si = max(vmax_si, vmin_si + abs(vmin_si) * 1e-9 + 1e-30)
        self._drs.set_manual(vmin_si, vmax_si)

    def on_brightness_changed(self, v: int) -> None:
        """Shift display window centre; keep width fixed."""
        display_arr = self._get_display_arr()
        if display_arr is None or self._hist_panel.data_min_si is None:
            return
        vmin_si, vmax_si = self._drs.resolve(display_arr)
        if vmin_si is None:
            return
        width = vmax_si - vmin_si
        new_center = self._hist_panel.sl_to_si(v)
        new_min = new_center - width / 2.0
        new_max = new_center + width / 2.0
        if new_min >= new_max:
            return
        self._drs.set_manual(new_min, new_max)

    def on_contrast_changed(self, v: int) -> None:
        """Change display window width; keep centre fixed."""
        display_arr = self._get_display_arr()
        if display_arr is None or self._hist_panel.data_min_si is None:
            return
        vmin_si, vmax_si = self._drs.resolve(display_arr)
        if vmin_si is None:
            return
        center = (vmin_si + vmax_si) / 2.0
        data_range = (self._hist_panel.data_max_si or 0.0) - self._hist_panel.data_min_si
        if data_range <= 0:
            return
        width_frac = max(0.001, 1.0 - v / 1000.0)
        new_width = data_range * width_frac
        new_min = center - new_width / 2.0
        new_max = center + new_width / 2.0
        if new_min >= new_max:
            return
        self._drs.set_manual(new_min, new_max)
