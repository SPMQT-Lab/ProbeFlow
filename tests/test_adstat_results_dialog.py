"""Tests for the AdStat view-spec Qt renderer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QLabel, QTabWidget, QTableWidget

from probeflow.gui.dialogs.adstat_results import (
    AdStatPlotWidget,
    AdStatResultsDialog,
    AdStatResultView,
    _CURVE_BAND_COLOR,
    _CURVE_MODEL_COLOR,
    _CURVE_OBSERVED_COLOR,
    _curve_legend_entries,
    _empty_panel_message,
    _plot_title,
    _populated_x_range,
    _realspace_marker_style,
    _series_reference_curves,
    _series_curve_label,
    _ticks,
)


def test_ticks_are_regularly_spaced_nice_numbers():
    ticks = _ticks(0.474, 1.84)
    assert len(ticks) >= 3
    steps = np.diff(np.asarray(ticks))
    # All intervals are equal (regular grid)...
    assert np.allclose(steps, steps[0])
    # ...and the step is a "nice" 1/2/2.5/5 x 10**k value.
    assert steps[0] == pytest.approx(0.5)


def test_ticks_anchor_places_reference_on_a_gridline():
    ticks = _ticks(0.47, 1.84, anchor=1.0)
    assert any(abs(t - 1.0) < 1e-9 for t in ticks)


def test_populated_x_range_trims_empty_tail():
    # Cluster-size-style data: non-zero only at small x, zero out to 120.
    x = np.arange(0.0, 121.0)
    observed = np.zeros_like(x)
    observed[:3] = [100.0, 12.0, 3.0]
    fallback = (float(x.min()), float(x.max()))
    x_min, x_max = _populated_x_range(x, (observed, None, None), fallback)
    assert x_min <= 0.0
    # Trimmed to just past the last non-zero sample (size 2), not out to 120.
    assert 2.0 <= x_max <= 8.0


def test_series_curve_label_names_single_pooled_group():
    # A single pooled group carries a "0 " (value 0, unitless) coverage label.
    assert _series_curve_label("0 ") == "pooled mean"
    assert _series_curve_label("0") == "pooled mean"
    assert _series_curve_label("") == "pooled mean"
    # A real coverage label is preserved.
    assert _series_curve_label("0.5 ML") == "0.5 ML"


def test_series_reference_curves_parse_tutorial_single_image_reference():
    panel = SimpleNamespace(
        metadata={
            "reference_curves": (
                {
                    "x": [0.5, 1.5, 2.5],
                    "y": [1.2, 0.9, 1.0],
                    "label": "single image reference",
                    "color": "#ff9f1c",
                },
            )
        }
    )

    curves = _series_reference_curves(panel)

    assert len(curves) == 1
    assert curves[0]["label"] == "single image reference"
    assert curves[0]["color"] == "#ff9f1c"
    np.testing.assert_allclose(curves[0]["y"], [1.2, 0.9, 1.0])


def test_populated_x_range_falls_back_when_all_zero():
    x = np.arange(0.0, 10.0)
    zeros = np.zeros_like(x)
    fallback = (0.0, 9.0)
    assert _populated_x_range(x, (zeros,), fallback) == fallback


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _synthetic_view_spec():
    return SimpleNamespace(
        panels=(
            SimpleNamespace(
                statistic="realspace",
                title="real space",
                kind="realspace",
                x_label="x (nm)",
                y_label="y (nm)",
                reference_line=None,
                observed=np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]]),
                caption_lines=("particle_count: 3",),
                metadata={},
            ),
            SimpleNamespace(
                statistic="pair_correlation_g_r",
                title="pair correlation",
                kind="curve",
                x_label="r (nm)",
                y_label="g(r)",
                reference_line=1.0,
                x=np.array([0.5, 1.5, 2.5]),
                observed=np.array([0.0, 1.2, 0.8]),
                band_low=np.array([0.4, 0.5, 0.6]),
                band_high=np.array([1.6, 1.5, 1.4]),
                central=np.array([1.0, 1.0, 1.0]),
                coordinate_values={"r_nm": np.array([0.5, 1.5, 2.5])},
                caption_lines=("pair_correlation_g_r / homogeneous_poisson",),
                metadata={"n_simulations": 4},
            ),
            SimpleNamespace(
                statistic="diagnostics",
                title="diagnostics",
                kind="table",
                x_label="",
                y_label="",
                reference_line=None,
                table_columns=("severity", "code", "message"),
                table_rows=(("WARN", "SMALL_N", "few particles"),),
                caption_lines=(),
                metadata={},
            ),
        ),
        verdict_rows=(
            ("homogeneous_poisson", "pair_correlation_g_r", "consistent_with_null", "0.5", "0.1", "0/3", "4"),
        ),
        status_lines=("small-N result is underpowered",),
        explainer=SimpleNamespace(
            friendly_name="random placement",
            plain_summary="Places particles randomly in the analysis region.",
            useful_for="Baseline comparison.",
            cautions=("Does not prove absence of interactions.",),
        ),
    )


def test_adstat_results_dialog_renders_summary_plots_and_tables(qapp):
    dlg = AdStatResultsDialog(_synthetic_view_spec(), source_label="Feature result")

    tabs = dlg.findChild(QTabWidget)
    tables = dlg.findChildren(QTableWidget)

    assert dlg.windowTitle() == "Particle Statistics results"
    assert dlg.tab_count == 5
    assert dlg.tab_titles == ("Summary", "Technical details", "real space", "pair correlation", "diagnostics")
    assert tabs.tabText(0) == "Summary"
    assert tabs.tabText(1) == "Technical details"
    assert len(tables) >= 2
    assert tables[0].rowCount() == 1
    assert sorted(plot.panel_kind for plot in dlg.findChildren(AdStatPlotWidget)) == ["curve", "realspace"]

    dlg.close()
    dlg.deleteLater()


def test_adstat_results_dialog_tolerates_partial_panel_contract(qapp):
    spec = SimpleNamespace(
        panels=(
            SimpleNamespace(
                title="bad realspace",
                kind="realspace",
                x_label="x",
                y_label="y",
                reference_line="not numeric",
                observed=np.array([1.0, 2.0, 3.0]),
                caption_lines=(),
                metadata={},
            ),
            SimpleNamespace(
                title="diagnostics",
                kind="table",
                table_columns=("severity",),
                table_rows=(("WARN", "SMALL_N"),),
                caption_lines=(),
                metadata={},
            ),
        ),
        verdict_rows=(),
        status_lines=(),
        explainer=None,
    )

    dlg = AdStatResultsDialog(spec, source_label="Partial contract")

    assert dlg.tab_titles == ("Summary", "bad realspace", "diagnostics")
    tables = dlg.findChildren(QTableWidget)
    assert tables[-1].horizontalHeaderItem(1).text() == "col 2"

    dlg.close()
    dlg.deleteLater()


def test_adstat_result_view_keeps_real_and_sandbox_modes_visually_distinct(qapp):
    real = AdStatResultView(_synthetic_view_spec(), source_label="Feature result")
    sandbox = AdStatResultView(
        _synthetic_view_spec(),
        source_label="Sandbox",
        data_mode="sandbox",
    )

    assert real.data_mode == "real"
    assert real.banner_text == ""
    assert sandbox.data_mode == "sandbox"
    assert sandbox.banner_text == "TEST MODE - GENERATED DATA"
    assert _realspace_marker_style("real")["marker"] == "o"
    assert _realspace_marker_style("real")["color"] == "#2f7ed8"
    assert _realspace_marker_style("sandbox")["marker"] == "^"
    assert _realspace_marker_style("sandbox")["color"] == "#f28e2b"

    real.close()
    sandbox.close()
    real.deleteLater()
    sandbox.deleteLater()


def test_adstat_result_view_show_panels_toggles_plot_tabs(qapp):
    full = AdStatResultView(_synthetic_view_spec(), source_label="Full")
    slim = AdStatResultView(
        _synthetic_view_spec(),
        source_label="Slim",
        show_panels=False,
    )

    # Default keeps the per-panel plot/table tabs (used by the standalone dialog).
    assert "pair correlation" in full.tab_titles
    assert "real space" in full.tab_titles
    # Embedded view shows only the verdict summary + technical details.
    assert slim.tab_titles == ("Summary", "Technical details")

    full.close()
    slim.close()
    full.deleteLater()
    slim.deleteLater()


def test_adstat_result_view_updates_tabs_when_spec_changes(qapp):
    view = AdStatResultView(_synthetic_view_spec(), source_label="Feature result")
    assert view.tab_count == 5

    view.set_view_spec(
        SimpleNamespace(
            panels=(),
            verdict_rows=(),
            status_lines=("ready",),
            explainer=None,
        )
    )

    assert view.tab_titles == ("Summary",)

    view.close()
    view.deleteLater()


def test_adstat_result_view_unparents_old_pages_when_spec_changes(qapp):
    view = AdStatResultView(_synthetic_view_spec(), source_label="Feature result")
    tabs = view.findChild(QTabWidget)
    old_pages = [tabs.widget(index) for index in range(tabs.count())]

    view.set_view_spec(
        SimpleNamespace(
            panels=(),
            verdict_rows=(),
            status_lines=("ready",),
            explainer=None,
        )
    )

    assert view.tab_titles == ("Summary",)
    assert all(page.parent() is None for page in old_pages)

    view.close()
    view.deleteLater()


def test_adstat_plot_widget_paints_with_qt_renderer(qapp):
    plot = AdStatPlotWidget(_synthetic_view_spec().panels[1])
    plot.resize(480, 360)

    pixmap = plot.grab()

    assert not pixmap.isNull()
    assert pixmap.width() == 480
    assert pixmap.height() == 360

    plot.close()
    plot.deleteLater()


def test_adstat_plot_widget_supports_observed_only_curve_mode(qapp):
    plot = AdStatPlotWidget(_synthetic_view_spec().panels[1], curve_mode="observed_only")

    assert plot.curve_mode == "observed_only"

    plot.close()
    plot.deleteLater()


def test_adstat_plot_widget_supports_curve_visibility_flags(qapp):
    panel = _synthetic_view_spec().panels[1]
    comparison = AdStatPlotWidget(panel)
    model_only = AdStatPlotWidget(panel, show_observed_curve=False)
    observed_only = AdStatPlotWidget(panel, show_model_curves=False)
    none_visible = AdStatPlotWidget(
        panel,
        show_observed_curve=False,
        show_model_curves=False,
    )

    assert comparison.show_observed_curve is True
    assert comparison.show_model_curves is True
    assert model_only.show_observed_curve is False
    assert model_only.show_model_curves is True
    assert observed_only.show_observed_curve is True
    assert observed_only.show_model_curves is False
    assert none_visible.show_observed_curve is False
    assert none_visible.show_model_curves is False
    assert _empty_panel_message("curve", no_visible_curves=True) == (
        "No selected plot layers visible"
    )

    for plot in (comparison, model_only, observed_only, none_visible):
        plot.close()
        plot.deleteLater()


def test_adstat_curve_plot_labels_data_vs_model(qapp):
    panel = _synthetic_view_spec().panels[1]
    legend = _curve_legend_entries(has_band=True, has_central=True)

    assert _plot_title(panel) == "Pair correlation g(r)"
    assert [entry[0] for entry in legend] == [
        "model envelope",
        "model median",
        "observed data",
    ]
    assert _CURVE_OBSERVED_COLOR == "#ff9f1c"
    assert _CURVE_MODEL_COLOR == "#7cc7ff"
    assert _CURVE_BAND_COLOR == "#5ea3ff"
    assert [entry[0] for entry in _curve_legend_entries(
        has_band=True,
        has_central=True,
        has_observed=False,
    )] == ["model envelope", "model median"]
    assert [entry[0] for entry in _curve_legend_entries(
        has_band=False,
        has_central=False,
        has_observed=True,
    )] == ["observed data"]


def test_adstat_result_view_groups_models_with_human_labels(qapp):
    view = AdStatResultView(_synthetic_view_spec(), source_label="Feature result")
    tabs = view.findChild(QTabWidget)

    summary_labels = "\n".join(
        label.text() for label in tabs.widget(0).findChildren(QLabel)
    )
    technical_tables = tabs.widget(1).findChildren(QTableWidget)
    technical_cells = "\n".join(
        technical_tables[0].item(row, column).text()
        for row in range(technical_tables[0].rowCount())
        for column in range(technical_tables[0].columnCount())
        if technical_tables[0].item(row, column) is not None
    )

    assert "Random placement" in summary_labels
    assert "Pair correlation" in summary_labels
    assert "homogeneous_poisson" not in summary_labels
    assert "homogeneous_poisson" in technical_cells

    view.close()
    view.deleteLater()


def test_adstat_generated_view_spec_opens_in_results_dialog(qapp):
    pytest.importorskip("adstat")

    from probeflow.analysis.adstat_adapter import compare_point_source_view_spec
    from probeflow.gui.roi_context import PointSource

    points_px = np.array(
        [[1.0, 1.0], [4.0, 1.0], [9.0, 5.0], [2.0, 5.0]],
        dtype=float,
    )
    source = PointSource(
        label="Feature result",
        source_type="feature_finder",
        points_px=points_px,
        points_m=points_px * np.array([0.5e-9, 1.25e-9]),
        metadata={"detection_mode": "maxima"},
    )
    scan = SimpleNamespace(scan_range_m=(6e-9, 10e-9), dims=(12, 8))

    spec = compare_point_source_view_spec(
        source,
        scan=scan,
        image_shape=(8, 12),
        pair_bin_width_nm=1.0,
        pair_max_radius_nm=4.0,
        n_simulations=4,
        random_seed=17,
    )
    dlg = AdStatResultsDialog(spec, source_label="Feature result")
    tabs = dlg.findChild(QTabWidget)
    tab_titles = [tabs.tabText(index) for index in range(tabs.count())]

    assert dlg.tab_count >= 2
    assert tab_titles[0] == "Summary"
    assert "Technical details" in tab_titles
    assert "real space" in tab_titles

    dlg.close()
    dlg.deleteLater()
