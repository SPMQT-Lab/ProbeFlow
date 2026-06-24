"""Tests for the ProbeFlow Particle Statistics tool."""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from probeflow.gui.dialogs.adstat_workbench import AdStatWorkbenchDialog
from probeflow.gui.dialogs.adstat_results import AdStatPlotWidget
from probeflow.gui.dialogs.particle_statistics import (
    FocusedStatisticPanel,
    ParticleFieldView,
    ParticleStatisticsDialog,
    _TUTORIALS,
    _aspect_fit_rect,
    _panel_for_statistic,
    _series_focus_read_text,
    _single_curve_reference_from_spec,
    _with_series_reference_curve,
)


LESSON_ORDER = [
    "welcome",
    "point_pattern",
    "model_baseline_observed",
    "model_baseline_model",
    "model_baseline_overlay",
    "image_to_statistic",
    "simulation_envelope",
    "verdict",
    "homogeneous_poisson",
    "clustered",
    "hard_core_meaning",
    "hard_core_parameters",
    "hard_core_statistic",
    "ordered_cluster_vs_order",
    "ordered_radial_spacing",
    "ordered_directional_pairs",
    "ordered_bond_order",
    "ordered_square_order",
    "ordered_mixed",
    "feature_biased",
    "other_statistics",
    "pooling_single",
    "pooling_two",
    "model_simulations_sandbox",
    "real_workflow",
    "final_caution",
]


def _plain_words(text: str) -> list[str]:
    plain = re.sub(r"<[^>]+>", " ", text)
    return re.findall(r"[A-Za-z0-9']+", plain)


def test_tutorial_structure_is_concept_first_and_guarded():
    keys = [ex.key for ex in _TUTORIALS]
    assert keys == LESSON_ORDER

    for tutorial in _TUTORIALS:
        for step in tutorial.steps:
            label = f"{tutorial.key}/{step.title}"
            assert step.title, label
            assert step.question, label
            assert step.look_for, label
            assert len(_plain_words(" ".join((step.title, step.question, step.look_for)))) <= 60
            assert len(step.visible_controls or step.controls) <= 1, label
            assert step.visible_panel in {"field", "plot", "results", "controls"}, label

    feature = next(ex for ex in _TUTORIALS if ex.key == "feature_biased")
    feature_text = " ".join(
        f"{step.question} {step.look_for} {step.caution} {step.more_detail}"
        for step in feature.steps
    ).lower()
    assert "independent" in feature_text
    assert "circular" in feature_text

    pooling = next(ex for ex in _TUTORIALS if ex.key == "pooling_two")
    pooling_text = " ".join(f"{step.question} {step.caution}" for step in pooling.steps).lower()
    assert "independent" in pooling_text
    assert "same experimental condition" in pooling_text

    verdict = next(ex for ex in _TUTORIALS if ex.key == "verdict")
    verdict_text = " ".join(f"{step.question} {step.look_for}" for step in verdict.steps).lower()
    assert "consistent with this model" in verdict_text
    assert "prove" not in verdict_text


def test_tutorial_model_baseline_stages_layers_and_language():
    observed = next(ex for ex in _TUTORIALS if ex.key == "model_baseline_observed").steps[0]
    model = next(ex for ex in _TUTORIALS if ex.key == "model_baseline_model").steps[0]
    overlay = next(ex for ex in _TUTORIALS if ex.key == "model_baseline_overlay").steps[0]

    assert observed.show_observed is True
    assert observed.show_simulated is False
    assert model.show_observed is False
    assert model.show_simulated is True
    assert overlay.show_observed is True
    assert overlay.show_simulated is True
    overlay_text = f"{overlay.question} {overlay.look_for} {overlay.more_detail}".lower()
    assert "not the final test" in overlay_text
    assert "statistical curve" in overlay_text
    assert all(not step.show_technical_details for step in (observed, model, overlay))


def test_tutorial_hard_core_lessons_have_physical_radius_guardrails():
    meaning = next(ex for ex in _TUTORIALS if ex.key == "hard_core_meaning").steps[0]
    parameters = next(ex for ex in _TUTORIALS if ex.key == "hard_core_parameters").steps[0]
    statistic = next(ex for ex in _TUTORIALS if ex.key == "hard_core_statistic").steps[0]

    assert meaning.model == "hard_core_random"
    assert meaning.show_observed is False
    assert meaning.show_simulated is True
    assert "model_hard_core_radius" in meaning.visible_controls
    meaning_text = f"{meaning.question} {meaning.look_for}".lower()
    assert "minimum allowed separation" in meaning_text
    assert "overlap" in meaning_text
    assert parameters.model_hard_core_radius_nm
    assert meaning.model_hard_core_radius_nm
    assert parameters.model_hard_core_radius_nm > meaning.model_hard_core_radius_nm
    assert "slow" in parameters.caution.lower()
    assert statistic.focus_statistic == "nearest_neighbor_distribution"


def test_tutorial_ordered_islands_teaches_local_order_sequence():
    ordered = [
        next(ex for ex in _TUTORIALS if ex.key == key).steps[0]
        for key in (
            "ordered_cluster_vs_order",
            "ordered_radial_spacing",
            "ordered_directional_pairs",
            "ordered_bond_order",
            "ordered_square_order",
            "ordered_mixed",
        )
    ]

    assert all(step.pattern == "ordered_islands" for step in ordered)
    assert ordered[0].visible_controls == ("ordered_lattice",)
    assert ordered[-1].visible_controls == ("ordered_background",)
    text = " ".join(
        f"{step.question} {step.look_for} {step.caution}"
        for step in ordered
    ).lower()
    assert "clustering" in text
    assert "direction" in text
    assert "ψ6" in text
    assert "ψ4" in text
    assert "square" in text
    assert "does not prove" in text
    assert ordered[1].focus_statistic == "pair_correlation_g_r"
    assert ordered[2].focus_statistic == "pair_correlation_g_r_theta"
    assert ordered[3].focus_statistic == "bond_order_psi6"
    assert ordered[4].focus_statistic == "bond_order_psi4"


def test_tutorial_pooling_uses_one_then_two_images():
    single = next(ex for ex in _TUTORIALS if ex.key == "pooling_single").steps[0]
    pooled = next(ex for ex in _TUTORIALS if ex.key == "pooling_two").steps[0]

    assert single.pool_images == 0
    assert pooled.pool_images == 2
    assert single.direct_labels == ("single image",)
    assert pooled.direct_labels == ("pooled: 2 images",)
    assert "independent" in pooled.question.lower()
    assert "same experimental condition" in pooled.caution.lower()
    assert "blue" in pooled.look_for.lower()
    assert "orange" in pooled.look_for.lower()
    assert "not a model envelope" in pooled.more_detail.lower()


def test_particle_statistics_landing_page_shows_three_workflows(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="landing")

    assert dlg.current_mode == "landing"
    assert not dlg._landing_panel.isHidden()
    assert dlg._workspace_panel.isHidden()
    assert dlg._mode_cb.isHidden()
    assert dlg._run_btn.isHidden()
    assert dlg._start_tutorial_btn.isHidden()

    cards = dlg._landing_panel.findChildren(QPushButton)
    labels = {button.text() for button in cards}
    assert labels == {"Choose point source", "Open simulations", "Start tutorial"}

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_landing_cards_enter_workflows(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="landing")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.findChild(QPushButton, "particleStatisticsLandingAnalyzeButton").click()
    assert dlg.current_mode == "real"
    assert dlg._landing_panel.isHidden()
    assert not dlg._workspace_panel.isHidden()
    assert not dlg._real_data_group.isHidden()

    dlg.set_current_mode("landing")
    dlg.findChild(QPushButton, "particleStatisticsLandingSimulationsButton").click()
    assert dlg.current_mode == "model_simulations"
    assert not dlg._generated_data_group.isHidden()
    assert dlg._generated_data_group.title() == "Generated pattern"

    dlg.set_current_mode("landing")
    dlg.findChild(QPushButton, "particleStatisticsLandingTutorialButton").click()
    assert dlg.current_mode == "learn"
    assert not dlg._tutorial_panel.isHidden()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_can_return_to_landing_page(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="sandbox")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")
    raises = []
    dlg._raise_self = lambda: raises.append(dlg.current_mode)

    assert dlg.current_mode == "model_simulations"
    assert not dlg._landing_btn.isHidden()

    dlg._landing_btn.click()
    assert dlg.current_mode == "landing"
    assert not dlg._landing_panel.isHidden()
    assert dlg._workspace_panel.isHidden()
    assert raises[-1] == "landing"

    dlg.findChild(QPushButton, "particleStatisticsLandingTutorialButton").click()
    assert dlg.current_mode == "learn"
    raises.clear()
    dlg._show_workflows_action.trigger()
    assert dlg.current_mode == "landing"
    assert raises[-1] == "landing"

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_window_layout_actions_use_config(qapp, monkeypatch):
    import probeflow.gui.dialogs.particle_statistics as particle_module

    saved = {}

    monkeypatch.setattr(particle_module, "load_config", lambda: {"layout": {}})
    monkeypatch.setattr(particle_module, "save_config", lambda cfg: saved.update(cfg))

    dlg = ParticleStatisticsDialog(initial_mode="landing")

    assert dlg.minimumWidth() >= 1300
    assert dlg.minimumHeight() >= 850

    dlg._use_wide_layout_action.trigger()
    dlg._reset_window_size_action.trigger()
    assert "layout" in saved
    assert "particle_statistics" not in saved["layout"]

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_raise_does_not_touch_owner_window(qapp):
    raised: list[str] = []

    class _Owner:
        def window(self):
            return self

        def isVisible(self):
            return True

        def raise_(self):
            raised.append("owner")

    dlg = ParticleStatisticsDialog(initial_mode="real")
    dlg.raise_ = lambda: raised.append("self")  # type: ignore[method-assign]
    dlg.activateWindow = lambda: None  # type: ignore[method-assign]
    dlg.isVisible = lambda: True  # type: ignore[method-assign]
    dlg.parent = lambda: _Owner()  # type: ignore[method-assign]

    dlg._raise_now()

    assert raised == ["self"]  # never the owner window

    dlg.isVisible = lambda: False  # type: ignore[method-assign]
    dlg.close()
    dlg.deleteLater()


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_particle_statistics_tutorial_controls_resolve_and_have_metadata(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    registry = dlg._tutorial_control_widgets()
    used = {
        key
        for tutorial in _TUTORIALS
        for step in tutorial.steps
        for key in (step.visible_controls or step.controls)
    }

    assert used
    assert {key for key in used if key not in registry or not registry[key]} == set()

    for tutorial in _TUTORIALS:
        for step in tutorial.steps:
            label = f"{tutorial.key}/{step.title}"
            assert step.question, label
            assert step.look_for, label
            if step.visible_panel == "results":
                assert step.statistic_label or step.focus_statistic == "model_summary", label
            if step.visible_controls:
                assert step.primary_action, label

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_tutorial_highlights_controls_and_clears(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("feature_biased")

    assert "#2fb344" in dlg._feature_layer_cb.styleSheet()
    assert "#2fb344" in dlg._next_tutorial_btn.styleSheet()

    dlg.next_tutorial_step()

    assert "#e0b020" in dlg._feature_layer_cb.styleSheet()
    assert "#2fb344" in dlg._observed_layer_cb.styleSheet()

    dlg.exit_tutorial()

    assert dlg._feature_layer_cb.styleSheet() == ""
    assert dlg._observed_layer_cb.styleSheet() == ""
    assert dlg._tutorial_panel.isHidden()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_hard_core_tutorial_highlights_radius(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("hard_core_meaning")

    assert dlg._generated_model_cb.currentData() == "hard_core_random"
    assert dlg._field.layer_visibility["observed"] is False
    assert dlg._field.layer_visibility["simulated"] is True
    assert dlg._hard_core_radius_spin.value() == pytest.approx(1.5)
    assert dlg._model_hard_core_radius_spin.value() == pytest.approx(1.5)
    assert "#2fb344" in dlg._model_hard_core_radius_spin.styleSheet()
    assert "minimum allowed separation" in dlg._tutorial_step_lbl.text().lower()

    dlg.next_tutorial_step()

    assert dlg.current_tutorial_key == "hard_core_parameters"
    assert dlg._hard_core_radius_spin.value() == pytest.approx(3.0)
    assert dlg._model_hard_core_radius_spin.value() == pytest.approx(3.0)
    assert "slow" in dlg._tutorial_careful_lbl.text().lower()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_tutorial_real_steps_keep_drawer_visible(qapp):
    dlg = ParticleStatisticsDialog(
        point_sources=[_point_source()],
        scan=SimpleNamespace(scan_range_m=(8e-9, 6e-9), dims=(8, 6)),
        image_shape=(6, 8),
        initial_mode="learn",
    )

    dlg.load_tutorial_example("real_workflow")

    assert dlg.current_mode == "learn"
    assert dlg._mode_cb.currentData() == "real"
    assert not dlg._tutorial_panel.isHidden()
    assert not dlg._feature_sets_group.isHidden()
    assert dlg._generated_data_group.isHidden()
    assert "#2fb344" in dlg._feature_sets_list.styleSheet()

    dlg.next_tutorial_step()

    assert dlg.current_tutorial_key == "final_caution"
    assert dlg._mode_cb.currentData() == "real"
    assert not dlg._tutorial_panel.isHidden()
    assert dlg._result_view.technical_details_visible is False

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_real_workflow_keeps_generated_examples_separate(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("real_workflow")

    assert dlg.current_mode == "learn"
    assert dlg._mode_cb.currentData() == "real"
    assert not dlg._tutorial_panel.isHidden()
    assert dlg._generated_banner.isHidden()
    assert "#2fb344" in dlg._feature_sets_list.styleSheet()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_setup_columns_restore_after_tutorial_exit(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("hard_core_meaning")

    assert not dlg._setup_model_column.isHidden()
    assert dlg._setup_data_column.isHidden()
    assert dlg._setup_statistic_column.isHidden()

    dlg.exit_tutorial()

    assert [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())] == [
        "Setup",
        "Results",
    ]
    assert not dlg._tabs.tabBar().isHidden()
    assert not dlg._setup_data_column.isHidden()
    assert not dlg._setup_model_column.isHidden()
    assert not dlg._setup_statistic_column.isHidden()
    assert not dlg._real_data_group.isHidden()
    assert not dlg._real_model_group.isHidden()
    assert dlg._generated_data_group.isHidden()
    assert dlg._generated_model_group.isHidden()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_top_menus_sync_with_controls(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="sandbox")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    menu_bar = dlg.layout().menuBar()
    assert [action.text() for action in menu_bar.actions()] == [
        "Workflow",
        "Data",
        "Model",
        "Statistic",
        "Export",
        "View",
        "Definitions",
    ]

    assert dlg._show_observed_action.isChecked()
    dlg._show_observed_action.trigger()
    assert not dlg._observed_layer_cb.isChecked()
    assert not dlg._show_observed_action.isChecked()

    assert dlg._show_model_action.isChecked()
    dlg._show_model_action.trigger()
    assert not dlg._simulation_layer_cb.isChecked()
    assert not dlg._show_model_action.isChecked()

    dlg._model_actions["hard_core_random"].trigger()
    assert dlg._generated_model_cb.currentData() == "hard_core_random"
    assert dlg._model_actions["hard_core_random"].isChecked()

    assert dlg._link_hard_core_radii_action.isChecked()
    dlg._link_hard_core_radii_action.trigger()
    assert not dlg._link_hard_core_radii_cb.isChecked()
    assert not dlg._link_hard_core_radii_action.isChecked()

    dlg._statistic_actions["nearest_neighbor_distribution"].trigger()
    assert dlg.focused_statistic == "nearest_neighbor_distribution"
    assert dlg._statistic_actions["nearest_neighbor_distribution"].isChecked()

    dlg._show_definitions_action.trigger()
    # The Definitions menu opens the shared ProbeFlow Definitions document at the
    # Particle Statistics tab (not an isolated in-dialog tab).
    assert dlg._definitions_dialog.current_reference_tab() == "particle_statistics"

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_model_simulations_mode_is_available(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="sandbox")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    assert dlg.current_mode == "model_simulations"
    assert dlg._mode_cb.currentData() == "sandbox"
    assert [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())] == [
        "Setup",
        "Results",
    ]
    assert not dlg._setup_data_column.isHidden()
    assert not dlg._setup_model_column.isHidden()
    assert not dlg._setup_statistic_column.isHidden()
    assert not dlg._generated_data_group.isHidden()
    assert dlg._generated_data_group.title() == "Generated pattern"
    assert not dlg._generated_model_group.isHidden()
    assert dlg._generated_model_group.title() == "Comparison model"
    assert not dlg._hard_core_radius_spin.isHidden()
    assert not dlg._model_hard_core_radius_spin.isHidden()
    assert dlg._generated_data_group.layout().rowCount() <= 8
    reference_text = " ".join(
        label.text() for label in dlg._model_reference_cards.findChildren(QLabel)
    )
    assert "Generated pattern" in reference_text
    assert "Comparison model" in reference_text
    assert "Measured-feature Poisson" not in reference_text
    assert len(_plain_words(reference_text)) <= 32
    assert dlg._tutorial_panel.isHidden()
    assert dlg._generated_banner.text() == "Model simulations"

    _set_index = dlg._pattern_cb.findData("no_overlap")
    dlg._pattern_cb.setCurrentIndex(_set_index)
    dlg._generated_model_cb.setCurrentIndex(
        dlg._generated_model_cb.findData("hard_core_random")
    )
    dlg._n_spin.setValue(100)
    dlg._seed_spin.setValue(12)
    assert dlg._link_hard_core_radii_cb.isChecked()
    dlg._hard_core_radius_spin.setValue(3.0)
    assert dlg._model_hard_core_radius_spin.value() == pytest.approx(3.0)
    dlg._link_hard_core_radii_cb.setChecked(False)
    assert not dlg._link_hard_core_radii_cb.isChecked()
    dlg._model_hard_core_radius_spin.setValue(2.0)

    assert dlg._sandbox_state.config.pattern == "no_overlap"
    assert dlg._sandbox_state.active_model == "hard_core_random"
    assert dlg._sandbox_state.config.seed == 12
    assert dlg._sandbox_state.config.hard_core_radius_nm == pytest.approx(3.0)
    assert dlg._sandbox_state.config.model_hard_core_radius_nm == pytest.approx(2.0)
    assert dlg.field_point_count == 100

    first_points = dlg._field._model.observed_xy_nm.copy()
    dlg._model_hard_core_radius_spin.setValue(4.0)
    assert dlg._sandbox_state.config.model_hard_core_radius_nm == pytest.approx(4.0)
    np.testing.assert_allclose(first_points, dlg._field._model.observed_xy_nm)

    dlg._hard_core_radius_spin.setValue(4.0)
    assert dlg._sandbox_state.config.hard_core_radius_nm == pytest.approx(4.0)
    assert not np.array_equal(first_points, dlg._field._model.observed_xy_nm)

    dlg._n_spin.setValue(200)
    dlg._hard_core_radius_spin.setValue(5.0)
    assert "slow" in dlg._sandbox_warning_lbl.text().lower()

    dlg._statistic_buttons["nearest_neighbor_distribution"].click()
    assert dlg.focused_statistic == "nearest_neighbor_distribution"

    dlg._pattern_cb.setCurrentIndex(dlg._pattern_cb.findData("ordered_islands"))
    assert not dlg._ordered_lattice_cb.isHidden()
    assert not dlg._ordered_background_cb.isHidden()
    dlg._ordered_lattice_cb.setCurrentIndex(dlg._ordered_lattice_cb.findData("square"))
    dlg._ordered_background_cb.setCurrentIndex(
        dlg._ordered_background_cb.findData("clustered")
    )
    assert dlg._sandbox_state.config.ordered_island_lattice == "square"
    assert dlg._sandbox_state.config.ordered_island_background == "clustered"

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_generated_hard_core_radius_link(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="sandbox")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg._generated_model_cb.setCurrentIndex(
        dlg._generated_model_cb.findData("hard_core_random")
    )

    assert dlg._link_hard_core_radii_cb.isChecked()
    assert not dlg._model_hard_core_radius_spin.isEnabled()

    dlg._hard_core_radius_spin.setValue(3.5)
    assert dlg._model_hard_core_radius_spin.value() == pytest.approx(3.5)
    assert dlg._sandbox_state.config.model_hard_core_radius_nm == pytest.approx(3.5)

    dlg._link_hard_core_radii_cb.setChecked(False)
    assert dlg._model_hard_core_radius_spin.isEnabled()
    dlg._model_hard_core_radius_spin.setValue(2.0)
    dlg._hard_core_radius_spin.setValue(4.0)
    assert dlg._sandbox_state.config.hard_core_radius_nm == pytest.approx(4.0)
    assert dlg._sandbox_state.config.model_hard_core_radius_nm == pytest.approx(2.0)

    dlg._link_hard_core_radii_cb.setChecked(True)
    assert not dlg._model_hard_core_radius_spin.isEnabled()
    assert dlg._model_hard_core_radius_spin.value() == pytest.approx(4.0)
    assert dlg._sandbox_state.config.model_hard_core_radius_nm == pytest.approx(4.0)

    dlg.close()
    dlg.deleteLater()


def _point_source(label: str = "Feature maxima"):
    points_px = np.array([[1.0, 2.0], [4.0, 5.0], [8.0, 3.0]], dtype=float)
    return SimpleNamespace(
        label=label,
        source_type="feature_maxima",
        points_px=points_px,
        points_m=points_px * 1e-9,
        metadata={"selection_scope": "full_image"},
    )


def _curve_panel():
    return SimpleNamespace(
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
        caption_lines=(),
        metadata={"n_simulations": 4},
    )


def test_particle_statistics_opens_real_mode_with_field_and_request(qapp):
    active_roi = object()
    active_mask = np.ones((6, 8), dtype=bool)
    scan = SimpleNamespace(scan_range_m=(8e-9, 6e-9), dims=(8, 6))
    dlg = ParticleStatisticsDialog(
        point_sources=[_point_source()],
        scan=scan,
        active_area_roi=active_roi,
        active_mask=active_mask,
        image_shape=(6, 8),
        theme={"text.color": "#111111"},
        initial_mode="real",
    )

    request = dlg._request_from_controls()

    assert dlg.windowTitle() == "Particle Statistics"
    assert dlg.current_mode == "real"
    assert dlg.field_point_count == 3
    assert dlg._field.data_mode == "real"
    assert dlg._field.marker_style["marker"] == "o"
    assert request.point_source_label == "Feature maxima"
    assert request.region_mode == "roi"
    assert request.roi_or_mask is active_roi
    assert request.models == ("poisson",)
    assert request.n_simulations == 100
    assert request.random_seed == 0
    assert dlg._result_view.banner_text == ""

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_real_mode_exposes_models_and_sim_controls(qapp):
    scan = SimpleNamespace(scan_range_m=(8e-9, 6e-9), dims=(8, 6))
    dlg = ParticleStatisticsDialog(
        point_sources=[_point_source()],
        scan=scan,
        image_shape=(6, 8),
        initial_mode="real",
    )

    models = {dlg._real_model_cb.itemData(i) for i in range(dlg._real_model_cb.count())}
    assert {"poisson", "hard_core_random"} <= models

    dlg._real_sim_spin.setValue(40)
    dlg._real_seed_spin.setValue(7)
    index = dlg._real_model_cb.findData("hard_core_random")
    dlg._real_model_cb.setCurrentIndex(index)

    request = dlg._request_from_controls()
    assert request.models == ("hard_core_random",)
    assert request.n_simulations == 40
    assert request.random_seed == 7

    dlg.close()
    dlg.deleteLater()


def _feature_set(name, n, seed):
    from probeflow.measurements.feature_sets import FeatureSet

    rng = np.random.default_rng(seed)
    xy_nm = rng.uniform(0.0, 100.0, size=(n, 2))
    return FeatureSet.from_points(
        name=name,
        points_px=xy_nm,
        points_m=xy_nm * 1e-9,
        scan_range_m=(100e-9, 100e-9),
        image_shape=(256, 256),
        image_label=name,
    )


def test_particle_statistics_lists_and_selects_feature_sets(qapp):
    sets = [_feature_set("A", 120, 1), _feature_set("B", 110, 2)]
    dlg = ParticleStatisticsDialog(
        point_sources=[_point_source()],
        scan=SimpleNamespace(scan_range_m=(8e-9, 6e-9), dims=(8, 6)),
        image_shape=(6, 8),
        feature_sets=sets,
        initial_mode="real",
    )

    assert dlg._feature_sets_list.count() == 2
    dlg.select_feature_set(sets[1].set_id)
    selected = dlg._selected_feature_sets()
    assert [fs.name for fs in selected] == ["B"]

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_runs_combined_feature_sets(qapp):
    if not pytest.importorskip("adstat"):
        pytest.skip("AdStat not installed")
    sets = [_feature_set("A", 120, 1), _feature_set("B", 110, 2)]
    dlg = ParticleStatisticsDialog(feature_sets=sets, initial_mode="real")

    # Drive the worker synchronously rather than via the thread pool.
    from probeflow.gui.dialogs.particle_statistics import _ParticleFeatureSetWorker
    from probeflow.gui.viewer.tool_launch import AdStatStatisticsRequest

    for index in range(dlg._feature_sets_list.count()):
        dlg._feature_sets_list.item(index).setCheckState(__import__("PySide6").QtCore.Qt.Checked)
    request = AdStatStatisticsRequest(
        point_source_label="Combined",
        region_mode="full",
        models=("poisson",),
        n_simulations=6,
        random_seed=0,
    )
    worker = _ParticleFeatureSetWorker(
        generation=dlg._generation, feature_sets=dlg._selected_feature_sets(), request=request
    )
    captured = {}
    worker.signals.finished.connect(
        lambda gen, spec, err: captured.update(spec=spec, err=err)
    )
    worker.work()

    assert captured["err"] == ""
    dlg._on_feature_sets_worker_finished(dlg._generation, captured["spec"], "")
    assert "Combined" in dlg._status_lbl.text()
    assert dlg._result_view.tab_count >= 1

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_pooling_b_adds_tutorial_reference_curve(qapp):
    pytest.importorskip("adstat")
    from probeflow.analysis.adstat_adapter import (
        compare_point_set_record_view_spec,
        compare_point_set_records_view_spec,
    )

    sets = [_feature_set("A", 120, 11), _feature_set("B", 120, 12)]
    single = compare_point_set_record_view_spec(
        sets[0].to_point_set_record(),
        n_simulations=4,
        random_seed=0,
    )
    reference = _single_curve_reference_from_spec(single, "pair_correlation_g_r")
    pooled = compare_point_set_records_view_spec(
        [feature_set.to_point_set_record() for feature_set in sets],
        n_simulations=4,
        random_seed=0,
    )
    normal_panel = _panel_for_statistic(pooled, "pair_correlation_g_r")

    assert reference is not None
    assert normal_panel is not None
    assert not (normal_panel.metadata or {}).get("reference_curves")

    tutorial_spec = _with_series_reference_curve(
        pooled,
        "pair_correlation_g_r",
        reference,
    )
    tutorial_panel = _panel_for_statistic(tutorial_spec, "pair_correlation_g_r")

    assert tutorial_panel is not None
    assert tutorial_panel.metadata["reference_curves"][0]["label"] == (
        "single image reference"
    )
    assert "single-image reference" in _series_focus_read_text(tutorial_panel)
    assert "model simulations" not in _series_focus_read_text(tutorial_panel)


def test_particle_statistics_exposes_measured_feature_model(qapp):
    sets = [_feature_set("particles", 120, 1), _feature_set("edges", 15, 2)]
    dlg = ParticleStatisticsDialog(feature_sets=sets, initial_mode="real")

    models = {dlg._real_model_cb.itemData(i) for i in range(dlg._real_model_cb.count())}
    assert "measured_feature_poisson" in models

    # The feature-layer picker is enabled only for the measured-feature model.
    index = dlg._real_model_cb.findData("measured_feature_poisson")
    dlg._real_model_cb.setCurrentIndex(index)
    assert dlg._feature_layer_set_cb.isEnabled()
    dlg._real_model_cb.setCurrentIndex(dlg._real_model_cb.findData("poisson"))
    assert not dlg._feature_layer_set_cb.isEnabled()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_measured_feature_validation_and_run(qapp):
    if not pytest.importorskip("adstat"):
        pytest.skip("AdStat not installed")
    from probeflow.gui.dialogs.particle_statistics import _ParticleFeatureSetWorker
    from probeflow.gui.viewer.tool_launch import AdStatStatisticsRequest

    particles = _feature_set("particles", 120, 1)
    edges = _feature_set("edges", 18, 2)
    dlg = ParticleStatisticsDialog(feature_sets=[particles, edges], initial_mode="real")
    dlg._real_model_cb.setCurrentIndex(dlg._real_model_cb.findData("measured_feature_poisson"))

    # No feature layer chosen yet → guarded with a helpful message, no crash.
    dlg.select_feature_set(particles.set_id)
    dlg.run_selected_feature_sets()
    assert "Feature layer" in dlg._status_lbl.text()

    # Choose the edges set as the feature layer, then drive the worker synchronously.
    dlg._feature_layer_set_cb.setCurrentIndex(dlg._feature_layer_set_cb.findData(edges.set_id))
    assert dlg._selected_feature_layer().set_id == edges.set_id
    request = AdStatStatisticsRequest(
        point_source_label="particles vs edges",
        region_mode="full",
        models=("measured_feature_poisson",),
        n_simulations=6,
        random_seed=0,
    )
    worker = _ParticleFeatureSetWorker(
        generation=dlg._generation,
        feature_sets=[particles],
        request=request,
        feature_layer=edges,
    )
    captured = {}
    worker.signals.finished.connect(lambda gen, spec, err: captured.update(spec=spec, err=err))
    worker.work()
    assert captured["err"] == ""
    assert captured["spec"].metadata.get("active_model") == "measured_feature_poisson"

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_can_deep_link_to_generated_mode(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")

    assert dlg.current_mode == "learn"
    assert dlg._field.data_mode == "generated"
    assert dlg._field.marker_style["marker"] != "o"
    assert dlg._result_view.banner_text == ""
    assert dlg.current_tutorial_key == "welcome"
    if dlg._sandbox_state is not None:
        dlg.load_tutorial_example("point_pattern")
        assert dlg.field_point_count > 0
        assert dlg._field.direct_labels == ("observed particles",)
        assert dlg._generated_banner.text() == "Tutorial: generated example"
    else:
        assert "AdStat" in dlg._status_lbl.text()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_real_mode_surfaces_failed_context(qapp):
    dlg = ParticleStatisticsDialog(
        point_sources=[_point_source()],
        scan=SimpleNamespace(scan_range_m=(8e-9, 6e-9), dims=(8, 6)),
        image_shape=(6, 8),
    )

    dlg._generation = 3
    dlg._on_real_worker_finished(
        3,
        SimpleNamespace(
            ready=False,
            status_message="ProbeFlow's AdStat adapter requires the optional 'adstat' package.",
            view_spec=None,
            point_source_label="Feature maxima",
        ),
    )

    assert "optional 'adstat' package" in dlg._status_lbl.text()
    assert dlg._result_view.data_mode == "real"
    assert dlg._result_view.banner_text == ""

    dlg.close()
    dlg.deleteLater()


def test_particle_field_view_keeps_real_and_generated_styles_distinct(qapp):
    view = ParticleFieldView()
    view.set_points([[1.0, 1.0]], field_size_nm=(10.0, 10.0), mode="real")
    assert view.data_mode == "real"
    assert view.marker_style["marker"] == "o"
    assert view.marker_style["color"] == "#2f7ed8"

    view.set_points([[1.0, 1.0]], field_size_nm=(10.0, 10.0), mode="generated")
    assert view.data_mode == "generated"
    assert view.marker_style["marker"] == "^"
    assert view.marker_style["color"] == "#f28e2b"

    view.close()
    view.deleteLater()


def test_particle_field_view_preserves_physical_aspect_ratio():
    square = _aspect_fit_rect(QRectF(0, 0, 800, 300), 100.0, 100.0)
    wide = _aspect_fit_rect(QRectF(0, 0, 800, 300), 200.0, 100.0)

    assert square.width() == pytest.approx(square.height())
    assert wide.width() / wide.height() == pytest.approx(2.0)
    assert wide.width() <= 800
    assert wide.height() <= 300


def test_focused_statistic_panel_renders_concept_and_plot(qapp):
    panel = FocusedStatisticPanel()

    panel.set_statistic("pair_correlation_g_r")
    assert panel.focused_statistic == "pair_correlation_g_r"
    assert panel.has_plot is False
    assert "Pair correlation" in panel._title.text()
    assert "Question:" not in panel._body.text()

    panel.set_statistic("pair_correlation_g_r", panel=_curve_panel())

    assert panel.has_plot is True
    assert panel.findChildren(AdStatPlotWidget)[0].panel_kind == "curve"
    assert "How to read this plot" not in panel._body.text()

    panel.set_statistic(
        "pair_correlation_g_r",
        panel=_curve_panel(),
        curve_mode="observed_only",
    )

    assert panel.findChildren(AdStatPlotWidget)[0].curve_mode == "observed_only"
    assert "model appears in the next step" in panel._body.text()

    directional = _curve_panel()
    directional.statistic = "pair_correlation_g_r_theta"
    panel.set_statistic("pair_correlation_g_r_theta", panel=directional)
    assert "distance-angle" in panel._title.text()
    assert "θ = pair direction" in panel._annotation.text()

    psi6 = _curve_panel()
    psi6.statistic = "bond_order_psi6"
    panel.set_statistic("bond_order_psi6", panel=psi6)
    assert "triangular-like" in panel._title.text()
    assert "0 = random-like" in panel._annotation.text()

    panel.close()
    panel.deleteLater()


def test_particle_statistics_layer_toggles_are_display_only(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("point_pattern")
    original_count = dlg.field_point_count
    dlg._observed_layer_cb.setChecked(False)

    assert dlg.field_point_count == original_count
    assert dlg._field.layer_visibility["observed"] is False
    assert "supported plots" in dlg._layer_hint_lbl.text()
    assert "still used for comparison" in dlg._layer_hint_lbl.text()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_layer_toggles_update_focus_plot_curves(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="sandbox")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg._sandbox_state.stage(n=20, n_simulations=2)
    state = dlg._sandbox_state
    state.run()
    dlg._on_generated_worker_finished(dlg._sandbox_generation, state, "")

    plot = dlg._focus_panel.findChildren(AdStatPlotWidget)[0]
    assert plot.show_observed_curve is True
    assert plot.show_model_curves is True
    assert dlg._result_view.tab_count >= 1

    dlg._observed_layer_cb.setChecked(False)
    plot = dlg._focus_panel.findChildren(AdStatPlotWidget)[0]
    assert plot.show_observed_curve is False
    assert plot.show_model_curves is True
    assert dlg._result_view.tab_count >= 1
    assert "supported plots" in dlg._layer_hint_lbl.text()

    dlg._simulation_layer_cb.setChecked(False)
    plot = dlg._focus_panel.findChildren(AdStatPlotWidget)[0]
    assert plot.show_observed_curve is False
    assert plot.show_model_curves is False

    dlg._observed_layer_cb.setChecked(True)
    plot = dlg._focus_panel.findChildren(AdStatPlotWidget)[0]
    assert plot.show_observed_curve is True
    assert plot.show_model_curves is False

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_tutorial_loads_examples_and_layer_steps(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("feature_biased")

    assert dlg.current_tutorial_key == "feature_biased"
    assert dlg._pattern_cb.currentData() == "feature_biased"
    assert dlg._generated_model_cb.currentData() == "measured_feature_poisson"
    assert dlg._field.layer_visibility["observed"] is False
    assert dlg._field.layer_visibility["features"] is True

    dlg.next_tutorial_step()

    assert dlg.current_tutorial_step == 1
    assert dlg._field.layer_visibility["observed"] is True

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_tutorial_marks_next_action_green(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("image_to_statistic")

    assert "#2fb344" in dlg._next_tutorial_btn.styleSheet()
    assert dlg._run_tutorial_btn.styleSheet() == ""
    assert dlg._field.layer_visibility["observed"] is True
    assert dlg._field.layer_visibility["simulated"] is True
    # Navigation names its destination and points forward (not the call-to-action).
    assert dlg._next_tutorial_btn.text().strip().endswith("▸")
    assert dlg._next_tutorial_btn.isEnabled()
    assert "Look for:" in dlg._tutorial_step_lbl.text()
    assert not dlg._tutorial_why_frame.isVisibleTo(dlg)

    dlg.next_tutorial_step()

    assert dlg.current_tutorial_key == "simulation_envelope"
    assert "#2fb344" in dlg._next_tutorial_btn.styleSheet()
    assert dlg._next_tutorial_btn.text().strip().endswith("▸")
    assert dlg._field.layer_visibility["observed"] is True
    assert dlg._field.layer_visibility["simulated"] is True

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_first_example_is_staged_field_lesson(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    assert dlg.current_tutorial_key == "welcome"
    assert dlg._current_tutorial_step_obj().visible_panel == "field"
    assert dlg.focus_has_plot is False
    assert dlg._tabs.isHidden()
    assert dlg._focus_panel.isHidden()
    assert dlg._result_view.technical_details_visible is False

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_tutorial_reveals_single_panel_at_a_time(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("image_to_statistic")
    assert dlg._tabs.isHidden()
    assert not dlg._focus_panel.isHidden()
    assert dlg._generated_data_group.isHidden()
    assert dlg._generated_model_group.isHidden()
    assert dlg._result_view.technical_details_visible is False

    dlg.load_tutorial_example("verdict")
    assert not dlg._tabs.isHidden()
    assert dlg._tabs.tabText(dlg._tabs.currentIndex()) == "Results"
    assert dlg._tabs.tabBar().isHidden()
    assert not dlg._focus_panel.isHidden()
    assert dlg._result_view.technical_details_visible is False

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_results_tab_has_no_duplicate_plots(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    # Run a comparison so verdict rows exist, then check the embedded Results view
    # only exposes the verdict summary + technical details (no per-statistic plots).
    dlg.load_tutorial_example("simulation_envelope")
    dlg._sandbox_state.stage(n=40, n_simulations=4)
    state = dlg._sandbox_state
    state.run()
    dlg._on_generated_worker_finished(dlg._sandbox_generation, state, "")

    titles = set(dlg._result_view.tab_titles)
    assert titles <= {"Summary", "Technical details"}
    assert "Summary" in titles
    assert "Technical details" not in titles
    assert not any("pair" in t.lower() or "pattern" in t.lower() for t in titles)

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_tutorial_is_available_from_workflow_menu(qapp):
    dlg = ParticleStatisticsDialog(
        point_sources=[_point_source()],
        scan=SimpleNamespace(scan_range_m=(8e-9, 6e-9), dims=(8, 6)),
        image_shape=(6, 8),
        initial_mode="real",
    )
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    assert dlg.current_mode == "real"
    assert dlg._start_tutorial_btn.isHidden()
    assert dlg._start_tutorial_action.isEnabled()

    dlg._start_tutorial_action.trigger()

    assert dlg.current_mode == "learn"
    assert dlg.current_tutorial_key == "welcome"
    assert dlg.current_tutorial_step == 0
    assert not dlg._start_tutorial_action.isEnabled()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_exit_and_restart_tutorial(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("clustered")
    assert dlg.current_mode == "learn"

    dlg.exit_tutorial()
    assert dlg.current_mode == "real"
    assert not dlg._tabs.isHidden()
    assert not dlg._tabs.tabBar().isHidden()
    assert not dlg._info_panel.isHidden()
    assert not dlg._layer_group.isHidden()
    assert dlg._result_view.technical_details_visible is True

    dlg.restart_tutorial()
    assert dlg.current_mode == "learn"
    assert dlg.current_tutorial_key == "welcome"
    assert dlg.current_tutorial_step == 0

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_exit_clears_and_ignores_late_tutorial_result(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    # Produce a generated result, then exit to real mode.
    dlg._sandbox_state.stage(n=20, n_simulations=2)
    state = dlg._sandbox_state
    state.run()
    dlg._on_generated_worker_finished(dlg._sandbox_generation, state, "")
    assert dlg.focus_has_plot is True

    stale_generation = dlg._sandbox_generation
    dlg.exit_tutorial()

    assert dlg.current_mode == "real"
    assert dlg._sandbox_generation != stale_generation  # in-flight run invalidated
    assert dlg._result_view.data_mode == "real"
    assert dlg.focus_has_plot is False

    # A late tutorial worker (old or current generation) must not repopulate real view.
    dlg._on_generated_worker_finished(stale_generation, state, "")
    dlg._on_generated_worker_finished(dlg._sandbox_generation, state, "")
    assert dlg._result_view.data_mode == "real"
    assert dlg.focus_has_plot is False

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_clear_real_view(qapp):
    dlg = ParticleStatisticsDialog(
        point_sources=[_point_source()],
        scan=SimpleNamespace(scan_range_m=(8e-9, 6e-9), dims=(8, 6)),
        image_shape=(6, 8),
        initial_mode="real",
    )

    dlg.clear_real_view()
    assert dlg.focus_has_plot is False
    assert dlg._result_view.data_mode == "real"

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_statistic_buttons_are_compact_and_selectable(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    pair_button = dlg._statistic_buttons["pair_correlation_g_r"]
    neighbor_button = dlg._statistic_buttons["nearest_neighbor_distribution"]
    directional_button = dlg._statistic_buttons["pair_correlation_g_r_theta"]
    psi6_button = dlg._statistic_buttons["bond_order_psi6"]

    # Buttons carry only the statistic name (no multi-line description baked in).
    assert pair_button.text() == "Pair correlation"
    assert directional_button.text() == "Pair distance-angle map"
    assert psi6_button.text() == "ψ6 triangular order"
    assert "\n" not in pair_button.text()
    assert pair_button.isCheckable()
    assert any(
        "General spatial pattern" in label.text()
        for label in dlg._statistic_group_labels
    )
    assert any(
        "Local order" in label.text()
        for label in dlg._statistic_group_labels
    )
    # The default-focused statistic reads as selected.
    assert pair_button.isChecked()
    assert pair_button.styleSheet() != ""
    assert neighbor_button.styleSheet() == ""
    assert "Selected: Pair correlation" in dlg._selected_statistic_help_lbl.text()
    assert all(label.isHidden() for label in dlg._statistic_description_labels.values())

    # Local-order statistics are opt-in: their cards are disabled until enabled.
    assert not directional_button.isEnabled()
    dlg._include_ordering_cb.setChecked(True)
    assert directional_button.isEnabled()
    directional_button.click()
    assert "Selected: Pair distance-angle map" in dlg._selected_statistic_help_lbl.text()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_tutorial_focuses_visible_statistic_card(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("image_to_statistic")
    tutorial_text = dlg._tutorial_step_lbl.text()

    assert dlg.focused_statistic == "pair_correlation_g_r"
    assert dlg.focus_has_plot is False
    assert not dlg._focus_panel.isHidden()
    assert "Pair correlation" in dlg._focus_panel._title.text()
    assert "Pair correlation" not in tutorial_text or "Pair correlation" in dlg._focus_panel._title.text()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_generated_run_keeps_tutorial_focus(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.next_tutorial_step()
    dlg._statistic_buttons["nearest_neighbor_distribution"].click()
    assert dlg.focused_statistic == "nearest_neighbor_distribution"

    dlg._sandbox_state.stage(n=20, n_simulations=2)
    state = dlg._sandbox_state
    state.run()
    dlg._on_generated_worker_finished(dlg._sandbox_generation, state, "")

    assert dlg.focused_statistic == "pair_correlation_g_r"
    assert "Pair correlation" in dlg._focus_panel._title.text()

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_tutorial_run_shows_actual_focus_plot(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("image_to_statistic")

    # Simulate the comparison that fires when the plotting lesson is shown.
    dlg._sandbox_state.stage(n=20, n_simulations=2)
    state = dlg._sandbox_state
    state.run()
    dlg._on_generated_worker_finished(dlg._sandbox_generation, state, "")

    # The statistic lesson shows a real plot (observed-only emphasis), not a concept card.
    assert dlg.current_tutorial_step == 0
    assert dlg.focused_statistic == "pair_correlation_g_r"
    assert dlg.focus_has_plot is True
    assert dlg._result_view.tab_count >= 1
    assert dlg._focus_panel.findChildren(AdStatPlotWidget)[0].curve_mode == "observed_only"

    dlg.next_tutorial_step()

    # Stepping adds the model envelope to the same live chart; no recompute.
    before_result = dlg._sandbox_state.result
    assert dlg.current_tutorial_key == "simulation_envelope"
    assert dlg.current_tutorial_step == 0
    assert dlg._tabs.tabText(dlg._tabs.currentIndex()) == "Setup"
    assert dlg._field.layer_visibility["simulated"] is True
    assert dlg.focus_has_plot is True
    assert dlg._focus_panel.findChildren(AdStatPlotWidget)[0].curve_mode == "comparison"

    dlg._statistic_buttons["ripley_l_function"].click()

    assert dlg.focused_statistic == "ripley_l_function"
    assert dlg.focus_has_plot is True
    assert dlg._sandbox_state.result is before_result

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_tutorial_green_action_queues_next_example(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("verdict")

    assert dlg.current_tutorial_key == "verdict"
    assert dlg.current_tutorial_step == 0
    assert "#2fb344" in dlg._next_tutorial_btn.styleSheet()
    assert dlg._next_tutorial_btn.text().strip().endswith("▸")

    dlg.next_tutorial_step()

    assert dlg.current_tutorial_key == "homogeneous_poisson"
    assert dlg.current_tutorial_step == 0
    assert dlg._pattern_cb.currentData() == "random"

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_feature_biased_tutorial_avoids_unimplemented_statistic(qapp):
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.load_tutorial_example("feature_biased")
    seen = []
    for _index in range(4):
        seen.append(dlg._tutorial_step_lbl.text().lower())
        dlg.next_tutorial_step()

    assert "feature-distance" not in "\n".join(seen)
    assert "feature distance" not in "\n".join(seen)

    dlg.close()
    dlg.deleteLater()


def test_adstat_workbench_wrapper_opens_particle_statistics(qapp):
    dlg = AdStatWorkbenchDialog(initial_mode="learn")

    assert dlg.windowTitle() == "Particle Statistics"
    assert dlg.current_mode == "learn"

    dlg.close()
    dlg.deleteLater()


def test_particle_statistics_shared_store_add(qapp):
    """add_feature_sets writes to the shared store and shows in the list."""
    from probeflow.measurements.feature_sets import FeatureSet, FeatureSetStore

    store = FeatureSetStore()
    dlg = ParticleStatisticsDialog(feature_set_store=store, initial_mode="real")
    fs = FeatureSet.from_points(
        name="imported set",
        points_px=[[1, 2], [3, 4]],
        points_m=[[1e-9, 2e-9], [3e-9, 4e-9]],
        scan_range_m=(10e-9, 10e-9),
        image_shape=(10, 10),
    )
    dlg.add_feature_sets([fs])
    assert len(store) == 1
    texts = [
        dlg._feature_sets_list.item(i).text()
        for i in range(dlg._feature_sets_list.count())
    ]
    assert any("imported set" in t for t in texts)


def test_particle_statistics_shared_store_visible_across_dialogs(qapp):
    """Two dialogs sharing one store both see a set added through either."""
    from probeflow.measurements.feature_sets import FeatureSet, FeatureSetStore

    store = FeatureSetStore()
    dlg1 = ParticleStatisticsDialog(feature_set_store=store, initial_mode="real")
    dlg2 = ParticleStatisticsDialog(feature_set_store=store, initial_mode="real")
    fs = FeatureSet.from_points(
        name="shared set",
        points_px=[[1, 2]],
        points_m=[[1e-9, 2e-9]],
        scan_range_m=(10e-9, 10e-9),
        image_shape=(10, 10),
    )
    dlg1.add_feature_sets([fs])
    dlg2._populate_feature_sets()  # resync from the shared store
    texts = [
        dlg2._feature_sets_list.item(i).text()
        for i in range(dlg2._feature_sets_list.count())
    ]
    assert any("shared set" in t for t in texts)


def test_particle_statistics_import_csv_from_disk(qapp, tmp_path, monkeypatch):
    """The import button loads a CSV into the store via the calibration dialog."""
    from PySide6.QtWidgets import QDialog, QFileDialog

    from probeflow.measurements.feature_sets import FeatureSetStore

    csv = tmp_path / "pts.csv"
    csv.write_text("x_nm,y_nm\n5,10\n15,20\n", encoding="utf-8")

    store = FeatureSetStore()
    dlg = ParticleStatisticsDialog(feature_set_store=store, initial_mode="real")

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(csv), ""))
    )

    class _FakeImportDialog:
        def __init__(self, preview, **kwargs):
            self._preview = preview

        def exec(self):
            return QDialog.Accepted

        def result_calibration(self):
            return "nm", (100e-9, 100e-9), (100, 100)

    monkeypatch.setattr(
        "probeflow.gui.dialogs.import_points.ImportPointsDialog", _FakeImportDialog
    )

    dlg.import_points_from_disk()
    assert len(store) == 1
    assert store.all()[0].point_count == 2


def test_particle_statistics_export_csv(qapp, tmp_path, monkeypatch):
    """The Export menu writes per-statistic CSVs from the current result."""
    from types import SimpleNamespace

    from PySide6.QtWidgets import QFileDialog

    from probeflow.analysis.adstat_adapter import compare_point_source_view_spec

    dlg = ParticleStatisticsDialog(initial_mode="real")
    rng = np.random.default_rng(0)
    pts_nm = rng.uniform(5.0, 95.0, size=(30, 2))
    source = SimpleNamespace(
        label="run", source_type="s",
        points_px=pts_nm / (100.0 / 256.0), points_m=pts_nm * 1e-9, metadata={},
    )
    scan = SimpleNamespace(scan_range_m=(100e-9, 100e-9), dims=(256, 256))
    dlg._last_view_spec = compare_point_source_view_spec(
        source, scan=scan, image_shape=(256, 256), n_simulations=4, random_seed=0
    )

    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path))
    )
    dlg._export_results_csv()
    files = list(tmp_path.glob("*.csv"))
    assert files
    assert any("verdicts" in f.name for f in files)


def test_particle_statistics_export_guarded_without_result(qapp):
    """Exporting before any run is a no-op with a status message, not a crash."""
    dlg = ParticleStatisticsDialog(initial_mode="real")
    dlg._export_results_csv()  # no result yet → should not raise
    dlg._export_results_json()


def test_particle_statistics_ordering_is_opt_in(qapp):
    """Local-order checks are off by default; the checkbox enables them + the cards."""
    dlg = ParticleStatisticsDialog(initial_mode="real")
    assert dlg._include_ordering_enabled() is False
    assert dlg._request_from_controls().include_ordering is False
    for button in dlg._ordering_stat_buttons.values():
        assert not button.isEnabled()

    dlg._include_ordering_cb.setChecked(True)
    assert dlg._request_from_controls().include_ordering is True
    for button in dlg._ordering_stat_buttons.values():
        assert button.isEnabled()


def test_particle_statistics_tutorial_navigation_never_dead_ends(qapp):
    """Previous/Next are always present, name a destination, and never strand the user."""
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    dlg.start_tutorial()
    # Both navigation buttons are present (not hidden) with the big title + progress.
    assert not dlg._prev_tutorial_btn.isHidden()
    assert not dlg._next_tutorial_btn.isHidden()
    assert dlg._tutorial_title_lbl.text()
    assert "Lesson 1 of" in dlg._tutorial_progress_lbl.text()
    # First lesson: Previous disabled, Next enabled and forward-pointing.
    assert not dlg._prev_tutorial_btn.isEnabled()
    assert dlg._next_tutorial_btn.isEnabled()
    assert dlg._next_tutorial_btn.text().strip().endswith("▸")

    total = dlg._tutorial_position()[1]
    # Walk forward to the end; Next must stay enabled until the final lesson, so a
    # "run" step (e.g. the hard-core Generate-points step) can never trap the user.
    guard = 0
    while dlg._next_tutorial_btn.isEnabled() and guard <= total + 2:
        dlg.next_tutorial_step()
        guard += 1
    pos, tot = dlg._tutorial_position()
    assert pos == tot  # reached the last lesson
    assert not dlg._next_tutorial_btn.isEnabled()
    assert dlg._prev_tutorial_btn.isEnabled()

    # Walk all the way back to the first lesson.
    guard = 0
    while dlg._prev_tutorial_btn.isEnabled() and guard <= tot + 2:
        dlg.previous_tutorial_step()
        guard += 1
    assert dlg._tutorial_position()[0] == 1
    dlg.close()


def test_exit_tutorial_and_landing_keep_dialog_in_front(qapp):
    """Exiting the tutorial (or returning to Workflows) re-raises Particle Statistics."""
    dlg = ParticleStatisticsDialog(initial_mode="learn")
    if dlg._sandbox_state is None:
        pytest.skip("AdStat sandbox is not installed")

    calls: list[str] = []
    dlg._raise_self = lambda: calls.append("raise")  # type: ignore[method-assign]

    dlg.exit_tutorial()
    assert calls, "exit_tutorial should re-raise the dialog to the front"
    assert dlg.current_mode == "real"

    calls.clear()
    dlg.return_to_landing_page()
    assert calls, "return_to_landing_page should re-raise the dialog to the front"
    assert dlg.current_mode == "landing"
    dlg.close()
