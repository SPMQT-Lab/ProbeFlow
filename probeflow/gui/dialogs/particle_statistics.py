"""ProbeFlow-native Particle Statistics tool powered by AdStat."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenuBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from probeflow.analysis.adstat_adapter import (
    ORDERING_STATISTICS,
    adstat_sandbox_context,
    adstat_sandbox_preview,
    adstat_sandbox_state,
    adstat_sandbox_view_spec,
    compare_point_set_record_view_spec,
)
from probeflow.gui.dialogs.adstat_results import AdStatPlotWidget, AdStatResultView
from probeflow.gui.config import load_config, save_config
from probeflow.gui.desktop_layout import (
    apply_screen_fraction_geometry,
    qbytearray_to_b64,
    restore_geometry_or_default,
)
from probeflow.gui.viewer.tool_launch import (
    AdStatStatisticsRequest,
    adstat_workbench_launch_context,
)
from probeflow.gui.workers import _PooledWorker


_PATTERN_LABELS = {
    "random": "Random",
    "clustered": "Clustered",
    "no_overlap": "No overlap / hard-core",
    "ordered_islands": "Ordered islands / lattice chunks",
    "feature_biased": "Feature-biased",
}

_MODEL_LABELS = {
    "homogeneous_poisson": "Homogeneous Poisson",
    "hard_core_random": "Hard-core random",
    "measured_feature_poisson": "Measured-feature Poisson",
    "poisson": "Homogeneous Poisson",
}
_ORDERED_ISLAND_LATTICE_LABELS = {
    "triangular": "Triangular / hexagonal-like",
    "square": "Square-like",
}
_ORDERED_ISLAND_BACKGROUND_LABELS = {
    "none": "None",
    "random": "Random particles",
    "clustered": "Disordered clusters",
}

_SETUP_COLUMN_STYLE = """
QFrame#particleStatisticsDataColumn {
    border: 1px solid rgba(242, 142, 43, 0.90);
    border-radius: 6px;
    background: rgba(242, 142, 43, 0.05);
}
QFrame#particleStatisticsModelColumn {
    border: 1px solid rgba(186, 85, 211, 0.90);
    border-radius: 6px;
    background: rgba(186, 85, 211, 0.05);
}
QFrame#particleStatisticsStatisticColumn {
    border: 1px solid rgba(47, 129, 247, 0.90);
    border-radius: 6px;
    background: rgba(47, 129, 247, 0.05);
}
"""

_TUTORIAL_ACTION_STYLE = (
    "QPushButton { "
    "background-color: #2fb344; color: #071b0b; border: 2px solid #83e89b; "
    "font-weight: 800; padding: 5px 12px; } "
    "QPushButton:hover { background-color: #39c956; } "
    "QPushButton:pressed { background-color: #238636; color: #ffffff; } "
    "QPushButton:disabled { background-color: #25352a; color: #8aa891; "
    "border: 1px solid #3e5c46; }"
)
_TUTORIAL_EXIT_STYLE = (
    "QPushButton { "
    "background-color: #b3382f; color: #ffffff; border: 1px solid #e8675c; "
    "font-weight: 700; padding: 5px 12px; } "
    "QPushButton:hover { background-color: #c9433a; } "
    "QPushButton:pressed { background-color: #8f2a23; }"
)
_TUTORIAL_ACTIVE_CONTROL_STYLE = (
    "QWidget { background-color: rgba(47, 179, 68, 0.22); "
    "border: 2px solid #2fb344; }"
)
_TUTORIAL_VISITED_CONTROL_STYLE = (
    "QWidget { background-color: rgba(224, 176, 32, 0.18); "
    "border: 1px solid #e0b020; }"
)
# Statistic selector buttons look like normal ProbeFlow buttons; the active one
# (whose plot is currently shown) gets a blue highlight.
_STAT_SELECTED_STYLE = (
    "QPushButton { border: 2px solid #2f81f7; background: #243044; font-weight: 700; }"
)
_REAL_EMPTY_STATE_MESSAGE = (
    "No points yet — detect features with Feature Finder and 'Send to Particle "
    "Statistics', tick a saved feature set, or click Start tutorial to learn the tool."
)

_DEFAULT_FOCUS_STATISTIC = "pair_correlation_g_r"
_MODEL_SUMMARY_FOCUS = "model_summary"
_STATISTIC_GROUPS = (
    (
        "General spatial pattern",
        (
            "pair_correlation_g_r",
            "nearest_neighbor_distribution",
            "ripley_l_function",
            "cluster_size_counts",
        ),
    ),
    (
        "Local order / ordered islands",
        (
            "pair_correlation_g_r_theta",
            "bond_order_psi6",
            "bond_order_psi4",
        ),
    ),
)
_STATISTIC_ORDER = tuple(
    statistic for _group, statistics in _STATISTIC_GROUPS for statistic in statistics
)
_STATISTIC_LABELS = {
    "pair_correlation_g_r": "Pair correlation",
    "pair_correlation_g_r_theta": "Pair distance-angle map",
    "bond_order_psi6": "ψ6 triangular order",
    "bond_order_psi4": "ψ4 square order",
    "nearest_neighbor_distribution": "Nearest neighbors",
    "ripley_l_function": "Ripley L",
    "cluster_size_counts": "Cluster sizes",
    _MODEL_SUMMARY_FOCUS: "Model verdict summary",
}
_STATISTIC_TITLES = {
    "pair_correlation_g_r": "Pair correlation g(r)",
    "pair_correlation_g_r_theta": "Pair distance-angle map",
    "bond_order_psi6": "ψ6 local order - triangular-like neighborhoods",
    "bond_order_psi4": "ψ4 local order - square-like neighborhoods",
    "nearest_neighbor_distribution": "Nearest-neighbor distances",
    "ripley_l_function": "Ripley L",
    "cluster_size_counts": "Cluster sizes",
    _MODEL_SUMMARY_FOCUS: "Model verdict summary",
}
_FALLBACK_STAT_GUIDES = {
    "pair_correlation_g_r": {
        "title": "Pair correlation g(r)",
        "focus_question": "At each distance, are there too many or too few pairs?",
        "before_run": "Computing the observed curve against a simulated envelope…",
        "how_to_read": "Above the envelope means more pairs than expected; below means fewer.",
    },
    "nearest_neighbor_distribution": {
        "title": "Nearest-neighbor distances",
        "focus_question": "How far is each point from its closest neighbor?",
        "before_run": "Computing the closest-neighbor distribution…",
        "how_to_read": "Small distances indicate close pairs; larger distances indicate separation.",
    },
    "ripley_l_function": {
        "title": "Ripley L",
        "focus_question": "Does structure accumulate across increasing distance?",
        "before_run": "Computing cumulative neighbor structure…",
        "how_to_read": "Above the envelope suggests cumulative clustering; below suggests depletion.",
    },
    "cluster_size_counts": {
        "title": "Cluster sizes",
        "focus_question": "How many isolated points, pairs, and larger groups exist?",
        "before_run": "Counting groups against the selected model…",
        "how_to_read": "More large groups than expected suggests clustering.",
    },
    "pair_correlation_g_r_theta": {
        "title": "Pair distance-angle map",
        "focus_question": "Do pairs repeat at particular distances and directions?",
        "before_run": "Computing directional pair density…",
        "how_to_read": "Repeated angular features suggest preferred neighbor directions.",
    },
    "bond_order_psi6": {
        "title": "ψ6 local order - triangular-like neighborhoods",
        "focus_question": "Do local neighbors look triangular or hexagonal-like?",
        "before_run": "Computing sixfold local bond-order values…",
        "how_to_read": "More particles near |ψ6| = 1 suggests triangular-like local order.",
    },
    "bond_order_psi4": {
        "title": "ψ4 local order - square-like neighborhoods",
        "focus_question": "Do local neighbors look square-like?",
        "before_run": "Computing fourfold local bond-order values…",
        "how_to_read": "More particles near |ψ4| = 1 suggests square-like local order.",
    },
    _MODEL_SUMMARY_FOCUS: {
        "title": "Model verdict summary",
        "focus_question": "Which model assumptions remain plausible after the comparison?",
        "before_run": "Comparing verdicts for random, no-overlap, and feature-biased models…",
        "how_to_read": "Read model verdicts as consistency checks, not mechanism proof.",
    },
}
_SHORT_STAT_READS = {
    "pair_correlation_g_r": "Orange = observed data; blue = model simulations. Above band = more pairs.",
    "pair_correlation_g_r_theta": "Radial g(r) collapses direction; this keeps distance and angle.",
    "bond_order_psi6": "Values near 1 mean strong sixfold-like local angles.",
    "bond_order_psi4": "Values near 1 mean strong fourfold-like local angles.",
    "nearest_neighbor_distribution": "Left shift = closer neighbors; right shift = more spacing.",
    "ripley_l_function": "Above band = accumulated clustering; below band = depletion.",
    "cluster_size_counts": "More large groups = more clustering.",
    _MODEL_SUMMARY_FOCUS: "Read consistency by model, not as mechanism proof.",
}
_STATISTIC_ANNOTATIONS = {
    "pair_correlation_g_r_theta": (
        "r = pair distance; θ = pair direction; colour = relative pair density."
    ),
    "bond_order_psi6": (
        "0 = random-like local angles; 1 = strong sixfold-like local angles."
    ),
    "bond_order_psi4": (
        "0 = random-like local angles; 1 = strong fourfold-like local angles."
    ),
}
_PARTICLE_STATISTICS_LAYOUT_KEY = "particle_statistics"


@dataclass(frozen=True)
class ParticleTutorialStep:
    title: str
    body: str = ""
    question: str = ""
    look_for: str = ""
    mode: str = "generated"
    visible_panel: str = "field"
    visible_controls: tuple[str, ...] = ()
    primary_action: str = "Next"
    action_kind: str = "next"
    model_label: str = ""
    statistic_label: str = ""
    caution: str = ""
    more_detail: str = ""
    target_tab: str = "Data"
    controls: tuple[str, ...] = ()
    focus_statistic: str = _DEFAULT_FOCUS_STATISTIC
    focus_curve_mode: str = "comparison"
    curve_mode: str = ""
    action_button: str = "next"
    action_text: str = ""
    advance_after_run: bool = False
    compute_on_show: bool = False
    show_technical_details: bool = False
    direct_labels: tuple[str, ...] = ()
    pattern: str | None = None
    model: str | None = None
    n: int | None = None
    seed: int | None = None
    simulations: int | None = None
    width_nm: float | None = None
    height_nm: float | None = None
    hard_core_radius_nm: float | None = None
    model_hard_core_radius_nm: float | None = None
    ordered_island_lattice: str | None = None
    ordered_island_background: str | None = None
    show_observed: bool = True
    show_simulated: bool = True
    show_features: bool = True
    show_region: bool = True
    pool_images: int = 0
    intro_card: bool = False
    intro_region: str = ""  # which central panel this card introduces: field|plot|info
    intro_panel_text: str = ""  # the one-line label shown in that panel
    what_changes: str = ""
    expected_effect: str = ""
    where_to_check: str = ""
    why: str = ""
    statistic_hint: str = ""
    limitation: str = ""


@dataclass(frozen=True)
class ParticleTutorialExample:
    key: str
    title: str
    summary: str
    steps: tuple[ParticleTutorialStep, ...]


_TUTORIALS: tuple[ParticleTutorialExample, ...] = (
    ParticleTutorialExample(
        key="welcome",
        title="00 - Welcome",
        summary="What Particle Statistics attempts to solve.",
        steps=(
            ParticleTutorialStep(
                title="Welcome",
                question="Particle Statistics tests whether particle positions are consistent with simple spatial models.",
                look_for="Start with the point field: the particles are the data.",
                pattern="random",
                model="homogeneous_poisson",
                n=80,
                seed=7,
                simulations=40,
                show_simulated=False,
                show_features=False,
                curve_mode="observed_only",
                direct_labels=("observed particles",),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="point_pattern",
        title="01 - Observed point pattern",
        summary="Particles become calibrated x,y data.",
        steps=(
            ParticleTutorialStep(
                title="Observed point pattern",
                question="The image is reduced to calibrated x,y particle positions.",
                look_for="The observed particles are the only layer shown.",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=7,
                simulations=60,
                show_simulated=False,
                show_features=False,
                curve_mode="observed_only",
                direct_labels=("observed particles",),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="model_baseline_observed",
        title="02 - Model baseline A",
        summary="Observed particles only.",
        steps=(
            ParticleTutorialStep(
                title="Observed data",
                question="These are the observed particle positions. They are the data we want to test.",
                look_for="Only observed particles are visible.",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=7,
                simulations=60,
                show_simulated=False,
                show_features=False,
                direct_labels=("observed particles",),
                primary_action="Show a random model",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="model_baseline_model",
        title="03 - Model baseline B",
        summary="One simulated model layout.",
        steps=(
            ParticleTutorialStep(
                title="Model simulation",
                question="This is one simulated random layout from the model, not the data.",
                look_for="Only the model simulation is visible.",
                model_label="Homogeneous Poisson",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=7,
                simulations=60,
                show_observed=False,
                show_simulated=True,
                show_features=False,
                direct_labels=("model simulation",),
                primary_action="Overlay observed particles",
                more_detail="One simulated layout is not enough for a verdict. It only shows what the model can generate.",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="model_baseline_overlay",
        title="04 - Model baseline C",
        summary="Observed particles and one model simulation together.",
        steps=(
            ParticleTutorialStep(
                title="Overlay",
                question="Observed particles and one model simulation are shown together.",
                look_for="Visual comparison helps orientation, but it is not the final test.",
                model_label="Homogeneous Poisson",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=7,
                simulations=60,
                show_simulated=True,
                show_features=False,
                direct_labels=("observed data", "model simulation"),
                primary_action="Show the statistic",
                more_detail="Our eye may see patterns, but the statistical curve compares the data to many model simulations.",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="image_to_statistic",
        title="05 - From image to statistic",
        summary="Visual comparison is not enough.",
        steps=(
            ParticleTutorialStep(
                title="From image to statistic",
                question="The spatial view is only the starting point.",
                look_for="Pair correlation turns positions into a measurable comparison.",
                model_label="Homogeneous Poisson",
                statistic_label="Pair correlation g(r)",
                visible_panel="plot",
                target_tab="Setup",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=7,
                simulations=60,
                show_simulated=True,
                show_features=False,
                focus_statistic="pair_correlation_g_r",
                curve_mode="observed_only",
                compute_on_show=True,
                primary_action="Show model band",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="simulation_envelope",
        title="06 - Simulation envelope",
        summary="Orange observed statistic versus blue model band.",
        steps=(
            ParticleTutorialStep(
                title="Simulation envelope",
                question="Many model layouts form the expected blue band.",
                look_for="Orange is observed data; blue is the model envelope.",
                model_label="Homogeneous Poisson",
                statistic_label="Pair correlation g(r)",
                visible_panel="plot",
                target_tab="Setup",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=7,
                simulations=60,
                show_simulated=True,
                show_features=False,
                focus_statistic="pair_correlation_g_r",
                curve_mode="comparison",
                compute_on_show=True,
                primary_action="Read the verdict",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="verdict",
        title="07 - Verdict language",
        summary="Consistent or inconsistent with this model.",
        steps=(
            ParticleTutorialStep(
                title="Verdict language",
                question="Inside the band means consistent with this model.",
                look_for="Leaving the band means this statistic rules out the model.",
                model_label="Homogeneous Poisson",
                statistic_label="Pair correlation g(r)",
                visible_panel="results",
                target_tab="Results",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=7,
                simulations=60,
                show_simulated=True,
                show_features=False,
                focus_statistic=_MODEL_SUMMARY_FOCUS,
                compute_on_show=True,
                primary_action="Explain Poisson model",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="homogeneous_poisson",
        title="08 - Homogeneous Poisson",
        summary="Random placement with one average density.",
        steps=(
            ParticleTutorialStep(
                title="Homogeneous Poisson",
                question="Homogeneous Poisson means independent random placement with one average density.",
                look_for="If appropriate, g(r) fluctuates around 1 and stays inside the band.",
                model_label="Homogeneous Poisson",
                statistic_label="Pair correlation g(r)",
                caution="Not inconsistent with this model does not prove there is no interaction.",
                visible_panel="plot",
                target_tab="Setup",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=7,
                simulations=60,
                show_simulated=True,
                show_features=False,
                focus_statistic="pair_correlation_g_r",
                curve_mode="comparison",
                compute_on_show=True,
                primary_action="Try clustered pattern",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="clustered",
        title="09 - Clustered pattern",
        summary="Too many close pairs.",
        steps=(
            ParticleTutorialStep(
                title="Clustered pattern",
                question="Clustered points create too many close particle pairs.",
                look_for="A small-distance peak rises above the random-placement band.",
                model_label="Homogeneous Poisson",
                statistic_label="Pair correlation g(r)",
                caution="This shows spatial clustering, not the physical cause.",
                visible_panel="plot",
                target_tab="Setup",
                pattern="clustered",
                model="homogeneous_poisson",
                n=120,
                seed=8,
                simulations=60,
                show_simulated=True,
                show_features=False,
                focus_statistic="pair_correlation_g_r",
                curve_mode="comparison",
                compute_on_show=True,
                primary_action="Try ordered islands",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="hard_core_meaning",
        title="10 - Hard-core model meaning",
        summary="Minimum separation and no direct overlap.",
        steps=(
            ParticleTutorialStep(
                title="Hard-core meaning",
                question="A hard-core radius sets a minimum allowed separation.",
                look_for="Think average particle or molecule size: particles cannot overlap.",
                model_label="Hard-core random",
                visible_panel="controls",
                visible_controls=("model_hard_core_radius",),
                pattern="no_overlap",
                model="hard_core_random",
                n=100,
                seed=9,
                simulations=40,
                hard_core_radius_nm=1.5,
                model_hard_core_radius_nm=1.5,
                show_observed=False,
                show_simulated=True,
                show_features=False,
                direct_labels=("model simulation", "minimum separation"),
                primary_action="Try radius",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="hard_core_parameters",
        title="11 - Hard-core parameter sandbox",
        summary="Change radius and particle number; generate points.",
        steps=(
            ParticleTutorialStep(
                title="Try radius",
                question="Increase the hard-core radius to strengthen exclusion.",
                look_for="Larger radius removes more very close neighbours.",
                model_label="Hard-core random",
                visible_panel="controls",
                visible_controls=("model_hard_core_radius",),
                pattern="no_overlap",
                model="hard_core_random",
                n=100,
                seed=9,
                simulations=40,
                hard_core_radius_nm=3.0,
                model_hard_core_radius_nm=3.0,
                show_observed=False,
                show_simulated=True,
                show_features=False,
                primary_action="Generate points",
                action_kind="run",
                caution="Large N or hard-core radius can be slow because overlapping placements are rejected.",
                direct_labels=("model simulation", "larger radius"),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="hard_core_statistic",
        title="12 - Hard-core statistic",
        summary="Nearest-neighbor distances lose very short separations.",
        steps=(
            ParticleTutorialStep(
                title="Nearest neighbors",
                question="The nearest-neighbor plot should lose very short distances.",
                look_for="Close neighbours are forbidden, so the left side is depleted.",
                model_label="Hard-core random",
                statistic_label="Nearest-neighbor distance",
                visible_panel="plot",
                target_tab="Setup",
                pattern="no_overlap",
                model="hard_core_random",
                n=100,
                seed=9,
                simulations=60,
                hard_core_radius_nm=3.0,
                model_hard_core_radius_nm=3.0,
                show_simulated=True,
                show_features=False,
                focus_statistic="nearest_neighbor_distribution",
                curve_mode="comparison",
                compute_on_show=True,
                primary_action="Try local order",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_cluster_vs_order",
        title="13 - Order A",
        summary="Clustering is not the same as local order.",
        steps=(
            ParticleTutorialStep(
                title="Clustering is not order",
                question="Clustering means particles are near each other; order means neighbor positions repeat.",
                look_for="The island has a regular internal pattern, not just a dense group.",
                model_label="Homogeneous Poisson",
                statistic_label="Pair correlation g(r)",
                caution="This suggests local order; it does not prove a crystal or mechanism.",
                visible_panel="field",
                target_tab="Setup",
                pattern="ordered_islands",
                ordered_island_lattice="triangular",
                ordered_island_background="none",
                model="homogeneous_poisson",
                n=120,
                seed=12,
                simulations=60,
                show_simulated=False,
                show_features=False,
                visible_controls=("ordered_lattice",),
                direct_labels=("ordered islands",),
                primary_action="Show radial spacing",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_radial_spacing",
        title="14 - Order B",
        summary="Radial g(r) shows spacing but removes direction.",
        steps=(
            ParticleTutorialStep(
                title="Radial distances",
                question="Radial g(r) shows preferred neighbor distances, but removes direction.",
                look_for="Sharp peaks suggest repeated spacings, not full 2D order.",
                model_label="Homogeneous Poisson",
                statistic_label="Pair correlation g(r)",
                caution="Peaks in g(r) are useful, but they do not identify island symmetry.",
                visible_panel="plot",
                target_tab="Setup",
                pattern="ordered_islands",
                ordered_island_lattice="triangular",
                ordered_island_background="none",
                model="homogeneous_poisson",
                n=120,
                seed=12,
                simulations=60,
                show_simulated=True,
                show_features=False,
                focus_statistic="pair_correlation_g_r",
                curve_mode="comparison",
                compute_on_show=True,
                primary_action="Show directions",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_directional_pairs",
        title="15 - Order C",
        summary="Directional pair density keeps angle.",
        steps=(
            ParticleTutorialStep(
                title="Distance and angle",
                question="The pair distance-angle map keeps distance and direction together.",
                look_for="Repeated angular features appear at preferred neighbor distances.",
                model_label="Homogeneous Poisson",
                statistic_label="Pair distance-angle map",
                caution="Read directional structure against the matched null, not by eye alone.",
                more_detail="r is pair distance; theta is pair direction; colour is relative pair density.",
                visible_panel="plot",
                target_tab="Setup",
                pattern="ordered_islands",
                ordered_island_lattice="triangular",
                ordered_island_background="none",
                model="homogeneous_poisson",
                n=120,
                seed=12,
                simulations=60,
                show_simulated=True,
                show_features=False,
                focus_statistic="pair_correlation_g_r_theta",
                curve_mode="comparison",
                compute_on_show=True,
                primary_action="Show ψ6",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_bond_order",
        title="16 - Order D",
        summary="ψ6 asks whether each particle has triangular-like neighbors.",
        steps=(
            ParticleTutorialStep(
                title="ψ6 local order",
                question="ψ6 measures sixfold-like local order around each particle.",
                look_for="Values near 1 mean strong triangular-like neighbor geometry.",
                model_label="Homogeneous Poisson",
                statistic_label="ψ6 triangular order",
                caution="The value depends on the neighbor cutoff and detection quality.",
                more_detail="Nearby neighbor angles are checked for 60 degree repetition. Random patterns stay lower; triangular islands shift toward 1.",
                visible_panel="plot",
                target_tab="Setup",
                pattern="ordered_islands",
                ordered_island_lattice="triangular",
                ordered_island_background="none",
                model="homogeneous_poisson",
                n=120,
                seed=12,
                simulations=60,
                show_simulated=True,
                show_features=False,
                focus_statistic="bond_order_psi6",
                curve_mode="comparison",
                compute_on_show=True,
                primary_action="Show ψ4",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_square_order",
        title="17 - Order E",
        summary="ψ4 is the square-like local-order check.",
        steps=(
            ParticleTutorialStep(
                title="ψ4 local order",
                question="ψ4 measures square-like local order around each particle.",
                look_for="Square islands shift ψ4 toward 1 more than ψ6.",
                model_label="Homogeneous Poisson",
                statistic_label="ψ4 square order",
                caution="Choose the symmetry that matches the structure you want to test.",
                more_detail="Use ψ6 for triangular-like islands and ψ4 for square-like islands. This suggests local angular order; it is not a lattice fit.",
                visible_panel="plot",
                target_tab="Setup",
                pattern="ordered_islands",
                ordered_island_lattice="square",
                ordered_island_background="none",
                model="homogeneous_poisson",
                n=120,
                seed=14,
                simulations=60,
                show_simulated=True,
                show_features=False,
                focus_statistic="bond_order_psi4",
                curve_mode="comparison",
                compute_on_show=True,
                primary_action="Mix in disorder",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_mixed",
        title="18 - Order F",
        summary="Mixed ordered and disordered regions need local metrics.",
        steps=(
            ParticleTutorialStep(
                title="Mixed islands",
                question="Real images can mix ordered islands, disordered clusters, and isolated particles.",
                look_for="Local order can remain visible even when the field is mixed.",
                model_label="Homogeneous Poisson",
                statistic_label="ψ6 triangular order",
                caution="A single global metric can hide local structure; inspect the image too.",
                more_detail="Global plots average over the whole field. Local metrics help flag ordered regions inside mixed images.",
                visible_panel="plot",
                visible_controls=("ordered_background",),
                target_tab="Setup",
                pattern="ordered_islands",
                ordered_island_lattice="triangular",
                ordered_island_background="clustered",
                model="homogeneous_poisson",
                n=120,
                seed=13,
                simulations=60,
                show_simulated=True,
                show_features=False,
                focus_statistic="bond_order_psi6",
                curve_mode="comparison",
                compute_on_show=True,
                primary_action="Try feature-biased model",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="feature_biased",
        title="19 - Feature-biased model",
        summary="Association with an independently measured feature layer.",
        steps=(
            ParticleTutorialStep(
                title="Feature layer",
                question="Feature-biased models test association with an independent feature layer.",
                look_for="Inspect the feature layer before adding particles.",
                model_label="Measured-feature Poisson",
                caution="The feature layer must be independent of the particles being tested.",
                more_detail="Reusing particles as their own feature layer makes the result circular.",
                pattern="feature_biased",
                model="measured_feature_poisson",
                n=120,
                seed=10,
                simulations=60,
                show_observed=False,
                show_simulated=False,
                show_features=True,
                visible_controls=("layer_features",),
                direct_labels=("feature layer",),
                primary_action="Add particles",
            ),
            ParticleTutorialStep(
                title="Particles and features",
                question="Do particles sit preferentially near the independent features?",
                look_for="Compare orange particles with the feature layer.",
                model_label="Measured-feature Poisson",
                caution="If the same detections define both layers, the test is circular.",
                pattern="feature_biased",
                model="measured_feature_poisson",
                n=120,
                seed=10,
                simulations=60,
                show_simulated=False,
                show_features=True,
                visible_controls=("layer_observed",),
                direct_labels=("observed particles", "feature layer"),
                primary_action="Read feature verdict",
            ),
            ParticleTutorialStep(
                title="Feature-biased verdict",
                question="Which model assumption stays consistent for these particles?",
                look_for="Read the model verdict cards, not just the picture.",
                model_label="Measured-feature Poisson",
                statistic_label="Model verdict summary",
                caution="Independent feature measurement is required.",
                visible_panel="results",
                target_tab="Results",
                pattern="feature_biased",
                model="measured_feature_poisson",
                n=120,
                seed=10,
                simulations=60,
                show_simulated=True,
                show_features=True,
                focus_statistic=_MODEL_SUMMARY_FOCUS,
                compute_on_show=True,
                primary_action="Review statistics",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="other_statistics",
        title="20 - Statistics reference",
        summary="Four statistics answer different spatial questions.",
        steps=(
            ParticleTutorialStep(
                title="Nearest neighbors",
                question="How far is each particle from its closest neighbor?",
                look_for="Use this for overlap, exclusion, or close aggregation.",
                statistic_label="Nearest-neighbor distance",
                visible_panel="plot",
                target_tab="Setup",
                pattern="no_overlap",
                model="hard_core_random",
                n=100,
                seed=9,
                simulations=60,
                hard_core_radius_nm=3.0,
                model_hard_core_radius_nm=3.0,
                focus_statistic="nearest_neighbor_distribution",
                curve_mode="comparison",
                compute_on_show=True,
            ),
            ParticleTutorialStep(
                title="Ripley L",
                question="Does structure accumulate over increasing length scale?",
                look_for="Use Ripley L to see whether structure persists.",
                statistic_label="Ripley L",
                visible_panel="plot",
                target_tab="Setup",
                pattern="clustered",
                model="homogeneous_poisson",
                n=120,
                seed=8,
                simulations=60,
                focus_statistic="ripley_l_function",
                curve_mode="comparison",
                compute_on_show=True,
            ),
            ParticleTutorialStep(
                title="Cluster sizes",
                question="How many isolated particles, pairs, and larger groups exist?",
                look_for="Cluster sizes depend on the chosen linking distance.",
                statistic_label="Cluster-size distribution",
                caution="Document the linking distance when reporting cluster sizes.",
                visible_panel="plot",
                target_tab="Setup",
                pattern="clustered",
                model="homogeneous_poisson",
                n=120,
                seed=8,
                simulations=60,
                focus_statistic="cluster_size_counts",
                curve_mode="comparison",
                compute_on_show=True,
                primary_action="Learn pooling",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="pooling_single",
        title="21 - Pooling A",
        summary="One generated image gives one noisy statistic.",
        steps=(
            ParticleTutorialStep(
                title="Single image",
                question="One image gives one noisy estimate of the statistic.",
                look_for="The curve is jagged and the model envelope is wide.",
                model_label="Homogeneous Poisson",
                statistic_label="Pair correlation g(r)",
                visible_panel="plot",
                target_tab="Setup",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=11,
                simulations=40,
                focus_statistic="pair_correlation_g_r",
                curve_mode="comparison",
                compute_on_show=True,
                direct_labels=("single image",),
                primary_action="Add second image",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="pooling_two",
        title="22 - Pooling B",
        summary="Pool two independent generated images.",
        steps=(
            ParticleTutorialStep(
                title="Two-image pool",
                question="Now pool two independent images from the same condition.",
                look_for=(
                    "Blue is pooled mean/spread; orange is one-image reference."
                ),
                model_label="Homogeneous Poisson",
                statistic_label="Pair correlation g(r)",
                caution="Pool only independent images from the same experimental condition.",
                visible_panel="plot",
                target_tab="Setup",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=11,
                simulations=30,
                pool_images=2,
                focus_statistic="pair_correlation_g_r",
                curve_mode="comparison",
                compute_on_show=True,
                direct_labels=("pooled: 2 images",),
                primary_action="Compare one vs two",
                more_detail=(
                    "The orange reference is one generated image. The blue pooled "
                    "curve uses two independent images; its band shows image-to-image "
                    "spread, not a model envelope."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="model_simulations_sandbox",
        title="23 - Model simulations sandbox",
        summary="Explore model parameters outside the linear tutorial.",
        steps=(
            ParticleTutorialStep(
                title="Model simulations",
                question="Use Model simulations to ask what each model can generate.",
                look_for="Change model, N, seed, field size, radius, then generate points.",
                mode="sandbox",
                visible_panel="controls",
                visible_controls=("generated_model",),
                target_tab="Setup",
                pattern="no_overlap",
                model="hard_core_random",
                n=100,
                seed=9,
                simulations=40,
                hard_core_radius_nm=3.0,
                model_hard_core_radius_nm=3.0,
                show_simulated=True,
                show_features=False,
                primary_action="Move to real data",
                more_detail="The sandbox uses the same point field, layer controls, statistic plot, and results view as real analysis.",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="real_workflow",
        title="24 - Real ProbeFlow workflow",
        summary="Feature Finder to Particle Statistics to verdict.",
        steps=(
            ParticleTutorialStep(
                title="Real workflow",
                question="Real analysis begins with Feature Finder positions.",
                look_for="Use saved feature sets, then choose region, model, statistic, simulations, and verdicts.",
                mode="real",
                model_label="Choose model from real controls",
                visible_panel="controls",
                visible_controls=("feature_sets",),
                target_tab="Setup",
                show_simulated=False,
                show_features=False,
                primary_action="Final caution",
                more_detail="Workflow: detect particles, send them to Particle Statistics, confirm calibration and region, choose a model and statistic, run simulations, read verdicts, and pool comparable images.",
                caution=(
                    "Pooling is only valid for independent images from the same "
                    "experimental condition. Pixel-level point ROIs also limit the "
                    "smallest meaningful distance bin."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="final_caution",
        title="25 - Final caution",
        summary="Statistical verdicts need physical interpretation.",
        steps=(
            ParticleTutorialStep(
                title="Final caution",
                question="Particle Statistics can rule out simple spatial models, not prove a mechanism.",
                look_for="Use verdicts as evidence, then interpret with the experiment.",
                mode="real",
                visible_panel="results",
                target_tab="Results",
                focus_statistic=_MODEL_SUMMARY_FOCUS,
                primary_action="Restart tutorial",
                action_kind="restart",
                caution="Physical mechanism requires experimental interpretation.",
            ),
        ),
    ),
)


@dataclass(frozen=True)
class ParticleFieldModel:
    """Display-only point field for the Particle Statistics window."""

    observed_xy_nm: np.ndarray
    width_nm: float
    height_nm: float
    mode: str = "real"
    source_label: str = ""
    region_label: str = "Full field"
    model_label: str = ""
    status: str = ""
    mask: np.ndarray | None = None
    simulated_xy_nm: np.ndarray | None = None
    feature_xy_nm: np.ndarray | None = None
    show_observed: bool = True
    show_simulated: bool = True
    show_features: bool = True
    show_region: bool = True
    direct_labels: tuple[str, ...] = ()


class ParticleFieldView(QWidget):
    """Qt-native field renderer for real and generated particle patterns."""

    def __init__(self, *, theme: dict | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("particleStatisticsField")
        self.setMinimumSize(440, 300)
        self._theme = theme or {}
        self._model = ParticleFieldModel(
            observed_xy_nm=np.empty((0, 2), dtype=float),
            width_nm=100.0,
            height_nm=100.0,
            status="No points to display.",
        )

    @property
    def point_count(self) -> int:
        return int(len(self._model.observed_xy_nm))

    @property
    def data_mode(self) -> str:
        return self._model.mode

    @property
    def marker_style(self) -> dict[str, str]:
        return _marker_style(self._model.mode)

    @property
    def layer_visibility(self) -> dict[str, bool]:
        model = self._model
        return {
            "observed": bool(model.show_observed),
            "simulated": bool(model.show_simulated),
            "features": bool(model.show_features),
            "region": bool(model.show_region),
        }

    @property
    def layer_availability(self) -> dict[str, bool]:
        model = self._model
        return {
            "observed": bool(len(model.observed_xy_nm)),
            "simulated": bool(model.simulated_xy_nm is not None and len(model.simulated_xy_nm)),
            "features": bool(model.feature_xy_nm is not None and len(model.feature_xy_nm)),
            "region": bool(model.mask is not None and model.mask.size),
        }

    @property
    def direct_labels(self) -> tuple[str, ...]:
        return tuple(self._model.direct_labels)

    def set_field_model(self, model: ParticleFieldModel) -> None:
        self._model = model
        self.update()

    def set_layer_visibility(
        self,
        *,
        observed: bool | None = None,
        simulated: bool | None = None,
        features: bool | None = None,
        region: bool | None = None,
    ) -> None:
        model = self._model
        self.set_field_model(
            replace(
                model,
                show_observed=model.show_observed if observed is None else bool(observed),
                show_simulated=model.show_simulated if simulated is None else bool(simulated),
                show_features=model.show_features if features is None else bool(features),
                show_region=model.show_region if region is None else bool(region),
            )
        )

    def set_direct_labels(self, labels: tuple[str, ...] | list[str]) -> None:
        model = self._model
        self.set_field_model(
            replace(model, direct_labels=tuple(str(label) for label in labels if label))
        )

    def set_points(
        self,
        observed_xy_nm: Any,
        *,
        field_size_nm: tuple[float, float],
        mode: str = "real",
        source_label: str = "",
        region_label: str = "Full field",
        model_label: str = "",
        status: str = "",
        mask: Any = None,
        simulated_xy_nm: Any = None,
        feature_xy_nm: Any = None,
        direct_labels: tuple[str, ...] = (),
    ) -> None:
        self.set_field_model(
            ParticleFieldModel(
                observed_xy_nm=_xy_array(observed_xy_nm),
                width_nm=float(field_size_nm[0]),
                height_nm=float(field_size_nm[1]),
                mode=_normalise_field_mode(mode),
                source_label=str(source_label or ""),
                region_label=str(region_label or "Full field"),
                model_label=str(model_label or ""),
                status=str(status or ""),
                mask=_mask_or_none(mask),
                simulated_xy_nm=_xy_array_or_none(simulated_xy_nm),
                feature_xy_nm=_xy_array_or_none(feature_xy_nm),
                direct_labels=tuple(str(label) for label in direct_labels if label),
            )
        )

    def set_region(self, *, region_label: str, mask: Any = None) -> None:
        model = self._model
        self.set_field_model(
            replace(
                model,
                region_label=str(region_label or "Full field"),
                mask=_mask_or_none(mask),
            )
        )

    def set_mode(self, mode: str) -> None:
        model = self._model
        self.set_field_model(
            replace(model, mode=_normalise_field_mode(mode))
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bg = _theme_qcolor(self._theme, ("figure.facecolor", "surface", "bg"), "#161a20")
        fg = _theme_qcolor(self._theme, ("text.color", "fg"), "#e8edf4")
        border = _theme_qcolor(self._theme, ("border", "sep"), "#3b4250")
        painter.fillRect(self.rect(), bg)

        plot_rect = self._plot_rect()
        if plot_rect.width() < 40 or plot_rect.height() < 40:
            return

        model = self._model
        painter.save()
        painter.setPen(QPen(border, 1.0))
        painter.setBrush(QColor("#101419"))
        painter.drawRect(plot_rect)
        if model.show_region:
            self._draw_mask(painter, plot_rect)
        painter.restore()

        transform = _FieldTransform(plot_rect, model.width_nm, model.height_nm)
        plot_w = plot_rect.width()
        painter.save()
        painter.setClipRect(plot_rect.adjusted(1, 1, -1, -1))
        visible_layers = 0
        if model.mode == "generated" and model.show_simulated and model.simulated_xy_nm is not None:
            visible_layers += int(len(model.simulated_xy_nm) > 0)
            _draw_marker_series(
                painter,
                transform,
                model.simulated_xy_nm,
                marker="o",
                color="#b96adf",
                radius=_field_marker_radius(plot_w, len(model.simulated_xy_nm)),
                hollow=True,
            )
        if model.mode == "generated" and model.show_features and model.feature_xy_nm is not None:
            visible_layers += int(len(model.feature_xy_nm) > 0)
            _draw_marker_series(
                painter,
                transform,
                model.feature_xy_nm,
                marker="x",
                color="#9b5de5",
                radius=max(4.0, _field_marker_radius(plot_w, len(model.feature_xy_nm))),
            )
        if model.show_observed:
            visible_layers += int(len(model.observed_xy_nm) > 0)
            style = _marker_style(model.mode)
            _draw_marker_series(
                painter,
                transform,
                model.observed_xy_nm,
                marker=style["marker"],
                color=style["color"],
                radius=_field_marker_radius(plot_w, len(model.observed_xy_nm)),
            )
        painter.restore()

        self._draw_chrome(painter, plot_rect, fg, border)
        self._draw_labels(painter, fg)
        self._draw_legend(painter, plot_rect, fg)
        self._draw_direct_labels(painter, plot_rect, fg)
        if visible_layers == 0 and model.status:
            painter.save()
            painter.setPen(fg)
            font = QFont(painter.font())
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(plot_rect.adjusted(20, 20, -20, -20), Qt.AlignCenter | Qt.TextWordWrap, model.status)
            painter.restore()

    def _plot_rect(self) -> QRectF:
        margin_l, margin_t, margin_r, margin_b = 68.0, 52.0, 28.0, 86.0
        available = QRectF(
            margin_l,
            margin_t,
            max(1.0, self.width() - margin_l - margin_r),
            max(1.0, self.height() - margin_t - margin_b),
        )
        return _aspect_fit_rect(available, self._model.width_nm, self._model.height_nm)

    def _draw_mask(self, painter: QPainter, plot_rect: QRectF) -> None:
        mask = self._model.mask
        if mask is None or mask.size == 0:
            return
        rows, cols = mask.shape
        step_y = max(1, rows // 96)
        step_x = max(1, cols // 96)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 229, 255, 42))
        for y in range(0, rows, step_y):
            y1 = min(rows, y + step_y)
            for x in range(0, cols, step_x):
                x1 = min(cols, x + step_x)
                if not bool(mask[y:y1, x:x1].any()):
                    continue
                rx = plot_rect.left() + (x / cols) * plot_rect.width()
                ry = plot_rect.top() + (y / rows) * plot_rect.height()
                rw = ((x1 - x) / cols) * plot_rect.width()
                rh = ((y1 - y) / rows) * plot_rect.height()
                painter.drawRect(QRectF(rx, ry, max(1.0, rw), max(1.0, rh)))

    def _draw_chrome(
        self,
        painter: QPainter,
        plot_rect: QRectF,
        fg: QColor,
        border: QColor,
    ) -> None:
        model = self._model
        painter.save()
        painter.setPen(QPen(border, 1.0))
        painter.drawRect(plot_rect)
        painter.setPen(QPen(fg, 1.0))
        font = QFont(painter.font())
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(
            QRectF(plot_rect.left(), plot_rect.bottom() + 8, plot_rect.width(), 22),
            Qt.AlignCenter,
            f"x: 0 to {model.width_nm:g} nm",
        )
        painter.save()
        painter.translate(20, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot_rect.height() / 2, 0, plot_rect.height(), 22), Qt.AlignCenter, f"y: 0 to {model.height_nm:g} nm")
        painter.restore()
        painter.restore()

    def _draw_labels(self, painter: QPainter, fg: QColor) -> None:
        model = self._model
        painter.save()
        painter.setPen(fg)
        title_font = QFont(painter.font())
        title_font.setPointSize(13)
        title_font.setBold(True)
        painter.setFont(title_font)
        title = "Generated particle field" if model.mode == "generated" else "Observed particle field"
        painter.drawText(QRectF(14, 8, self.width() - 28, 26), Qt.AlignLeft, title)
        body_font = QFont(painter.font())
        body_font.setPointSize(10)
        body_font.setBold(False)
        painter.setFont(body_font)
        detail = "  ".join(
            part
            for part in (
                model.source_label,
                model.region_label,
                model.model_label,
                f"N={len(model.observed_xy_nm)}",
            )
            if part
        )
        painter.drawText(QRectF(14, 32, self.width() - 28, 22), Qt.AlignLeft, detail)
        painter.restore()

    def _draw_legend(self, painter: QPainter, plot_rect: QRectF, fg: QColor) -> None:
        model = self._model
        legend = []
        if model.show_observed:
            legend.append(("observed", _marker_style(model.mode)["color"], _marker_style(model.mode)["marker"], False))
        if model.mode == "generated" and model.show_simulated and model.simulated_xy_nm is not None:
            legend.append(("model sample", "#b96adf", "o", True))
        if model.mode == "generated" and model.show_features and model.feature_xy_nm is not None:
            legend.append(("feature layer", "#9b5de5", "x", False))
        if not legend:
            return
        # Horizontal strip in the bottom margin (below the x-axis label) so the legend
        # never sits on top of the points.
        painter.save()
        font = QFont(painter.font())
        font.setPointSize(10)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        gap = 18.0
        sample_w = 20.0
        items = [(label, color, marker, hollow, float(metrics.horizontalAdvance(label))) for label, color, marker, hollow in legend]
        total = sum(sample_w + tw for *_unused, tw in items) + gap * (len(items) - 1)
        y = plot_rect.bottom() + 40.0
        x = max(plot_rect.left(), plot_rect.center().x() - total / 2.0)
        for label, color, marker, hollow, text_w in items:
            _draw_marker(painter, QPointF(x + 7, y), marker, QColor(color), 4.4, hollow=hollow)
            painter.setPen(QPen(fg))
            painter.drawText(QRectF(x + sample_w, y - 9.0, text_w + 4.0, 18.0), Qt.AlignLeft | Qt.AlignVCenter, label)
            x += sample_w + text_w + gap
        painter.restore()

    def _draw_direct_labels(self, painter: QPainter, plot_rect: QRectF, fg: QColor) -> None:
        labels = tuple(self._model.direct_labels)
        if not labels:
            return
        painter.save()
        font = QFont(painter.font())
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        x = plot_rect.left() + 10.0
        y = plot_rect.top() + 10.0
        for label in labels[:4]:
            text_w = float(metrics.horizontalAdvance(label))
            box = QRectF(x, y, text_w + 18.0, 24.0)
            painter.setPen(QPen(QColor("#2fb344"), 1.2))
            painter.setBrush(QColor(47, 179, 68, 52))
            painter.drawRoundedRect(box, 4.0, 4.0)
            painter.setPen(QPen(fg))
            painter.drawText(box.adjusted(9, 0, -9, 0), Qt.AlignLeft | Qt.AlignVCenter, label)
            y += 28.0
        painter.restore()


class FocusedStatisticPanel(QFrame):
    """Large teaching panel for the currently focused statistic."""

    def __init__(self, *, theme: dict | None = None, parent=None):
        super().__init__(parent)
        self.setObjectName("particleStatisticsFocusedStatistic")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumSize(440, 300)
        self._theme = theme or {}
        self._statistic_id = _DEFAULT_FOCUS_STATISTIC
        self._has_plot = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._title = QLabel("", self)
        self._title.setObjectName("particleStatisticsFocusTitle")
        self._title.setStyleSheet("font-weight: 700;")
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        self._body = QLabel("", self)
        self._body.setObjectName("particleStatisticsFocusBody")
        self._body.setWordWrap(True)
        self._body.setMaximumHeight(54)
        self._body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._body)

        self._annotation = QLabel("", self)
        self._annotation.setObjectName("particleStatisticsFocusAnnotation")
        self._annotation.setWordWrap(True)
        self._annotation.setStyleSheet(
            "color: #9ecbff; border: 1px solid rgba(47, 129, 247, 0.45); "
            "padding: 3px 6px;"
        )
        self._annotation.setVisible(False)
        layout.addWidget(self._annotation)

        self._plot_host = QWidget(self)
        self._plot_layout = QVBoxLayout(self._plot_host)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._plot_layout.setSpacing(0)
        layout.addWidget(self._plot_host, 1)

    @property
    def focused_statistic(self) -> str:
        return self._statistic_id

    @property
    def has_plot(self) -> bool:
        return self._has_plot

    def set_statistic(
        self,
        statistic_id: str,
        *,
        panel: Any = None,
        data_mode: str = "real",
        curve_mode: str = "comparison",
        has_result: bool = False,
        empty_message: str | None = None,
        show_observed_curve: bool = True,
        show_model_curves: bool = True,
    ) -> None:
        self._statistic_id = str(statistic_id or _DEFAULT_FOCUS_STATISTIC)
        self._clear_plot()
        self._has_plot = panel is not None
        title = _display_statistic_title(self._statistic_id)
        question = _guide_text(self._statistic_id, "focus_question")
        self._title.setText(title)
        annotation = _plot_annotation_text(self._statistic_id)
        self._annotation.setText(annotation)
        self._annotation.setVisible(bool(annotation and panel is not None))
        if panel is not None:
            if str(getattr(panel, "kind", "")) == "series_curve":
                quick_read = _series_focus_read_text(panel)
            else:
                quick_read = _focus_read_text(self._statistic_id, curve_mode)
            body = f"<b>{question}</b> {quick_read}"
            self._body.setText(body)
            plot = AdStatPlotWidget(
                panel,
                theme=self._theme,
                data_mode=data_mode,
                curve_mode=curve_mode,
                show_observed_curve=show_observed_curve,
                show_model_curves=show_model_curves,
                parent=self._plot_host,
            )
            plot.setMinimumHeight(240)
            self._plot_layout.addWidget(plot, 1)
            return

        if self._statistic_id == _MODEL_SUMMARY_FOCUS and has_result:
            message = "Read the grouped model verdict cards in Results. The focus here is which model assumption stayed plausible."
        elif has_result:
            message = "This result does not include this panel. Choose another statistic card or rerun with a supported model."
        elif empty_message:
            message = empty_message
        elif data_mode == "sandbox":
            message = "Computing the statistic…"
        else:
            message = "Run a comparison to plot this statistic against the model envelope."
        self._body.setText(
            f"<b>{question}</b> {_guide_text(self._statistic_id, 'before_run')}"
        )
        placeholder = QLabel(message, self._plot_host)
        placeholder.setObjectName("particleStatisticsFocusPlaceholder")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setWordWrap(True)
        placeholder.setStyleSheet("border: 1px solid #3b4250; padding: 12px;")
        self._plot_layout.addWidget(placeholder, 1)

    def _clear_plot(self) -> None:
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()


class _ParticleRealWorkerSignals(QObject):
    finished = Signal(int, object)


class _ParticleRealWorker(_PooledWorker):
    def __init__(
        self,
        *,
        generation: int,
        point_sources: list[Any],
        scan: Any,
        image_shape: tuple[int, int] | None,
        request: AdStatStatisticsRequest,
    ):
        super().__init__(_ParticleRealWorkerSignals())
        self._generation = int(generation)
        self._point_sources = list(point_sources)
        self._scan = scan
        self._image_shape = image_shape
        self._request = request

    def work(self) -> None:
        context = adstat_workbench_launch_context(
            self._point_sources,
            scan=self._scan,
            image_shape=self._image_shape,
            request=self._request,
        )
        self.signals.finished.emit(self._generation, context)


class _ParticleFeatureSetWorkerSignals(QObject):
    finished = Signal(int, object, str)


class _ParticleFeatureSetWorker(_PooledWorker):
    """Run a single-set or pooled multi-set comparison from saved feature sets."""

    def __init__(
        self,
        *,
        generation: int,
        feature_sets: list[Any],
        request: AdStatStatisticsRequest,
        feature_layer: Any = None,
    ):
        super().__init__(_ParticleFeatureSetWorkerSignals())
        self._generation = int(generation)
        self._feature_sets = list(feature_sets)
        self._request = request
        self._feature_layer = feature_layer

    def work(self) -> None:
        from probeflow.analysis.adstat_adapter import (
            compare_point_set_record_view_spec,
            compare_point_set_records_view_spec,
        )

        try:
            records = [fs.to_point_set_record() for fs in self._feature_sets]
            models = self._request.models or ("poisson",)
            feature_layers = (
                [self._feature_layer.to_feature_layer()]
                if self._feature_layer is not None
                else ()
            )
            if len(records) == 1:
                spec = compare_point_set_record_view_spec(
                    records[0],
                    models=models,
                    feature_layers=feature_layers,
                    n_simulations=self._request.n_simulations,
                    random_seed=self._request.random_seed,
                    include_ordering=self._request.include_ordering,
                )
            else:
                spec = compare_point_set_records_view_spec(
                    records,
                    models=models,
                    n_simulations=self._request.n_simulations,
                    random_seed=self._request.random_seed,
                )
        except Exception as exc:  # noqa: BLE001 - report to GUI shell
            self.signals.finished.emit(self._generation, None, str(exc))
            return
        self.signals.finished.emit(self._generation, spec, "")


class _ParticleSandboxWorkerSignals(QObject):
    finished = Signal(int, object, str)


class _ParticleSandboxWorker(_PooledWorker):
    def __init__(self, state: Any, operation: str, generation: int):
        super().__init__(_ParticleSandboxWorkerSignals())
        self._state = state
        self._operation = operation
        self._generation = int(generation)

    def work(self) -> None:
        try:
            if self._operation == "new_pattern":
                self._state.new_random_pattern()
            elif self._operation == "reset":
                self._state.reset()
            else:
                self._state.run()
        except Exception as exc:  # noqa: BLE001 - report to GUI shell
            self.signals.finished.emit(self._generation, None, str(exc))
            return
        self.signals.finished.emit(self._generation, self._state, "")


class ParticleStatisticsDialog(QDialog):
    """Standalone Particle Statistics tool window."""

    def __init__(
        self,
        *,
        point_sources: list[Any] | None = None,
        scan: Any = None,
        active_area_roi: Any = None,
        active_mask: Any = None,
        image_shape: tuple[int, int] | None = None,
        feature_sets: Any = (),
        feature_set_store: Any = None,
        theme: dict | None = None,
        initial_mode: str = "landing",
        parent=None,
        pool: QThreadPool | None = None,
        context_refresh_fn: Any = None,
    ):
        super().__init__(parent)
        self.setObjectName("particleStatisticsDialog")
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
        )
        self.setWindowTitle("Particle Statistics")
        self.setMinimumSize(1300, 850)
        self.resize(1500, 900)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._theme = theme or {}
        self._pool = pool or QThreadPool.globalInstance()
        self._context_refresh_fn = context_refresh_fn
        self._point_sources = list(point_sources or ())
        # When a shared FeatureSetStore is provided it is the live source of
        # truth; ``self._feature_sets`` is a cache resynced from it on each
        # populate. Otherwise the static ``feature_sets`` tuple is used.
        self._feature_set_store_ref = feature_set_store
        self._feature_sets = list(
            feature_set_store.all() if feature_set_store is not None else (feature_sets or ())
        )
        self._scan = scan
        self._active_area_roi = active_area_roi
        self._active_mask = _valid_mask(active_mask, image_shape)
        self._image_shape = image_shape
        self._generation = 0
        self._sandbox_generation = 0
        self._sandbox_context = None
        self._sandbox_state = None
        self._updating_generated_controls = False
        self._updating_layer_controls = False
        self._updating_mode = False
        self._updating_tutorial_highlights = False
        self._force_close = False
        self._active_mode = ""
        self._tutorial_active = str(initial_mode).lower() == "learn"
        self._tutorial_step_index = 0
        self._tutorial_run_in_progress = False
        self._pooling_reference_curve: dict[str, Any] | None = None
        self._focused_statistic = _DEFAULT_FOCUS_STATISTIC
        self._focused_curve_mode = "comparison"
        self._last_view_spec = _empty_view_spec("Run a comparison to populate result panels.")
        self._field = ParticleFieldView(theme=self._theme, parent=self)
        self._focus_panel = FocusedStatisticPanel(theme=self._theme, parent=self)
        self._info_lbl = QLabel("")
        self._info_lbl.setObjectName("particleStatisticsInfo")
        self._info_lbl.setWordWrap(True)
        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("particleStatisticsStatus")
        self._status_lbl.setWordWrap(True)
        self._result_view = AdStatResultView(
            self._last_view_spec,
            source_label="Particle Statistics",
            theme=self._theme,
            data_mode="real",
            show_banner=False,
            # The per-statistic plots and point-pattern panel are already the
            # always-visible top panel and left field; show only the verdict
            # summary and technical details here to avoid duplication.
            show_panels=False,
            parent=self,
        )
        self._statistic_buttons: dict[str, QPushButton] = {}
        self._controls: list[QWidget] = []
        self._generated_controls: list[QWidget] = []

        try:
            self._sandbox_context = adstat_sandbox_context()
            self._sandbox_state = adstat_sandbox_state()
        except ImportError as exc:
            self._sandbox_error = str(exc)
        else:
            self._sandbox_error = ""

        self._build()
        self._restore_particle_statistics_layout()
        self._sync_generated_controls_from_state()
        initial = str(initial_mode).lower().replace(" ", "_")
        start_mode = (
            "generated"
            if self._tutorial_active
            else "sandbox"
            if initial in {"sandbox", "model_simulations"}
            else "landing"
            if initial in {"landing", "start", "home"}
            else "real"
        )
        self._set_mode(start_mode)
        if self._tutorial_active:
            self._apply_tutorial_step(self._current_tutorial_step_obj(), stage_generated=True)

    @property
    def current_mode(self) -> str:
        """Public workflow mode, decoupled from the internal data-surface name.

        Two vocabularies coexist on purpose:

        - ``self._active_mode`` is the internal *control/data surface*:
          ``landing`` (workflow chooser), ``real`` (scan points),
          ``generated`` (tutorial-staged generated data), or ``sandbox``
          (free-play Model simulations).  ``self._field.data_mode`` mirrors the
          data source as ``real``/``generated``/``sandbox``.
        - This property reports the *workflow the user is in*: ``landing``,
          ``real``, ``learn`` (the guided tutorial, whatever generated surface it
          stages), or ``model_simulations`` (the sandbox).

        So ``_active_mode == "generated"`` maps to ``learn`` while the tutorial is
        active, and ``_active_mode == "sandbox"`` maps to ``model_simulations``.
        """
        if self._active_mode == "landing":
            return "landing"
        if self._tutorial_active:
            return "learn"
        return "model_simulations" if self._active_mode == "sandbox" else "real"

    @property
    def field_point_count(self) -> int:
        return self._field.point_count

    @property
    def current_tutorial_key(self) -> str:
        if not hasattr(self, "_tutorial_cb"):
            return ""
        return str(self._tutorial_cb.currentData() or "")

    @property
    def current_tutorial_step(self) -> int:
        return int(self._tutorial_step_index)

    @property
    def focused_statistic(self) -> str:
        return self._focused_statistic

    @property
    def focus_has_plot(self) -> bool:
        return self._focus_panel.has_plot

    def set_current_mode(self, initial_mode: str) -> None:
        # Maps a public workflow name to an internal data surface (see the
        # ``current_mode`` docstring): ``learn`` -> ``generated`` surface with the
        # tutorial active; ``sandbox``/``model_simulations`` -> ``sandbox``.
        mode = str(initial_mode).lower().replace(" ", "_")
        tutorial = mode == "learn"
        target = (
            "generated"
            if tutorial
            else "sandbox"
            if mode in {"sandbox", "model_simulations"}
            else "landing"
            if mode in {"landing", "start", "home"}
            else "real"
        )
        self._set_mode(target, tutorial_active=tutorial)
        if tutorial:
            self._apply_tutorial_step(self._current_tutorial_step_obj(), stage_generated=True)
            self._ensure_tutorial_comparison(force=True)
        else:
            self._clear_tutorial_highlights()

    def return_to_landing_page(self) -> None:
        """Return to the three-card workflow chooser without closing the dialog."""

        self._sandbox_generation += 1
        self._tutorial_run_in_progress = False
        self._tutorial_active = False
        self._clear_tutorial_highlights()
        self._set_mode("landing", tutorial_active=False)
        # Keep this dialog in front after the mode switch (see exit_tutorial).
        self._raise_self()

    def focus_statistic(self, statistic_id: str, *, curve_mode: str | None = None) -> None:
        self._focused_statistic = str(statistic_id or _DEFAULT_FOCUS_STATISTIC)
        self._focused_curve_mode = str(curve_mode or "comparison")
        self._refresh_focus_panel()
        self._sync_statistic_buttons()
        self._raise_self()

    def refresh_probe_context(
        self,
        *,
        point_sources: list[Any] | None = None,
        scan: Any = None,
        active_area_roi: Any = None,
        active_mask: Any = None,
        image_shape: tuple[int, int] | None = None,
        feature_sets: Any = None,
        feature_set_store: Any = None,
    ) -> None:
        """Refresh real ProbeFlow inputs without resetting generated examples."""

        selected_source = self._current_source_label() if hasattr(self, "_source_cb") else ""
        selected_region = str(self._region_cb.currentData() or "full") if hasattr(self, "_region_cb") else "full"
        self._point_sources = list(point_sources or ())
        if feature_set_store is not None:
            self._feature_set_store_ref = feature_set_store
        if feature_sets is not None and self._feature_set_store_ref is None:
            self._feature_sets = list(feature_sets or ())
        self._scan = scan
        self._active_area_roi = active_area_roi
        self._image_shape = image_shape
        self._active_mask = _valid_mask(active_mask, image_shape)
        if hasattr(self, "_source_cb"):
            self._populate_sources()
            _set_combo_value(self._source_cb, selected_source)
        if hasattr(self, "_region_cb"):
            self._populate_regions()
            _set_combo_value(self._region_cb, selected_region)
        self._populate_feature_sets()
        if getattr(self, "_active_mode", "real") == "real":
            self._refresh_real_field()

    def force_close(self) -> None:
        self._force_close = True
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.isVisible():
            self._save_particle_statistics_layout()
        if self._force_close:
            super().closeEvent(event)
            return
        event.ignore()
        self.hide()

    def _restore_particle_statistics_layout(self) -> None:
        cfg = load_config()
        layout = cfg.get("layout", {}).get(_PARTICLE_STATISTICS_LAYOUT_KEY, {})
        restore_geometry_or_default(self, layout.get("geometry"), 0.92)

    def _save_particle_statistics_layout(self) -> None:
        cfg = load_config()
        layout_root = cfg.setdefault("layout", {})
        layout = layout_root.setdefault(_PARTICLE_STATISTICS_LAYOUT_KEY, {})
        layout["geometry"] = qbytearray_to_b64(self.saveGeometry())
        save_config(cfg)

    def _use_wide_layout(self) -> None:
        apply_screen_fraction_geometry(self, 0.92)

    def _reset_particle_statistics_window_size(self) -> None:
        cfg = load_config()
        if isinstance(cfg.get("layout"), dict):
            cfg["layout"].pop(_PARTICLE_STATISTICS_LAYOUT_KEY, None)
        save_config(cfg)
        self._use_wide_layout()

    def _set_result_view_spec(
        self,
        view_spec: Any,
        *,
        source_label: str,
        data_mode: str,
    ) -> None:
        self._last_view_spec = view_spec
        self._result_view.set_view_spec(
            view_spec,
            source_label=source_label,
            data_mode=data_mode,
        )
        self._refresh_focus_panel()

    def _refresh_focus_panel(self) -> None:
        if not hasattr(self, "_focus_panel"):
            return
        panel = _panel_for_statistic(self._last_view_spec, self._focused_statistic)
        empty_message = None
        if not _view_spec_has_result(self._last_view_spec):
            empty_message = str(
                getattr(self._last_view_spec, "metadata", {}).get("message", "") or ""
            ) or None
        self._focus_panel.set_statistic(
            self._focused_statistic,
            panel=panel,
            data_mode="sandbox" if self._active_mode in {"generated", "sandbox"} else "real",
            curve_mode=self._focused_curve_mode,
            has_result=_view_spec_has_result(self._last_view_spec),
            empty_message=empty_message,
            show_observed_curve=self._show_observed_curve_in_focus(),
            show_model_curves=self._show_model_curves_in_focus(),
        )

    def _show_observed_curve_in_focus(self) -> bool:
        if not hasattr(self, "_observed_layer_cb"):
            return True
        return bool(self._observed_layer_cb.isChecked())

    def _show_model_curves_in_focus(self) -> bool:
        if not hasattr(self, "_simulation_layer_cb"):
            return True
        # In real-data mode the field has no one-simulation overlay, but the focus
        # plot still has a model envelope. Keep it visible there.
        if self._active_mode not in {"generated", "sandbox"}:
            return True
        return bool(self._simulation_layer_cb.isChecked())

    def _sync_statistic_buttons(self) -> None:
        for statistic_id, button in getattr(self, "_statistic_buttons", {}).items():
            selected = statistic_id == self._focused_statistic
            button.setChecked(selected)
            # Normal ProbeFlow button when unselected; highlighted when its plot is shown.
            button.setStyleSheet(_STAT_SELECTED_STYLE if selected else "")
        self._refresh_selected_statistic_help()
        self._sync_workflow_actions()

    def _refresh_selected_statistic_help(self) -> None:
        if not hasattr(self, "_selected_statistic_help_lbl"):
            return
        label = _STATISTIC_LABELS.get(self._focused_statistic, self._focused_statistic)
        question = _guide_text(self._focused_statistic, "focus_question")
        self._selected_statistic_help_lbl.setText(
            f"<b>Selected: {label}</b><br>{question}"
        )

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)
        outer.setMenuBar(self._view_menu_bar())

        toolbar = QHBoxLayout()
        title = QLabel("Particle Statistics")
        title.setObjectName("dialogTitle")
        title.setStyleSheet("font-weight: 700;")
        toolbar.addWidget(title)
        self._landing_btn = QPushButton("Workflows", self)
        self._landing_btn.setObjectName("particleStatisticsReturnToLanding")
        self._landing_btn.setToolTip("Return to the workflow start page.")
        self._landing_btn.clicked.connect(self.return_to_landing_page)
        toolbar.addWidget(self._landing_btn)
        self._start_tutorial_btn = QPushButton("Start tutorial", self)
        self._start_tutorial_btn.setObjectName("particleStatisticsStartTutorial")
        self._start_tutorial_btn.setStyleSheet(_TUTORIAL_ACTION_STYLE)
        self._start_tutorial_btn.setToolTip("Open the guided walkthrough with generated example data.")
        self._start_tutorial_btn.clicked.connect(self.start_tutorial)
        toolbar.addWidget(self._start_tutorial_btn)
        toolbar.addStretch(1)
        self._mode_label = QLabel("Mode:")
        toolbar.addWidget(self._mode_label)
        self._mode_cb = QComboBox(self)
        self._mode_cb.setObjectName("particleStatisticsMode")
        self._mode_cb.addItem("Analyze scan points", "real")
        self._mode_cb.addItem("Learn with tutorial", "generated")
        self._mode_cb.addItem("Model simulations", "sandbox")
        self._mode_cb.currentIndexChanged.connect(self._on_mode_changed)
        toolbar.addWidget(self._mode_cb)
        self._run_btn = QPushButton("Run comparison", self)
        self._run_btn.setObjectName("particleStatisticsRun")
        self._run_btn.clicked.connect(self._run_comparison)
        toolbar.addWidget(self._run_btn)
        outer.addLayout(toolbar)

        self._landing_panel = self._landing_page()
        outer.addWidget(self._landing_panel, 1)

        self._workspace_panel = QWidget(self)
        workspace = QVBoxLayout(self._workspace_panel)
        workspace.setContentsMargins(0, 0, 0, 0)
        workspace.setSpacing(6)

        self._generated_banner = QLabel("TEST MODE - GENERATED DATA")
        self._generated_banner.setObjectName("particleStatisticsGeneratedBanner")
        self._generated_banner.setAlignment(Qt.AlignCenter)
        self._generated_banner.setStyleSheet(
            "background: #f59f00; color: #1f1300; font-weight: 800; "
            "padding: 6px; border: 1px solid #b36b00;"
        )
        workspace.addWidget(self._generated_banner)
        self._tutorial_panel = self._tutorial_drawer()
        workspace.addWidget(self._tutorial_panel)

        split = QSplitter(Qt.Vertical, self)
        self._main_splitter = split
        split.addWidget(self._top_panel())
        split.addWidget(self._workflow_tabs())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([520, 380])
        split.setChildrenCollapsible(False)
        workspace.addWidget(split, 1)
        outer.addWidget(self._workspace_panel, 1)

    def _view_menu_bar(self) -> QMenuBar:
        menu_bar = QMenuBar(self)
        workflow_menu = menu_bar.addMenu("Workflow")
        self._show_workflows_action = QAction("Workflow start page", self)
        self._show_workflows_action.triggered.connect(self.return_to_landing_page)
        workflow_menu.addAction(self._show_workflows_action)
        self._analyze_scan_points_action = QAction("Analyze scan points", self)
        self._analyze_scan_points_action.triggered.connect(
            lambda: self.set_current_mode("real")
        )
        workflow_menu.addAction(self._analyze_scan_points_action)
        self._model_simulations_action = QAction("Model simulations", self)
        self._model_simulations_action.triggered.connect(
            lambda: self.set_current_mode("sandbox")
        )
        workflow_menu.addAction(self._model_simulations_action)
        self._start_tutorial_action = QAction("Start tutorial", self)
        self._start_tutorial_action.triggered.connect(self.start_tutorial)
        workflow_menu.addAction(self._start_tutorial_action)
        workflow_menu.addSeparator()
        self._run_comparison_action = QAction("Run comparison", self)
        self._run_comparison_action.triggered.connect(self._run_comparison)
        workflow_menu.addAction(self._run_comparison_action)

        data_menu = menu_bar.addMenu("Data")
        self._show_observed_action = QAction("Show observed/fake data", self)
        self._show_observed_action.setCheckable(True)
        self._show_observed_action.triggered.connect(
            lambda checked=False: self._set_layer_action("observed", checked)
        )
        data_menu.addAction(self._show_observed_action)
        self._show_feature_layer_action = QAction("Show feature layer", self)
        self._show_feature_layer_action.setCheckable(True)
        self._show_feature_layer_action.triggered.connect(
            lambda checked=False: self._set_layer_action("features", checked)
        )
        data_menu.addAction(self._show_feature_layer_action)
        self._show_region_action = QAction("Show region / mask", self)
        self._show_region_action.setCheckable(True)
        self._show_region_action.triggered.connect(
            lambda checked=False: self._set_layer_action("region", checked)
        )
        data_menu.addAction(self._show_region_action)
        data_menu.addSeparator()
        self._new_pattern_action = QAction("New generated pattern", self)
        self._new_pattern_action.triggered.connect(self.new_generated_pattern)
        data_menu.addAction(self._new_pattern_action)
        self._reset_pattern_action = QAction("Reset generated pattern", self)
        self._reset_pattern_action.triggered.connect(self.reset_generated)
        data_menu.addAction(self._reset_pattern_action)
        data_menu.addSeparator()
        self._refresh_sources_action = QAction("Refresh real sources", self)
        self._refresh_sources_action.triggered.connect(self.refresh_probe_sources)
        data_menu.addAction(self._refresh_sources_action)
        self._clear_real_action = QAction("Clear real field", self)
        self._clear_real_action.triggered.connect(self.clear_real_view)
        data_menu.addAction(self._clear_real_action)

        model_menu = menu_bar.addMenu("Model")
        self._show_model_action = QAction("Show model simulation/envelope", self)
        self._show_model_action.setCheckable(True)
        self._show_model_action.triggered.connect(
            lambda checked=False: self._set_layer_action("simulated", checked)
        )
        model_menu.addAction(self._show_model_action)
        self._link_hard_core_radii_action = QAction(
            "Link data/model hard-core radius",
            self,
        )
        self._link_hard_core_radii_action.setCheckable(True)
        self._link_hard_core_radii_action.triggered.connect(
            self._set_link_hard_core_radii
        )
        model_menu.addAction(self._link_hard_core_radii_action)
        model_menu.addSeparator()
        self._model_action_group = QActionGroup(self)
        self._model_action_group.setExclusive(True)
        self._model_actions: dict[str, QAction] = {}
        for model_id in (
            "poisson",
            "hard_core_random",
            "measured_feature_poisson",
        ):
            action = QAction(_MODEL_LABELS.get(model_id, model_id), self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, value=model_id: self._choose_model_from_menu(value)
            )
            self._model_action_group.addAction(action)
            self._model_actions[model_id] = action
            model_menu.addAction(action)

        statistic_menu = menu_bar.addMenu("Statistic")
        self._statistic_action_group = QActionGroup(self)
        self._statistic_action_group.setExclusive(True)
        self._statistic_actions: dict[str, QAction] = {}
        for group_title, statistic_ids in _STATISTIC_GROUPS:
            section = QAction(group_title, self)
            section.setEnabled(False)
            statistic_menu.addAction(section)
            for statistic_id in statistic_ids:
                action = QAction(_STATISTIC_LABELS.get(statistic_id, statistic_id), self)
                action.setCheckable(True)
                action.triggered.connect(
                    lambda _checked=False, value=statistic_id: self.focus_statistic(value)
                )
                self._statistic_action_group.addAction(action)
                self._statistic_actions[statistic_id] = action
                statistic_menu.addAction(action)

        export_menu = menu_bar.addMenu("Export")
        self._export_csv_action = QAction("Export curves + verdicts (CSV folder)…", self)
        self._export_csv_action.setToolTip(
            "Write one CSV per statistic (g(r), nearest-neighbour, Ripley L, …) "
            "plus a verdicts table, for reproducing the plots elsewhere."
        )
        self._export_csv_action.triggered.connect(self._export_results_csv)
        export_menu.addAction(self._export_csv_action)
        self._export_json_action = QAction("Export full result (JSON)…", self)
        self._export_json_action.setToolTip(
            "Write the entire result (all panels, curves, and verdicts) to one JSON file."
        )
        self._export_json_action.triggered.connect(self._export_results_json)
        export_menu.addAction(self._export_json_action)

        view_menu = menu_bar.addMenu("View")
        self._use_wide_layout_action = QAction("Use wide layout", self)
        self._use_wide_layout_action.triggered.connect(self._use_wide_layout)
        view_menu.addAction(self._use_wide_layout_action)
        self._reset_window_size_action = QAction(
            "Reset Particle Statistics window size", self
        )
        self._reset_window_size_action.triggered.connect(
            self._reset_particle_statistics_window_size
        )
        view_menu.addAction(self._reset_window_size_action)

        definitions_menu = menu_bar.addMenu("Definitions")
        self._show_definitions_action = QAction("Show Definitions tab", self)
        self._show_definitions_action.triggered.connect(self.show_definitions_tab)
        definitions_menu.addAction(self._show_definitions_action)
        self._definitions_tutorial_action = QAction("Start tutorial", self)
        self._definitions_tutorial_action.triggered.connect(self.start_tutorial)
        definitions_menu.addAction(self._definitions_tutorial_action)
        return menu_bar

    def _landing_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("particleStatisticsLanding")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(18)
        heading = QLabel("Particle Statistics", page)
        heading.setObjectName("particleStatisticsLandingTitle")
        heading.setStyleSheet("font-size: 24px; font-weight: 800;")
        heading.setAlignment(Qt.AlignCenter)
        layout.addWidget(heading)
        summary = QLabel(
            "Compare detected particle positions with simulated spatial models.",
            page,
        )
        summary.setObjectName("particleStatisticsLandingSummary")
        summary.setWordWrap(True)
        summary.setAlignment(Qt.AlignCenter)
        layout.addWidget(summary)
        prompt = QLabel("Choose a workflow:", page)
        prompt.setAlignment(Qt.AlignCenter)
        prompt.setStyleSheet("font-weight: 700;")
        layout.addWidget(prompt)
        cards = QGridLayout()
        cards.setHorizontalSpacing(14)
        cards.setVerticalSpacing(14)
        cards.addWidget(
            self._landing_card(
                page,
                object_name="particleStatisticsLandingAnalyze",
                title="Analyze scan points",
                body="Use detected particles from Feature Finder or saved feature sets.",
                button_text="Choose point source",
                mode="real",
            ),
            0,
            0,
        )
        cards.addWidget(
            self._landing_card(
                page,
                object_name="particleStatisticsLandingSimulations",
                title="Model simulations",
                body=(
                    "Generate random, clustered, hard-core, feature-biased, "
                    "or ordered patterns."
                ),
                button_text="Open simulations",
                mode="sandbox",
            ),
            0,
            1,
        )
        cards.addWidget(
            self._landing_card(
                page,
                object_name="particleStatisticsLandingTutorial",
                title="Tutorial",
                body="Learn the workflow with guided generated examples.",
                button_text="Start tutorial",
                mode="learn",
            ),
            0,
            2,
        )
        layout.addLayout(cards)
        layout.addStretch(1)
        return page

    def _landing_card(
        self,
        parent: QWidget,
        *,
        object_name: str,
        title: str,
        body: str,
        button_text: str,
        mode: str,
    ) -> QFrame:
        card = QFrame(parent)
        card.setObjectName(object_name)
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(
            "QFrame { border: 1px solid #3b4250; border-radius: 6px; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        title_label = QLabel(f"<b>{title}</b>", card)
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        body_label = QLabel(body, card)
        body_label.setWordWrap(True)
        layout.addWidget(body_label)
        layout.addStretch(1)
        button = QPushButton(button_text, card)
        button.setObjectName(f"{object_name}Button")
        button.clicked.connect(lambda _checked=False, value=mode: self._choose_landing_workflow(value))
        layout.addWidget(button)
        return card

    def _choose_landing_workflow(self, mode: str) -> None:
        if mode == "learn":
            self.start_tutorial()
            return
        self._set_mode("sandbox" if mode == "sandbox" else "real", tutorial_active=False)

    def _set_layer_action(self, layer: str, checked: bool) -> None:
        mapping = {
            "observed": "_observed_layer_cb",
            "simulated": "_simulation_layer_cb",
            "features": "_feature_layer_cb",
            "region": "_region_layer_cb",
        }
        attr = mapping.get(str(layer))
        checkbox = getattr(self, attr, None) if attr else None
        if isinstance(checkbox, QCheckBox):
            checkbox.setChecked(bool(checked))
        self._sync_workflow_actions()
        self._raise_self()

    def _set_link_hard_core_radii(self, checked: bool) -> None:
        checkbox = getattr(self, "_link_hard_core_radii_cb", None)
        if isinstance(checkbox, QCheckBox):
            checkbox.setChecked(bool(checked))
        self._sync_workflow_actions()
        self._raise_self()

    def _choose_model_from_menu(self, model_id: str) -> None:
        if self._active_mode in {"generated", "sandbox"}:
            sandbox_id = (
                "homogeneous_poisson" if str(model_id) == "poisson" else str(model_id)
            )
            _set_combo_value(self._generated_model_cb, sandbox_id)
            self._raise_self()
            return
        _set_combo_value(self._real_model_cb, str(model_id))
        self._raise_self()

    def show_definitions_tab(self) -> None:
        """Open the shared ProbeFlow Definitions document at the Particle Statistics tab."""
        from probeflow.gui.dialogs.definitions import _DefinitionsDialog

        dlg = getattr(self, "_definitions_dialog", None)
        try:
            if dlg is not None:
                dlg.isVisible()
        except RuntimeError:
            dlg = None
        if dlg is None:
            dlg = _DefinitionsDialog(
                self._theme, self, initial_tab="particle_statistics"
            )
            self._definitions_dialog = dlg
        else:
            dlg.set_reference_tab("particle_statistics")
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _top_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._field, 2)
        layout.addWidget(self._focus_panel, 4)

        info_panel = QFrame(panel)
        info_panel.setObjectName("particleStatisticsInfoPanel")
        info_panel.setFrameShape(QFrame.StyledPanel)
        info_panel.setMinimumWidth(260)
        info_panel.setMaximumWidth(340)
        self._info_panel = info_panel
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(8)
        heading = QLabel("Field")
        heading.setStyleSheet("font-weight: 700;")
        info_layout.addWidget(heading)
        info_layout.addWidget(self._info_lbl)
        info_layout.addWidget(self._status_lbl)

        layer_group = QGroupBox("View layers", info_panel)
        self._layer_group = layer_group
        layer_layout = QVBoxLayout(layer_group)
        layer_layout.setContentsMargins(8, 8, 8, 8)
        layer_layout.setSpacing(4)
        self._observed_layer_cb = QCheckBox("Observed points", layer_group)
        self._observed_layer_cb.setObjectName("particleStatisticsLayerObserved")
        self._simulation_layer_cb = QCheckBox("Model simulation", layer_group)
        self._simulation_layer_cb.setObjectName("particleStatisticsLayerSimulation")
        self._feature_layer_cb = QCheckBox("Feature layer", layer_group)
        self._feature_layer_cb.setObjectName("particleStatisticsLayerFeature")
        self._region_layer_cb = QCheckBox("Region / mask", layer_group)
        self._region_layer_cb.setObjectName("particleStatisticsLayerRegion")
        for checkbox in (
            self._observed_layer_cb,
            self._simulation_layer_cb,
            self._feature_layer_cb,
            self._region_layer_cb,
        ):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._on_layer_toggled)
            layer_layout.addWidget(checkbox)
        self._layer_hint_lbl = QLabel("", layer_group)
        self._layer_hint_lbl.setObjectName("particleStatisticsLayerHint")
        self._layer_hint_lbl.setWordWrap(True)
        layer_layout.addWidget(self._layer_hint_lbl)
        info_layout.addWidget(layer_group)
        info_layout.addStretch(1)
        layout.addWidget(info_panel)
        return panel

    def _workflow_tabs(self) -> QTabWidget:
        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("particleStatisticsTabs")
        self._tabs.addTab(self._scrollable(self._setup_tab()), "Setup")
        self._tabs.addTab(self._scrollable(self._results_tab()), "Results")
        # Both the feature-layer picker and the model combo now exist, so set
        # the picker's enabled state for the default model.
        self._sync_feature_layer_controls()
        return self._tabs

    def _scrollable(self, page: QWidget) -> QScrollArea:
        """Wrap a tab page so tall content (e.g. the Results plots) scrolls instead
        of being clipped below the window when the bottom pane is short."""
        area = QScrollArea(self)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(page)
        return area

    def _setup_tab(self) -> QWidget:
        page = QWidget(self)
        page.setStyleSheet(_SETUP_COLUMN_STYLE)
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._setup_data_column, data_layout = self._setup_column(
            page, "particleStatisticsDataColumn", "Data / observed pattern"
        )
        self._build_data_column(data_layout)
        layout.addWidget(self._setup_data_column, 1)

        self._setup_model_column, model_layout = self._setup_column(
            page, "particleStatisticsModelColumn", "Model / null hypothesis"
        )
        self._build_model_column(model_layout)
        layout.addWidget(self._setup_model_column, 1)

        self._setup_statistic_column, statistic_layout = self._setup_column(
            page, "particleStatisticsStatisticColumn", "Statistic / question"
        )
        self._build_statistic_column(statistic_layout)
        layout.addWidget(self._setup_statistic_column, 1)
        return page

    def _setup_column(
        self, parent: QWidget, object_name: str, title: str
    ) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame(parent)
        frame.setObjectName(object_name)
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        header = QLabel(f"<b>{title}</b>", frame)
        header.setWordWrap(True)
        layout.addWidget(header)
        return frame, layout

    def _build_data_column(self, layout: QVBoxLayout) -> None:
        parent = layout.parentWidget()
        self._real_data_group = QGroupBox("Analyze scan points", parent)
        real_form = QFormLayout(self._real_data_group)
        self._source_cb = QComboBox(self._real_data_group)
        self._source_cb.setObjectName("particleStatisticsSource")
        self._populate_sources()
        self._source_cb.currentIndexChanged.connect(self._refresh_real_field)
        real_form.addRow("Point source:", self._source_cb)
        self._region_cb = QComboBox(self._real_data_group)
        self._region_cb.setObjectName("particleStatisticsRegion")
        self._populate_regions()
        self._region_cb.currentIndexChanged.connect(self._refresh_real_field)
        real_form.addRow("Region:", self._region_cb)
        source_buttons = QHBoxLayout()
        self._refresh_sources_btn = QPushButton("Refresh sources", self._real_data_group)
        self._refresh_sources_btn.setToolTip("Reload ProbeFlow point sources, active ROI, and active mask from the viewer.")
        self._refresh_sources_btn.clicked.connect(self.refresh_probe_sources)
        source_buttons.addWidget(self._refresh_sources_btn)
        self._clear_real_btn = QPushButton("Clear", self._real_data_group)
        self._clear_real_btn.setObjectName("particleStatisticsClearReal")
        self._clear_real_btn.setToolTip("Clear the field, plot, and results.")
        self._clear_real_btn.clicked.connect(self.clear_real_view)
        source_buttons.addWidget(self._clear_real_btn)
        real_form.addRow("", source_buttons)
        layout.addWidget(self._real_data_group)

        self._feature_sets_group = QGroupBox("Saved feature sets", parent)
        sets_layout = QVBoxLayout(self._feature_sets_group)
        sets_layout.setContentsMargins(8, 8, 8, 8)
        sets_layout.setSpacing(6)
        sets_hint = QLabel(
            "Tick one set for a single-image comparison, or two or more to pool them "
            "(replicates) into one combined verdict."
        )
        sets_hint.setWordWrap(True)
        sets_layout.addWidget(sets_hint)
        self._feature_sets_list = QListWidget(self._feature_sets_group)
        self._feature_sets_list.setObjectName("particleStatisticsFeatureSets")
        self._feature_sets_list.setMinimumHeight(120)
        sets_layout.addWidget(self._feature_sets_list, 1)
        feature_layer_row = QHBoxLayout()
        feature_layer_row.addWidget(QLabel("Feature layer:"))
        self._feature_layer_set_cb = QComboBox(self._feature_sets_group)
        self._feature_layer_set_cb.setObjectName("particleStatisticsFeatureLayer")
        self._feature_layer_set_cb.setToolTip(
            "For the Measured-feature Poisson model: an independently-measured set "
            "(e.g. step edges) the particles may follow. Must differ from the tested set."
        )
        feature_layer_row.addWidget(self._feature_layer_set_cb, 1)
        sets_layout.addLayout(feature_layer_row)
        self._run_feature_sets_btn = QPushButton("Run selected sets", self._feature_sets_group)
        self._run_feature_sets_btn.setObjectName("particleStatisticsRunFeatureSets")
        self._run_feature_sets_btn.clicked.connect(self.run_selected_feature_sets)
        sets_layout.addWidget(self._run_feature_sets_btn)

        io_row = QHBoxLayout()
        self._import_sets_btn = QPushButton("Load points from disk…", self._feature_sets_group)
        self._import_sets_btn.setObjectName("particleStatisticsImportSets")
        self._import_sets_btn.setToolTip(
            "Import a CSV position table or a ProbeFlow JSON file as a feature set."
        )
        self._import_sets_btn.clicked.connect(self.import_points_from_disk)
        io_row.addWidget(self._import_sets_btn)
        self._save_sets_btn = QPushButton("Save feature sets…", self._feature_sets_group)
        self._save_sets_btn.setObjectName("particleStatisticsSaveSets")
        self._save_sets_btn.setToolTip("Save all current feature sets to a JSON file.")
        self._save_sets_btn.clicked.connect(self.save_feature_sets_to_disk)
        io_row.addWidget(self._save_sets_btn)
        sets_layout.addLayout(io_row)

        layout.addWidget(self._feature_sets_group)
        self._populate_feature_sets()

        self._generated_data_group = QGroupBox("Generated / fake data", parent)
        gen_form = QFormLayout(self._generated_data_group)
        gen_form.setContentsMargins(8, 8, 8, 8)
        gen_form.setHorizontalSpacing(8)
        gen_form.setVerticalSpacing(4)
        self._pattern_cb = QComboBox(self._generated_data_group)
        self._pattern_cb.setObjectName("particleStatisticsPattern")
        self._populate_combo(
            self._pattern_cb,
            getattr(self._sandbox_context, "SANDBOX_PATTERNS", ()),
            _PATTERN_LABELS,
        )
        self._pattern_cb.currentIndexChanged.connect(self._stage_generated_from_controls)
        gen_form.addRow("Generated pattern:", self._pattern_cb)

        self._ordered_lattice_cb = QComboBox(self._generated_data_group)
        self._ordered_lattice_cb.setObjectName("particleStatisticsOrderedLattice")
        self._populate_combo(
            self._ordered_lattice_cb,
            getattr(self._sandbox_context, "ORDERED_ISLAND_LATTICES", ("triangular", "square")),
            _ORDERED_ISLAND_LATTICE_LABELS,
        )
        self._ordered_lattice_cb.currentIndexChanged.connect(
            self._stage_generated_from_controls
        )
        self._ordered_lattice_lbl = QLabel("Island lattice:", self._generated_data_group)
        gen_form.addRow(self._ordered_lattice_lbl, self._ordered_lattice_cb)

        self._ordered_background_cb = QComboBox(self._generated_data_group)
        self._ordered_background_cb.setObjectName(
            "particleStatisticsOrderedBackground"
        )
        self._populate_combo(
            self._ordered_background_cb,
            getattr(
                self._sandbox_context,
                "ORDERED_ISLAND_BACKGROUNDS",
                ("none", "random", "clustered"),
            ),
            _ORDERED_ISLAND_BACKGROUND_LABELS,
        )
        self._ordered_background_cb.currentIndexChanged.connect(
            self._stage_generated_from_controls
        )
        self._ordered_background_lbl = QLabel("Background:", self._generated_data_group)
        gen_form.addRow(self._ordered_background_lbl, self._ordered_background_cb)

        self._n_spin = QSpinBox(self._generated_data_group)
        self._n_spin.setObjectName("particleStatisticsN")
        self._n_spin.setRange(2, 500)
        self._n_spin.valueChanged.connect(self._stage_generated_from_controls)

        self._seed_spin = QSpinBox(self._generated_data_group)
        self._seed_spin.setObjectName("particleStatisticsSeed")
        self._seed_spin.setRange(0, 2_147_483_647)
        self._seed_spin.valueChanged.connect(self._stage_generated_from_controls)
        count_seed_row = QHBoxLayout()
        count_seed_row.setContentsMargins(0, 0, 0, 0)
        count_seed_row.setSpacing(6)
        count_seed_row.addWidget(self._n_spin, 1)
        count_seed_row.addWidget(QLabel("Seed:", self._generated_data_group))
        count_seed_row.addWidget(self._seed_spin, 1)
        gen_form.addRow("N:", count_seed_row)

        self._hard_core_radius_spin = QDoubleSpinBox(self._generated_data_group)
        self._hard_core_radius_spin.setObjectName("particleStatisticsHardCoreRadius")
        self._hard_core_radius_spin.setRange(0.0, 50.0)
        self._hard_core_radius_spin.setDecimals(2)
        self._hard_core_radius_spin.setSingleStep(0.5)
        self._hard_core_radius_spin.setSuffix(" nm")
        self._hard_core_radius_spin.setToolTip(
            "Minimum allowed separation for no-overlap / hard-core generated points."
        )
        self._hard_core_radius_spin.valueChanged.connect(self._stage_generated_from_controls)
        gen_form.addRow("Data hard-core radius:", self._hard_core_radius_spin)

        self._width_spin = QDoubleSpinBox(self._generated_data_group)
        self._width_spin.setObjectName("particleStatisticsFieldWidth")
        self._width_spin.setRange(10.0, 1000.0)
        self._width_spin.setDecimals(1)
        self._width_spin.setSuffix(" nm")
        self._width_spin.valueChanged.connect(self._stage_generated_from_controls)

        self._height_spin = QDoubleSpinBox(self._generated_data_group)
        self._height_spin.setObjectName("particleStatisticsFieldHeight")
        self._height_spin.setRange(10.0, 1000.0)
        self._height_spin.setDecimals(1)
        self._height_spin.setSuffix(" nm")
        self._height_spin.valueChanged.connect(self._stage_generated_from_controls)
        field_size_row = QHBoxLayout()
        field_size_row.setContentsMargins(0, 0, 0, 0)
        field_size_row.setSpacing(6)
        field_size_row.addWidget(self._width_spin, 1)
        field_size_row.addWidget(QLabel("H:", self._generated_data_group))
        field_size_row.addWidget(self._height_spin, 1)
        gen_form.addRow("Field size:", field_size_row)

        self._sandbox_warning_lbl = QLabel("", self._generated_data_group)
        self._sandbox_warning_lbl.setObjectName("particleStatisticsSandboxWarning")
        self._sandbox_warning_lbl.setWordWrap(True)
        gen_form.addRow("", self._sandbox_warning_lbl)
        button_row = QHBoxLayout()
        self._new_pattern_btn = QPushButton("New pattern", self._generated_data_group)
        self._new_pattern_btn.setObjectName("particleStatisticsNewPattern")
        self._new_pattern_btn.clicked.connect(self.new_generated_pattern)
        button_row.addWidget(self._new_pattern_btn)
        self._reset_btn = QPushButton("Reset", self._generated_data_group)
        self._reset_btn.setObjectName("particleStatisticsReset")
        self._reset_btn.clicked.connect(self.reset_generated)
        button_row.addWidget(self._reset_btn)
        gen_form.addRow("", button_row)
        layout.addWidget(self._generated_data_group)
        layout.addStretch(1)
        self._generated_controls.extend(
            [
                self._pattern_cb,
                self._ordered_lattice_cb,
                self._ordered_background_cb,
                self._n_spin,
                self._hard_core_radius_spin,
                self._width_spin,
                self._height_spin,
                self._seed_spin,
                self._new_pattern_btn,
                self._reset_btn,
            ]
        )

    def _build_model_column(self, layout: QVBoxLayout) -> None:
        parent = layout.parentWidget()
        self._real_model_group = QGroupBox("Real-data model", parent)
        real_form = QFormLayout(self._real_model_group)
        real_form.setContentsMargins(8, 8, 8, 8)
        real_form.setHorizontalSpacing(8)
        real_form.setVerticalSpacing(4)
        self._real_model_cb = QComboBox(self._real_model_group)
        self._real_model_cb.setObjectName("particleStatisticsRealModel")
        self._real_model_cb.addItem("Homogeneous Poisson", "poisson")
        self._real_model_cb.addItem("Hard-core random", "hard_core_random")
        self._real_model_cb.addItem("Measured-feature Poisson", "measured_feature_poisson")
        self._real_model_cb.currentIndexChanged.connect(self._on_real_model_changed)
        real_form.addRow("Comparison model:", self._real_model_cb)
        self._real_sim_spin = QSpinBox(self._real_model_group)
        self._real_sim_spin.setObjectName("particleStatisticsRealSimulations")
        self._real_sim_spin.setRange(1, 500)
        self._real_sim_spin.setValue(100)
        self._real_sim_spin.setToolTip("Number of null-model simulations used to build the comparison envelope.")
        real_form.addRow("Simulations:", self._real_sim_spin)
        self._real_seed_spin = QSpinBox(self._real_model_group)
        self._real_seed_spin.setObjectName("particleStatisticsRealSeed")
        self._real_seed_spin.setRange(0, 2_147_483_647)
        self._real_seed_spin.setValue(0)
        self._real_seed_spin.setToolTip("Random seed for reproducible simulation envelopes.")
        real_form.addRow("Seed:", self._real_seed_spin)
        real_note = QLabel(
            "Use measured-feature only with a different, independently measured feature layer."
        )
        real_note.setWordWrap(True)
        real_form.addRow("", real_note)
        layout.addWidget(self._real_model_group)

        self._generated_model_group = QGroupBox("Model comparison", parent)
        gen_form = QFormLayout(self._generated_model_group)
        gen_form.setContentsMargins(8, 8, 8, 8)
        gen_form.setHorizontalSpacing(8)
        gen_form.setVerticalSpacing(4)
        self._generated_model_cb = QComboBox(self._generated_model_group)
        self._generated_model_cb.setObjectName("particleStatisticsGeneratedModel")
        self._populate_combo(
            self._generated_model_cb,
            getattr(self._sandbox_context, "SANDBOX_MODELS", ()),
            _MODEL_LABELS,
        )
        self._generated_model_cb.currentIndexChanged.connect(self._on_generated_model_changed)
        gen_form.addRow("Comparison model:", self._generated_model_cb)
        self._link_hard_core_radii_cb = QCheckBox(
            "Link data/model hard-core radius",
            self._generated_model_group,
        )
        self._link_hard_core_radii_cb.setObjectName(
            "particleStatisticsLinkHardCoreRadii"
        )
        self._link_hard_core_radii_cb.setChecked(True)
        self._link_hard_core_radii_cb.setToolTip(
            "When checked, the comparison model uses the same hard-core radius "
            "as the generated/fake data."
        )
        self._link_hard_core_radii_cb.toggled.connect(
            self._on_link_hard_core_radii_toggled
        )
        gen_form.addRow("", self._link_hard_core_radii_cb)
        self._model_hard_core_radius_spin = QDoubleSpinBox(self._generated_model_group)
        self._model_hard_core_radius_spin.setObjectName(
            "particleStatisticsModelHardCoreRadius"
        )
        self._model_hard_core_radius_spin.setRange(0.0, 50.0)
        self._model_hard_core_radius_spin.setDecimals(2)
        self._model_hard_core_radius_spin.setSingleStep(0.5)
        self._model_hard_core_radius_spin.setSuffix(" nm")
        self._model_hard_core_radius_spin.setToolTip(
            "Minimum separation assumed by the hard-core comparison model."
        )
        self._model_hard_core_radius_spin.valueChanged.connect(
            self._stage_generated_from_controls
        )
        gen_form.addRow("Model hard-core radius:", self._model_hard_core_radius_spin)
        self._sim_spin = QSpinBox(self._generated_model_group)
        self._sim_spin.setObjectName("particleStatisticsSimulations")
        self._sim_spin.setRange(1, 500)
        self._sim_spin.valueChanged.connect(self._stage_generated_from_controls)
        gen_form.addRow("Simulations:", self._sim_spin)
        layout.addWidget(self._generated_model_group)

        cards = QWidget(parent)
        self._model_reference_cards = cards
        card_layout = QVBoxLayout(cards)
        card_layout.setContentsMargins(0, 0, 0, 0)
        note = QLabel(
            "<b>Working distinction</b><br>"
            "Generated pattern makes the orange test data. "
            "Comparison model builds the blue envelope. "
            "Use Definitions for model details."
        )
        note.setWordWrap(True)
        note.setMaximumHeight(74)
        card_layout.addWidget(note)
        layout.addWidget(cards)
        self._generated_controls.extend(
            [
                self._generated_model_cb,
                self._link_hard_core_radii_cb,
                self._model_hard_core_radius_spin,
                self._sim_spin,
            ]
        )
        layout.addStretch(1)

    def _tutorial_drawer(self) -> QWidget:
        panel = QFrame(self)
        panel.setObjectName("particleStatisticsTutorial")
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setStyleSheet(
            "QFrame#particleStatisticsTutorial { "
            "background-color: rgba(245, 159, 0, 0.10); border: 1px solid #9f6b00; }"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(5)

        # Hidden example picker — kept because ``current_tutorial_key`` reads it and
        # the load/highlight machinery references it; navigation is now linear.
        self._tutorial_picker_lbl = QLabel("Guided example")
        self._tutorial_picker_lbl.setVisible(False)
        self._tutorial_cb = QComboBox(panel)
        self._tutorial_cb.setObjectName("particleStatisticsTutorialExample")
        for example in _TUTORIALS:
            self._tutorial_cb.addItem(example.title, example.key)
        self._tutorial_cb.currentIndexChanged.connect(self._on_tutorial_changed)
        self._tutorial_cb.setVisible(False)
        layout.addWidget(self._tutorial_picker_lbl)
        layout.addWidget(self._tutorial_cb)

        # Navigation row: previous lesson | current lesson (prominent) | next lesson.
        nav = QHBoxLayout()
        self._prev_tutorial_btn = QPushButton("◂ Previous", panel)
        self._prev_tutorial_btn.setObjectName("particleStatisticsPrevTutorial")
        self._prev_tutorial_btn.setMaximumWidth(260)
        self._prev_tutorial_btn.setToolTip("Go back to the previous lesson.")
        self._prev_tutorial_btn.clicked.connect(self.previous_tutorial_step)
        nav.addWidget(self._prev_tutorial_btn, 0)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        self._tutorial_progress_lbl = QLabel("", panel)
        self._tutorial_progress_lbl.setAlignment(Qt.AlignCenter)
        self._tutorial_progress_lbl.setStyleSheet("color: #c9d1d9; font-size: 9pt;")
        self._tutorial_title_lbl = QLabel("", panel)
        self._tutorial_title_lbl.setObjectName("particleStatisticsTutorialTitle")
        self._tutorial_title_lbl.setAlignment(Qt.AlignCenter)
        self._tutorial_title_lbl.setWordWrap(True)
        self._tutorial_title_lbl.setStyleSheet("font-size: 15pt; font-weight: 800;")
        title_box.addWidget(self._tutorial_progress_lbl)
        title_box.addWidget(self._tutorial_title_lbl)
        nav.addLayout(title_box, 1)

        self._next_tutorial_btn = QPushButton("Next ▸", panel)
        self._next_tutorial_btn.setObjectName("particleStatisticsNextTutorial")
        self._next_tutorial_btn.setMaximumWidth(260)
        self._next_tutorial_btn.setToolTip("Go on to the next lesson.")
        self._next_tutorial_btn.clicked.connect(self.next_tutorial_step)
        nav.addWidget(self._next_tutorial_btn, 0)
        layout.addLayout(nav)

        # Action + meta row: the call-to-action (run/load) is separate from
        # navigation; More detail / Restart / Exit are always available.
        meta = QHBoxLayout()
        self._run_tutorial_btn = QPushButton("Run this example", panel)
        self._run_tutorial_btn.setObjectName("particleStatisticsRunTutorial")
        self._run_tutorial_btn.clicked.connect(self.run_current_tutorial_example)
        meta.addWidget(self._run_tutorial_btn)
        self._load_tutorial_btn = QPushButton("Load example", panel)
        self._load_tutorial_btn.setObjectName("particleStatisticsLoadTutorial")
        self._load_tutorial_btn.clicked.connect(self.load_current_tutorial_example)
        meta.addWidget(self._load_tutorial_btn)
        self._tutorial_detail_btn = QToolButton(panel)
        self._tutorial_detail_btn.setObjectName("particleStatisticsTutorialDetail")
        self._tutorial_detail_btn.setCheckable(True)
        self._tutorial_detail_btn.setText("More detail ▸")
        self._tutorial_detail_btn.toggled.connect(self._on_tutorial_detail_toggled)
        meta.addWidget(self._tutorial_detail_btn)
        meta.addStretch(1)
        self._restart_tutorial_btn = QPushButton("Restart tutorial", panel)
        self._restart_tutorial_btn.setObjectName("particleStatisticsRestartTutorial")
        self._restart_tutorial_btn.clicked.connect(self.restart_tutorial)
        meta.addWidget(self._restart_tutorial_btn)
        self._exit_tutorial_btn = QPushButton("Exit tutorial", panel)
        self._exit_tutorial_btn.setObjectName("particleStatisticsExitTutorial")
        self._exit_tutorial_btn.setStyleSheet(_TUTORIAL_EXIT_STYLE)
        self._exit_tutorial_btn.setToolTip("Leave the guided tutorial and switch to analysing real scan points.")
        self._exit_tutorial_btn.clicked.connect(self.exit_tutorial)
        meta.addWidget(self._exit_tutorial_btn)
        layout.addLayout(meta)

        self._tutorial_step_lbl = QLabel("", panel)
        self._tutorial_step_lbl.setWordWrap(True)
        self._tutorial_step_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._tutorial_step_lbl)

        # "Why it matters" always visible with a green importance bar (SEMITIP-style),
        # so the eye lands on the takeaway without reading everything.
        self._tutorial_why_frame, self._tutorial_why_lbl = _bar_text_row(panel, "#2fb344")
        layout.addWidget(self._tutorial_why_frame)

        # Collapsible depth: "What to look for" (blue) + "Careful" (red).
        self._tutorial_detail_container = QWidget(panel)
        detail_layout = QVBoxLayout(self._tutorial_detail_container)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(4)
        self._tutorial_look_frame, self._tutorial_look_lbl = _bar_text_row(self._tutorial_detail_container, "#2f81f7")
        self._tutorial_careful_frame, self._tutorial_careful_lbl = _bar_text_row(self._tutorial_detail_container, "#b3382f")
        detail_layout.addWidget(self._tutorial_look_frame)
        detail_layout.addWidget(self._tutorial_careful_frame)
        self._tutorial_detail_container.setVisible(False)
        layout.addWidget(self._tutorial_detail_container)
        return panel

    def _build_statistic_column(self, layout: QVBoxLayout) -> None:
        parent = layout.parentWidget()
        layout.setSpacing(8)
        intro = QLabel(
            "<b>Choose the question for the chart above.</b>",
            parent,
        )
        self._statistics_intro_lbl = intro
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._include_ordering_cb = QCheckBox(
            "Include local-order checks (ψ4/ψ6, angle map)", parent
        )
        self._include_ordering_cb.setObjectName("particleStatisticsIncludeOrdering")
        self._include_ordering_cb.setToolTip(
            "Local-order statistics answer a different question — is there square "
            "or triangular lattice order? — and are sensitive to the neighbour "
            "cutoff and edge effects. Off by default; enable to compute and show "
            "ψ4, ψ6, and the angular pair map."
        )
        self._include_ordering_cb.toggled.connect(self._on_include_ordering_toggled)
        layout.addWidget(self._include_ordering_cb)

        self._statistic_buttons = {}
        self._ordering_stat_buttons: dict[str, QPushButton] = {}
        self._statistic_description_labels: dict[str, QLabel] = {}
        self._statistic_group_labels: list[QLabel] = []
        for group_title, statistic_ids in _STATISTIC_GROUPS:
            group_label = QLabel(f"<b>{group_title}</b>", parent)
            group_label.setObjectName(
                "particleStatisticsStatisticGroup_"
                + group_title.lower().replace(" ", "_").replace("/", "_")
            )
            self._statistic_group_labels.append(group_label)
            layout.addWidget(group_label)
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 4)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(6)
            for index, statistic_id in enumerate(statistic_ids):
                button = QPushButton(
                    _STATISTIC_LABELS.get(statistic_id, statistic_id), parent
                )
                button.setObjectName(f"particleStatisticsFocus_{statistic_id}")
                button.setCheckable(True)
                button.setMinimumWidth(150)
                button.setMinimumHeight(30)
                button.setToolTip(_statistic_row_description(statistic_id))
                button.clicked.connect(
                    lambda _checked=False, value=statistic_id: self.focus_statistic(value)
                )
                self._statistic_buttons[statistic_id] = button
                if statistic_id in ORDERING_STATISTICS:
                    self._ordering_stat_buttons[statistic_id] = button
                desc = QLabel(_statistic_row_description(statistic_id), parent)
                desc.setObjectName(f"particleStatisticsDesc_{statistic_id}")
                desc.setWordWrap(True)
                desc.setVisible(False)
                self._statistic_description_labels[statistic_id] = desc
                grid.addWidget(button, index // 2, index % 2)
            layout.addLayout(grid)
        self._selected_statistic_help_lbl = QLabel("", parent)
        self._selected_statistic_help_lbl.setObjectName(
            "particleStatisticsSelectedStatisticHelp"
        )
        self._selected_statistic_help_lbl.setWordWrap(True)
        self._selected_statistic_help_lbl.setStyleSheet(
            "border: 1px solid rgba(47, 129, 247, 0.45); padding: 5px;"
        )
        layout.addWidget(self._selected_statistic_help_lbl)
        layout.addStretch(1)
        self._sync_statistic_buttons()
        self._sync_ordering_card_state()

    def _results_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._result_view, 1)
        return page

    def refresh_probe_sources(self) -> None:
        if callable(self._context_refresh_fn):
            context = self._context_refresh_fn()
            if isinstance(context, dict):
                self.refresh_probe_context(**context)
                return
        self._refresh_real_field()

    def _on_layer_toggled(self, *_args) -> None:
        if self._updating_layer_controls:
            return
        self._apply_layer_controls()

    def _sync_layer_controls(self) -> None:
        if not hasattr(self, "_observed_layer_cb"):
            return
        availability = self._field.layer_availability
        previous = getattr(self, "_last_layer_availability", {})
        mapping = {
            "observed": self._observed_layer_cb,
            "simulated": self._simulation_layer_cb,
            "features": self._feature_layer_cb,
            "region": self._region_layer_cb,
        }
        self._updating_layer_controls = True
        try:
            for key, checkbox in mapping.items():
                available = bool(availability.get(key, False))
                checkbox.setEnabled(available)
                if not available:
                    checkbox.setChecked(False)
                elif not bool(previous.get(key, False)):
                    checkbox.setChecked(True)
        finally:
            self._updating_layer_controls = False
        self._last_layer_availability = availability
        self._apply_layer_controls()

    def _apply_layer_controls(self) -> None:
        if not hasattr(self, "_observed_layer_cb"):
            return
        self._field.set_layer_visibility(
            observed=self._observed_layer_cb.isChecked(),
            simulated=self._simulation_layer_cb.isChecked(),
            features=self._feature_layer_cb.isChecked(),
            region=self._region_layer_cb.isChecked(),
        )
        self._update_layer_hint()
        self._refresh_focus_panel()
        self._sync_workflow_actions()

    def _set_layer_controls_from_step(self, step: ParticleTutorialStep) -> None:
        if not hasattr(self, "_observed_layer_cb"):
            return
        availability = self._field.layer_availability
        self._updating_layer_controls = True
        try:
            values = {
                "observed": step.show_observed,
                "simulated": step.show_simulated,
                "features": step.show_features,
                "region": step.show_region,
            }
            for key, checkbox in (
                ("observed", self._observed_layer_cb),
                ("simulated", self._simulation_layer_cb),
                ("features", self._feature_layer_cb),
                ("region", self._region_layer_cb),
            ):
                if availability.get(key, False):
                    checkbox.setChecked(bool(values[key]))
        finally:
            self._updating_layer_controls = False
        self._apply_layer_controls()

    def _update_layer_hint(self) -> None:
        if not hasattr(self, "_layer_hint_lbl"):
            return
        availability = self._field.layer_availability
        hidden: list[str] = []
        if availability.get("observed") and not self._observed_layer_cb.isChecked():
            hidden.append(
                "Observed/fake data hidden from the field and supported plots; "
                "still used for comparison."
            )
        if availability.get("simulated") and not self._simulation_layer_cb.isChecked():
            hidden.append(
                "Model simulation hidden from the field and supported plots; "
                "still used for comparison."
            )
        if availability.get("features") and not self._feature_layer_cb.isChecked():
            hidden.append("Feature layer hidden from view.")
        if availability.get("region") and not self._region_layer_cb.isChecked():
            hidden.append("Region or mask hidden from view.")
        self._layer_hint_lbl.setText(" ".join(hidden))

    @staticmethod
    def _tutorial_step_controls(step: ParticleTutorialStep) -> tuple[str, ...]:
        return tuple(step.visible_controls or step.controls)

    @staticmethod
    def _tutorial_step_curve_mode(step: ParticleTutorialStep) -> str:
        return str(step.curve_mode or step.focus_curve_mode or "comparison")

    def _refresh_generated_banner(self) -> None:
        if not hasattr(self, "_generated_banner"):
            return
        is_generated = self._active_mode in {"generated", "sandbox"}
        self._generated_banner.setVisible(is_generated)
        if self._tutorial_active and is_generated:
            self._generated_banner.setText("Tutorial: generated example")
            self._generated_banner.setStyleSheet(
                "background: rgba(47, 179, 68, 0.14); color: #d8ffe0; "
                "font-weight: 700; padding: 4px; border: 1px solid #2fb344;"
            )
            return
        if self._active_mode == "sandbox":
            self._generated_banner.setText("Model simulations")
            self._generated_banner.setStyleSheet(
                "background: rgba(47, 129, 247, 0.14); color: #dceaff; "
                "font-weight: 700; padding: 4px; border: 1px solid #2f81f7;"
            )
            return
        self._generated_banner.setText("TEST MODE - GENERATED DATA")
        self._generated_banner.setStyleSheet(
            "background: #f59f00; color: #1f1300; font-weight: 800; "
            "padding: 6px; border: 1px solid #b36b00;"
        )

    def _apply_staged_tutorial_visibility(
        self,
        step: ParticleTutorialStep | None = None,
    ) -> None:
        if not hasattr(self, "_tabs"):
            return
        if step is None and self._tutorial_active:
            step = self._current_tutorial_step_obj()
        if not self._tutorial_active or step is None:
            self._tabs.setVisible(True)
            self._tabs.tabBar().setVisible(True)
            self._focus_panel.setVisible(True)
            self._info_panel.setVisible(True)
            self._layer_group.setVisible(True)
            self._setup_data_column.setVisible(True)
            self._setup_model_column.setVisible(True)
            self._setup_statistic_column.setVisible(True)
            is_generated = self._active_mode in {"generated", "sandbox"}
            self._real_data_group.setVisible(not is_generated)
            self._real_model_group.setVisible(not is_generated)
            self._feature_sets_group.setVisible(not is_generated)
            self._generated_data_group.setVisible(is_generated)
            self._generated_model_group.setVisible(is_generated)
            self._model_reference_cards.setVisible(True)
            self._statistics_intro_lbl.setVisible(True)
            for label in getattr(self, "_statistic_group_labels", ()):
                label.setVisible(True)
            self._selected_statistic_help_lbl.setVisible(True)
            for statistic_id, button in self._statistic_buttons.items():
                button.setVisible(True)
                label = self._statistic_description_labels.get(statistic_id)
                if label is not None:
                    label.setVisible(False)
            self._result_view.set_technical_details_visible(True)
            self._refresh_generated_banner()
            return

        controls = set(self._tutorial_step_controls(step))
        panel = str(step.visible_panel or "field").lower()
        is_generated = self._active_mode in {"generated", "sandbox"}
        layer_keys = {"layer_observed", "layer_simulated", "layer_features", "layer_region"}
        generated_data_keys = {
            "pattern",
            "n",
            "generated_seed",
            "field_size",
            "hard_core_radius",
            "data_hard_core_radius",
        }
        generated_model_keys = {
            "generated_model",
            "generated_simulations",
            "hard_core_radius",
            "link_hard_core_radii",
            "model_hard_core_radius",
        }
        real_data_keys = {"source", "region"}
        real_model_keys = {"real_model", "real_simulations", "real_seed"}
        statistic_keys = {
            "stat_pair",
            "stat_nearest",
            "stat_ripley",
            "stat_clusters",
            "stat_directional",
            "stat_psi6",
            "stat_psi4",
        }

        tabs_visible = panel in {
            "controls",
            "results",
            "learn",
            "definitions",
            "statistics_reference",
        }
        self._tabs.setVisible(tabs_visible)
        self._tabs.tabBar().setVisible(False)
        self._focus_panel.setVisible(panel in {"plot", "results"})
        self._info_panel.setVisible(panel in {"controls", "info"} or bool(controls & layer_keys))
        self._layer_group.setVisible(bool(controls & layer_keys))

        show_generated_data = (
            is_generated and panel == "controls" and bool(controls & generated_data_keys)
        )
        show_generated_model = (
            is_generated and panel == "controls" and bool(controls & generated_model_keys)
        )
        show_real_data = (
            (not is_generated)
            and panel == "controls"
            and (not controls or bool(controls & real_data_keys))
        )
        show_feature_sets = (
            (not is_generated)
            and panel == "controls"
            and (not controls or "feature_sets" in controls or "feature_layer" in controls)
        )
        show_real_model = (
            (not is_generated) and panel == "controls" and bool(controls & real_model_keys)
        )
        show_statistics_reference = panel == "statistics_reference"
        show_statistic_controls = (
            show_statistics_reference or (panel == "controls" and bool(controls & statistic_keys))
        )

        self._generated_data_group.setVisible(show_generated_data)
        self._generated_model_group.setVisible(show_generated_model)
        self._real_data_group.setVisible(show_real_data)
        self._feature_sets_group.setVisible(
            show_feature_sets
        )
        self._real_model_group.setVisible(show_real_model)
        self._model_reference_cards.setVisible(False)
        self._setup_data_column.setVisible(
            show_generated_data or show_real_data or show_feature_sets
        )
        self._setup_model_column.setVisible(show_generated_model or show_real_model)
        self._setup_statistic_column.setVisible(show_statistic_controls)

        self._statistics_intro_lbl.setVisible(show_statistics_reference)
        for label in getattr(self, "_statistic_group_labels", ()):
            label.setVisible(show_statistic_controls)
        self._selected_statistic_help_lbl.setVisible(show_statistic_controls)
        for statistic_id, button in self._statistic_buttons.items():
            visible = show_statistics_reference or (
                show_statistic_controls and statistic_id == step.focus_statistic
            )
            button.setVisible(visible)
            label = self._statistic_description_labels.get(statistic_id)
            if label is not None:
                label.setVisible(False)

        self._result_view.set_technical_details_visible(bool(step.show_technical_details))
        self._refresh_generated_banner()

    def _tutorial_by_key(self, key: str) -> ParticleTutorialExample:
        for example in _TUTORIALS:
            if example.key == key:
                return example
        return _TUTORIALS[0]

    def _current_tutorial(self) -> ParticleTutorialExample:
        return self._tutorial_by_key(self.current_tutorial_key)

    def _current_tutorial_step_obj(self) -> ParticleTutorialStep:
        tutorial = self._current_tutorial()
        index = max(0, min(self._tutorial_step_index, len(tutorial.steps) - 1))
        self._tutorial_step_index = index
        return tutorial.steps[index]

    def _next_tutorial_key(self) -> str:
        current = self.current_tutorial_key
        for index, example in enumerate(_TUTORIALS):
            if example.key == current and index < len(_TUTORIALS) - 1:
                return _TUTORIALS[index + 1].key
        return ""

    def _next_tutorial_title(self) -> str:
        key = self._next_tutorial_key()
        return self._tutorial_by_key(key).title if key else ""

    def _on_tutorial_changed(self) -> None:
        self._tutorial_step_index = 0
        self.load_current_tutorial_example()

    def start_tutorial(self) -> None:
        """Enter tutorial mode at the first guided example (the workspace tour)."""
        self.load_tutorial_example(_TUTORIALS[0].key)

    def restart_tutorial(self) -> None:
        """Return to the first guided example (the workspace tour) at step one."""
        self.load_tutorial_example(_TUTORIALS[0].key)

    def exit_tutorial(self) -> None:
        """Leave the guided tutorial and switch to the real-data (Analyze) workflow."""
        # Invalidate any in-flight tutorial comparison so its late result cannot
        # repopulate the real view, then switch and clear.
        self._sandbox_generation += 1
        self._tutorial_run_in_progress = False
        self._tutorial_active = False
        self._clear_tutorial_highlights()
        self._set_mode("real", tutorial_active=False)
        self.clear_real_view()
        # Switching modes rebuilds the field/tabs, which can drop this modeless
        # dialog behind the Browse/Image Viewer windows on macOS. Keep it in front
        # so exiting the tutorial leaves the user on Particle Statistics.
        self._raise_self()

    def clear_real_view(self) -> None:
        """Reset the real-data field, focused statistic, and result panels to empty."""
        self._set_result_view_spec(
            _empty_view_spec("Run a comparison to populate result panels."),
            source_label="Particle Statistics",
            data_mode="real",
        )
        self._refresh_real_field()
        self._status_lbl.setText(_REAL_EMPTY_STATE_MESSAGE)

    def load_tutorial_example(self, key: str) -> None:
        _set_combo_value(self._tutorial_cb, key)
        self._tutorial_step_index = 0
        step = self._current_tutorial_step_obj()
        self._tutorial_active = True
        self._set_mode(self._tutorial_step_mode(step), tutorial_active=True)
        self._apply_tutorial_step(step, stage_generated=True)
        self._ensure_tutorial_comparison(force=True)

    def load_current_tutorial_example(self) -> None:
        step = self._current_tutorial_step_obj()
        self._tutorial_active = True
        self._set_mode(self._tutorial_step_mode(step), tutorial_active=True)
        self._apply_tutorial_step(step, stage_generated=True)
        self._ensure_tutorial_comparison(force=True)

    def _tutorial_step_mode(self, step: ParticleTutorialStep) -> str:
        mode = str(step.mode).lower()
        if mode == "real":
            return "real"
        if mode == "sandbox":
            return "sandbox"
        return "generated"

    def _ensure_tutorial_comparison(self, *, force: bool = False) -> None:
        """Compute the statistic up front so the chart is live from step one.

        The statistic is the core of this module, so the focus panel should show a
        real curve immediately rather than a text concept card. Stepping then just
        toggles emphasis (observed-only -> model envelope) on the existing result.

        Only fires while the dialog is visible: production always shows the window,
        while bare construction (e.g. in tests) must not spawn a background worker
        that mutates the shared sandbox state.
        """
        if self._active_mode not in {"generated", "sandbox"} or self._sandbox_state is None:
            return
        if not self.isVisible():
            return
        step = self._current_tutorial_step_obj()
        if step.intro_card:
            # Soft-intro cards are read-only; the panels are set by _apply_intro_card.
            return
        if not step.compute_on_show:
            return
        if step.pool_images > 0:
            return
        if not force and _view_spec_has_result(self._last_view_spec):
            return
        self._set_result_view_spec(
            _empty_view_spec("Computing the statistic…"),
            source_label="Generated examples",
            data_mode="sandbox",
        )
        self._start_generated_worker("run")

    def run_current_tutorial_example(self) -> None:
        step = self._current_tutorial_step_obj()
        self._set_mode(self._tutorial_step_mode(step), tutorial_active=True)
        self._apply_tutorial_step(step, stage_generated=True)
        self._tutorial_run_in_progress = True
        if self._active_mode in {"generated", "sandbox"}:
            self._start_generated_worker("run")
        else:
            self._start_real_worker()

    def previous_tutorial_step(self) -> None:
        if self._tutorial_step_index <= 0:
            prev_key = self._prev_tutorial_key()
            if not prev_key:
                return
            prev = self._tutorial_by_key(prev_key)
            _set_combo_value(self._tutorial_cb, prev_key)
            self._tutorial_step_index = max(0, len(prev.steps) - 1)
        else:
            self._tutorial_step_index -= 1
        step = self._current_tutorial_step_obj()
        self._set_mode(self._tutorial_step_mode(step), tutorial_active=True)
        self._apply_tutorial_step(step, stage_generated=True)
        self._ensure_tutorial_comparison(force=True)

    def next_tutorial_step(self) -> None:
        tutorial = self._current_tutorial()
        if self._tutorial_step_index >= len(tutorial.steps) - 1:
            next_key = self._next_tutorial_key()
            if next_key:
                self.load_tutorial_example(next_key)
            else:
                self._apply_tutorial_step(self._current_tutorial_step_obj(), stage_generated=True)
            return
        self._tutorial_step_index += 1
        step = self._current_tutorial_step_obj()
        self._set_mode(self._tutorial_step_mode(step), tutorial_active=True)
        self._apply_tutorial_step(step, stage_generated=True)
        self._ensure_tutorial_comparison(force=True)

    def _on_tutorial_detail_toggled(self, expanded: bool) -> None:
        if not hasattr(self, "_tutorial_detail_container"):
            return
        self._tutorial_detail_btn.setText("Less detail ▾" if expanded else "More detail ▸")
        has_detail = self._tutorial_look_frame.isVisibleTo(self._tutorial_detail_container) or bool(
            self._tutorial_look_lbl.text() or self._tutorial_careful_lbl.text()
        )
        self._tutorial_detail_container.setVisible(bool(expanded) and has_detail)

    def _apply_tutorial_step(
        self,
        step: ParticleTutorialStep,
        *,
        stage_generated: bool,
    ) -> None:
        self._tutorial_active = True
        self._set_mode(self._tutorial_step_mode(step), tutorial_active=True)
        self._set_ordering_for_step(step)
        if step.intro_card:
            self._apply_intro_card(step)
            return
        if stage_generated and self._active_mode in {"generated", "sandbox"} and self._sandbox_state is not None:
            config = self._sandbox_state.config
            pattern = step.pattern or config.pattern
            n = int(step.n if step.n is not None else config.n)
            seed = int(step.seed if step.seed is not None else config.seed)
            simulations = int(step.simulations if step.simulations is not None else config.n_simulations)
            width_nm = float(step.width_nm if step.width_nm is not None else config.width_nm)
            height_nm = float(step.height_nm if step.height_nm is not None else config.height_nm)
            hard_core_radius_nm = float(
                step.hard_core_radius_nm
                if step.hard_core_radius_nm is not None
                else config.hard_core_radius_nm
            )
            config_model_radius = getattr(
                config, "model_hard_core_radius_nm", None
            )
            if config_model_radius is None:
                config_model_radius = config.hard_core_radius_nm
            model_hard_core_radius_nm = float(
                step.model_hard_core_radius_nm
                if step.model_hard_core_radius_nm is not None
                else config_model_radius
            )
            config_lattice = getattr(config, "ordered_island_lattice", "triangular")
            config_background = getattr(config, "ordered_island_background", "none")
            ordered_island_lattice = step.ordered_island_lattice or config_lattice
            ordered_island_background = (
                step.ordered_island_background or config_background
            )
            active_model = str(getattr(self._sandbox_state, "active_model", ""))
            needs_model_update = bool(step.model and active_model != str(step.model))
            needs_stage = (
                str(config.pattern) != str(pattern)
                or int(config.n) != n
                or int(config.seed) != seed
                or int(config.n_simulations) != simulations
                or float(config.width_nm) != width_nm
                or float(config.height_nm) != height_nm
                or float(config.hard_core_radius_nm) != hard_core_radius_nm
                or float(config_model_radius) != model_hard_core_radius_nm
                or str(config_lattice) != str(ordered_island_lattice)
                or str(config_background) != str(ordered_island_background)
            )
            if needs_model_update:
                self._sandbox_state.set_model(step.model)
            if needs_stage:
                changes = {
                    "pattern": pattern,
                    "n": n,
                    "seed": seed,
                    "width_nm": width_nm,
                    "height_nm": height_nm,
                    "hard_core_radius_nm": hard_core_radius_nm,
                    "n_simulations": simulations,
                }
                if hasattr(config, "model_hard_core_radius_nm"):
                    changes["model_hard_core_radius_nm"] = model_hard_core_radius_nm
                if hasattr(config, "ordered_island_lattice"):
                    changes["ordered_island_lattice"] = ordered_island_lattice
                if hasattr(config, "ordered_island_background"):
                    changes["ordered_island_background"] = ordered_island_background
                self._sandbox_state.stage(**changes)
            self._sync_generated_controls_from_state()
            self._refresh_generated_field()
        elif self._active_mode == "real":
            if step.model:
                model = "poisson" if step.model == "homogeneous_poisson" else step.model
                _set_combo_value(self._real_model_cb, model)
            if step.simulations is not None:
                self._real_sim_spin.setValue(int(step.simulations))
            if step.seed is not None:
                self._real_seed_spin.setValue(int(step.seed))
            self._refresh_real_field()
        self._set_layer_controls_from_step(step)
        self._field.set_direct_labels(step.direct_labels)
        self.focus_statistic(
            step.focus_statistic or _DEFAULT_FOCUS_STATISTIC,
            curve_mode=self._tutorial_step_curve_mode(step),
        )
        self._select_tutorial_tab(step.target_tab)
        self._refresh_tutorial_text()
        self._apply_staged_tutorial_visibility(step)
        self._apply_tutorial_highlights()
        if step.pool_images > 0:
            self._run_generated_pooling_demo(step)
        self._raise_self()

    def _apply_intro_card(self, step: ParticleTutorialStep) -> None:
        """Soft-intro card: keep the three central panels blank, introducing one
        region at a time with a short label, so the eye stays on the top-bar text."""
        region = step.intro_region
        # Left point field: blank, with the region label only when this card owns it.
        self._field.set_points(
            np.empty((0, 2), dtype=float),
            field_size_nm=(100.0, 100.0),
            mode="generated",
            source_label="",
            region_label="",
            status=step.intro_panel_text if region == "field" else "",
        )
        # Right Field-info panel: blank unless this card introduces it.
        self._info_lbl.setText(step.intro_panel_text if region == "info" else "")
        self._status_lbl.setText("")
        # Centre statistic plot: a neutral placeholder, labelled only on its card.
        self._focused_statistic = _DEFAULT_FOCUS_STATISTIC
        self._focus_panel.set_statistic(
            _DEFAULT_FOCUS_STATISTIC,
            panel=None,
            data_mode="sandbox",
            empty_message=step.intro_panel_text if region == "plot" else " ",
        )
        self._sync_statistic_buttons()
        self._select_tutorial_tab(step.target_tab)
        self._refresh_tutorial_text()
        self._apply_staged_tutorial_visibility(step)
        self._apply_tutorial_highlights()
        self._raise_self()

    def _run_generated_pooling_demo(self, step: ParticleTutorialStep) -> None:
        """Pool several independent generated images live, to show statistics smoothing.

        Builds ``step.pool_images`` synthetic random fields (varying seed) into feature
        sets and pools them with the same engine path as real saved sets, so the
        learner sees the jagged single-image curve become a smooth pooled one.
        """
        if self._sandbox_state is None or not self.isVisible():
            return
        from probeflow.measurements.feature_sets import FeatureSet

        config = self._sandbox_state.config
        base_seed = int(step.seed if step.seed is not None else config.seed)
        n = int(step.n if step.n is not None else config.n)
        image_shape = (256, 256)
        sets: list[Any] = []
        self._pooling_reference_curve = None
        for index in range(int(step.pool_images)):
            cfg = replace(config, pattern="random", n=n, seed=base_seed + index)
            try:
                preview = adstat_sandbox_preview(cfg, active_model="homogeneous_poisson")
            except Exception:  # noqa: BLE001 - optional engine errors surface as empty
                return
            xy_nm = np.asarray(preview.xy_nm, dtype=float)
            width_nm = float(preview.width_nm)
            height_nm = float(preview.height_nm)
            denom = np.array([max(width_nm, 1e-9), max(height_nm, 1e-9)])
            points_px = xy_nm / denom * np.array([image_shape[1], image_shape[0]])
            sets.append(
                FeatureSet.from_points(
                    name=f"image {index + 1}",
                    points_px=points_px,
                    points_m=xy_nm * 1e-9,
                    scan_range_m=(width_nm * 1e-9, height_nm * 1e-9),
                    image_shape=image_shape,
                    image_label=f"image {index + 1}",
                )
            )
        if len(sets) < 2:
            return
        try:
            single_spec = compare_point_set_record_view_spec(
                sets[0].to_point_set_record(),
                models=("poisson",),
                n_simulations=int(step.simulations or 30),
                random_seed=0,
            )
            self._pooling_reference_curve = _single_curve_reference_from_spec(
                single_spec,
                step.focus_statistic or _DEFAULT_FOCUS_STATISTIC,
            )
        except Exception:  # noqa: BLE001 - optional teaching aid only
            self._pooling_reference_curve = None
        self._sandbox_generation += 1
        generation = self._sandbox_generation
        self._set_controls_enabled(False)
        self._set_result_view_spec(
            _empty_view_spec("Pooling images…"),
            source_label="Generated examples",
            data_mode="sandbox",
        )
        self._status_lbl.setText(f"Pooling {len(sets)} images…")
        request = AdStatStatisticsRequest(
            point_source_label="Pooled images",
            region_mode="full",
            roi_or_mask=None,
            models=("poisson",),
            n_simulations=int(step.simulations or 30),
            random_seed=0,
        )
        worker = _ParticleFeatureSetWorker(
            generation=generation, feature_sets=sets, request=request
        )
        worker.signals.finished.connect(self._on_pooling_demo_finished)
        self._pool.start(worker)

    def _on_pooling_demo_finished(self, generation: int, spec: Any, error: str) -> None:
        if int(generation) != self._sandbox_generation:
            return
        self._set_controls_enabled(True)
        if self._active_mode not in {"generated", "sandbox"}:
            return
        if error or spec is None:
            self._status_lbl.setText(error or "Pooling failed.")
            self._pooling_reference_curve = None
            return
        step = self._current_tutorial_step_obj()
        if self._pooling_reference_curve is not None:
            spec = _with_series_reference_curve(
                spec,
                step.focus_statistic or _DEFAULT_FOCUS_STATISTIC,
                self._pooling_reference_curve,
            )
        self._pooling_reference_curve = None
        self._set_result_view_spec(
            spec, source_label="Generated examples — pooled", data_mode="sandbox"
        )
        self.focus_statistic(
            step.focus_statistic or _DEFAULT_FOCUS_STATISTIC,
            curve_mode=self._tutorial_step_curve_mode(step),
        )
        self._apply_staged_tutorial_visibility(step)
        self._status_lbl.setText("Pooled comparison complete.")
        self._raise_self()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # First show in generated mode: compute the statistic so the chart is live.
        self._ensure_tutorial_comparison()

    def _raise_self(self) -> None:
        """Keep the dialog above the main window while the tutorial is driven.

        Rebuilding the field, swapping tabs, re-creating plot widgets, and async
        comparison runs can drop this modeless dialog behind the Browse/main window
        on macOS. A synchronous ``raise_()`` inside the click handler is too early:
        Qt/Cocoa can re-order the stack *after* the handler returns. Defer the raise
        to the next event-loop tick so it lands after the stack has settled.
        """
        if not self.isVisible():
            return
        QTimer.singleShot(0, self._raise_now)

    def _raise_now(self) -> None:
        try:
            if not self.isVisible():
                return
            # Raise only this dialog. Raising the owning Image Viewer too (an earlier
            # belt-and-suspenders) reordered the Image Viewer/Browse windows in the
            # background on every async completion.
            self.raise_()
            self.activateWindow()
        except Exception:
            pass

    def _select_tutorial_tab(self, tab_name: str) -> None:
        if not hasattr(self, "_tabs"):
            return
        target = str(tab_name or "").lower()
        for index in range(self._tabs.count()):
            if self._tabs.tabText(index).lower() == target:
                self._tabs.setCurrentIndex(index)
                return

    def _refresh_tutorial_text(self) -> None:
        if not hasattr(self, "_tutorial_step_lbl"):
            return
        tutorial = self._current_tutorial()
        step = self._current_tutorial_step_obj()
        has_next = (
            self._tutorial_step_index < len(tutorial.steps) - 1
            or bool(self._next_tutorial_key())
        )
        has_prev = self._tutorial_step_index > 0 or bool(self._prev_tutorial_key())

        # Prominent current-lesson title + linear progress.
        self._tutorial_title_lbl.setText(step.title)
        position, total = self._tutorial_position()
        self._tutorial_progress_lbl.setText(f"Lesson {position} of {total}")

        # Navigation always available and named by its destination, so the eye
        # never confuses a forward action with the current lesson.
        self._prev_tutorial_btn.setVisible(True)
        self._prev_tutorial_btn.setEnabled(has_prev)
        self._prev_tutorial_btn.setText(
            self._elide("◂ " + (self._prev_step_title() or "Previous"))
        )
        self._next_tutorial_btn.setVisible(True)
        self._next_tutorial_btn.setEnabled(has_next)
        self._next_tutorial_btn.setText(
            self._elide((self._next_step_title() or "Finish") + " ▸")
        )

        action_button = self._tutorial_action_button(step)
        run_text = step.primary_action or step.action_text or "Run this example"
        self._run_tutorial_btn.setText(run_text if action_button == "run" else "Run this example")
        self._load_tutorial_btn.setText(
            step.primary_action or step.action_text or "Load example"
        )
        # The call-to-action button is only for steps that actually run or load;
        # navigation steps are handled by Previous/Next above.
        self._load_tutorial_btn.setVisible(action_button == "load")
        self._run_tutorial_btn.setVisible(action_button == "run")
        self._restart_tutorial_btn.setVisible(True)
        self._restart_tutorial_btn.setText("Restart tutorial")

        # Body: the question, what to look for, and context chips (title is above).
        question = step.question or step.body
        look_for = step.look_for or step.statistic_hint
        parts = [question]
        if look_for:
            parts.append(f"<span style='color:#7cc7ff;'><b>Look for:</b> {look_for}</span>")
        context_chips = []
        if step.model_label:
            context_chips.append(f"Model: {step.model_label}")
        if step.statistic_label:
            context_chips.append(f"Statistic: {step.statistic_label}")
        if context_chips:
            parts.append(
                "<span style='color:#c9d1d9; font-size: 10pt;'>"
                + " &nbsp; ".join(context_chips)
                + "</span>"
            )
        if hasattr(self, "_tutorial_why_lbl"):
            self._tutorial_why_lbl.setText("")
            self._tutorial_why_frame.setVisible(False)
        self._tutorial_step_lbl.setText("<br>".join(p for p in parts if p))
        self._refresh_tutorial_detail(step)
        self._refresh_tutorial_action_style()

    @staticmethod
    def _elide(text: str, limit: int = 30) -> str:
        text = str(text)
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    def _tutorial_position(self) -> tuple[int, int]:
        """1-based index of the current step across all examples, and the total."""
        total = 0
        position = 1
        current_key = self.current_tutorial_key
        for example in _TUTORIALS:
            if example.key == current_key:
                position = total + min(self._tutorial_step_index, len(example.steps) - 1) + 1
            total += len(example.steps)
        return position, max(total, 1)

    def _prev_tutorial_key(self) -> str:
        current = self.current_tutorial_key
        for index, example in enumerate(_TUTORIALS):
            if example.key == current and index > 0:
                return _TUTORIALS[index - 1].key
        return ""

    def _prev_step_title(self) -> str:
        tutorial = self._current_tutorial()
        if self._tutorial_step_index > 0:
            return tutorial.steps[self._tutorial_step_index - 1].title
        prev_key = self._prev_tutorial_key()
        if prev_key:
            prev = self._tutorial_by_key(prev_key)
            return prev.steps[-1].title if prev.steps else prev.title
        return ""

    def _next_step_title(self) -> str:
        tutorial = self._current_tutorial()
        if self._tutorial_step_index < len(tutorial.steps) - 1:
            return tutorial.steps[self._tutorial_step_index + 1].title
        next_key = self._next_tutorial_key()
        if next_key:
            nxt = self._tutorial_by_key(next_key)
            return nxt.steps[0].title if nxt.steps else nxt.title
        return ""

    def _refresh_tutorial_detail(self, step: ParticleTutorialStep) -> None:
        if not hasattr(self, "_tutorial_detail_container"):
            return
        look_rows = []
        for label, value in (
            ("Model", step.model_label),
            ("Statistic", step.statistic_label),
            ("More detail", step.more_detail),
            ("Change", step.what_changes),
            ("Expected", step.expected_effect),
            ("Check", step.where_to_check),
        ):
            if value:
                look_rows.append(f"<b>{label}:</b> {value}")
        self._tutorial_look_lbl.setText("<br>".join(look_rows))
        self._tutorial_look_frame.setVisible(bool(look_rows))
        caution = step.caution or step.limitation
        self._tutorial_careful_lbl.setText(
            f"<b>Careful:</b> {caution}" if caution else ""
        )
        self._tutorial_careful_frame.setVisible(bool(caution))
        has_detail = bool(look_rows or caution)
        self._tutorial_look_frame.setVisible(bool(look_rows))
        self._tutorial_detail_btn.setEnabled(has_detail)
        self._tutorial_detail_container.setVisible(has_detail and self._tutorial_detail_btn.isChecked())

    def _tutorial_action_button(self, step: ParticleTutorialStep) -> str:
        action = str(step.action_kind or step.action_button or "").lower()
        if action == "run":
            return "run"
        if action == "load":
            return "load"
        if action == "restart":
            return "restart"
        if action == "next_example":
            return "next_example" if self._next_tutorial_key() else "none"
        if self._next_tutorial_btn.isEnabled():
            return "next"
        return "none"

    def _refresh_tutorial_action_style(self) -> None:
        if not hasattr(self, "_run_tutorial_btn"):
            return
        buttons = (
            self._load_tutorial_btn,
            self._run_tutorial_btn,
            self._prev_tutorial_btn,
            self._next_tutorial_btn,
            self._restart_tutorial_btn,
        )
        for button in buttons:
            button.setStyleSheet("")
        # The red Exit button keeps its style at all times.
        self._exit_tutorial_btn.setStyleSheet(_TUTORIAL_EXIT_STYLE)
        if not self._tutorial_active:
            return
        action_button = self._tutorial_action_button(self._current_tutorial_step_obj())
        if action_button in {"next", "next_example"} and self._next_tutorial_btn.isEnabled():
            self._next_tutorial_btn.setStyleSheet(_TUTORIAL_ACTION_STYLE)
        elif action_button == "run" and self._run_tutorial_btn.isEnabled():
            self._run_tutorial_btn.setStyleSheet(_TUTORIAL_ACTION_STYLE)
        elif action_button == "load" and self._load_tutorial_btn.isEnabled():
            self._load_tutorial_btn.setStyleSheet(_TUTORIAL_ACTION_STYLE)
        elif action_button == "restart":
            self._restart_tutorial_btn.setStyleSheet(_TUTORIAL_ACTION_STYLE)
        self._apply_tutorial_highlights()

    def _tutorial_control_widgets(self) -> dict[str, tuple[QWidget, ...]]:
        statistic = getattr(self, "_statistic_buttons", {})
        return {
            "mode": (self._mode_cb,),
            "pattern": (self._pattern_cb,),
            "ordered_lattice": (self._ordered_lattice_cb,),
            "ordered_background": (self._ordered_background_cb,),
            "n": (self._n_spin,),
            "generated_seed": (self._seed_spin,),
            "generated_model": (self._generated_model_cb,),
            "generated_simulations": (self._sim_spin,),
            "field_size": (self._width_spin, self._height_spin),
            "data_hard_core_radius": (self._hard_core_radius_spin,),
            "link_hard_core_radii": (self._link_hard_core_radii_cb,),
            "hard_core_radius": (self._model_hard_core_radius_spin,),
            "model_hard_core_radius": (self._model_hard_core_radius_spin,),
            "source": (self._source_cb,),
            "region": (self._region_cb,),
            "real_model": (self._real_model_cb,),
            "real_simulations": (self._real_sim_spin,),
            "real_seed": (self._real_seed_spin,),
            "feature_sets": (self._feature_sets_list,),
            "feature_layer": (self._feature_layer_set_cb,),
            "run_comparison": (self._run_btn,),
            "run_tutorial": (self._run_tutorial_btn,),
            "run_selected_sets": (self._run_feature_sets_btn,),
            "load_tutorial": (self._load_tutorial_btn,),
            "next": (self._next_tutorial_btn,),
            "restart": (self._restart_tutorial_btn,),
            "layer_observed": (self._observed_layer_cb,),
            "layer_simulated": (self._simulation_layer_cb,),
            "layer_features": (self._feature_layer_cb,),
            "layer_region": (self._region_layer_cb,),
            "stat_pair": tuple(
                w for w in (statistic.get("pair_correlation_g_r"),) if w is not None
            ),
            "stat_nearest": tuple(
                w for w in (statistic.get("nearest_neighbor_distribution"),) if w is not None
            ),
            "stat_ripley": tuple(
                w for w in (statistic.get("ripley_l_function"),) if w is not None
            ),
            "stat_clusters": tuple(
                w for w in (statistic.get("cluster_size_counts"),) if w is not None
            ),
            "stat_directional": tuple(
                w for w in (statistic.get("pair_correlation_g_r_theta"),) if w is not None
            ),
            "stat_psi6": tuple(
                w for w in (statistic.get("bond_order_psi6"),) if w is not None
            ),
            "stat_psi4": tuple(
                w for w in (statistic.get("bond_order_psi4"),) if w is not None
            ),
        }

    def _tutorial_action_control(self, step: ParticleTutorialStep) -> str:
        action = self._tutorial_action_button(step)
        if action in {"next", "next_example"}:
            return "next"
        if action == "run":
            return (
                "run_tutorial"
                if self._active_mode in {"generated", "sandbox"} and self._tutorial_active
                else "run_comparison"
            )
        if action == "load":
            return "load_tutorial"
        if action == "restart":
            return "restart"
        return ""

    def _clear_tutorial_highlights(self) -> None:
        if not hasattr(self, "_run_tutorial_btn") or self._updating_tutorial_highlights:
            return
        self._updating_tutorial_highlights = True
        try:
            seen: set[int] = set()
            for widgets in self._tutorial_control_widgets().values():
                for widget in widgets:
                    if id(widget) in seen:
                        continue
                    seen.add(id(widget))
                    if isinstance(widget, QPushButton) and widget in self._statistic_buttons.values():
                        style = _STAT_SELECTED_STYLE if widget.isChecked() else ""
                    elif widget is self._exit_tutorial_btn:
                        style = _TUTORIAL_EXIT_STYLE
                    elif widget is self._start_tutorial_btn:
                        style = _TUTORIAL_ACTION_STYLE
                    else:
                        style = ""
                    widget.setStyleSheet(style)
            self._exit_tutorial_btn.setStyleSheet(_TUTORIAL_EXIT_STYLE)
        finally:
            self._updating_tutorial_highlights = False

    def _apply_tutorial_highlights(self) -> None:
        if not getattr(self, "_tutorial_active", False):
            self._clear_tutorial_highlights()
            return
        if not hasattr(self, "_tutorial_cb") or self._updating_tutorial_highlights:
            return
        self._updating_tutorial_highlights = True
        try:
            self._updating_tutorial_highlights = False
            self._clear_tutorial_highlights()
            self._updating_tutorial_highlights = True
            registry = self._tutorial_control_widgets()
            tutorial = self._current_tutorial()
            step = self._current_tutorial_step_obj()
            active = list(self._tutorial_step_controls(step))
            action_control = self._tutorial_action_control(step)
            if action_control:
                active.append(action_control)
            active_set = set(active)
            visited: list[str] = []
            for earlier in tutorial.steps[: self._tutorial_step_index]:
                for key in self._tutorial_step_controls(earlier):
                    if key not in active_set and key not in visited:
                        visited.append(key)
            for key in visited:
                for widget in registry.get(key, ()):
                    widget.setStyleSheet(_TUTORIAL_VISITED_CONTROL_STYLE)
            for key in active:
                for widget in registry.get(key, ()):
                    if isinstance(widget, QPushButton):
                        widget.setStyleSheet(_TUTORIAL_ACTION_STYLE)
                    else:
                        widget.setStyleSheet(_TUTORIAL_ACTIVE_CONTROL_STYLE)
        finally:
            self._updating_tutorial_highlights = False

    def _on_mode_changed(self) -> None:
        if self._updating_mode:
            return
        mode = str(self._mode_cb.currentData() or "real")
        tutorial = mode == "generated"
        self._set_mode(mode, tutorial_active=tutorial)
        if tutorial:
            self._apply_tutorial_step(self._current_tutorial_step_obj(), stage_generated=True)
            self._ensure_tutorial_comparison()
        elif mode == "sandbox":
            self._refresh_generated_field()
        else:
            self._clear_tutorial_highlights()

    def _set_mode(self, mode: str, *, tutorial_active: bool | None = None) -> None:
        mode = mode if mode in {"landing", "real", "generated", "sandbox"} else "real"
        if tutorial_active is not None:
            self._tutorial_active = bool(tutorial_active)
        if mode == "landing":
            self._tutorial_active = False
        previous = self._active_mode
        is_landing = mode == "landing"
        is_sandbox_like = mode in {"generated", "sandbox"}
        target_index = {"real": 0, "generated": 1, "sandbox": 2}.get(mode)
        if target_index is not None and self._mode_cb.currentIndex() != target_index:
            self._updating_mode = True
            try:
                self._mode_cb.setCurrentIndex(target_index)
            finally:
                self._updating_mode = False
        self._active_mode = mode
        self._landing_panel.setVisible(is_landing)
        self._workspace_panel.setVisible(not is_landing)
        self._landing_btn.setVisible(not is_landing and not self._tutorial_active)
        self._mode_label.setVisible(not is_landing and not self._tutorial_active)
        self._mode_cb.setVisible(not is_landing and not self._tutorial_active)
        self._run_btn.setVisible(not is_landing and not self._tutorial_active)
        self._tutorial_panel.setVisible(self._tutorial_active)
        self._start_tutorial_btn.setVisible(False)
        if is_landing:
            self._run_btn.setEnabled(False)
            self._clear_tutorial_highlights()
            self._sync_workflow_actions()
            self._raise_self()
            return
        self._real_data_group.setVisible(not is_sandbox_like)
        self._real_model_group.setVisible(not is_sandbox_like)
        if hasattr(self, "_feature_sets_group"):
            self._feature_sets_group.setVisible(not is_sandbox_like)
        if hasattr(self, "_generated_data_group"):
            self._generated_data_group.setTitle(
                "Generated pattern" if mode == "sandbox" else "Generated / fake data"
            )
        if hasattr(self, "_generated_model_group"):
            self._generated_model_group.setTitle(
                "Comparison model" if mode == "sandbox" else "Model comparison"
            )
        self._generated_data_group.setVisible(is_sandbox_like)
        self._generated_model_group.setVisible(is_sandbox_like)
        if previous != mode:
            self._set_result_view_spec(
                _empty_view_spec("Run a comparison to populate result panels."),
                source_label=self._sandbox_source_label() if is_sandbox_like else "Particle Statistics",
                data_mode="sandbox" if is_sandbox_like else "real",
            )
        if is_sandbox_like:
            self._refresh_generated_field()
            self._refresh_tutorial_text()
        else:
            self._refresh_real_field()
            self._refresh_tutorial_text()
        if self._tutorial_active:
            self._apply_staged_tutorial_visibility(self._current_tutorial_step_obj())
            self._apply_tutorial_highlights()
        else:
            self._apply_staged_tutorial_visibility(None)
            self._clear_tutorial_highlights()
        self._sync_workflow_actions()
        self._raise_self()

    def _sync_workflow_actions(self) -> None:
        if not hasattr(self, "_run_comparison_action"):
            return
        is_landing = self._active_mode == "landing"
        is_tutorial = bool(self._tutorial_active)
        self._show_workflows_action.setEnabled(not is_landing)
        self._analyze_scan_points_action.setEnabled(not (self._active_mode == "real" and not is_tutorial))
        self._model_simulations_action.setEnabled(self._active_mode != "sandbox")
        self._start_tutorial_action.setEnabled(not is_tutorial)
        self._run_comparison_action.setEnabled(
            bool(self._run_btn.isEnabled() and self._run_btn.isVisible())
        )
        if not hasattr(self, "_show_observed_action"):
            return
        is_sandbox_like = self._active_mode in {"generated", "sandbox"}
        workspace_active = not is_landing
        global_controls_enabled = workspace_active and not is_tutorial
        for action, checkbox in (
            (self._show_observed_action, getattr(self, "_observed_layer_cb", None)),
            (self._show_model_action, getattr(self, "_simulation_layer_cb", None)),
            (self._show_feature_layer_action, getattr(self, "_feature_layer_cb", None)),
            (self._show_region_action, getattr(self, "_region_layer_cb", None)),
        ):
            if isinstance(checkbox, QCheckBox):
                action.setChecked(checkbox.isChecked())
                action.setEnabled(workspace_active and checkbox.isEnabled())
            else:
                action.setEnabled(False)
        self._new_pattern_action.setEnabled(
            global_controls_enabled and is_sandbox_like and self._new_pattern_btn.isEnabled()
        )
        self._reset_pattern_action.setEnabled(
            global_controls_enabled and is_sandbox_like and self._reset_btn.isEnabled()
        )
        self._refresh_sources_action.setEnabled(
            global_controls_enabled and self._active_mode == "real"
        )
        self._clear_real_action.setEnabled(
            global_controls_enabled and self._active_mode == "real"
        )
        if hasattr(self, "_link_hard_core_radii_cb"):
            self._link_hard_core_radii_action.setChecked(
                self._link_hard_core_radii_cb.isChecked()
            )
            self._link_hard_core_radii_action.setEnabled(
                global_controls_enabled and self._link_hard_core_radii_cb.isEnabled()
            )
        current_model = (
            str(self._generated_model_cb.currentData() or "")
            if is_sandbox_like
            else str(self._real_model_cb.currentData() or "")
        )
        if current_model == "homogeneous_poisson":
            current_model = "poisson"
        for model_id, action in self._model_actions.items():
            action.setChecked(model_id == current_model)
            action.setEnabled(global_controls_enabled)
        for statistic_id, action in self._statistic_actions.items():
            action.setChecked(statistic_id == self._focused_statistic)
            action.setEnabled(workspace_active and not is_tutorial)
        self._show_definitions_action.setEnabled(True)
        self._definitions_tutorial_action.setEnabled(not is_tutorial)

    def _populate_sources(self) -> None:
        self._source_cb.clear()
        for source in self._point_sources:
            label = str(getattr(source, "label", "") or "Point source")
            source_type = str(getattr(source, "source_type", "") or "points")
            self._source_cb.addItem(
                f"{label} ({_point_count(source)} points, {source_type.replace('_', ' ')})",
                label,
            )
        if not self._point_sources:
            self._source_cb.addItem("No point source available", "")

    def _sync_feature_sets_from_store(self) -> None:
        """Resync the cached set list from the shared store, when one is present."""
        if self._feature_set_store_ref is not None:
            self._feature_sets = list(self._feature_set_store_ref.all())

    def add_feature_sets(self, sets: Any, *, select_first: bool = True) -> None:
        """Add imported/built feature sets to the store and refresh the list."""
        added = list(sets or ())
        if not added:
            return
        if self._feature_set_store_ref is not None:
            for fs in added:
                self._feature_set_store_ref.add(fs)
        else:
            self._feature_sets = list(self._feature_sets) + added
        self._set_mode("real")
        self._populate_feature_sets()
        if select_first:
            self.select_feature_set(added[0].set_id)

    def import_points_from_disk(self) -> None:
        """Import a CSV/JSON point table from disk as one or more feature sets."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from probeflow.measurements.point_table_io import (
            load_point_table,
            sniff_point_table,
        )

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import point table",
            "",
            "Point tables (*.csv *.json);;CSV files (*.csv);;JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            preview = sniff_point_table(path)
        except Exception as exc:  # noqa: BLE001 - report to the user
            QMessageBox.warning(self, "Import failed", f"Could not read file:\n{exc}")
            return

        units = scan_range_m = image_shape = None
        if preview.needs_calibration:
            from probeflow.gui.dialogs.import_points import ImportPointsDialog

            cal_dlg = ImportPointsDialog(preview, theme=self._theme, parent=self)
            if cal_dlg.exec() != QDialog.Accepted:
                return
            units, scan_range_m, image_shape = cal_dlg.result_calibration()

        try:
            sets = load_point_table(
                path, units=units, scan_range_m=scan_range_m, image_shape=image_shape
            )
        except Exception as exc:  # noqa: BLE001 - report to the user
            QMessageBox.warning(self, "Import failed", f"Could not load points:\n{exc}")
            return
        if not sets:
            self._status_lbl.setText("No points found in file.")
            return
        self.add_feature_sets(sets)
        total = sum(s.point_count for s in sets)
        self._status_lbl.setText(
            f"Imported {len(sets)} feature set(s), {total} points. Tick to analyse."
        )

    def save_feature_sets_to_disk(self) -> None:
        """Save all current feature sets to a JSON file (round-trips on import)."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        from probeflow.measurements.feature_sets import FeatureSetStore

        self._sync_feature_sets_from_store()
        if not self._feature_sets:
            self._status_lbl.setText("No feature sets to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save feature sets", "probeflow_feature_sets.json", "JSON files (*.json)"
        )
        if not path:
            return
        store = self._feature_set_store_ref or FeatureSetStore(list(self._feature_sets))
        try:
            store.save(path)
        except Exception as exc:  # noqa: BLE001 - report to the user
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._status_lbl.setText(f"Saved {len(self._feature_sets)} feature set(s) to disk.")

    def _export_base_label(self) -> str:
        label = self._current_source_label() if hasattr(self, "_source_cb") else ""
        return label or "particle_statistics"

    def _export_results_csv(self) -> None:
        """Export per-statistic curve CSVs + a verdicts CSV to a chosen folder."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        if not _view_spec_has_result(self._last_view_spec):
            self._status_lbl.setText("Run a comparison before exporting.")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Export statistics CSVs to folder"
        )
        if not out_dir:
            return
        from probeflow.measurements.adstat_export import export_result_csvs

        try:
            written = export_result_csvs(
                self._last_view_spec, out_dir, base=self._export_base_label()
            )
        except Exception as exc:  # noqa: BLE001 - report to the user
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        if not written:
            self._status_lbl.setText("No curve/verdict data to export for this result.")
            return
        self._status_lbl.setText(f"Exported {len(written)} CSV file(s) to {out_dir}.")

    def _export_results_json(self) -> None:
        """Export the entire result view spec to a single JSON file."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        if not _view_spec_has_result(self._last_view_spec):
            self._status_lbl.setText("Run a comparison before exporting.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export result JSON",
            "particle_statistics_result.json",
            "JSON files (*.json)",
        )
        if not path:
            return
        from probeflow.measurements.adstat_export import export_result_json

        try:
            export_result_json(self._last_view_spec, path)
        except Exception as exc:  # noqa: BLE001 - report to the user
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self._status_lbl.setText("Exported result JSON.")

    def _populate_feature_sets(self) -> None:
        if not hasattr(self, "_feature_sets_list"):
            return
        self._sync_feature_sets_from_store()
        checked_ids = set(self._checked_feature_set_ids())
        self._feature_sets_list.clear()
        for fs in self._feature_sets:
            label = f"{fs.name}  ·  {fs.point_count} pts"
            item = QListWidgetItem(label, self._feature_sets_list)
            item.setData(Qt.UserRole, fs.set_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if fs.set_id in checked_ids else Qt.Unchecked
            )
        empty = not self._feature_sets
        self._feature_sets_group.setVisible(self._active_mode not in {"generated", "sandbox"})
        self._run_feature_sets_btn.setEnabled(not empty)
        if empty:
            placeholder = QListWidgetItem(
                "No saved sets yet — use 'Send to Particle Statistics' in Feature Finder.",
                self._feature_sets_list,
            )
            placeholder.setFlags(Qt.NoItemFlags)
        if hasattr(self, "_feature_layer_set_cb"):
            previous = str(self._feature_layer_set_cb.currentData() or "")
            self._feature_layer_set_cb.blockSignals(True)
            self._feature_layer_set_cb.clear()
            self._feature_layer_set_cb.addItem("(none)", "")
            for fs in self._feature_sets:
                self._feature_layer_set_cb.addItem(f"{fs.name}  ·  {fs.point_count} pts", fs.set_id)
            _set_combo_value(self._feature_layer_set_cb, previous)
            self._feature_layer_set_cb.blockSignals(False)
            self._sync_feature_layer_controls()

    def _on_real_model_changed(self) -> None:
        self._sync_feature_layer_controls()
        self._refresh_real_field()
        self._sync_workflow_actions()

    def _sync_feature_layer_controls(self) -> None:
        """Enable the feature-layer picker only for the measured-feature model."""
        if not hasattr(self, "_feature_layer_set_cb") or not hasattr(self, "_real_model_cb"):
            return
        is_feature = str(self._real_model_cb.currentData() or "") == "measured_feature_poisson"
        self._feature_layer_set_cb.setEnabled(is_feature)

    def _selected_feature_layer(self) -> Any | None:
        if not hasattr(self, "_feature_layer_set_cb"):
            return None
        set_id = str(self._feature_layer_set_cb.currentData() or "")
        if not set_id:
            return None
        for fs in self._feature_sets:
            if fs.set_id == set_id:
                return fs
        return None

    def _checked_feature_set_ids(self) -> list[str]:
        if not hasattr(self, "_feature_sets_list"):
            return []
        ids: list[str] = []
        for index in range(self._feature_sets_list.count()):
            item = self._feature_sets_list.item(index)
            if item.checkState() == Qt.Checked:
                set_id = item.data(Qt.UserRole)
                if set_id:
                    ids.append(str(set_id))
        return ids

    def _selected_feature_sets(self) -> list[Any]:
        checked = set(self._checked_feature_set_ids())
        return [fs for fs in self._feature_sets if fs.set_id in checked]

    def select_feature_set(self, set_id: str) -> None:
        """Check a saved set by id (used by the Feature Finder hand-off)."""
        if not hasattr(self, "_feature_sets_list"):
            return
        for index in range(self._feature_sets_list.count()):
            item = self._feature_sets_list.item(index)
            if str(item.data(Qt.UserRole) or "") == str(set_id):
                item.setCheckState(Qt.Checked)
            elif item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Unchecked)

    def run_selected_feature_sets(self) -> None:
        sets = self._selected_feature_sets()
        if not sets:
            self._status_lbl.setText("Tick one or more saved feature sets first.")
            return
        model = str(self._real_model_cb.currentData() or "poisson")
        feature_layer = None
        if model == "measured_feature_poisson":
            if len(sets) != 1:
                self._status_lbl.setText(
                    "Measured-feature Poisson runs on a single image — tick exactly one tested set."
                )
                return
            feature_layer = self._selected_feature_layer()
            if feature_layer is None:
                self._status_lbl.setText(
                    "Choose an independent Feature layer set for the Measured-feature model."
                )
                return
            if feature_layer.set_id == sets[0].set_id:
                self._status_lbl.setText(
                    "The feature layer must be a different set from the tested particles."
                )
                return
        self._set_mode("real")
        self._generation += 1
        generation = self._generation
        self._set_controls_enabled(False)
        if feature_layer is not None:
            label = f"{sets[0].name} vs feature layer {feature_layer.name}"
        else:
            label = sets[0].name if len(sets) == 1 else f"Combined: {len(sets)} images"
        self._status_lbl.setText(f"Computing {label}…")
        request = AdStatStatisticsRequest(
            point_source_label=label,
            region_mode="full",
            roi_or_mask=None,
            models=(model,),
            n_simulations=int(self._real_sim_spin.value()),
            random_seed=int(self._real_seed_spin.value()),
        )
        worker = _ParticleFeatureSetWorker(
            generation=generation,
            feature_sets=sets,
            request=request,
            feature_layer=feature_layer,
        )
        worker.signals.finished.connect(self._on_feature_sets_worker_finished)
        self._pool.start(worker)

    def _on_feature_sets_worker_finished(self, generation: int, spec: Any, error: str) -> None:
        if int(generation) != self._generation:
            return
        self._set_controls_enabled(True)
        if error or spec is None:
            message = error or "Combined comparison failed."
            self._status_lbl.setText(message)
            self._set_result_view_spec(
                _empty_view_spec(message), source_label="Feature sets", data_mode="real"
            )
            return
        sets = self._selected_feature_sets()
        total = sum(fs.point_count for fs in sets)
        label = (
            sets[0].name if len(sets) == 1
            else f"Combined: {len(sets)} images, {total} points"
        )
        self._set_result_view_spec(spec, source_label=label, data_mode="real")
        self._status_lbl.setText(f"Comparison complete for {label}.")
        self._raise_self()

    def _populate_regions(self) -> None:
        self._region_cb.clear()
        if self._active_area_roi is not None:
            self._region_cb.addItem("Active area ROI", "roi")
        if self._active_mask is not None:
            self._region_cb.addItem("Active mask", "mask")
        self._region_cb.addItem("Full image", "full")

    def _sync_generated_controls_from_state(self) -> None:
        if self._sandbox_state is None:
            for widget in self._generated_controls:
                widget.setEnabled(False)
            return
        config = self._sandbox_state.config
        self._updating_generated_controls = True
        try:
            _set_combo_value(self._pattern_cb, config.pattern)
            _set_combo_value(self._generated_model_cb, self._sandbox_state.active_model)
            self._n_spin.setValue(int(config.n))
            self._width_spin.setValue(float(config.width_nm))
            self._height_spin.setValue(float(config.height_nm))
            self._seed_spin.setValue(int(config.seed))
            self._sim_spin.setValue(int(config.n_simulations))
            self._hard_core_radius_spin.setValue(float(config.hard_core_radius_nm))
            model_radius = getattr(config, "model_hard_core_radius_nm", None)
            if model_radius is None:
                model_radius = config.hard_core_radius_nm
            if (
                hasattr(self, "_link_hard_core_radii_cb")
                and self._link_hard_core_radii_cb.isChecked()
            ):
                model_radius = config.hard_core_radius_nm
            self._model_hard_core_radius_spin.setValue(float(model_radius))
            if hasattr(config, "ordered_island_lattice"):
                _set_combo_value(
                    self._ordered_lattice_cb,
                    getattr(config, "ordered_island_lattice", "triangular"),
                )
            if hasattr(config, "ordered_island_background"):
                _set_combo_value(
                    self._ordered_background_cb,
                    getattr(config, "ordered_island_background", "none"),
                )
        finally:
            self._updating_generated_controls = False
        self._sync_generated_model_radius_controls()
        self._sync_ordered_island_controls()
        self._refresh_sandbox_warning()

    def _refresh_sandbox_warning(self) -> None:
        if not hasattr(self, "_sandbox_warning_lbl"):
            return
        if self._sandbox_state is None:
            self._sandbox_warning_lbl.setText("")
            return
        config = self._sandbox_state.config
        area = max(float(config.width_nm) * float(config.height_nm), 1e-9)
        model_radius = getattr(config, "model_hard_core_radius_nm", None)
        if model_radius is None:
            model_radius = config.hard_core_radius_nm
        radius_for_warning = max(
            float(config.hard_core_radius_nm), float(model_radius)
        )
        exclusion_area = (
            int(config.n) * math.pi * (radius_for_warning / 2.0) ** 2
        )
        warnings = list(getattr(self._sandbox_state, "warnings", ()) or ())
        if exclusion_area / area > 0.25:
            warnings.append(
                "Large N or hard-core radius can be slow because overlapping placements "
                "are rejected."
            )
        self._sandbox_warning_lbl.setText("\n".join(warnings))

    def _refresh_real_field(self) -> None:
        if self._active_mode in {"landing", "generated", "sandbox"}:
            return
        source = self._selected_source()
        width_nm, height_nm = _field_size_nm(self._scan, self._image_shape)
        mask, region_label = self._selected_region_mask_and_label()
        if source is None and not self._feature_sets:
            status = _REAL_EMPTY_STATE_MESSAGE
            points = np.empty((0, 2), dtype=float)
            source_label = ""
        elif source is None:
            status = "Tick a saved feature set below and Run selected sets, or detect points with Feature Finder."
            points = np.empty((0, 2), dtype=float)
            source_label = ""
        else:
            points = _source_points_nm(source)
            source_label = str(getattr(source, "label", "") or "Point source")
            status = "Ready. Choose a point source and analysis region, then run comparison."
        self._field.set_points(
            points,
            field_size_nm=(width_nm, height_nm),
            mode="real",
            source_label=source_label,
            region_label=region_label,
            model_label=str(self._real_model_cb.currentText() or "Homogeneous Poisson"),
            status=status,
            mask=mask,
        )
        self._update_info(status=status)
        self._sync_layer_controls()
        self._run_btn.setEnabled(self._scan is not None and source is not None)
        self._sync_workflow_actions()

    def _refresh_generated_field(self) -> None:
        if self._sandbox_state is None:
            message = self._sandbox_error or "Generated examples require the optional AdStat engine."
            self._field.set_points(
                np.empty((0, 2), dtype=float),
                field_size_nm=(100.0, 100.0),
                mode="generated",
                source_label=self._sandbox_source_label(),
                region_label="Synthetic field",
                status=message,
            )
            self._update_info(status=message)
            self._sync_layer_controls()
            self._run_btn.setEnabled(False)
            self._sync_workflow_actions()
            return
        config = self._sandbox_state.config
        try:
            preview = adstat_sandbox_preview(
                config,
                active_model=str(self._sandbox_state.active_model),
            )
        except Exception as exc:  # noqa: BLE001 - show optional engine errors in the UI
            message = f"Could not preview generated pattern: {exc}"
            self._field.set_points(
                np.empty((0, 2), dtype=float),
                field_size_nm=(float(config.width_nm), float(config.height_nm)),
                mode="generated",
                source_label=self._sandbox_source_label(),
                region_label="Synthetic field",
                status=message,
            )
            self._update_info(status=message)
            self._sync_layer_controls()
            self._run_btn.setEnabled(False)
            self._sync_workflow_actions()
            return
        status = self._sandbox_state.status_text()
        self._field.set_points(
            preview.xy_nm,
            field_size_nm=(preview.width_nm, preview.height_nm),
            mode="generated",
            source_label=(
                self._sandbox_source_label()
                if self._active_mode == "sandbox"
                else _PATTERN_LABELS.get(config.pattern, str(config.pattern))
            ),
            region_label="Synthetic field",
            model_label=_MODEL_LABELS.get(str(self._sandbox_state.active_model), str(self._sandbox_state.active_model)),
            status=status,
            simulated_xy_nm=preview.simulated_xy_nm,
            feature_xy_nm=preview.feature_xy_nm if config.pattern == "feature_biased" else None,
        )
        self._update_info(status=status)
        self._refresh_sandbox_warning()
        self._sync_layer_controls()
        self._run_btn.setEnabled(True)
        self._sync_workflow_actions()

    def _stage_generated_from_controls(self) -> None:
        if self._updating_generated_controls or self._sandbox_state is None:
            return
        try:
            changes = {
                "pattern": str(self._pattern_cb.currentData()),
                "n": int(self._n_spin.value()),
                "width_nm": float(self._width_spin.value()),
                "height_nm": float(self._height_spin.value()),
                "seed": int(self._seed_spin.value()),
                "hard_core_radius_nm": float(self._hard_core_radius_spin.value()),
                "n_simulations": int(self._sim_spin.value()),
            }
            if hasattr(self._sandbox_state.config, "model_hard_core_radius_nm"):
                model_radius = self._linked_or_model_hard_core_radius()
                changes["model_hard_core_radius_nm"] = model_radius
            if hasattr(self._sandbox_state.config, "ordered_island_lattice"):
                changes["ordered_island_lattice"] = str(
                    self._ordered_lattice_cb.currentData() or "triangular"
                )
            if hasattr(self._sandbox_state.config, "ordered_island_background"):
                changes["ordered_island_background"] = str(
                    self._ordered_background_cb.currentData() or "none"
                )
            self._sandbox_state.stage(**changes)
        except ValueError as exc:
            self._status_lbl.setText(str(exc))
            self._sync_generated_controls_from_state()
            return
        self._refresh_sandbox_warning()
        self._sync_ordered_island_controls()
        if str(self._mode_cb.currentData() or "real") in {"generated", "sandbox"}:
            self._refresh_generated_field()

    def _on_generated_model_changed(self) -> None:
        if self._updating_generated_controls or self._sandbox_state is None:
            return
        self._sandbox_state.set_model(str(self._generated_model_cb.currentData()))
        self._sync_generated_model_radius_controls()
        self._sync_workflow_actions()
        if str(self._mode_cb.currentData() or "real") in {"generated", "sandbox"}:
            self._refresh_generated_field()

    def _sync_generated_model_radius_controls(self) -> None:
        if not hasattr(self, "_model_hard_core_radius_spin"):
            return
        is_hard_core = str(self._generated_model_cb.currentData() or "") == "hard_core_random"
        linked = (
            hasattr(self, "_link_hard_core_radii_cb")
            and self._link_hard_core_radii_cb.isChecked()
        )
        if hasattr(self, "_link_hard_core_radii_cb"):
            self._link_hard_core_radii_cb.setEnabled(
                self._generated_model_cb.isEnabled() and is_hard_core
            )
        self._model_hard_core_radius_spin.setEnabled(
            self._generated_model_cb.isEnabled() and is_hard_core and not linked
        )
        self._sync_workflow_actions()

    def _linked_or_model_hard_core_radius(self) -> float:
        data_radius = float(self._hard_core_radius_spin.value())
        linked = (
            hasattr(self, "_link_hard_core_radii_cb")
            and self._link_hard_core_radii_cb.isChecked()
        )
        if linked:
            if abs(float(self._model_hard_core_radius_spin.value()) - data_radius) > 1e-9:
                blocked = self._model_hard_core_radius_spin.blockSignals(True)
                try:
                    self._model_hard_core_radius_spin.setValue(data_radius)
                finally:
                    self._model_hard_core_radius_spin.blockSignals(blocked)
            return data_radius
        return float(self._model_hard_core_radius_spin.value())

    def _on_link_hard_core_radii_toggled(self, checked: bool) -> None:
        if not hasattr(self, "_model_hard_core_radius_spin"):
            return
        if checked:
            blocked = self._model_hard_core_radius_spin.blockSignals(True)
            try:
                self._model_hard_core_radius_spin.setValue(
                    float(self._hard_core_radius_spin.value())
                )
            finally:
                self._model_hard_core_radius_spin.blockSignals(blocked)
        self._sync_generated_model_radius_controls()
        self._stage_generated_from_controls()

    def _sync_ordered_island_controls(self) -> None:
        if not hasattr(self, "_ordered_lattice_cb"):
            return
        is_ordered = str(self._pattern_cb.currentData() or "") == "ordered_islands"
        for widget in (
            self._ordered_lattice_lbl,
            self._ordered_lattice_cb,
            self._ordered_background_lbl,
            self._ordered_background_cb,
        ):
            widget.setVisible(is_ordered)
        enabled = self._pattern_cb.isEnabled() and is_ordered
        self._ordered_lattice_cb.setEnabled(enabled)
        self._ordered_background_cb.setEnabled(enabled)

    def _run_comparison(self) -> None:
        if self._active_mode in {"generated", "sandbox"}:
            self._start_generated_worker("run")
        elif self._active_mode == "real":
            self._start_real_worker()

    def _start_real_worker(self) -> None:
        if str(self._real_model_cb.currentData() or "") == "measured_feature_poisson":
            self._status_lbl.setText(
                "Measured-feature Poisson uses saved sets: tick one tested set, pick a "
                "Feature layer set, then 'Run selected sets'."
            )
            return
        source = self._selected_source()
        if self._scan is None or source is None:
            self._refresh_real_field()
            return
        self._generation += 1
        generation = self._generation
        self._set_controls_enabled(False)
        self._status_lbl.setText("computing...")
        worker = _ParticleRealWorker(
            generation=generation,
            point_sources=self._point_sources,
            scan=self._scan,
            image_shape=self._image_shape,
            request=self._request_from_controls(),
        )
        worker.signals.finished.connect(self._on_real_worker_finished)
        self._pool.start(worker)

    def _start_generated_worker(self, operation: str) -> None:
        if self._sandbox_state is None:
            self._tutorial_run_in_progress = False
            self._refresh_generated_field()
            return
        self._sandbox_generation += 1
        generation = self._sandbox_generation
        self._set_controls_enabled(False)
        self._status_lbl.setText("computing...")
        worker = _ParticleSandboxWorker(self._sandbox_state, operation, generation)
        worker.signals.finished.connect(self._on_generated_worker_finished)
        self._pool.start(worker)

    def new_generated_pattern(self) -> None:
        self._start_generated_worker("new_pattern")

    def reset_generated(self) -> None:
        self._start_generated_worker("reset")

    def _on_real_worker_finished(self, generation: int, context: Any) -> None:
        if int(generation) != self._generation:
            return
        self._set_controls_enabled(True)
        if not getattr(context, "ready", False):
            message = str(getattr(context, "status_message", "") or "Particle Statistics analysis failed.")
            self._status_lbl.setText(message)
            self._set_result_view_spec(
                _empty_view_spec(message),
                source_label=getattr(context, "point_source_label", None) or self._current_source_label(),
                data_mode="real",
            )
            return
        label = getattr(context, "point_source_label", None) or self._current_source_label()
        self._set_result_view_spec(context.view_spec, source_label=label, data_mode="real")
        self._status_lbl.setText(f"Particle Statistics comparison complete for {label}.")
        self._raise_self()

    def _on_generated_worker_finished(self, generation: int, state: Any, error: str) -> None:
        if int(generation) != self._sandbox_generation:
            return
        if self._active_mode not in {"generated", "sandbox"}:
            # The user left the tutorial while this comparison was running; do not
            # overwrite the real-data view with stale generated results.
            self._set_controls_enabled(True)
            return
        self._set_controls_enabled(True)
        was_tutorial_run = self._tutorial_run_in_progress
        self._tutorial_run_in_progress = False
        if error:
            self._status_lbl.setText(str(error))
            return
        self._sandbox_state = state
        self._sync_generated_controls_from_state()
        spec = adstat_sandbox_view_spec(
            self._sandbox_state, include_ordering=self._include_ordering_enabled()
        )
        self._set_result_view_spec(
            spec,
            source_label=self._sandbox_source_label(),
            data_mode="sandbox",
        )
        self._refresh_generated_field()
        step = self._current_tutorial_step_obj() if self._tutorial_active else None
        tutorial = self._current_tutorial() if self._tutorial_active else None
        if self._tutorial_active and step is not None:
            self.focus_statistic(
                step.focus_statistic or _DEFAULT_FOCUS_STATISTIC,
                curve_mode=self._tutorial_step_curve_mode(step),
            )
        if (
            was_tutorial_run
            and step is not None
            and tutorial is not None
            and step.advance_after_run
            and self._tutorial_step_index < len(tutorial.steps) - 1
        ):
            self._tutorial_step_index += 1
            self._apply_tutorial_step(
                self._current_tutorial_step_obj(),
                stage_generated=False,
            )
        # An async comparison finishing rebuilds the result panels; re-assert z-order
        # so the dialog does not slip behind the Browse/main window.
        self._raise_self()

    def _request_from_controls(self) -> AdStatStatisticsRequest:
        return AdStatStatisticsRequest(
            point_source_label=self._current_source_label(),
            region_mode=str(self._region_cb.currentData() or "full"),
            roi_or_mask=self._selected_region_object(),
            models=(str(self._real_model_cb.currentData() or "poisson"),),
            n_simulations=int(self._real_sim_spin.value()),
            random_seed=int(self._real_seed_spin.value()),
            include_ordering=self._include_ordering_enabled(),
        )

    def _include_ordering_enabled(self) -> bool:
        cb = getattr(self, "_include_ordering_cb", None)
        return bool(cb.isChecked()) if cb is not None else False

    def _sync_ordering_card_state(self) -> None:
        enabled = self._include_ordering_enabled()
        for statistic_id, button in getattr(self, "_ordering_stat_buttons", {}).items():
            button.setEnabled(enabled)
            button.setToolTip(
                _statistic_row_description(statistic_id)
                if enabled
                else "Enable 'Include local-order checks' to compute this statistic."
            )
        if not enabled and self._focused_statistic in ORDERING_STATISTICS:
            self.focus_statistic(_DEFAULT_FOCUS_STATISTIC)

    def _on_include_ordering_toggled(self, checked: bool) -> None:
        self._sync_ordering_card_state()
        # Re-run / re-render so ordering verdicts appear or disappear in place.
        if self._active_mode in {"generated", "sandbox"}:
            state = self._sandbox_state
            if state is not None and getattr(state, "result", None) is not None:
                spec = adstat_sandbox_view_spec(state, include_ordering=checked)
                self._set_result_view_spec(
                    spec, source_label=self._sandbox_source_label(), data_mode="sandbox"
                )
        elif self._active_mode == "real" and _view_spec_has_result(self._last_view_spec):
            self._start_real_worker()

    def _set_ordering_for_step(self, step: ParticleTutorialStep) -> None:
        """Enable ordering only when a tutorial step teaches an ordering statistic."""
        cb = getattr(self, "_include_ordering_cb", None)
        if cb is None:
            return
        want = str(getattr(step, "focus_statistic", "")) in ORDERING_STATISTICS
        if cb.isChecked() == want:
            return
        cb.blockSignals(True)
        cb.setChecked(want)
        cb.blockSignals(False)
        self._sync_ordering_card_state()
        # Re-render an existing generated result so the ψ panels match the step.
        state = self._sandbox_state
        if (
            self._active_mode in {"generated", "sandbox"}
            and state is not None
            and getattr(state, "result", None) is not None
        ):
            spec = adstat_sandbox_view_spec(state, include_ordering=want)
            self._set_result_view_spec(
                spec, source_label=self._sandbox_source_label(), data_mode="sandbox"
            )

    def _selected_source(self) -> Any | None:
        label = self._current_source_label()
        for source in self._point_sources:
            if str(getattr(source, "label", "")) == label:
                return source
        return self._point_sources[0] if self._point_sources else None

    def _current_source_label(self) -> str:
        return str(self._source_cb.currentData() or "")

    def _selected_region_object(self) -> Any:
        mode = str(self._region_cb.currentData() or "full")
        if mode == "roi":
            return self._active_area_roi
        if mode == "mask":
            return self._active_mask
        return None

    def _selected_region_mask_and_label(self) -> tuple[np.ndarray | None, str]:
        mode = str(self._region_cb.currentData() or "full")
        if mode == "roi":
            return _mask_from_roi(self._active_area_roi, self._image_shape), "Active area ROI"
        if mode == "mask":
            return self._active_mask, "Active mask"
        return None, "Full image"

    def _update_info(self, *, status: str) -> None:
        model = self._field._model
        # In a soft-intro card the Field panel would only be clutter; keep it to a
        # one-line promise of what it will hold once an example runs.
        if (
            model.mode == "generated"
            and hasattr(self, "_tutorial_cb")
            and self._current_tutorial_step_obj().intro_card
        ):
            self._info_lbl.setText("Field summarises the model and its settings once an example runs.")
            self._status_lbl.setText(status)
            return
        lines = [
            f"Mode: {self._sandbox_source_label() if model.mode == 'generated' else 'Real scan points'}",
            f"Points: {len(model.observed_xy_nm)}",
        ]
        if model.model_label:
            lines.append(f"Model: {model.model_label}")
        # The detail below is useful but not needed at a glance; keep it last.
        lines.append(f"Field: {model.width_nm:g} x {model.height_nm:g} nm")
        if model.source_label:
            lines.append(f"Source: {model.source_label}")
        if model.region_label and model.region_label != "Synthetic field":
            lines.append(f"Region: {model.region_label}")
        self._info_lbl.setText("\n".join(lines))
        self._status_lbl.setText(status)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._mode_cb,
            self._source_cb,
            self._region_cb,
            self._real_model_cb,
            self._real_sim_spin,
            self._real_seed_spin,
            self._pattern_cb,
            self._ordered_lattice_cb,
            self._ordered_background_cb,
            self._n_spin,
            self._width_spin,
            self._height_spin,
            self._seed_spin,
            self._generated_model_cb,
            self._sim_spin,
            self._hard_core_radius_spin,
            self._link_hard_core_radii_cb,
            self._model_hard_core_radius_spin,
            self._new_pattern_btn,
            self._reset_btn,
            self._landing_btn,
            self._run_btn,
            self._refresh_sources_btn,
            self._observed_layer_cb,
            self._simulation_layer_cb,
            self._feature_layer_cb,
            self._region_layer_cb,
            self._tutorial_cb,
            self._load_tutorial_btn,
            self._run_tutorial_btn,
            self._prev_tutorial_btn,
            self._next_tutorial_btn,
            self._tutorial_detail_btn,
            self._restart_tutorial_btn,
        ):
            widget.setEnabled(bool(enabled))
        if enabled:
            self._sync_layer_controls()
            self._sync_generated_model_radius_controls()
            self._sync_ordered_island_controls()
            self._refresh_tutorial_text()
        self._sync_workflow_actions()

    def _sandbox_source_label(self) -> str:
        return "Model simulations" if self._active_mode == "sandbox" else "Generated examples"

    @staticmethod
    def _populate_combo(combo: QComboBox, values: tuple[str, ...], labels: dict[str, str]) -> None:
        combo.clear()
        for value in values:
            combo.addItem(labels.get(value, value.replace("_", " ").title()), value)


class _FieldTransform:
    def __init__(self, rect: QRectF, width_nm: float, height_nm: float):
        self.rect = rect
        self.width_nm = max(float(width_nm), 1e-9)
        self.height_nm = max(float(height_nm), 1e-9)

    def point(self, xy: Any) -> QPointF:
        x = float(xy[0])
        y = float(xy[1])
        return QPointF(
            self.rect.left() + (x / self.width_nm) * self.rect.width(),
            self.rect.top() + (y / self.height_nm) * self.rect.height(),
        )


def _aspect_fit_rect(available: QRectF, width_nm: float, height_nm: float) -> QRectF:
    """Return the largest centered rect that preserves physical field aspect."""

    target_ratio = max(float(width_nm), 1e-9) / max(float(height_nm), 1e-9)
    available_ratio = max(float(available.width()), 1.0) / max(float(available.height()), 1.0)
    if target_ratio >= available_ratio:
        width = float(available.width())
        height = width / target_ratio
    else:
        height = float(available.height())
        width = height * target_ratio
    x = available.left() + (available.width() - width) / 2.0
    y = available.top() + (available.height() - height) / 2.0
    return QRectF(x, y, max(1.0, width), max(1.0, height))


def _draw_marker_series(
    painter: QPainter,
    transform: _FieldTransform,
    points: np.ndarray,
    *,
    marker: str,
    color: str,
    radius: float,
    hollow: bool = False,
) -> None:
    for xy in np.asarray(points, dtype=float):
        if not np.isfinite(xy).all():
            continue
        _draw_marker(painter, transform.point(xy), marker, QColor(color), radius, hollow=hollow)


def _draw_marker(
    painter: QPainter,
    center: QPointF,
    marker: str,
    color: QColor,
    radius: float,
    *,
    hollow: bool = False,
) -> None:
    painter.save()
    painter.setPen(QPen(color, 1.5))
    painter.setBrush(Qt.NoBrush if hollow else color)
    if marker == "^":
        polygon = QPolygonF(
            [
                QPointF(center.x(), center.y() - radius),
                QPointF(center.x() - radius, center.y() + radius),
                QPointF(center.x() + radius, center.y() + radius),
            ]
        )
        painter.drawPolygon(polygon)
    elif marker == "s":
        painter.drawRect(QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2))
    elif marker == "x":
        painter.drawLine(QPointF(center.x() - radius, center.y() - radius), QPointF(center.x() + radius, center.y() + radius))
        painter.drawLine(QPointF(center.x() - radius, center.y() + radius), QPointF(center.x() + radius, center.y() - radius))
    else:
        painter.drawEllipse(center, radius, radius)
    painter.restore()


def _marker_style(mode: str) -> dict[str, str]:
    if _normalise_field_mode(mode) == "generated":
        return {"marker": "^", "color": "#f28e2b"}
    return {"marker": "o", "color": "#2f7ed8"}


def _bar_text_row(parent: QWidget, color: str) -> tuple[QFrame, QLabel]:
    """A word-wrapped label with a coloured left importance bar (SEMITIP-style)."""
    frame = QFrame(parent)
    frame.setStyleSheet(
        f"QFrame {{ border-left: 4px solid {color}; }}"
    )
    row = QHBoxLayout(frame)
    row.setContentsMargins(8, 2, 0, 2)
    label = QLabel("", frame)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    row.addWidget(label, 1)
    return frame, label


def _field_marker_radius(plot_width_px: float, n_points: int) -> float:
    """Scale field markers to point density so dense fields do not overlap.

    Mean point spacing in pixels is ~ plot_width / sqrt(N); a fraction of that gives
    a radius that shrinks as more points are packed in (≈3.6 px at N=120, ≈2.4 px at
    N=500), clamped to a legible range.
    """
    n = max(int(n_points), 1)
    spacing_px = float(plot_width_px) / math.sqrt(n)
    return max(2.0, min(4.0, 0.13 * spacing_px))


def _normalise_field_mode(mode: str) -> str:
    return "generated" if str(mode).lower() in {"generated", "sandbox", "learn"} else "real"


def _xy_array(value: Any) -> np.ndarray:
    arr = _xy_array_or_none(value)
    return np.empty((0, 2), dtype=float) if arr is None else arr


def _xy_array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr


def _mask_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=bool)
    if arr.ndim != 2 or not arr.any():
        return None
    return arr


def _valid_mask(mask: Any, image_shape: tuple[int, int] | None) -> np.ndarray | None:
    arr = _mask_or_none(mask)
    if arr is None:
        return None
    if image_shape is not None and arr.shape != tuple(image_shape):
        return None
    return arr


def _mask_from_roi(roi: Any, image_shape: tuple[int, int] | None) -> np.ndarray | None:
    if roi is None or image_shape is None:
        return None
    to_mask = getattr(roi, "to_mask", None)
    if not callable(to_mask):
        return None
    try:
        return _valid_mask(to_mask(tuple(image_shape)), image_shape)
    except Exception:
        return None


def _source_points_nm(source: Any) -> np.ndarray:
    points_m = getattr(source, "points_m", None)
    if points_m is None:
        return np.empty((0, 2), dtype=float)
    return _xy_array(points_m) * 1e9


def _field_size_nm(scan: Any, image_shape: tuple[int, int] | None) -> tuple[float, float]:
    scan_range = getattr(scan, "scan_range_m", None)
    if scan_range is not None:
        return float(scan_range[0]) * 1e9, float(scan_range[1]) * 1e9
    if image_shape is not None:
        return float(image_shape[1]), float(image_shape[0])
    return 100.0, 100.0


def _point_count(source: Any) -> int:
    try:
        return int(len(getattr(source, "points_px")))
    except Exception:
        return 0


def _theme_qcolor(theme: dict, keys: tuple[str, ...], default: str) -> QColor:
    for key in keys:
        value = theme.get(key)
        if value:
            return QColor(str(value))
    return QColor(default)


def _set_combo_value(combo: QComboBox, value: str) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def _panel_for_statistic(view_spec: Any, statistic_id: str) -> Any | None:
    for panel in tuple(getattr(view_spec, "panels", ()) or ()):
        if str(getattr(panel, "statistic", "")) == str(statistic_id):
            return panel
    return None


def _view_spec_has_result(view_spec: Any) -> bool:
    return bool(
        tuple(getattr(view_spec, "panels", ()) or ())
        or tuple(getattr(view_spec, "verdict_rows", ()) or ())
    )


def _statistic_row_description(statistic_id: str) -> str:
    """The plain-language description shown beside a statistic's selector button."""
    question = _guide_text(statistic_id, "focus_question")
    if statistic_id == _DEFAULT_FOCUS_STATISTIC:
        return f"Key plot — {question}"
    return question


def _display_statistic_title(statistic_id: str) -> str:
    return _STATISTIC_TITLES.get(
        str(statistic_id),
        _guide_text(statistic_id, "title")
        or _STATISTIC_LABELS.get(str(statistic_id), str(statistic_id)),
    )


def _plot_annotation_text(statistic_id: str) -> str:
    return _STATISTIC_ANNOTATIONS.get(str(statistic_id), "")


def _focus_read_text(statistic_id: str, curve_mode: str) -> str:
    if statistic_id == _DEFAULT_FOCUS_STATISTIC and curve_mode == "observed_only":
        return "Orange = g(r) measured from the test data. The model appears in the next step."
    return _SHORT_STAT_READS.get(
        statistic_id,
        _guide_text(statistic_id, "how_to_read"),
    )


def _series_focus_read_text(panel: Any) -> str:
    metadata = getattr(panel, "metadata", {}) or {}
    if metadata.get("reference_curves"):
        return (
            "Blue = pooled mean; blue band = image-to-image spread; "
            "orange = single-image reference."
        )
    return "Blue = pooled mean; blue band = image-to-image spread across images."


def _single_curve_reference_from_spec(
    view_spec: Any,
    statistic_id: str,
) -> dict[str, Any] | None:
    panel = _panel_for_statistic(view_spec, statistic_id)
    if panel is None:
        return None
    x = _finite_curve_array(getattr(panel, "x", None))
    y = _finite_curve_array(getattr(panel, "observed", None))
    if x is None or y is None or len(x) != len(y):
        return None
    return {
        "x": x.copy(),
        "y": y.copy(),
        "label": "single image reference",
        "color": "#ff9f1c",
    }


def _with_series_reference_curve(
    view_spec: Any,
    statistic_id: str,
    reference_curve: dict[str, Any],
) -> Any:
    panels = []
    changed = False
    for panel in tuple(getattr(view_spec, "panels", ()) or ()):
        if (
            str(getattr(panel, "kind", "")) == "series_curve"
            and str(getattr(panel, "statistic", "")) == str(statistic_id)
        ):
            metadata = dict(getattr(panel, "metadata", {}) or {})
            curves = tuple(metadata.get("reference_curves", ()) or ())
            metadata["reference_curves"] = (*curves, reference_curve)
            panel = replace(panel, metadata=metadata)
            changed = True
        panels.append(panel)
    if not changed:
        return view_spec
    return replace(view_spec, panels=tuple(panels))


def _finite_curve_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or len(array) == 0:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _guide_text(statistic_id: str, field: str) -> str:
    guide = _statistic_guide(statistic_id)
    if guide is not None:
        value = getattr(guide, field, "")
        if value:
            return str(value)
    fallback = _FALLBACK_STAT_GUIDES.get(str(statistic_id), {})
    return str(fallback.get(field, ""))


def _statistic_guide(statistic_id: str) -> Any | None:
    try:
        from adstat.education.howto import get_statistic_guide
    except Exception:
        return None
    try:
        return get_statistic_guide(statistic_id)
    except Exception:
        return None


def _empty_view_spec(message: str) -> Any:
    return type(
        "EmptyParticleStatisticsSpec",
        (),
        {
            "panels": (),
            "verdict_rows": (),
            "status_lines": (message,),
            "explainer": None,
            "metadata": {"has_result": False, "message": message},
        },
    )()
