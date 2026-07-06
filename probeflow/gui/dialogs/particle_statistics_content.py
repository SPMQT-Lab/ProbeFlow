"""User-facing content for the Particle Statistics tool.

All prose and metadata lives here — pattern/model display labels, statistic
titles, guides and quick-read texts, and the guided tutorial lessons
(`_TUTORIALS`).  Pure data + text lookups: no Qt, so lessons and wording can
be edited (or reviewed) without touching the dialog implementation in
:mod:`probeflow.gui.dialogs.particle_statistics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

_REAL_EMPTY_STATE_MESSAGE = (
    "No points yet — detect features with Feature Finder and 'Send to Particle "
    "Statistics', tick a saved feature set, or click Start tutorial to learn the tool."
)

_DEFAULT_FOCUS_STATISTIC = "pair_correlation_g_r"
_MODEL_SUMMARY_FOCUS = "model_summary"
_QUICK_SUMMARY_FOCUS = "quick_summary"
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
    _QUICK_SUMMARY_FOCUS: "Data summary",
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
    _QUICK_SUMMARY_FOCUS: "Descriptive summary — before any model",
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
    _QUICK_SUMMARY_FOCUS: {
        "title": "Data summary",
        "focus_question": "How many particles, how dense, how far apart?",
        "before_run": "Counting points and measuring nearest-neighbour distances…",
        "how_to_read": (
            "The dashed line marks the mean nearest-neighbour distance a random "
            "pattern of this density would have — a hint, not a test. Run a "
            "model comparison to test it."
        ),
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
    _QUICK_SUMMARY_FOCUS: (
        "Histogram left of the dashed line = closer than random; right = more spread out."
    ),
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
                why=(
                    "A visual impression of randomness is unreliable; a model "
                    "comparison replaces it with evidence."
                ),
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
                question=(
                    "The image is reduced to calibrated x,y particle positions — "
                    "that list is all the statistics ever see."
                ),
                look_for="The observed particles are the only layer shown.",
                why=(
                    "Everything downstream depends on detection quality and "
                    "calibration; wrong positions give confident-looking wrong verdicts."
                ),
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
        key="describe_data",
        title="02 - Describe the data",
        summary="Count, density, and spacing before any model.",
        steps=(
            ParticleTutorialStep(
                title="Describe the data",
                question=(
                    "How many particles, how dense, how far apart? The Data "
                    "summary answers this with no model run at all."
                ),
                look_for=(
                    "The dashed line marks the mean nearest-neighbour distance a "
                    "random pattern of this density would have."
                ),
                caution=(
                    "The dashed line is a hint, not a test. The simulation "
                    "envelope later in this tutorial is the test."
                ),
                why=(
                    "Descriptive numbers catch unit and detection mistakes early, "
                    "and the observed-versus-random spacing hint tells you which "
                    "model comparison is worth running."
                ),
                statistic_label="Data summary",
                visible_panel="plot",
                target_tab="Setup",
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=7,
                simulations=60,
                show_simulated=False,
                show_features=False,
                focus_statistic=_QUICK_SUMMARY_FOCUS,
            ),
        ),
    ),
    ParticleTutorialExample(
        key="model_baseline_observed",
        title="03 - Model baseline A",
        summary="Observed particles only.",
        steps=(
            ParticleTutorialStep(
                title="Observed data",
                question="These are the observed particle positions. They are the data we want to test.",
                look_for="Only observed particles are visible.",
                why=(
                    "A test needs a concrete null: 'random' must be something we "
                    "can simulate, not a feeling."
                ),
                pattern="random",
                model="homogeneous_poisson",
                n=120,
                seed=7,
                simulations=60,
                show_simulated=False,
                show_features=False,
                direct_labels=("observed particles",),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="model_baseline_model",
        title="04 - Model baseline B",
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
                why=(
                    "Simulation makes the null hypothesis concrete: this is what "
                    "'independent random placement' actually looks like."
                ),
                more_detail="One simulated layout is not enough for a verdict. It only shows what the model can generate.",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="model_baseline_overlay",
        title="05 - Model baseline C",
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
                why=(
                    "Eyes find patterns in pure noise; if you cannot tell data "
                    "from model here, you need a statistic — that is the point."
                ),
                more_detail="Our eye may see patterns, but the statistical curve compares the data to many model simulations.",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="image_to_statistic",
        title="06 - From image to statistic",
        summary="Visual comparison is not enough.",
        steps=(
            ParticleTutorialStep(
                title="From image to statistic",
                question="How do positions become something testable?",
                look_for="Pair correlation turns positions into a measurable comparison.",
                why=(
                    "A statistic compresses the pattern into a curve that can be "
                    "compared against many simulations, not one impression."
                ),
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
            ),
        ),
    ),
    ParticleTutorialExample(
        key="simulation_envelope",
        title="07 - Simulation envelope",
        summary="Orange observed statistic versus blue model band.",
        steps=(
            ParticleTutorialStep(
                title="Simulation envelope",
                question="Many model layouts form the expected blue band.",
                look_for="Orange is observed data; blue is the model envelope.",
                what_changes="Sixty random layouts are simulated instead of one.",
                expected_effect=(
                    "this pattern really is random, so the orange curve should "
                    "stay inside the blue band."
                ),
                where_to_check="the pair-correlation plot.",
                why=(
                    "The band shows the statistic's spread under the model; only "
                    "excursions beyond it count as evidence."
                ),
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
            ),
        ),
    ),
    ParticleTutorialExample(
        key="verdict",
        title="08 - Verdict language",
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
                why=(
                    "'Consistent' means not ruled out — never proof of "
                    "randomness; a small or noisy sample may simply lack power."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="homogeneous_poisson",
        title="09 - Homogeneous Poisson",
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
                why=(
                    "Poisson is the reference point: every departure is measured "
                    "against independent placement at one average density."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="clustered",
        title="10 - Clustered pattern",
        summary="Too many close pairs.",
        steps=(
            ParticleTutorialStep(
                title="Clustered pattern",
                question="What do too many close pairs do to g(r)?",
                look_for="A small-distance peak rises above the random-placement band.",
                what_changes="The generated pattern is now clustered (same N, same field).",
                expected_effect="an excess of close pairs: g(r) should rise above the band at small r.",
                where_to_check="the left edge of the pair-correlation plot.",
                why=(
                    "The signature of aggregation lives at small distances; "
                    "learning it here trains your eye for real data."
                ),
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
            ),
        ),
    ),
    ParticleTutorialExample(
        key="hard_core_meaning",
        title="11 - Hard-core model meaning",
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
                why=(
                    "Finite particle size alone forbids close pairs; the null "
                    "must include exclusion before any depletion can be called "
                    "'interaction'."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="hard_core_parameters",
        title="12 - Hard-core parameter sandbox",
        summary="Change radius and particle number; generate points.",
        steps=(
            ParticleTutorialStep(
                title="Try radius",
                question="Increase the hard-core radius to strengthen exclusion.",
                look_for="Larger radius removes more very close neighbours.",
                what_changes="The hard-core radius is raised from 1.5 to 3 nm.",
                expected_effect="the shortest separations vanish and points look more evenly spread.",
                where_to_check="the point field after generating.",
                why=(
                    "Sweeping a parameter in the sandbox teaches what each knob "
                    "does before it matters on real data."
                ),
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
        title="13 - Hard-core statistic",
        summary="Nearest-neighbor distances lose very short separations.",
        steps=(
            ParticleTutorialStep(
                title="Nearest neighbors",
                question="Where does a minimum separation show up in the statistics?",
                look_for="Close neighbours are forbidden, so the left side is depleted.",
                expected_effect="the nearest-neighbour histogram should be empty below the 3 nm radius.",
                where_to_check="the left side of the nearest-neighbour plot.",
                why=(
                    "Exclusion appears as a missing left tail in nearest-neighbour "
                    "distances — the most direct fingerprint of a minimum separation."
                ),
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
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_cluster_vs_order",
        title="14 - Order A",
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
                why=(
                    "Dense is not the same as ordered: a 'clustered' verdict says "
                    "nothing about internal arrangement."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_radial_spacing",
        title="15 - Order B",
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
                why=(
                    "Peaks in g(r) reveal repeated spacings but average away "
                    "direction — one number per distance."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_directional_pairs",
        title="16 - Order C",
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
                why=(
                    "Keeping the pair angle exposes lattice directions that "
                    "radial averaging hides."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_bond_order",
        title="17 - Order D",
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
                why=(
                    "ψ6 asks each particle a local question — do your neighbours "
                    "sit at 60° steps? — so ordered islands stand out even in "
                    "mixed fields."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_square_order",
        title="18 - Order E",
        summary="ψ4 is the square-like local-order check.",
        steps=(
            ParticleTutorialStep(
                title="ψ4 local order",
                question="ψ4 measures square-like local order around each particle.",
                look_for="Square islands shift ψ4 toward 1 more than ψ6.",
                what_changes="The generated islands are now square lattices, not triangular.",
                expected_effect="ψ4 should shift toward 1 while ψ6 stays lower.",
                where_to_check="this ψ4 histogram against the previous lesson's ψ6.",
                why=(
                    "Choosing the symmetry to test is a physics decision; the "
                    "statistic can only answer the question you pose."
                ),
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
            ),
        ),
    ),
    ParticleTutorialExample(
        key="ordered_mixed",
        title="19 - Order F",
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
                why=(
                    "Global averages dilute local order; a mixed field needs "
                    "local metrics plus your eyes on the image."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="feature_biased",
        title="20 - Feature-biased model",
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
                why=(
                    "Association tests need an independently measured reference "
                    "layer — it must not come from the tested particles."
                ),
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
                why=(
                    "Judging association by eye invites confirmation bias; the "
                    "matched null does the counting."
                ),
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
                why=(
                    "Model verdict cards summarise which placement assumptions "
                    "survive — read them per model, not as one number."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="other_statistics",
        title="21 - Statistics reference",
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
                why=(
                    "Each statistic answers one question; nearest-neighbour "
                    "distances are the sharpest probe of contact and exclusion."
                ),
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
                why=(
                    "Ripley's L accumulates structure over scale, catching "
                    "clustering that a single distance can miss."
                ),
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
                why=(
                    "Group-size counts turn 'it looks clumpy' into how many "
                    "pairs, triples, and islands — always report the linking "
                    "distance with them."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="pooling_single",
        title="22 - Pooling A",
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
                why=(
                    "One image is one noisy draw; deciding from it alone risks "
                    "overreading fluctuations."
                ),
            ),
        ),
    ),
    ParticleTutorialExample(
        key="pooling_two",
        title="23 - Pooling B",
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
                what_changes="A second independent image of the same condition joins the pool.",
                expected_effect="the pooled curve smooths and its spread narrows.",
                where_to_check="the blue pooled band against the orange single-image reference.",
                why=(
                    "Replication is the honest way to strengthen a verdict — pool "
                    "independent images, never repeated detections of the same one."
                ),
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
        title="24 - Model simulations sandbox",
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
                why=(
                    "Free experimentation with the models builds the intuition "
                    "the linear lessons cannot."
                ),
                more_detail="The sandbox uses the same point field, layer controls, statistic plot, and results view as real analysis.",
            ),
        ),
    ),
    ParticleTutorialExample(
        key="real_workflow",
        title="25 - Real ProbeFlow workflow",
        summary="Feature Finder to Particle Statistics to verdict.",
        steps=(
            ParticleTutorialStep(
                title="Real workflow",
                question="Real analysis begins with Feature Finder positions.",
                look_for=(
                    "With real points loaded, the Data summary card appears "
                    "first; then choose region, model, statistic, simulations, "
                    "and read the verdicts."
                ),
                mode="real",
                model_label="Choose model from real controls",
                visible_panel="controls",
                visible_controls=("feature_sets",),
                target_tab="Setup",
                show_simulated=False,
                show_features=False,
                why=(
                    "The full path — detect, describe, choose region and model, "
                    "simulate, read verdicts — is the same one you just "
                    "practised on generated data."
                ),
                more_detail="Workflow: detect particles, send them to Particle Statistics, read the Data summary (count, density, spacing), confirm calibration and region, choose a model and statistic, run simulations, read verdicts, and pool comparable images.",
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
        title="26 - Final caution",
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
                why="Statistics rules models out; only physics rules mechanisms in.",
            ),
        ),
    ),
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
