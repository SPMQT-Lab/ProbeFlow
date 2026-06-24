# Particle Statistics Tutorial Structure and Flow

This document describes the tutorial mode for the ProbeFlow Particle Statistics
tool. It is written as a detailed reference for teaching, review, and future
maintenance. The implementation lives primarily in
`probeflow/gui/dialogs/particle_statistics.py`.

The tutorial is inspired by the SEMITIP/Poisson Solver teaching pattern: each
step should tell the student what problem is being addressed, what control or
result area matters, what is expected to change, and where to check the result.
The goal is not only to demonstrate the interface, but to make particle-pattern
analysis intelligible to students who are still learning what the statistics
mean.

## Teaching Goal

Particle Statistics asks a spatial question:

> Given a set of particle positions, are those positions consistent with a
> simple null model, or do they show clustering, spacing, or association with an
> independently measured feature?

The tutorial mode is designed to teach four ideas in order:

1. A point field can be turned into measurable spatial statistics.
2. A null model gives the baseline for deciding whether an observed pattern is
   unusual.
3. Different statistics answer different geometric questions.
4. Real analysis requires detected feature sets, calibrated image scale,
   independent images for pooling, and careful model choice.

The tutorial deliberately starts with generated data. That avoids requiring a
specific scan file and lets students see known patterns before applying the
same workflow to their own detections.

## Main UI Surfaces

The dialog opens on a **workflow start page** (the landing page) with three
cards: **Analyze scan points** (real data), **Model simulations** (free-play
generated data), and **Tutorial** (this guided mode). A menu bar mirrors the
same actions. **Model simulations** is a separate, non-guided sandbox workflow;
it shares the generated-data backend but is not part of the tutorial sequence.

Once a workflow is chosen, the Particle Statistics dialog is divided into a few
teaching regions:

| Region | Role in the tutorial |
| --- | --- |
| Menu bar | Workflow / Data / Model / Statistic / View / Definitions menus mirroring the controls. |
| Top toolbar | Contains `Workflows`, `Start tutorial`, the mode selector (Analyze scan points / Learn with tutorial / Model simulations), and `Run comparison`. |
| Landing page | The workflow start page shown before a mode is chosen. |
| Tutorial drawer | The guided lesson selector, navigation buttons, action hints, and detail text. |
| Point field | Shows observed points, generated points, model simulations, feature layers, and regions. |
| Focus plot | Shows the currently selected statistic or verdict summary. |
| Field/info panel | Summarizes mode, point count, model, source, region, and layer visibility. |
| Data tab | Holds generated-data controls in generated mode and real-data controls in real mode. |
| Model tab | Holds generated model controls or real model/simulation/seed controls. |
| Statistics tab | Holds the statistic selector cards and the "Include local-order checks" toggle. |
| Results tab | Holds verdict cards and technical details. |
| Learn tab | Holds compact reference notes and definitions. |

The local-order statistics (ψ4, ψ6, angular `g(r, θ)`) are **opt-in**: off by
default and hidden from the verdict so a plain randomness analysis is not
cluttered with lattice checks. The ordered tutorial examples tick the
"Include local-order checks" box automatically (driven by each step's
`focus_statistic`); leaving those steps turns it back off.

The tutorial drawer remains the central teaching surface. It has:

- A guided-example selector.
- `Load example` and `Run this example`.
- `Previous` and `Next`.
- `More detail`.
- `Restart tutorial`.
- `Exit tutorial`.
- An always-visible green "Why it matters" row.
- A collapsible detail section with "Change", "Expected", "Check", "What to
  look for", and "Careful" notes.

## Tutorial State Model

The tutorial distinguishes between two concepts that used to be coupled:

| Concept | Meaning |
| --- | --- |
| Tutorial active | The guided drawer is visible and `current_mode` reports `learn`. |
| Active data mode | The visible analysis controls are either generated examples or real scan points. |

This is important because the tutorial starts in generated mode but ends by
showing real-data controls. The final lessons need the tutorial drawer to stay
open while the Data and Model tabs show real point sources, saved feature sets,
real models, simulation count, seed, and pooling controls.

Internally:

- `self._tutorial_active` records whether the guided tutorial is active.
- `self._active_mode` records whether the current data source is `generated` or
  `real`.
- `current_mode` returns `learn` when tutorial mode is active, even if the
  current step is displaying real-data controls.
- `Exit tutorial` switches back to normal real analysis, clears tutorial
  highlights, invalidates stale generated workers, and clears the real view.

## Step Data Model

Each tutorial step is represented by `ParticleTutorialStep`. It combines
student-facing text with machine-readable UI guidance.

| Field | Purpose |
| --- | --- |
| `title` | Short step title shown in the drawer. |
| `body` | Main student instruction. |
| `mode` | `generated` or `real`; controls which workflow controls are visible. |
| `target_tab` | The tab the tutorial should select for this step. |
| `controls` | Stable tutorial control keys to highlight. |
| `focus_statistic` | Statistic plotted in the focus panel. |
| `focus_curve_mode` | Plot emphasis, such as observed-only or model comparison. |
| `action_button` | Which action is the intended next action: next, next example, run, load, restart, or none. |
| `action_text` | Custom call-to-action text for the green button. |
| `advance_after_run` | Whether a run completion should advance to the next step. |
| `pattern` | Generated point pattern to stage. |
| `model` | Generated or real model to select. |
| `n` | Generated particle count. |
| `seed` | Random seed. |
| `simulations` | Number of model simulations. |
| `show_observed` | Whether observed particles are visible in the field. |
| `show_simulated` | Whether model simulations are visible in the field. |
| `show_features` | Whether independent feature layers are visible. |
| `show_region` | Whether region or mask overlays are visible. |
| `pool_images` | Number of generated images to pool for the pooling demo. |
| `intro_card` | Marks read-only orientation cards that do not run a comparison. |
| `intro_region` | Which central panel an intro card is introducing. |
| `intro_panel_text` | One-line text placed in the introduced panel. |
| `what_changes` | What input or view changes in this step. |
| `expected_effect` | What the student should expect to happen. |
| `where_to_check` | Where in the UI the student should verify it. |
| `why` | The conceptual reason this step matters. |
| `statistic_hint` | What to look for in the plot or control. |
| `limitation` | A caution or interpretation limit. |

The tutorial uses `_complete_tutorial_metadata()` to fill default
`what_changes`, `expected_effect`, and `where_to_check` text for steps that
have controls, actions, or result tabs. This keeps the SEMITIP-style metadata
complete without forcing every simple step to repeat boilerplate.

## Control Highlighting

The tutorial uses a registry of stable control keys. Each key maps to one or
more widgets. Tutorial steps name keys, not widget objects.

Current control keys include:

| Key | Highlighted control |
| --- | --- |
| `mode` | Mode combo box. |
| `pattern` | Generated pattern combo. |
| `n` | Generated particle-count spinbox. |
| `generated_seed` | Generated seed spinbox. |
| `generated_model` | Generated model combo. |
| `generated_simulations` | Generated simulation-count spinbox. |
| `source` | Real point-source combo. |
| `region` | Real region combo. |
| `real_model` | Real model combo. |
| `real_simulations` | Real simulation-count spinbox. |
| `real_seed` | Real seed spinbox. |
| `feature_sets` | Saved feature-set list. |
| `feature_layer` | Feature-layer picker for measured-feature models. |
| `run_comparison` | Main `Run comparison` button. |
| `run_tutorial` | Tutorial `Run this example` button. |
| `run_selected_sets` | Saved feature-set run button. |
| `load_tutorial` | Tutorial `Load example` button. |
| `next` | Tutorial `Next` button. |
| `restart` | Tutorial `Restart tutorial` button. |
| `layer_observed` | Observed-points layer checkbox. |
| `layer_simulated` | Model-simulation layer checkbox. |
| `layer_features` | Feature-layer checkbox. |
| `layer_region` | Region/mask layer checkbox. |
| `stat_pair` | Pair correlation statistic card. |
| `stat_nearest` | Nearest-neighbor statistic card. |
| `stat_ripley` | Ripley L statistic card. |
| `stat_clusters` | Cluster-size statistic card. |

Highlight colors:

- Green means "this is the current control or action".
- Yellow means "this control has already been covered in the current lesson".
- Normal styling is restored when leaving tutorial mode or changing steps.

The tutorial also highlights the active action button. For example, a step with
`action_button="next_example"` highlights `Next`, while a final step with
`action_button="restart"` highlights `Restart tutorial`.

## Lesson Sequence

The tutorial is a sequence of guided examples. Each example has a stable key,
a title, a summary, and a tuple of steps.

### 1. `intro`: Start Here

Purpose:

- Introduce the three main visual regions without running a comparison.
- Explain that the tool looks for spatial rules beyond random chance.
- Prepare students for the idea that one image is noisy and pooling matters.

Flow:

1. The point field: where particles are drawn.
2. The statistic plot: where point positions become a curve or verdict.
3. The model summary: where model assumptions and results are summarized.
4. Getting good statistics: why multiple independent images matter.

These are soft intro cards. They keep the central panels mostly blank so
students are not overloaded before the first generated example.

### 2. `tour`: Workspace Tour

Purpose:

- Walk the student through every tab and statistic card.
- Show how generated data, model choice, statistic selection, and result
  interpretation are connected.

Flow:

1. Data tab: generated random points, particle count, seed, and layer checkboxes.
2. Model tab: homogeneous Poisson as the baseline null model.
3. Statistics tab: pair correlation `g(r)`.
4. Statistics tab: nearest-neighbor distance.
5. Statistics tab: Ripley L.
6. Statistics tab: cluster sizes.
7. Results tab: verdict cards and consistency language.
8. Learn tab: compact definitions and reference notes.

Teaching emphasis:

- The four statistic cards are different lenses on the same points.
- Switching statistics changes the focus plot, not the underlying point data.
- A verdict is statistical consistency or inconsistency with a model, not proof
  of a physical mechanism.

### 3. `random`: Random Placement

Purpose:

- Establish homogeneous Poisson placement as the baseline null model.

Flow:

1. Show a random generated pattern and its observed-only pair correlation.
2. Add the simulation envelope from many random layouts.
3. Read the verdict: random placement is not rejected.

Teaching emphasis:

- A single random layout proves nothing because it is itself random.
- The simulation envelope is the fair comparison.
- Failure to reject the null does not prove there are no interactions.

### 4. `more_particles`: More Particles

Purpose:

- Show how sample size affects statistical sensitivity.

Flow:

1. Increase the generated random field from the earlier example to many more
   points.
2. Compare against the same null model.
3. Read the verdict, which should still be consistent with random placement.

Teaching emphasis:

- More points make curves smoother.
- More points narrow the envelope and make smaller effects detectable.
- More points can also cost more computation.

### 5. `pooling`: Pooling Multiple Images

Purpose:

- Teach why independent images are the practical route to stronger statistics.

Flow:

1. Show one noisy single-image comparison.
2. Pool five independent generated images and show the smoother combined result.

Teaching emphasis:

- One image is often too noisy.
- Pooling independent images adds evidence without forcing one image to carry
  the whole conclusion.
- Images should only be pooled when they belong to the same condition.

### 6. `clustered`: Clustered Points

Purpose:

- Show how visual clustering becomes a measurable short-range excess.

Flow:

1. Show a clustered generated pattern and observed-only `g(r)`.
2. Compare the pattern against homogeneous Poisson simulations.
3. Read the verdict: inconsistent with random placement.

Teaching emphasis:

- Clustering creates extra close pairs.
- Extra close pairs appear as a small-distance bump in pair correlation.
- Rejecting random placement is evidence of structure, not proof of a specific
  attractive mechanism.

### 7. `hard_core`: No-Overlap / Hard-Core

Purpose:

- Show the opposite of clustering: points avoid close neighbors.

Flow:

1. Show a no-overlap generated pattern and focus nearest-neighbor distance.
2. Compare against the hard-core model and simulation envelope.
3. Read the verdict and compare it with pure random placement.

Teaching emphasis:

- Exclusion removes very short neighbor distances.
- Nearest-neighbor distance is the clearest first statistic for this case.
- Apparent spacing can have several causes: particle size, detection merging,
  substrate registry, or other effects.

### 8. `feature_biased`: Feature-Biased Points

Purpose:

- Teach how to test association with an independently measured feature layer.

Flow:

1. Show the independent feature layer alone.
2. Add the particles and ask whether they sit near the features.
3. Compare model verdicts.
4. Explain why the feature layer must be independent.
5. Switch into real mode and show the real saved-set workflow for this model.

Teaching emphasis:

- The feature layer is the proposed influence, not the particles being tested.
- Reusing the particles as their own feature layer is circular.
- Measured-feature Poisson requires one tested set and a different feature-layer
  set from the same real image context.

### 9. `real_handoff`: Move to Real Scan Points

Purpose:

- Walk students from generated examples into actual ProbeFlow analysis.

Flow:

1. Detect features in Feature Finder.
2. Send the detected features to Particle Statistics.
3. Repeat across independent images of the same condition.
4. Tick saved sets and run a pooled comparison.

Teaching emphasis:

- Feature Finder creates the point set Particle Statistics needs.
- Sending to Particle Statistics preserves points and calibration in a
  viewer-session feature set.
- Multiple saved sets represent independent images and can be pooled.
- The final run uses real model, simulation count, seed, and saved-set controls.
- Particle Statistics can suggest clustering, spacing, randomness, or feature
  association. It does not prove a physical mechanism by itself.

## Generated Data Flow

Generated tutorial steps use the AdStat sandbox:

1. The step selects `pattern`, `n`, `seed`, `model`, and `simulations`.
2. The sandbox state is staged only when the step is in generated mode.
3. The point field refreshes with observed generated points, optional simulated
   points, and optional feature points.
4. The focus statistic is selected.
5. The target tab is selected.
6. If the dialog is visible and the step is not an intro card, the tutorial can
   compute a comparison so the plot is live.

Layer checkboxes are display-only. They affect what is drawn, not what is
computed. The tutorial uses this to teach observed points, model simulations,
feature layers, and region overlays one at a time.

## Real Data Flow

Real tutorial steps do not require the tutorial to own Feature Finder. Instead,
they explain the handoff and highlight the Particle Statistics controls that
students should inspect next.

Real steps show:

1. Point source and region controls.
2. Saved feature sets from Feature Finder.
3. Real model selection.
4. Simulation count and seed.
5. Measured-feature feature-layer picker.
6. `Run selected sets` for single-set or pooled saved-set analysis.

The tutorial stays active while these controls are visible. This is the key
structural change from the older version, where tutorial visibility was tied to
generated mode only.

## Interpretation Language

The tutorial uses careful language throughout:

- "Consistent with random" means the null model was not rejected.
- "Inconsistent with random" means the observed statistic departed from the
  null-model envelope.
- A model verdict is not a proof of mechanism.
- Feature association requires independently measured feature layers.
- Pooling requires independent images from the same condition.

This language is intentionally conservative for students. It separates what the
statistical comparison can support from what a physical interpretation would
require.

## Tests and Guardrails

The tutorial has tests in `tests/test_adstat_workbench_dialog.py`.

Important checks include:

- The tutorial example order is stable.
- The intro example contains read-only intro cards.
- Feature-biased model-summary steps do not strand the student on the
  Statistics tab.
- Every used tutorial control key resolves to an actual widget.
- Steps that point at controls, actions, or results carry metadata for
  `what_changes`, `expected_effect`, `where_to_check`, and `why`.
- Tutorial highlights apply to current controls.
- Previously covered controls become yellow.
- Highlights clear on tutorial exit.
- Real workflow steps keep the tutorial drawer visible while real controls are
  shown.
- Feature-biased handoff steps switch to real model controls and select
  measured-feature Poisson.
- Generated tutorial behavior is preserved: focus plots, example loading,
  layer toggles, next-example navigation, pooling demo, exit/restart behavior,
  and late generated worker suppression.

These tests are intentionally similar in spirit to the SEMITIP tutorial tests:
they protect both mechanics and teaching quality.

## Future Extension Points

This pass deliberately does not add a full primer, glossary dialog, artwork
package, or PDF export. Good future additions would be:

1. A Particle Statistics primer with pages for null models, pair correlation,
   nearest-neighbor distance, Ripley L, cluster sizes, pooling, and
   measured-feature controls.
2. Contextual primer links from tutorial steps, similar to SEMITIP
   `primer_key`.
3. A short glossary for terms such as null model, envelope, hard-core,
   measured feature, pooling, independent replicate, and verdict.
4. Optional handout/export support once the tutorial text stabilizes.
5. Light Feature Finder cues if the real-data workflow needs cross-dialog
   guidance later.

The current structure leaves room for those additions by keeping lesson content
separate from control highlighting and by making tutorial metadata explicit.
