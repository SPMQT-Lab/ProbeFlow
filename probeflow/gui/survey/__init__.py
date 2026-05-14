"""ProbeFlow Survey mode — open a ScanFlow campaign and polish images for PPTX.

Top-level workflow:
    ScanFlow → "Open in ProbeFlow"
        → CLI: probeflow gui --open-survey survey.json
        → ProbeFlowWindow loads SurveyPanel pre-populated with the manifest.

The panel lists each feature, lets the user click through and process each
.dat with ProbeFlow's existing viewer, save a polished PNG, then export the
final PPTX deck (slide 1 overview with numbered features, one slide per
feature with metadata: Vbias, current, drift, size, position).
"""

from probeflow.gui.survey.survey_panel import SurveyPanel

__all__ = ["SurveyPanel"]
