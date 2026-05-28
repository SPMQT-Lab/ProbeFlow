"""Main window compatibility location.

``ProbeFlowWindow`` lives in :mod:`probeflow.gui.app`; this module re-exports
it under the historical ``probeflow.gui.main_window`` import path.
"""

from probeflow.gui.app import ProbeFlowWindow

__all__ = ["ProbeFlowWindow"]
