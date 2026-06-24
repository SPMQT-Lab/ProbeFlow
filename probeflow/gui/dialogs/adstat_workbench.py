"""Compatibility wrapper for the Particle Statistics tool."""

from __future__ import annotations

from probeflow.gui.dialogs.particle_statistics import ParticleStatisticsDialog


class AdStatWorkbenchDialog(ParticleStatisticsDialog):
    """Legacy AdStat workbench name; opens Particle Statistics."""
