"""Thread-pool workers for the Particle Statistics dialog.

Each worker runs one long computation (real-data comparison, saved
feature-set comparison, or a sandbox simulation step) off the GUI thread and
reports back with a generation counter so stale results are discarded.
Extracted verbatim from ``particle_statistics.py``.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal

from probeflow.gui.viewer.tool_launch import (
    AdStatStatisticsRequest,
    adstat_workbench_launch_context,
)
from probeflow.gui.workers import _PooledWorker


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
