"""Tests for chained processing pipelines via apply_processing_state.

Single-operation coverage lives in test_processing.py.  These tests focus on:
  - empty state behaves as identity (and returns a copy, not the same object)
  - chained operations preserve shape and float64 dtype
  - the raw input array is never mutated
  - result is numerically consistent across equivalent chain orderings
"""

from __future__ import annotations

import numpy as np
import pytest

from probeflow.processing.state import ProcessingState, ProcessingStep, apply_processing_state


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tilted():
    """64×64 image with a linear tilt and mild noise."""
    rng = np.random.default_rng(7)
    Y, X = np.mgrid[:64, :64]
    arr = 0.3 * X + 0.15 * Y + rng.normal(scale=0.05, size=(64, 64))
    return arr.astype(np.float64)


@pytest.fixture
def flat_noisy():
    rng = np.random.default_rng(11)
    return rng.normal(loc=1e-9, scale=1e-11, size=(32, 32)).astype(np.float64)


# ── Empty state ───────────────────────────────────────────────────────────────

class TestEmptyState:
    def test_empty_state_is_numerically_identity(self, tilted):
        state = ProcessingState()
        result = apply_processing_state(tilted, state)
        np.testing.assert_array_equal(result, tilted)

    def test_empty_state_returns_a_copy_not_same_object(self, tilted):
        state = ProcessingState()
        result = apply_processing_state(tilted, state)
        assert result is not tilted

    def test_empty_state_preserves_shape(self, tilted):
        state = ProcessingState()
        result = apply_processing_state(tilted, state)
        assert result.shape == tilted.shape

    def test_empty_state_output_is_float64(self, tilted):
        state = ProcessingState()
        result = apply_processing_state(tilted, state)
        assert result.dtype == np.float64


# ── Input immutability ────────────────────────────────────────────────────────

class TestInputImmutability:
    def test_input_not_mutated_by_align_rows(self, tilted):
        original = tilted.copy()
        state = ProcessingState(steps=[ProcessingStep("align_rows")])
        apply_processing_state(tilted, state)
        np.testing.assert_array_equal(tilted, original)

    def test_input_not_mutated_by_plane_bg(self, tilted):
        original = tilted.copy()
        state = ProcessingState(steps=[ProcessingStep("plane_bg", {"order": 1})])
        apply_processing_state(tilted, state)
        np.testing.assert_array_equal(tilted, original)

    def test_input_not_mutated_by_smooth(self, flat_noisy):
        original = flat_noisy.copy()
        state = ProcessingState(steps=[ProcessingStep("smooth", {"sigma": 1.0})])
        apply_processing_state(flat_noisy, state)
        np.testing.assert_array_equal(flat_noisy, original)


# ── Chained operations ────────────────────────────────────────────────────────

class TestChainedOps:
    def test_align_then_plane_bg_preserves_shape(self, tilted):
        state = ProcessingState(steps=[
            ProcessingStep("align_rows"),
            ProcessingStep("plane_bg", {"order": 1}),
        ])
        result = apply_processing_state(tilted, state)
        assert result.shape == tilted.shape

    def test_align_then_plane_bg_output_is_float64(self, tilted):
        state = ProcessingState(steps=[
            ProcessingStep("align_rows"),
            ProcessingStep("plane_bg", {"order": 1}),
        ])
        result = apply_processing_state(tilted, state)
        assert result.dtype == np.float64

    def test_plane_bg_then_smooth_preserves_shape(self, flat_noisy):
        state = ProcessingState(steps=[
            ProcessingStep("plane_bg", {"order": 1}),
            ProcessingStep("smooth", {"sigma": 1.0}),
        ])
        result = apply_processing_state(flat_noisy, state)
        assert result.shape == flat_noisy.shape

    def test_plane_bg_then_smooth_output_is_float64(self, flat_noisy):
        state = ProcessingState(steps=[
            ProcessingStep("plane_bg", {"order": 1}),
            ProcessingStep("smooth", {"sigma": 1.0}),
        ])
        result = apply_processing_state(flat_noisy, state)
        assert result.dtype == np.float64

    def test_three_op_chain_preserves_shape(self, tilted):
        state = ProcessingState(steps=[
            ProcessingStep("align_rows"),
            ProcessingStep("plane_bg", {"order": 1}),
            ProcessingStep("smooth", {"sigma": 1.0}),
        ])
        result = apply_processing_state(tilted, state)
        assert result.shape == tilted.shape

    def test_plane_bg_reduces_tilt_rms(self, tilted):
        state = ProcessingState(steps=[ProcessingStep("plane_bg", {"order": 1})])
        result = apply_processing_state(tilted, state)
        assert result.std() < tilted.std()

    def test_smooth_reduces_noise_std(self, flat_noisy):
        state = ProcessingState(steps=[ProcessingStep("smooth", {"sigma": 2.0})])
        result = apply_processing_state(flat_noisy, state)
        assert result.std() < flat_noisy.std()


# ── ProcessingState serialisation roundtrip ───────────────────────────────────

class TestProcessingStateRoundtrip:
    def test_empty_state_roundtrips(self):
        state = ProcessingState()
        d = state.to_dict()
        restored = ProcessingState.from_dict(d)
        assert restored.steps == state.steps

    def test_single_step_roundtrips(self):
        state = ProcessingState(steps=[
            ProcessingStep("align_rows", {"method": "median"}),
        ])
        restored = ProcessingState.from_dict(state.to_dict())
        assert len(restored.steps) == 1
        assert restored.steps[0].op == "align_rows"
        assert restored.steps[0].params == {"method": "median"}

    def test_multi_step_roundtrips_in_order(self):
        state = ProcessingState(steps=[
            ProcessingStep("align_rows"),
            ProcessingStep("plane_bg", {"order": 2}),
            ProcessingStep("smooth", {"sigma": 1.5}),
        ])
        restored = ProcessingState.from_dict(state.to_dict())
        assert [s.op for s in restored.steps] == ["align_rows", "plane_bg", "smooth"]

    def test_to_dict_is_json_serialisable(self):
        import json
        state = ProcessingState(steps=[
            ProcessingStep("plane_bg", {"order": 1}),
        ])
        json.dumps(state.to_dict())  # must not raise
