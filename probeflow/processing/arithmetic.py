"""Image arithmetic operations for processing states."""

from __future__ import annotations

import numpy as np


_CONSTANT_OPS = frozenset({"add", "subtract", "multiply", "divide"})
_IMAGE_OPS = frozenset({"add", "subtract"})
_GENERATED_PATTERNS = frozenset({
    "checkerboard",
    "ramp_x",
    "ramp_y",
    "speckle",
    "impulse_grid",
})


def generate_arithmetic_pattern(
    shape: tuple[int, int],
    pattern: str,
    amplitude_si: float,
    *,
    period_px: int = 16,
    seed: int = 1,
) -> np.ndarray:
    """Return a deterministic generated arithmetic operand for *shape*."""
    if len(shape) != 2:
        raise ValueError(f"Generated arithmetic patterns require a 2-D shape, got {shape!r}.")
    ny, nx = (int(shape[0]), int(shape[1]))
    if ny <= 0 or nx <= 0:
        raise ValueError(f"Generated arithmetic pattern shape must be positive, got {shape!r}.")

    pattern_key = str(pattern)
    if pattern_key not in _GENERATED_PATTERNS:
        raise ValueError(f"Unknown generated arithmetic pattern: {pattern_key!r}")

    amplitude = float(amplitude_si)
    period = max(1, int(period_px))

    if pattern_key == "checkerboard":
        y, x = np.indices((ny, nx))
        sign = np.where(((y // period) + (x // period)) % 2 == 0, 1.0, -1.0)
        return sign.astype(np.float64) * amplitude

    if pattern_key == "ramp_x":
        row = np.linspace(-amplitude, amplitude, nx, dtype=np.float64)
        return np.broadcast_to(row, (ny, nx)).copy()

    if pattern_key == "ramp_y":
        col = np.linspace(-amplitude, amplitude, ny, dtype=np.float64)[:, None]
        return np.broadcast_to(col, (ny, nx)).copy()

    if pattern_key == "speckle":
        rng = np.random.default_rng(int(seed))
        return rng.normal(loc=0.0, scale=amplitude, size=(ny, nx)).astype(np.float64)

    result = np.zeros((ny, nx), dtype=np.float64)
    result[::period, ::period] = amplitude
    return result


def apply_arithmetic(
    arr: np.ndarray,
    *,
    operation: str,
    operand_type: str,
    value_si: float | None = None,
    factor: float | None = None,
    operand_image: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a simple arithmetic operation to a 2-D image.

    Constant add/subtract values are in the image's native SI units. Constant
    multiply/divide factors are dimensionless. Image operands must already be
    loaded as raw numeric arrays and match the target image shape exactly.
    """
    image = np.asarray(arr, dtype=np.float64)
    op = str(operation)
    kind = str(operand_type)

    if kind == "constant":
        if op not in _CONSTANT_OPS:
            raise ValueError(f"Unsupported constant arithmetic operation: {op!r}")
        if op in {"add", "subtract"}:
            value = float(0.0 if value_si is None else value_si)
            if op == "add":
                return image + value
            return image - value
        scalar = float(1.0 if factor is None else factor)
        if op == "multiply":
            return image * scalar
        if scalar == 0.0:
            raise ValueError("Cannot divide by zero in image arithmetic.")
        return image / scalar

    if kind in {"image", "generated"}:
        if op not in _IMAGE_OPS:
            raise ValueError(
                "Image arithmetic currently supports only add and subtract operands."
            )
        if operand_image is None:
            raise ValueError(f"{kind.title()} arithmetic requires an operand image.")
        operand = np.asarray(operand_image, dtype=np.float64)
        if operand.shape != image.shape:
            raise ValueError(
                f"Image arithmetic operand shape {operand.shape} does not match "
                f"current image shape {image.shape}."
            )
        if op == "add":
            return image + operand
        return image - operand

    raise ValueError(f"Unsupported arithmetic operand type: {kind!r}")


__all__ = ["apply_arithmetic", "generate_arithmetic_pattern"]
