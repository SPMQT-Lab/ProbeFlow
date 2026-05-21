"""Image arithmetic operations for processing states."""

from __future__ import annotations

import numpy as np


_CONSTANT_OPS = frozenset({"add", "subtract", "multiply", "divide"})
_IMAGE_OPS = frozenset({"add", "subtract"})


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

    if kind == "image":
        if op not in _IMAGE_OPS:
            raise ValueError(
                "Image arithmetic currently supports only add and subtract operands."
            )
        if operand_image is None:
            raise ValueError("Image arithmetic requires an operand image.")
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


__all__ = ["apply_arithmetic"]
