"""Active-mask layer: ``ImageMask`` and ``MaskSet`` — domain model only.

A *mask* is a boolean array over an image that downstream tools consume: it
can restrict statistics, exclude regions from a background fit, or convert to
ROI objects.  ``MaskSet`` is a deliberate structural twin of
:class:`probeflow.core.roi.ROISet` — same ``image_id`` / ``active_*_id`` /
``to_dict`` / ``from_dict`` shape — so its sidecar and viewer plumbing mirror
the ROI path.

Edge detection is the first producer of masks; thresholding, manual paint and
segmentation can plug into the same layer later.

This module is Qt-free and lives in ``core`` (alongside ``roi`` and
``processing_state``) so it can be imported without pulling in the GUI.
"""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from probeflow.core import op_vocab


# ── Compact boolean-array (de)serialisation ─────────────────────────────────────

def _pack_bool(data: np.ndarray) -> str:
    """Pack a boolean array into a base64 string of bits (row-major)."""
    bits = np.packbits(np.asarray(data, dtype=bool).ravel(order="C"))
    return base64.b64encode(bits.tobytes()).decode("ascii")


def _unpack_bool(packed: str, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`_pack_bool`."""
    raw = np.frombuffer(base64.b64decode(packed.encode("ascii")), dtype=np.uint8)
    n = int(shape[0]) * int(shape[1])
    bits = np.unpackbits(raw)[:n]
    return bits.astype(bool).reshape(shape)


# ── Auto-naming ─────────────────────────────────────────────────────────────────

def mask_name(method: str, params: dict[str, Any] | None = None) -> str:
    """Build a descriptive mask name from method + parameters.

    Examples: ``Canny_sigma1.5_p60-85``, ``Sobel_magnitude_p90``,
    ``Scharr_x_gradient``.
    """
    p = params or {}
    m = (method or "mask").lower()
    if m == "canny":
        sigma = p.get("sigma", 1.0)
        bits = [f"Canny_sigma{_fmt_num(sigma)}"]
        if str(p.get("threshold_mode", "percentile")) == "percentile":
            bits.append(f"p{_fmt_num(p.get('low', 70))}-{_fmt_num(p.get('high', 90))}")
        return "_".join(bits)
    if m in ("sobel", "scharr"):
        output = str(p.get("output", "magnitude"))
        label = f"{m.capitalize()}_{output}"
        if output != "orientation":
            if p.get("threshold_to_mask"):
                label += f"_p{_fmt_num(p.get('threshold', 90))}"
            elif output in ("x", "y"):
                label += "_gradient"
        return label
    return method


def _fmt_num(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f == int(f) else f"{f:g}"


# ── Mask model ──────────────────────────────────────────────────────────────────

@dataclass
class ImageMask:
    """One boolean mask over an image, with provenance.

    *data* is the source of truth; *shape* is cached for serialisation.  All
    coordinates are pixel-space (``(Ny, Nx)``), matching :class:`ROI` masks.
    """

    id: str
    name: str
    data: np.ndarray
    method: str = "manual"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=bool)
        if self.data.ndim != 2:
            raise ValueError(f"ImageMask data must be 2-D, got shape {self.data.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    @classmethod
    def new(
        cls,
        data: np.ndarray,
        *,
        method: str = "manual",
        parameters: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> "ImageMask":
        """Create a mask with a fresh UUID and an auto-generated name."""
        params = dict(parameters or {})
        mask_id = str(uuid.uuid4())
        resolved = name if name is not None else mask_name(method, params)
        return cls(id=mask_id, name=resolved, data=np.asarray(data, dtype=bool),
                   method=method, parameters=params)

    def count(self) -> int:
        """Number of True pixels."""
        return int(self.data.sum())

    # ── Geometric transformation ────────────────────────────────────────────────

    def transform(
        self,
        operation: str,
        params: dict[str, Any],
        image_shape: tuple[int, int],
    ) -> "ImageMask | None":
        """Return a copy of this mask transformed for *operation*, or ``None``.

        Structural twin of :meth:`probeflow.core.roi.ROI.transform`, but masks
        are rasters so the boolean array itself is transformed with the same
        numpy primitives the image uses (``np.fliplr`` / ``np.flipud`` /
        ``np.rot90``), keeping the mask pixel-aligned with the transformed
        image.

        Returns ``None`` (invalidate) for ``rotate_arbitrary`` and any other
        resampling / shape-changing op (scale, shear, affine): a boolean mask
        cannot be resampled without interpolation, and silently keeping a
        same-shape mask after such an op is exactly the staleness this guards
        against.  ``crop`` slices the array.  Identity fields (id/name/method/
        parameters) are preserved.
        """
        op = op_vocab.to_short(operation)
        data = self.data

        if op == "flip_horizontal":
            new_data = np.fliplr(data)
        elif op == "flip_vertical":
            new_data = np.flipud(data)
        elif op == "rot90_cw":
            new_data = np.rot90(data, k=3)
        elif op == "rot180":
            new_data = np.rot90(data, k=2)
        elif op == "rot270_cw":
            new_data = np.rot90(data, k=1)
        elif op == "crop":
            try:
                x0 = int(params["x0"]); y0 = int(params["y0"])
                x1 = int(params["x1"]); y1 = int(params["y1"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"crop requires integer x0/y0/x1/y1 params, got {params!r}"
                ) from exc
            # Inclusive bounds, matching ROI._transform_crop.
            sub = data[y0:y1 + 1, x0:x1 + 1]
            if sub.size == 0 or not sub.any():
                return None
            new_data = sub
        elif op in ("rotate_arbitrary", "scale_image", "shear",
                    "linear_undistort",
                    "affine_lattice_correction"):
            # Resampling / shape-changing ops invalidate the raster, exactly
            # as the docstring promises (the previous code raised ValueError
            # for scale/shear/affine despite documenting None).
            return None
        else:
            raise ValueError(f"ImageMask.transform: unknown operation {operation!r}")

        return ImageMask(
            id=self.id,
            name=self.name,
            data=np.ascontiguousarray(new_data, dtype=bool),
            method=self.method,
            parameters=dict(self.parameters),
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "method": self.method,
            "parameters": dict(self.parameters),
            "shape": list(self.shape),
            "data": _pack_bool(self.data),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImageMask":
        shape = tuple(int(v) for v in d["shape"])  # type: ignore[assignment]
        data = _unpack_bool(str(d["data"]), shape)  # type: ignore[arg-type]
        return cls(
            id=str(d["id"]),
            name=str(d["name"]),
            data=data,
            method=str(d.get("method", "manual")),
            parameters=dict(d.get("parameters", {})),
        )


@dataclass
class MaskSet:
    """An ordered collection of masks belonging to one image.

    Structural twin of :class:`probeflow.core.roi.ROISet`.  *image_id* ties the
    set to a scan; its value is opaque (typically the file stem or a UUID).
    """

    image_id: str
    masks: list[ImageMask] = field(default_factory=list)
    active_mask_id: str | None = None

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, mask: ImageMask) -> None:
        self.masks.append(mask)

    def remove(self, mask_id: str) -> None:
        self.masks = [m for m in self.masks if m.id != mask_id]
        if self.active_mask_id == mask_id:
            self.active_mask_id = None

    def get(self, mask_id: str) -> ImageMask | None:
        for m in self.masks:
            if m.id == mask_id:
                return m
        return None

    def get_by_name(self, name: str) -> ImageMask | None:
        for m in self.masks:
            if m.name == name:
                return m
        return None

    def set_active(self, mask_id: str | None) -> None:
        if mask_id is not None and self.get(mask_id) is None:
            raise ValueError(f"Mask {mask_id!r} not in this MaskSet")
        self.active_mask_id = mask_id

    def active(self) -> ImageMask | None:
        """Return the active mask, or None."""
        if self.active_mask_id is None:
            return None
        return self.get(self.active_mask_id)

    def replace(self, mask_id: str, data: np.ndarray) -> None:
        """Replace the *data* of an existing mask in place (e.g. after cleanup)."""
        mask = self.get(mask_id)
        if mask is None:
            raise ValueError(f"Mask {mask_id!r} not in this MaskSet")
        mask.data = np.asarray(data, dtype=bool)

    # ── Geometric transformation ────────────────────────────────────────────────

    def transform_all(
        self,
        operation: str,
        params: dict[str, Any],
        image_shape: tuple[int, int],
    ) -> list[str]:
        """Apply *operation* to every mask; return list of invalidated mask IDs.

        Structural twin of :meth:`probeflow.core.roi.ROISet.transform_all`.
        Invalidated masks are NOT removed from this set — that policy decision
        belongs to the caller (typically the viewer's display-op handler).
        """
        invalidated: list[str] = []
        new_masks: list[ImageMask] = []
        for mask in self.masks:
            transformed = mask.transform(operation, params, image_shape)
            if transformed is None:
                invalidated.append(mask.id)
                new_masks.append(mask)   # keep for now; caller decides
            else:
                new_masks.append(transformed)
        self.masks = new_masks
        return invalidated

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "masks": [m.to_dict() for m in self.masks],
            "active_mask_id": self.active_mask_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MaskSet":
        mask_set = cls(
            image_id=str(d["image_id"]),
            active_mask_id=d.get("active_mask_id"),
        )
        for mask_dict in d.get("masks", []):
            try:
                mask_set.masks.append(ImageMask.from_dict(mask_dict))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Failed to reconstruct ImageMask from dict: {exc!r} — "
                    f"offending dict keys: {sorted(mask_dict) if isinstance(mask_dict, dict) else mask_dict!r}"
                ) from exc
        # Drop a dangling active id rather than raising on load.
        if mask_set.active_mask_id is not None and mask_set.get(mask_set.active_mask_id) is None:
            mask_set.active_mask_id = None
        return mask_set
