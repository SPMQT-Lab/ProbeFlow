"""
Pure-python readers and writers for Nanonis ``.sxm`` files.

These functions have no Qt / matplotlib dependency so they can be reused by
the CLI, tests, and batch scripts without pulling in the GUI stack.

An ``.sxm`` file is a text header followed by a binary block of one or more
big-endian float32 image planes.  The text header ends with the marker
``:SCANIT_END:``; fixed cushion bytes after that marker are recorded in
``probeflow/data/file_cushions`` and used to locate the payload.

The public API:
    parse_sxm_header(path) -> dict
    sxm_data_info(hdr)     -> [(name, unit, direction), ...]
    sxm_dims(hdr)          -> (Nx, Ny)
    sxm_scan_range(hdr)    -> (width_m, height_m)
    orient_plane(arr, hdr, plane_idx) -> oriented array
    read_sxm_plane(path, plane_idx=0, orient=True) -> ndarray
    read_all_sxm_planes(path, orient=True) -> (hdr, list[ndarray])
    write_sxm_with_planes(src_sxm, out_sxm, new_planes) -> None
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from probeflow.core.resources import FILE_CUSHIONS_DIR

# ── Binary layout ────────────────────────────────────────────────────────────
# Cushion byte lengths are read lazily to avoid repeated file I/O.

_DEFAULT_CUSHION_DIR = FILE_CUSHIONS_DIR

_POST_END_BYTES_CACHE: Optional[bytes] = None
_PRE_PAYLOAD_BYTES_CACHE: Optional[bytes] = None

# Nanonis SXM files terminate their text header with this marker.
_SCANIT_END_MARKER = b":SCANIT_END:"


def _cushion_tail_lens(cushion_dir: Optional[Path] = None) -> Tuple[int, int]:
    """Return ``(len(post_end_bytes), len(pre_payload_bytes))`` from the cushion."""
    global _POST_END_BYTES_CACHE, _PRE_PAYLOAD_BYTES_CACHE
    if (cushion_dir is None and _POST_END_BYTES_CACHE is not None
            and _PRE_PAYLOAD_BYTES_CACHE is not None):
        return len(_POST_END_BYTES_CACHE), len(_PRE_PAYLOAD_BYTES_CACHE)
    cushion_dir = Path(cushion_dir) if cushion_dir else _DEFAULT_CUSHION_DIR
    post = (cushion_dir / "post_end_bytes.bin").read_bytes()
    pre = (cushion_dir / "pre_payload_bytes.bin").read_bytes()
    if cushion_dir == _DEFAULT_CUSHION_DIR:
        _POST_END_BYTES_CACHE = post
        _PRE_PAYLOAD_BYTES_CACHE = pre
    return len(post), len(pre)


def _data_offset_in_file(
    raw: bytes,
    cushion_dir: Optional[Path] = None,
) -> int:
    """Compute the byte offset of the binary payload in a given .sxm file.

    Locates ``:SCANIT_END:`` and adds the fixed post-marker + pre-payload
    padding lengths from the cushion, so header length can vary across files.
    """
    idx = raw.find(_SCANIT_END_MARKER)
    if idx < 0:
        raise ValueError("No :SCANIT_END: marker in .sxm file")
    post_len, pre_len = _cushion_tail_lens(cushion_dir)
    return idx + len(_SCANIT_END_MARKER) + post_len + pre_len


# ── Header parsing ───────────────────────────────────────────────────────────

def parse_sxm_header(sxm_path: Path) -> dict:
    """Return a dict of ``:KEY:`` → value strings from the .sxm text header.

    Raises
    ------
    ValueError
        If the file does not contain a complete SXM header ending in
        ``:SCANIT_END:``.
    """
    params: dict = {}
    current_key: Optional[str] = None
    buf: list[str] = []
    found_end = False

    def _flush() -> None:
        if current_key is not None:
            params[current_key] = " ".join(buf).strip()

    with open(sxm_path, "rb") as fh:
        for raw in fh:
            if raw.strip() == b":SCANIT_END:":
                found_end = True
                break
            line = raw.decode("latin-1", errors="replace").rstrip("\r\n")
            if line.startswith(":") and line.endswith(":") and len(line) > 2:
                _flush()
                current_key = line[1:-1]
                buf = []
            elif current_key is not None:
                s = line.strip()
                if s:
                    buf.append(s)
    if not found_end:
        raise ValueError(f"{sxm_path}: missing :SCANIT_END: marker")
    _flush()
    return params


def sxm_dims(hdr: dict) -> Tuple[int, int]:
    """Return (Nx, Ny) scanned pixel dimensions from the header."""
    nums = [int(x) for x in re.findall(r"\d+", hdr.get("SCAN_PIXELS", ""))]
    if len(nums) < 2:
        raise ValueError("SXM header is missing valid SCAN_PIXELS dimensions")
    Nx, Ny = nums[0], nums[1]
    if Nx <= 0 or Ny <= 0:
        raise ValueError(f"SXM header has invalid SCAN_PIXELS dimensions: {(Nx, Ny)}")
    return (Nx, Ny)


def sxm_scan_range(hdr: dict) -> Tuple[float, float]:
    """Return (width_m, height_m) from the SCAN_RANGE header entry."""
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                      hdr.get("SCAN_RANGE", ""))
    if len(nums) >= 2:
        try:
            return (float(nums[0]), float(nums[1]))
        except ValueError:
            pass
    return (0.0, 0.0)


def sxm_data_info(hdr: dict) -> list[dict[str, str]]:
    """Return channel rows parsed from the Nanonis ``DATA_INFO`` header.

    Nanonis stores this section as a tabular block, but ``parse_sxm_header``
    normalises header continuation lines to spaces.  The useful schema is:
    ``Channel Name Unit Direction Calibration Offset`` followed by one
    six-token row per acquired signal.  The returned ``name`` is display
    friendly, with Nanonis underscores converted back to spaces.
    """
    raw = str(hdr.get("DATA_INFO", "")).strip()
    if not raw:
        return []
    toks = raw.split()
    if "Offset" not in toks:
        return []
    start = toks.index("Offset") + 1
    rows: list[dict[str, str]] = []
    for i in range(start, len(toks), 6):
        chunk = toks[i:i + 6]
        if len(chunk) < 6:
            break
        _channel, name, unit, direction, calibration, offset = chunk
        rows.append(
            {
                "name": _display_channel_name(name),
                "unit": unit,
                "direction": direction.lower(),
                "calibration": calibration,
                "offset": offset,
            }
        )
    return rows


def sxm_plane_metadata(hdr: dict, n_planes: int) -> tuple[list[str], list[str]]:
    """Return plane names and units for the decoded SXM payload planes."""
    rows = sxm_data_info(hdr)
    names: list[str] = []
    units: list[str] = []

    for row in rows:
        base = row["name"]
        unit = row["unit"]
        direction = row["direction"]
        if direction == "both":
            directions = ("forward", "backward")
        elif direction in {"forward", "forw", "fwd"}:
            directions = ("forward",)
        elif direction in {"backward", "backw", "bwd"}:
            directions = ("backward",)
        else:
            directions = (direction,) if direction else ("",)
        for d in directions:
            label = f"{base} {d}".strip()
            names.append(label)
            units.append(unit)

    while len(names) < n_planes:
        names.append(f"Channel {len(names)}")
        units.append("")

    return names[:n_planes], units[:n_planes]


def _expanded_plane_directions(rows: list[dict[str, str]]) -> list[str]:
    directions: list[str] = []
    for row in rows:
        direction = row["direction"]
        if direction == "both":
            directions.extend(("forward", "backward"))
        elif direction in {"forward", "forw", "fwd"}:
            directions.append("forward")
        elif direction in {"backward", "backw", "bwd"}:
            directions.append("backward")
        elif direction:
            directions.append(direction)
        else:
            directions.append("forward")
    return directions


def _plane_direction(hdr: dict, plane_idx: int) -> str:
    directions = _expanded_plane_directions(sxm_data_info(hdr))
    if 0 <= plane_idx < len(directions):
        return directions[plane_idx]
    return "backward" if plane_idx % 2 == 1 else "forward"


def _display_channel_name(name: str) -> str:
    return name.replace("_", " ")


# ── Plane orientation ────────────────────────────────────────────────────────

def orient_plane(arr: np.ndarray, hdr: dict, plane_idx: int) -> np.ndarray:
    """Apply canonical display orientation to a raw SXM plane.

    Two corrections are needed to present an array with origin = top-left
    and forward scan direction:
      * ``SCAN_DIR='up'`` rows were acquired bottom-to-top → flip vertically.
      * DATA_INFO backward planes were acquired right-to-left → flip
        horizontally.  Headers without direction metadata fall back to the
        legacy odd-plane convention.
    """
    scan_dir = hdr.get("SCAN_DIR", "down").strip().lower()
    if scan_dir == "up":
        arr = np.flipud(arr)
    if _plane_direction(hdr, plane_idx) == "backward":
        arr = np.fliplr(arr)
    return arr


# ── Reading planes ───────────────────────────────────────────────────────────

def read_sxm_plane(
    sxm_path: Path,
    plane_idx: int = 0,
    orient: bool = True,
    cushion_dir: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Return a float64 array for one plane of an SXM file.

    A missing plane index returns ``None``. Malformed headers, corrupt payload
    offsets, and other file integrity errors raise instead of being hidden.
    """
    if plane_idx < 0:
        return None
    hdr = parse_sxm_header(sxm_path)
    Nx, Ny = sxm_dims(hdr)
    raw = Path(sxm_path).read_bytes()
    offset = _data_offset_in_file(raw, cushion_dir)
    plane_bytes = Ny * Nx * 4
    start = offset + plane_idx * plane_bytes
    if start + plane_bytes > len(raw):
        return None
    arr = np.frombuffer(raw[start:start + plane_bytes], dtype=">f4").copy()
    arr = arr.reshape((Ny, Nx))
    if orient:
        arr = orient_plane(arr, hdr, plane_idx)
    return arr.astype(np.float64)


def read_all_sxm_planes(
    sxm_path: Path,
    orient: bool = True,
    cushion_dir: Optional[Path] = None,
) -> Tuple[dict, List[np.ndarray]]:
    """Return ``(header_dict, planes)`` for all payload planes in an SXM file."""
    hdr = parse_sxm_header(sxm_path)
    Nx, Ny = sxm_dims(hdr)
    raw = Path(sxm_path).read_bytes()
    offset = _data_offset_in_file(raw, cushion_dir)
    plane_bytes = Ny * Nx * 4
    planes: List[np.ndarray] = []
    n_planes = (len(raw) - offset) // plane_bytes
    # Compute the expected plane count by expanding "both"-direction rows to 2 planes each.
    _rows = sxm_data_info(hdr)
    expected_planes = sum(
        2 if r.get("direction", "") == "both" else 1 for r in _rows
    )
    if expected_planes > 0 and n_planes < expected_planes:
        import warnings
        warnings.warn(
            f"{Path(sxm_path).name}: SXM payload contains {n_planes} complete "
            f"plane(s) but DATA_INFO header indicates {expected_planes}. "
            f"The file may have been incompletely written. "
            f"Missing planes: {expected_planes - n_planes}.",
            UserWarning,
            stacklevel=2,
        )
    for idx in range(n_planes):
        start = offset + idx * plane_bytes
        if start + plane_bytes > len(raw):
            break
        arr = np.frombuffer(raw[start:start + plane_bytes], dtype=">f4").copy()
        arr = arr.reshape((Ny, Nx))
        if orient:
            arr = orient_plane(arr, hdr, idx)
        planes.append(arr.astype(np.float64))
    return hdr, planes


def sxm_payload_plane_count(
    sxm_path: Path,
    hdr: Optional[dict] = None,
    cushion_dir: Optional[Path] = None,
) -> int:
    """Return the number of complete image planes in an SXM payload."""

    sxm_path = Path(sxm_path)
    if hdr is None:
        hdr = parse_sxm_header(sxm_path)
    Nx, Ny = sxm_dims(hdr)
    raw = sxm_path.read_bytes()
    offset = _data_offset_in_file(raw, cushion_dir)
    plane_bytes = Ny * Nx * 4
    if plane_bytes <= 0 or len(raw) <= offset:
        return 0
    return (len(raw) - offset) // plane_bytes


# ── Writing a modified .sxm ──────────────────────────────────────────────────

def _patch_comment_in_header(header_bytes: bytes, new_comment: str) -> bytes:
    """Replace the :COMMENT: value in an SXM header byte block.

    Locates the existing value (everything between ``:COMMENT:\\n`` and the
    next ``:KEY:`` line) and replaces it with *new_comment*.  The function is
    a no-op when no ``:COMMENT:`` section is found.
    """
    enc = new_comment.encode("latin-1", errors="replace")
    marker = b":COMMENT:"
    idx = header_bytes.find(marker)
    if idx < 0:
        return header_bytes

    # Skip to the byte immediately after ":COMMENT:\n"
    nl_idx = header_bytes.find(b"\n", idx + len(marker))
    if nl_idx < 0:
        return header_bytes
    value_start = nl_idx + 1

    # Advance line by line until a line that starts with ":" is found —
    # that is the start of the next :KEY: section.
    pos = value_start
    while pos < len(header_bytes):
        eol = header_bytes.find(b"\n", pos)
        if eol < 0:
            pos = len(header_bytes)
            break
        next_start = eol + 1
        if next_start < len(header_bytes) and header_bytes[next_start:next_start + 1] == b":":
            pos = next_start
            break
        pos = next_start

    return header_bytes[:value_start] + enc + b"\n" + header_bytes[pos:]


def write_sxm_with_planes(
    src_sxm: Path,
    out_sxm: Path,
    new_planes: List[np.ndarray],
    cushion_dir: Optional[Path] = None,
    comment_override: Optional[str] = None,
    *,
    overwrite: bool = False,
) -> None:
    """Rewrite ``out_sxm`` using the header of ``src_sxm`` and new plane data.

    The header block (up to and including the fixed binary prefix) is copied
    verbatim from the source file; only the float32 payload is replaced.

    ``new_planes`` must contain exactly as many (Ny, Nx) arrays as the source
    file had.  Arrays are cast to big-endian float32 on write.  Each array is
    expected to be in the **display orientation** returned by ``orient_plane``
    — on write we invert that orientation to match how Nanonis stores data.

    Safety guards (added 2026-05-28, review IO #1):
        * Raises ``ValueError`` if ``out_sxm`` resolves to the same file as
          ``src_sxm`` (would silently corrupt the source instrument file).
        * Raises ``FileExistsError`` if ``out_sxm`` already exists and
          ``overwrite=False`` (default).  Pass ``overwrite=True`` to allow
          replacing an existing artifact (e.g. when called from
          :func:`probeflow.io.writers.sxm.write_sxm` which has already done
          its own collision check upstream).
    """
    # Lazy import to avoid the import-cycle: io.common -> core -> io
    from probeflow.io.common import check_output_available, check_overwrite

    src_sxm = Path(src_sxm)
    out_sxm = Path(out_sxm)

    check_overwrite(src_sxm, out_sxm)
    check_output_available(out_sxm, overwrite=overwrite)

    hdr = parse_sxm_header(src_sxm)
    Nx, Ny = sxm_dims(hdr)

    raw = src_sxm.read_bytes()
    offset = _data_offset_in_file(raw, cushion_dir)
    plane_bytes = Ny * Nx * 4

    header_prefix = raw[:offset]
    if comment_override is not None:
        header_prefix = _patch_comment_in_header(header_prefix, comment_override)

    n_planes_src = 0
    while offset + (n_planes_src + 1) * plane_bytes <= len(raw):
        n_planes_src += 1
    if n_planes_src == 0:
        raise ValueError(f"{src_sxm}: no data planes detected")

    if len(new_planes) != n_planes_src:
        raise ValueError(
            f"Plane count mismatch: source has {n_planes_src} plane(s), "
            f"got {len(new_planes)}"
        )

    payload = bytearray()
    for idx, arr in enumerate(new_planes):
        if arr.shape != (Ny, Nx):
            raise ValueError(
                f"Plane {idx} shape {arr.shape} != expected {(Ny, Nx)}"
            )
        # Invert orientation so we write in Nanonis native order.
        undo = arr
        scan_dir = hdr.get("SCAN_DIR", "down").strip().lower()
        if idx % 2 == 1:
            undo = np.fliplr(undo)
        if scan_dir == "up":
            undo = np.flipud(undo)
        payload.extend(np.ascontiguousarray(undo, dtype=">f4").tobytes())

    tail_start = offset + n_planes_src * plane_bytes
    tail = raw[tail_start:]

    out_sxm.parent.mkdir(parents=True, exist_ok=True)
    out_sxm.write_bytes(bytes(header_prefix) + bytes(payload) + bytes(tail))
