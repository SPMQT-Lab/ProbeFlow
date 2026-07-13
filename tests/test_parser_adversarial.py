"""Adversarial parser tests: truncated, corrupted, and partial files.

2026-06-12 parser review. Two contracts:

* **Metadata ↔ full-parse agreement** — for any file, the metadata fast path
  and the full reader must agree: both succeed with consistent counts, or
  both fail. The review found the Createc VERT metadata path summarising a
  corrupt table as healthy (lenient ``np.fromstring``) while the full parse
  raised — so browse (and the metadata cache) showed a spectrum the viewer
  could not load.
* **Failures are loud and informative; partial files are explicit** —
  decoders either raise a ValueError naming the file and the problem, or
  degrade to the complete subset of the data with warnings recorded (the
  scan-still-being-written case), never silent garbage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TESTDATA = Path(__file__).resolve().parent.parent / "test_data"

DAT_SCAN_FIXTURES = sorted(TESTDATA.glob("createc_*.dat"))
SXM_FIXTURE = TESTDATA / "nanonis.sxm"
SM4_FIXTURE = TESTDATA / "rhk.sm4"


# ── Metadata ↔ full-parse agreement across the real corpus ────────────────────

class TestMetadataFullAgreement:
    def test_vert_corpus(self, createc_time_spec, createc_bias_spec):
        from probeflow.io.readers.createc_vert import read_createc_vert_report

        for path in (createc_time_spec, createc_bias_spec):
            full = read_createc_vert_report(path, include_arrays=True)
            meta = read_createc_vert_report(path, include_arrays=False)
            assert meta.raw_table_shape == full.raw_table_shape
            assert meta.column_names == full.column_names
            assert meta.bias_min_mv == pytest.approx(full.bias_min_mv)
            assert meta.bias_max_mv == pytest.approx(full.bias_max_mv)

    def test_nanonis_spec_corpus(self, nanonis_spec):
        from probeflow.io.spectroscopy import read_spec_file, read_spec_metadata

        full = read_spec_file(nanonis_spec)
        meta = read_spec_metadata(nanonis_spec)
        assert meta.metadata["n_points"] == full.metadata["n_points"]
        assert tuple(meta.channels) == tuple(full.channel_order)

    @pytest.mark.parametrize(
        "path", DAT_SCAN_FIXTURES, ids=lambda p: p.name)
    def test_createc_scan_corpus(self, path):
        from probeflow.core.metadata import read_scan_metadata
        from probeflow.core.scan_loader import load_scan

        full = load_scan(path)
        meta = read_scan_metadata(path)
        assert meta.shape == full.planes[0].shape
        assert tuple(meta.plane_names) == tuple(full.plane_names)


# ── Createc VERT: corruption and truncation ───────────────────────────────────

class TestVertCorruption:
    @pytest.fixture
    def vert_bytes(self, createc_bias_spec):
        return createc_bias_spec.read_bytes()

    def _write(self, tmp_path, data: bytes) -> Path:
        p = tmp_path / "mutated.VERT"
        p.write_bytes(data)
        return p

    def test_corrupt_row_fails_in_both_paths(self, tmp_path, vert_bytes):
        """The bug: a garbage token mid-table raised in the full parse but
        summarised as healthy in the metadata path."""
        import re

        from probeflow.io.readers.createc_vert import read_createc_vert_report

        text = vert_bytes.decode("latin-1")
        lines = text.splitlines(keepends=True)
        for i in range(len(lines) // 2, len(lines)):
            if re.match(r"\s*-?\d", lines[i]):
                lines[i] = lines[i][:10] + "GARBAGE_##\n"
                break
        p = self._write(tmp_path, "".join(lines).encode("latin-1"))

        with pytest.raises(ValueError, match="mutated.VERT"):
            read_createc_vert_report(p, include_arrays=True)
        with pytest.raises(ValueError, match="mutated.VERT"):
            read_createc_vert_report(p, include_arrays=False)

    def test_mid_row_truncation_fails_in_both_paths(self, tmp_path, vert_bytes):
        from probeflow.io.readers.createc_vert import read_createc_vert_report

        p = self._write(tmp_path, vert_bytes[: int(len(vert_bytes) * 0.5) + 7])
        results = []
        for include in (True, False):
            try:
                read_createc_vert_report(p, include_arrays=include)
                results.append("ok")
            except ValueError:
                results.append("raise")
        assert results[0] == results[1], (
            f"metadata/full divergence on truncated file: {results}"
        )

    def test_row_boundary_truncation_warns_about_row_count(
            self, tmp_path, vert_bytes):
        """Cutting whole trailing rows is the partial-sweep case: both paths
        load what is there and warn that the params line promised more."""
        from probeflow.io.readers.createc_vert import read_createc_vert_report

        text = vert_bytes.decode("latin-1")
        head, _, _tail = text.rpartition("\n")
        # Drop the last five lines cleanly.
        kept = "\n".join(head.splitlines()[:-5]) + "\n"
        p = self._write(tmp_path, kept.encode("latin-1"))

        for include in (True, False):
            rep = read_createc_vert_report(p, include_arrays=include)
            assert any("point(s), parsed" in w for w in rep.warnings), (
                "partial table not flagged"
            )

    def test_missing_data_marker(self, tmp_path, vert_bytes):
        from probeflow.io.readers.createc_vert import read_createc_vert_report

        pos = vert_bytes.find(b"DATA")
        p = self._write(tmp_path, vert_bytes[:pos])
        with pytest.raises(ValueError, match="missing DATA marker"):
            read_createc_vert_report(p)

    def test_header_only_chunked_reader_handles_boundary(self, tmp_path):
        """parse_createc_vert_header streams in 64 KiB chunks; a DATA marker
        straddling the chunk boundary must still be found."""
        from probeflow.io.readers.createc_vert import parse_createc_vert_header

        filler = b"Comment=" + b"x" * (65536 - 10) + b"\r\n"
        pad = b"A=1\r\n" * 3
        blob = filler + pad
        # Position DATA so it straddles the 64 KiB boundary.
        blob = blob[:65534] + b"\r\nDATA\r\nparams\r\n0 0 0 0\r\n"
        p = tmp_path / "straddle.VERT"
        p.write_bytes(b"[ParVERT30]\r\nKey=1\r\n" + blob)
        hdr = parse_createc_vert_header(p)
        assert hdr.get("Key") == "1"


# ── Createc DAT scans: corruption and truncation ──────────────────────────────

@pytest.mark.skipif(not DAT_SCAN_FIXTURES, reason="no DAT fixtures")
class TestCreatecDatCorruption:
    @pytest.fixture
    def dat_bytes(self):
        return DAT_SCAN_FIXTURES[0].read_bytes()

    def test_truncated_payload_names_the_problem(self, tmp_path, dat_bytes):
        from probeflow.io.readers.createc_dat import read_createc_dat_report

        p = tmp_path / "trunc.dat"
        p.write_bytes(dat_bytes[: len(dat_bytes) // 2])
        with pytest.raises(ValueError, match="corrupt or truncated"):
            read_createc_dat_report(p)

    def test_missing_marker(self, tmp_path, dat_bytes):
        from probeflow.io.readers.createc_dat import read_createc_dat_report

        pos = dat_bytes.find(b"DATA")
        p = tmp_path / "nomarker.dat"
        p.write_bytes(dat_bytes[:pos])
        with pytest.raises(ValueError, match="missing DATA marker"):
            read_createc_dat_report(p)

    def test_zero_dimensions_rejected(self, tmp_path, dat_bytes):
        from probeflow.io.readers.createc_dat import read_createc_dat_report

        mutated = dat_bytes.replace(b"Num.X", b"Nim.X")  # hide every spelling
        p = tmp_path / "nodims.dat"
        p.write_bytes(mutated)
        with pytest.raises(ValueError, match="invalid dimensions"):
            read_createc_dat_report(p)


# ── Partial-file tolerance (scan still being written) ─────────────────────────

class TestPartialFileTolerance:
    @pytest.mark.skipif(not SXM_FIXTURE.exists(), reason="no SXM fixture")
    def test_truncated_sxm_loads_complete_planes_with_warning(self, tmp_path):
        from probeflow.core.scan_loader import load_scan

        raw = SXM_FIXTURE.read_bytes()
        p = tmp_path / "half.sxm"
        p.write_bytes(raw[: len(raw) // 2])
        full = load_scan(SXM_FIXTURE)
        part = load_scan(p)

        assert 0 < part.n_planes < full.n_planes
        for plane in part.planes:
            assert np.isfinite(plane).all(), "partial plane contains garbage"
        assert any("incompletely written" in w for w in part.warnings), (
            "partial load not explained in scan.warnings"
        )

    @pytest.mark.skipif(not SM4_FIXTURE.exists(), reason="no SM4 fixture")
    def test_truncated_sm4_loads_complete_pages_with_warnings(self, tmp_path):
        from probeflow.core.scan_loader import load_scan

        raw = SM4_FIXTURE.read_bytes()
        p = tmp_path / "half.sm4"
        p.write_bytes(raw[: len(raw) // 2])
        part = load_scan(p)

        assert part.n_planes >= 1
        for plane in part.planes:
            assert np.isfinite(plane).all()
        assert part.warnings, "skipped pages/objects not recorded"

    @pytest.mark.skipif(not SXM_FIXTURE.exists(), reason="no SXM fixture")
    def test_viewer_scan_data_carries_the_warnings(self, tmp_path):
        """The reader explains the partial load; the viewer load path must
        carry that explanation instead of dropping it (scan.warnings was
        consumed nowhere before this review)."""
        from probeflow.gui.viewer.scan_load import load_scan_for_viewer

        raw = SXM_FIXTURE.read_bytes()
        p = tmp_path / "half.sxm"
        p.write_bytes(raw[: len(raw) // 2])
        data = load_scan_for_viewer(p, 0)
        assert data.raw_arr is not None
        assert data.scan_warnings
        assert "incompletely written" in data.scan_warnings[0]


# ── Nanonis spec edge cases ───────────────────────────────────────────────────

class TestNanonisSpecEdges:
    def test_single_column_file_parses_as_column(self, tmp_path):
        """loadtxt collapses one-column data to 1-D; it must come back as
        (N, 1), not one N-wide row that fails the column-count check."""
        from probeflow.io.readers.nanonis_spec import read_nanonis_spec

        p = tmp_path / "single.dat"
        contents = (
            "Experiment\tHistory Data\r\n"
            "[DATA]\r\n"
            "Current (A)\r\n"
            + "".join(f"{v:.3e}\r\n" for v in np.linspace(1e-12, 5e-12, 40))
        )
        # Write bytes so Windows text-mode newline translation cannot turn
        # the explicit CRLF sequences into CR-CR-LF blank lines.
        p.write_bytes(contents.encode("latin-1"))
        spec = read_nanonis_spec(p)
        assert spec.metadata["n_points"] == 40
        assert spec.channels["Current"].shape == (40,)

    def test_truncation_fails_in_both_paths(self, tmp_path, nanonis_spec):
        from probeflow.io.spectroscopy import read_spec_file, read_spec_metadata

        raw = nanonis_spec.read_bytes()
        p = tmp_path / "trunc.dat"
        p.write_bytes(raw[: int(len(raw) * 0.95)])
        with pytest.raises(ValueError):
            read_spec_file(p)
        with pytest.raises(ValueError):
            read_spec_metadata(p)
