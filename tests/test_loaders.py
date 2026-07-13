from pathlib import Path

import pytest

from probeflow.io.file_type import FileType
from probeflow.core.loaders import identify_scan_file, identify_spectrum_file


TESTDATA = Path(__file__).resolve().parents[1] / "test_data"
_CREATEC_SCAN = TESTDATA / "createc_scan_11nm.dat"
_NANONIS_SCAN = TESTDATA / "nanonis.sxm"


class TestIdentifyScanFile:
    def test_identifies_createc_scan(self):
        sig = identify_scan_file(_CREATEC_SCAN)
        assert sig.path == _CREATEC_SCAN
        assert sig.file_type == FileType.CREATEC_IMAGE
        assert sig.item_type == "scan"
        assert sig.source_format == "dat"

    def test_identifies_nanonis_scan(self):
        sig = identify_scan_file(_NANONIS_SCAN)
        assert sig.file_type == FileType.NANONIS_IMAGE
        assert sig.source_format == "sxm"

    def test_rejects_spectroscopy(self, nanonis_spec):
        with pytest.raises(ValueError, match="spectroscopy.*scan sniff stage"):
            identify_scan_file(nanonis_spec)


class TestIdentifySpectrumFile:
    def test_identifies_createc_spec(self, createc_time_spec):
        sig = identify_spectrum_file(createc_time_spec)
        assert sig.path == createc_time_spec
        assert sig.file_type == FileType.CREATEC_SPEC
        assert sig.item_type == "spectrum"
        assert sig.source_format == "createc_vert"

    def test_identifies_nanonis_spec(self, nanonis_spec):
        sig = identify_spectrum_file(nanonis_spec)
        assert sig.file_type == FileType.NANONIS_SPEC
        assert sig.source_format == "nanonis_dat_spectrum"

    def test_rejects_scan_image(self):
        with pytest.raises(ValueError, match="scan image.*spectroscopy sniff stage"):
            identify_spectrum_file(_NANONIS_SCAN)

    def test_unknown_file_raises(self, tmp_path):
        bad = tmp_path / "unknown.bin"
        bad.write_bytes(b"not a probeflow file")
        with pytest.raises(ValueError, match="sniff stage could not identify"):
            identify_spectrum_file(bad)
