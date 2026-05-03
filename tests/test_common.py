"""Unit tests for probeflow.io.common."""

import numpy as np
import pytest
from probeflow.io.common import (
    DAC_BITS_DEFAULT,
    DAC_VOLTAGE_REF,
    _f,
    _i,
    check_overwrite,
    detect_channels,
    find_hdr,
    get_dac_bits,
    i_scale_a_per_dac,
    parse_header,
    percentile_clip,
    sanitize,
    to_uint8,
    trim_stack,
    v_per_dac,
    z_scale_m_per_dac,
)


# ─── _f / _i ──────────────────────────────────────────────────────────────────

class TestSafeConversions:
    def test_f_normal(self):
        assert _f("3.14") == pytest.approx(3.14)

    def test_f_comma(self):
        assert _f("3,14") == pytest.approx(3.14)

    def test_f_none_returns_default(self):
        assert _f(None, 0.0) == 0.0

    def test_f_junk_returns_default(self):
        assert _f("abc", -1.0) == -1.0

    def test_i_normal(self):
        assert _i("512") == 512

    def test_i_float_string(self):
        assert _i("512.9") == 512

    def test_i_bad_returns_default(self):
        assert _i("nope", 0) == 0


# ─── v_per_dac ────────────────────────────────────────────────────────────────

class TestVPerDac:
    def test_20bit(self):
        assert v_per_dac(20) == pytest.approx(DAC_VOLTAGE_REF / 2**20)

    def test_default(self):
        assert v_per_dac() == v_per_dac(DAC_BITS_DEFAULT)

    def test_16bit(self):
        assert v_per_dac(16) == pytest.approx(10.0 / 65536)


# ─── parse_header ─────────────────────────────────────────────────────────────

class TestParseHeader:
    def test_basic(self):
        hb = b"Key1=value1\nKey2=value2\n"
        hdr = parse_header(hb)
        assert hdr["Key1"] == "value1"
        assert hdr["Key2"] == "value2"

    def test_path_key(self):
        hb = b"section/SubKey=val\n"
        hdr = parse_header(hb)
        assert hdr["SubKey"] == "val"

    def test_no_equals_ignored(self):
        hb = b"no_equals_here\nKey=ok\n"
        hdr = parse_header(hb)
        assert "no_equals_here" not in hdr
        assert hdr["Key"] == "ok"

    def test_empty_bytes(self):
        assert parse_header(b"") == {}


# ─── find_hdr ─────────────────────────────────────────────────────────────────

class TestFindHdr:
    def test_exact_match(self):
        hdr = {"Num.X": "256", "Num.Y": "256"}
        assert find_hdr(hdr, "Num.X") == "256"

    def test_case_insensitive(self):
        hdr = {"GainPre": "9"}
        assert find_hdr(hdr, "gainpre") == "9"

    def test_substring(self):
        hdr = {"T_AUXADC6[K]": "4.2"}
        assert find_hdr(hdr, "AUXADC6") == "4.2"

    def test_missing_returns_default(self):
        assert find_hdr({}, "missing", "fallback") == "fallback"

    def test_missing_returns_none_by_default(self):
        assert find_hdr({}, "missing") is None


# ─── get_dac_bits ─────────────────────────────────────────────────────────────

class TestGetDacBits:
    def test_explicit_20bit(self):
        hdr = {"DAC-Type": "20bit"}
        assert get_dac_bits(hdr) == 20

    def test_with_spaces(self):
        hdr = {"DAC-Type": "20 bit"}
        assert get_dac_bits(hdr) == 20

    def test_missing_returns_default(self):
        assert get_dac_bits({}) == DAC_BITS_DEFAULT

    def test_bad_value_returns_default(self):
        hdr = {"DAC-Type": "unknown"}
        assert get_dac_bits(hdr) == DAC_BITS_DEFAULT


# ─── sanitize ─────────────────────────────────────────────────────────────────

class TestSanitize:
    def test_spaces_replaced(self):
        assert sanitize("hello world") == "hello_world"

    def test_special_chars(self):
        assert sanitize("Z (m)") == "Z_m"

    def test_safe_chars_preserved(self):
        assert sanitize("file-name_01.ext") == "file-name_01.ext"


# ─── z_scale_m_per_dac ────────────────────────────────────────────────────────

class TestZScale:
    def test_uses_createc_dacto_z_as_nm_per_dac_when_present(self):
        hdr = {"Dacto[A]z": "5.0"}
        vpd = v_per_dac(20)
        zs = z_scale_m_per_dac(hdr, vpd)
        assert zs == pytest.approx(5.0 * 1e-9)

    def test_fallback_uses_zpiezoconst_without_gainz(self):
        hdr = {"GainZ": "10.0", "ZPiezoconst": "19.2"}
        vpd = v_per_dac(20)
        zs = z_scale_m_per_dac(hdr, vpd)
        expected = 2.0 * vpd * 19.2 * 1e-9
        assert zs == pytest.approx(expected, rel=1e-6)

    def test_returns_metres(self):
        hdr = {"Dacto[A]z": "1.0"}
        vpd = v_per_dac(20)
        zs = z_scale_m_per_dac(hdr, vpd)
        assert zs == pytest.approx(1e-9)

    def test_dacto_z_uses_gainz_reference_10(self):
        hdr = {"Dacto[A]z": "5.0", "GainZ": "10"}
        vpd = v_per_dac(20)
        zs = z_scale_m_per_dac(hdr, vpd)
        assert zs == pytest.approx(5.0 * 1e-9)

    def test_dacto_z_scales_legacy_gainz_three(self):
        hdr = {"Dacto[A]z": "5.0", "GainZ": "3"}
        vpd = v_per_dac(20)
        zs = z_scale_m_per_dac(hdr, vpd)
        assert zs == pytest.approx(5.0 * 0.3 * 1e-9)


# ─── i_scale_a_per_dac ────────────────────────────────────────────────────────

class TestIScale:
    def test_negative_by_default(self):
        hdr = {"GainPre": "9"}
        vpd = v_per_dac(20)
        is_ = i_scale_a_per_dac(hdr, vpd, negative=True)
        assert is_ < 0

    def test_positive_when_requested(self):
        hdr = {"GainPre": "9"}
        vpd = v_per_dac(20)
        is_ = i_scale_a_per_dac(hdr, vpd, negative=False)
        assert is_ > 0

    def test_magnitude(self):
        hdr = {"GainPre": "9"}
        vpd = v_per_dac(20)
        is_ = i_scale_a_per_dac(hdr, vpd, negative=False)
        assert is_ == pytest.approx(vpd / 1e9, rel=1e-6)


# ─── detect_channels ──────────────────────────────────────────────────────────

class TestDetectChannels:
    def _make_payload(self, n, Ny, Nx):
        arr = np.random.rand(n * Ny * Nx).astype("<f4")
        return arr.tobytes()

    def test_detects_4_channels(self):
        Ny, Nx = 16, 16
        payload = self._make_payload(4, Ny, Nx)
        stack, n = detect_channels(payload, Ny, Nx)
        assert n == 4
        assert stack.shape == (4, Ny, Nx)

    def test_detects_2_channels(self):
        Ny, Nx = 16, 16
        payload = self._make_payload(2, Ny, Nx)
        stack, n = detect_channels(payload, Ny, Nx)
        assert n == 2
        assert stack.shape == (2, Ny, Nx)

    def test_raises_on_too_small(self):
        payload = b"\x00" * 4  # 1 float, way too small
        with pytest.raises(ValueError, match="Payload too small"):
            detect_channels(payload, 16, 16)


# ─── trim_stack ───────────────────────────────────────────────────────────────

class TestTrimStack:
    def test_trims_zero_rows(self):
        Ny, Nx = 8, 4
        stack = np.ones((2, Ny, Nx), dtype=np.float32)
        # zero out last 2 rows
        stack[:, 6:, :] = 0.0
        trimmed, new_Ny = trim_stack(stack)
        assert new_Ny < Ny
        assert trimmed.shape[1] == new_Ny

    def test_no_trim_needed(self):
        stack = np.ones((2, 4, 4), dtype=np.float32)
        trimmed, new_Ny = trim_stack(stack)
        assert new_Ny == 4

    def test_all_zeros(self):
        stack = np.zeros((2, 4, 4), dtype=np.float32)
        trimmed, new_Ny = trim_stack(stack)
        assert new_Ny >= 1  # always returns at least 1 row


# ─── percentile_clip ──────────────────────────────────────────────────────────

class TestPercentileClip:
    def test_basic(self):
        arr = np.linspace(0, 100, 1000)
        vmin, vmax = percentile_clip(arr, 1, 99)
        assert vmin > 0
        assert vmax < 100

    def test_all_nan(self):
        arr = np.full((10,), np.nan)
        with pytest.raises(ValueError):
            percentile_clip(arr)

    def test_constant_array(self):
        arr = np.ones(100)
        vmin, vmax = percentile_clip(arr)
        assert vmin == 0.0
        assert vmax == 2.0


# ─── to_uint8 ─────────────────────────────────────────────────────────────────

class TestToUint8:
    def test_maps_range(self):
        arr = np.array([0.0, 0.5, 1.0])
        u8 = to_uint8(arr, 0.0, 1.0)
        assert u8[0] == 0
        assert u8[2] == 255

    def test_clips_below(self):
        arr = np.array([-10.0, 0.5, 1.0])
        u8 = to_uint8(arr, 0.0, 1.0)
        assert u8[0] == 0

    def test_output_dtype(self):
        arr = np.linspace(0, 1, 50)
        u8 = to_uint8(arr, 0.0, 1.0)
        assert u8.dtype == np.uint8


# ─── check_overwrite ──────────────────────────────────────────────────────────

class TestCheckOverwrite:
    def test_raises_when_same_path(self, tmp_path):
        f = tmp_path / "scan.sxm"
        f.touch()
        with pytest.raises(ValueError, match="overwrite"):
            check_overwrite(f, f)

    def test_no_raise_for_different_paths(self, tmp_path):
        inp = tmp_path / "scan.sxm"
        out = tmp_path / "scan_processed.sxm"
        check_overwrite(inp, out)
