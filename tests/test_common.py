"""Contract tests for probeflow.io.common."""

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


def test_safe_conversion_and_header_helpers_contract():
    assert _f("3.14") == pytest.approx(3.14)
    assert _f("3,14") == pytest.approx(3.14)
    assert _f(None, 0.0) == 0.0
    assert _f("abc", -1.0) == -1.0
    assert _i("512") == 512
    assert _i("512.9") == 512
    assert _i("nope", 0) == 0

    hdr = parse_header(b"Key1=value1\nsection/SubKey=val\nbad\n")
    assert hdr == {"Key1": "value1", "SubKey": "val"}
    assert parse_header(b"") == {}
    assert find_hdr({"GainPre": "9"}, "gainpre") == "9"
    assert find_hdr({"T_AUXADC6[K]": "4.2"}, "AUXADC6") == "4.2"
    assert find_hdr({}, "missing", "fallback") == "fallback"
    assert find_hdr({}, "missing") is None

    assert get_dac_bits({"DAC-Type": "20bit"}) == 20
    assert get_dac_bits({"DAC-Type": "20 bit"}) == 20
    assert get_dac_bits({}) == DAC_BITS_DEFAULT
    assert get_dac_bits({"DAC-Type": "unknown"}) == DAC_BITS_DEFAULT
    assert sanitize("hello world") == "hello_world"
    assert sanitize("Z (m)") == "Z_m"
    assert sanitize("file-name_01.ext") == "file-name_01.ext"


def test_dac_voltage_height_and_current_scales_contract():
    assert v_per_dac(20) == pytest.approx(DAC_VOLTAGE_REF / 2**20)
    assert v_per_dac() == v_per_dac(DAC_BITS_DEFAULT)
    assert v_per_dac(16) == pytest.approx(10.0 / 65536)

    vpd = v_per_dac(20)
    assert z_scale_m_per_dac({"Dacto[A]z": "1.0"}, vpd) == pytest.approx(1e-9)
    assert z_scale_m_per_dac({"Dacto[A]z": "5.0", "GainZ": "10"}, vpd) == pytest.approx(
        5.0e-9
    )
    assert z_scale_m_per_dac({"Dacto[A]z": "5.0", "GainZ": "3"}, vpd) == pytest.approx(
        5.0 * 0.3e-9
    )
    assert z_scale_m_per_dac({"GainZ": "10.0", "ZPiezoconst": "19.2"}, vpd) == pytest.approx(
        2.0 * vpd * 19.2e-9,
        rel=1e-6,
    )

    current_scale = i_scale_a_per_dac({"GainPre": "9"}, vpd, negative=False)
    assert current_scale == pytest.approx(vpd / 1e9, rel=1e-6)
    assert current_scale > 0
    assert i_scale_a_per_dac({"GainPre": "9"}, vpd, negative=True) < 0


def test_channel_detection_and_trim_contract():
    rng = np.random.default_rng(0)
    for n_channels in (2, 4):
        Ny, Nx = 16, 16
        payload = rng.random(n_channels * Ny * Nx).astype("<f4").tobytes()
        stack, detected = detect_channels(payload, Ny, Nx)
        assert detected == n_channels
        assert stack.shape == (n_channels, Ny, Nx)

    with pytest.raises(ValueError, match="Payload too small"):
        detect_channels(b"\x00" * 4, 16, 16)

    stack = np.ones((2, 8, 4), dtype=np.float32)
    stack[:, 6:, :] = 0.0
    trimmed, new_Ny = trim_stack(stack)
    assert new_Ny < 8
    assert trimmed.shape[1] == new_Ny

    untrimmed, untrimmed_Ny = trim_stack(np.ones((2, 4, 4), dtype=np.float32))
    assert untrimmed_Ny == 4
    assert untrimmed.shape[1] == 4

    zero_trimmed, zero_Ny = trim_stack(np.zeros((2, 4, 4), dtype=np.float32))
    assert zero_Ny >= 1
    assert zero_trimmed.shape[1] == zero_Ny


def test_clip_and_uint8_helpers_contract():
    vmin, vmax = percentile_clip(np.linspace(0, 100, 1000), 1, 99)
    assert 0 < vmin < vmax < 100

    with pytest.raises(ValueError):
        percentile_clip(np.full((10,), np.nan))

    assert percentile_clip(np.ones(100)) == (0.0, 2.0)

    u8 = to_uint8(np.array([-10.0, 0.5, 1.0]), 0.0, 1.0)
    assert u8.dtype == np.uint8
    assert u8[0] == 0
    assert u8[-1] == 255


def test_check_overwrite_contract(tmp_path):
    source = tmp_path / "scan.sxm"
    source.touch()

    with pytest.raises(ValueError, match="overwrite"):
        check_overwrite(source, source)

    check_overwrite(source, tmp_path / "scan_processed.sxm")
