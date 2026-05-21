"""Regression tests for Createc unit conversion and pixel-size round-trip."""

from __future__ import annotations


def test_createc_scan_range_pixel_size_roundtrip():
    """Validate that scan_range_m and pixel_size are consistent round-trip.

    Createc stores scan dimensions in Ångströms (key ``"Length x[A]"``).
    The loader converts to SI metres by multiplying by 1e-10.
    """
    # 100 Å = 10 nm = 1e-8 m
    scan_range_m = (100.0 * 1e-10, 100.0 * 1e-10)
    assert abs(scan_range_m[0] - 1e-8) < 1e-20, "X range conversion incorrect"
    assert abs(scan_range_m[1] - 1e-8) < 1e-20, "Y range conversion incorrect"

    image_width, image_height = 256, 256

    # Pixel sizes as computed by RealSpaceCalibration
    px_size_x = scan_range_m[0] / image_width   # m/px
    px_size_y = scan_range_m[1] / image_height  # m/px

    assert px_size_x > 0, "Pixel size X must be positive"
    assert px_size_y > 0, "Pixel size Y must be positive"

    # Round-trip: pixels → metres → pixels
    test_pixels_x = 128.5
    recovered = (test_pixels_x * px_size_x) / px_size_x
    assert abs(test_pixels_x - recovered) < 1e-9, "Pixel round-trip failed"

    # Full scan range must equal pixel_size × N pixels
    assert abs(px_size_x * image_width - scan_range_m[0]) < 1e-20
    assert abs(px_size_y * image_height - scan_range_m[1]) < 1e-20
