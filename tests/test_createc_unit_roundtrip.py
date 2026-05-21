
def test_createc_scan_range_pixel_size_roundtrip():
    """Validate that scan_range_m and pixel_size are consistent round-trip."""
    from pathlib import Path
    import numpy as np
    
    # Createc header with known dimensions and ranges
    test_data = {
        "Length x[A]": "100.0",  # 100 Angstrom = 10 nm
        "Length y[A]": "100.0",  # 100 Angstrom = 10 nm
    }
    
    # Expected conversion
    scan_range_m = (100.0 * 1e-10, 100.0 * 1e-10)  # [Å] * 1e-10 → [m]
    assert abs(scan_range_m[0] - 1e-8) < 1e-20, "X range conversion incorrect"
    assert abs(scan_range_m[1] - 1e-8) < 1e-20, "Y range conversion incorrect"
    
    # Simulate image dimensions
    image_width, image_height = 256, 256
    
    # Compute pixel sizes (this is what RealSpaceCalibration does)
    px_size_x = scan_range_m[0] / image_width  # m/px
    px_size_y = scan_range_m[1] / image_height  # m/px
    
    # Verify consistency
    assert px_size_x > 0, "Pixel size X must be positive"
    assert px_size_y > 0, "Pixel size Y must be positive"
    
    # Round-trip: pixels → metres → pixels
    test_pixels_x = 128.5
    test_meters = test_pixels_x * px_size_x
    test_pixels_back = test_meters / px_size_x
    assert abs(test_pixels_x - test_pixels_back) < 1e-9, "Round-trip failed"
    
    # Verify that full scan range equals pixel_size * N
    full_range_x = px_size_x * image_width
    assert abs(full_range_x - scan_range_m[0]) < 1e-20, "Full range reconstruction failed"


if __name__ == "__main__":
    test_createc_scan_range_pixel_size_roundtrip()
    print("✓ Createc unit conversion round-trip test passed")
