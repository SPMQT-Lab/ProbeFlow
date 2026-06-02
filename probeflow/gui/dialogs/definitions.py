"""Processing definitions/help dialog for ProbeFlow."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Mapping

from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class _DefinitionEntry:
    """Structured help content for one processing operation."""

    title: str
    params: tuple[str, ...]
    summary: str
    equations: tuple[str, ...]
    details: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()


_DEFINITION_ENTRIES: tuple[_DefinitionEntry, ...] = (
    _DefinitionEntry(
        title="Bad-line correction",
        params=(
            "method = step | mad",
            "polarity = bright | dark",
            "threshold_mad",
            "min_segment_length_px",
            "max_adjacent_bad_lines",
        ),
        summary=(
            "Detects short damaged segments on fast-scan rows by comparing each "
            "row to nearby rows, then replaces only the accepted segment from "
            "valid neighbouring scan lines."
        ),
        equations=(
            "MADN(v) = 1.4826 * median(|v - median(v)|)\n"
            "b_i[j] = per-column median over nearby finite row samples\n"
            "r_i[j] = z_i[j] - b_i[j]\n\n"
            "segment convention:\n"
            "  S = [j0, j1) spans columns k where j0 <= k < j1\n"
            "  reject S when length(S) < min_segment_length_px\n\n"
            "step mode:\n"
            "  d_i[j] = r_i[j+1] - r_i[j]\n"
            "  cutoff = threshold_mad * MADN(d)\n"
            "  bright segment S=(j0:j1) requires d_i[j0-1] > cutoff and d_i[j1-1] < -cutoff\n"
            "  dark segment reverses those signs\n\n"
            "mad mode:\n"
            "  cutoff = threshold_mad * MADN(r)\n"
            "  bright: r_i[j] - median(r_i) > cutoff\n"
            "  dark:   r_i[j] - median(r_i) < -cutoff\n\n"
            "repair:\n"
            "  skip connected bad-row groups wider than max_adjacent_bad_lines\n"
            "  above/below neighbours must be finite on S and not marked bad on S\n"
            "  z'_i,S = 0.5 * (z_above,S + z_below,S) when both neighbours are valid\n"
            "  z'_i,S = nearest valid neighbour segment when only one side is valid",
        ),
        details=(
            "The minimum segment length rejects single-pixel speckles. The maximum "
            "adjacent bad-line setting is a safety limit: broader blocks of damaged "
            "rows are skipped because local interpolation becomes unreliable.",
            "Preview detection computes candidate segments without modifying the image. "
            "Applying correction preserves pixels outside accepted segments.",
        ),
        cautions=(
            "This is intended for local line artifacts. It is not a whole-row "
            "levelling tool and should not be used to flatten real terrace steps.",
        ),
    ),
    _DefinitionEntry(
        title="Row alignment",
        params=("method = median | mean | linear",),
        summary=(
            "Removes independent row offsets, and optionally row slopes, from "
            "raw scan lines before later background or filter operations."
        ),
        equations=(
            "median: z'_i,j = z_i,j - median_j(z_i,j)\n"
            "mean:   z'_i,j = z_i,j - mean_j(z_i,j)\n"
            "linear: fit a_i*x_j + b_i to finite pixels in row i, x_j in [-1, 1]\n"
            "        z'_i,j = z_i,j - (a_i*x_j + b_i)",
        ),
        details=(
            "Median is the robust default for STM data because isolated crashes or "
            "adsorbates do not dominate the row reference. Mean can be useful for "
            "clean rows. Linear also removes within-row tilt.",
        ),
        cautions=(
            "If a row contains mostly a real sloped feature, alignment can remove part "
            "of that physical signal.",
        ),
    ),
    _DefinitionEntry(
        title="Simple background subtraction",
        params=(
            "order = 1..4",
            "fit ROI",
            "apply ROI",
            "exclude ROI",
            "step_tolerance",
            "step_threshold_deg",
        ),
        summary=(
            "Fits a two-dimensional polynomial background on selected finite "
            "pixels, then subtracts the fitted surface from the whole image or "
            "from an apply ROI."
        ),
        equations=(
            "normalised coordinates: x, y in [-1, 1]\n"
            "B(x, y) = sum c_pq * x^p * y^q, for p + q <= order\n"
            "optional step-tolerant fit mask:\n"
            "  slope = sqrt((dz/dx / pixel_size_x_m)^2 + (dz/dy / pixel_size_y_m)^2)\n"
            "  keep fit pixel only when slope < tan(step_threshold_deg)\n"
            "least squares on fit pixels: min_c sum_fit (z(x,y) - B(x,y))^2\n"
            "z'(x, y) = z(x, y) - B(x, y)",
        ),
        details=(
            "The viewer's Plane/background button uses a first-order plane by default: "
            "B(x,y) = c00 + c10*x + c01*y.",
            "Fit ROI and exclude ROI affect only the pixels used to estimate the "
            "background. Apply ROI controls which pixels are modified.",
            "Step tolerance excludes high-slope pixels from the fit only when enough "
            "pixels remain to solve the requested polynomial.",
        ),
        cautions=(
            "A polynomial fit extrapolates outside the fit region. If the fit ROI is "
            "small or not representative, the subtraction can create artificial "
            "curvature.",
        ),
    ),
    _DefinitionEntry(
        title="STM background subtraction",
        params=(
            "fit_region",
            "line_statistic",
            "model",
            "linear_x_first",
            "blur_length",
            "jump_threshold",
            "preserve_level",
        ),
        summary=(
            "Builds a one-dimensional background profile along the slow-scan "
            "direction, fits or smooths that profile, then subtracts the "
            "resulting line background from the full image."
        ),
        equations=(
            "optional x prefit per row:\n"
            "  L_i(x_j) = a_i*x_j + b_i\n"
            "  w_i,j = z_i,j - L_i(x_j)\n\n"
            "scan-line profile from selected fit pixels M:\n"
            "  p_i = median_j(w_i,j | M_i,j)  or  mean_j(w_i,j | M_i,j)\n\n"
            "model examples, y_i in [-1, 1]:\n"
            "  linear:       B_i = a + b*y_i\n"
            "  poly2/poly3:  B_i = polynomial in y_i\n"
            "  low_pass:     B_i = GaussianSmooth1D(p_i, blur_length)\n"
            "  line_by_line: B_i = interpolated p_i\n\n"
            "nonlinear models fitted with constrained curve_fit:\n"
            "  piezo_creep:    B_i = a + b*y_i + c*log(|y_i - d| + eps)\n"
            "  piezo_creep_x2: B_i = a + b*y_i + c*log(|y_i - d| + eps) + e*y_i^2\n"
            "  piezo_creep_x3: B_i = a + b*y_i + c*log(|y_i - d| + eps) + e*y_i^3\n"
            "  sqrt_creep:     B_i = a + b*y_i + c*sqrt(|y_i - d|)\n\n"
            "eps is a small numerical floor for logarithm evaluation.\n"
            "background_i,j = B_i + L_i(x_j) when x prefit is enabled\n"
            "z'_i,j = z_i,j - background_i,j + reference(background)",
        ),
        details=(
            "Active ROI limits where the line profile is estimated, but subtraction is "
            "still applied to the full image.",
            "The optional jump threshold removes large discontinuities in the profile "
            "before fitting. Preserve level adds back the median or mean of the fitted "
            "background so the corrected image keeps a familiar height reference.",
            "For logarithmic creep models, d is constrained before the scan start so "
            "the log singularity stays off the measured image; eps is only numerical "
            "protection. The sqrt model uses the same off-scan anchor convention to "
            "keep its derivative cusp outside the measured rows.",
        ),
        cautions=(
            "Line-by-line mode is aggressive because each row gets its own background "
            "value. It can suppress real slow-scan variation if the selected fit pixels "
            "include the signal of interest.",
        ),
    ),
    _DefinitionEntry(
        title="Gaussian smoothing",
        params=("sigma_px",),
        summary=(
            "Applies an isotropic two-dimensional Gaussian blur while handling "
            "NaN pixels by weighted normalisation."
        ),
        equations=(
            "G_sigma(x,y) = exp(-(x^2 + y^2) / (2*sigma_px^2))\n"
            "M_i,j = 1 for finite pixels, 0 otherwise\n"
            "z'_i,j = (G_sigma * (M*z))_i,j / (G_sigma * M)_i,j",
        ),
        details=(
            "The normalisation prevents missing pixels from bleeding NaNs into "
            "neighbouring valid pixels. Original NaN positions are restored after "
            "smoothing.",
        ),
        cautions=(
            "Large sigma values remove atomic corrugation and other fine spatial detail.",
        ),
    ),
    _DefinitionEntry(
        title="Gaussian high-pass",
        params=("sigma_px",),
        summary=(
            "Estimates broad structure with a Gaussian blur and subtracts it, "
            "leaving high-spatial-frequency contrast."
        ),
        equations=(
            "background = GaussianSmooth(z, sigma_px)\n"
            "z' = z - background",
        ),
        details=(
            "NaNs are handled with the same weighted Gaussian normalisation as Gaussian "
            "smoothing, then restored in the output.",
        ),
        cautions=(
            "High-pass filtering removes real long-wavelength topography as well as "
            "unwanted background.",
        ),
    ),
    _DefinitionEntry(
        title="FFT soft-border filtering",
        params=("mode = low_pass | high_pass", "cutoff", "border_frac"),
        summary=(
            "Applies a radial FFT filter after tapering image edges toward the "
            "finite-pixel mean to reduce wrap-around ringing."
        ),
        equations=(
            "m = mean_finite(z)\n"
            "w(x,y) = Tukey border window, controlled by border_frac\n"
            "F(kx,ky) = fftshift(fft2((z_fill - m) * w))\n"
            "R = sqrt((kx/kx_Nyquist)^2 + (ky/ky_Nyquist)^2)\n"
            "H_low  = 1 where R <= cutoff, else 0\n"
            "H_high = 1 where R >= cutoff, else 0\n"
            "z' = real(ifft2(ifftshift(H*F))) / safe(w) + m",
        ),
        details=(
            "This is a full forward-filter-inverse operation. The soft border is "
            "compensated after the inverse transform so the image interior stays on the "
            "original level.",
            "This compensation is the key difference from the simpler Fourier radial "
            "filter, where the Hann or Hamming window is not divided out after inverse "
            "transform.",
        ),
        cautions=(
            "Very small cutoffs or large border fractions can over-smooth the image and "
            "amplify edge compensation artifacts.",
        ),
    ),
    _DefinitionEntry(
        title="Fourier radial filtering",
        params=(
            "mode = low_pass | high_pass",
            "cutoff",
            "window = hanning | hamming | none",
        ),
        summary=(
            "Applies a global circular cutoff in the centred 2-D Fourier "
            "domain, optionally after a Hann or Hamming window."
        ),
        equations=(
            "m = mean_finite(z)\n"
            "F(kx,ky) = fftshift(fft2((z_fill - m) * window))\n"
            "R = sqrt((kx/kx_Nyquist)^2 + (ky/ky_Nyquist)^2)\n"
            "H_low  = 1 where R <= cutoff, else 0\n"
            "H_high = 1 where R >= cutoff, else 0\n"
            "z' = real(ifft2(ifftshift(H*F))) + m",
        ),
        details=(
            "The finite-pixel mean is subtracted before the FFT in both modes, then "
            "added back after the inverse FFT so the output stays on the original "
            "height reference. Low-pass keeps broad structure; high-pass keeps fine "
            "spatial detail.",
        ),
        cautions=(
            "Hard circular cutoffs can create ringing, especially when no window is used.",
            "Unlike FFT soft-border filtering, the Hann or Hamming taper is not "
            "compensated after the inverse transform, so border amplitudes can be "
            "attenuated.",
        ),
    ),
    _DefinitionEntry(
        title="Periodic notch filtering",
        params=("peaks = (dx, dy)", "radius_px"),
        summary=(
            "Suppresses selected periodic FFT peaks and their Hermitian "
            "conjugates with Gaussian notches, then transforms back to real "
            "space."
        ),
        equations=(
            "m = mean_finite(z)\n"
            "F = fftshift(fft2(z_fill - m))\n"
            "for each selected peak p and conjugate -p:\n"
            "  N_p(k) = 1 - exp(-|k - p|^2 / (2*radius_px^2))\n"
            "N(k) = product_p N_p(k)\n"
            "z' = real(ifft2(ifftshift(N*F))) + m",
        ),
        details=(
            "Peaks are pixel offsets from the centred FFT origin. Suppressing both a "
            "peak and its conjugate keeps the inverse transform real-valued.",
        ),
        cautions=(
            "Notching lattice peaks can make defects easier to see, but it can also "
            "remove real periodic structure that matters for lattice analysis.",
        ),
    ),
    _DefinitionEntry(
        title="Edge detection",
        params=("method = laplacian | log | dog", "sigma", "sigma2"),
        summary=(
            "Returns a Laplacian-family filter response for edge and feature "
            "enhancement. Positive and negative values mark opposite contrast "
            "directions."
        ),
        equations=(
            "laplacian: z' = Laplacian(z)\n"
            "LoG:       z' = Laplacian(G_sigma * z)\n"
            "DoG:       z' = (G_sigma * z) - (G_sigma2 * z), sigma2 >= sigma + 0.1",
        ),
        details=(
            "NaN and inf pixels are filled with the finite mean for the derivative "
            "calculation, then restored to NaN in the output.",
            "With the implemented DoG sign, bright compact features are positive when "
            "sigma2 is broader than sigma.",
        ),
        cautions=(
            "Second-derivative filters are noise-sensitive. Use LoG or DoG when raw "
            "Laplacian contrast is too harsh.",
        ),
    ),
    _DefinitionEntry(
        title="Manual zero reference",
        params=("set_zero_point", "set_zero_plane_points", "patch"),
        summary=(
            "Subtracts a user-picked height reference from the whole image. A "
            "single point sets Z=0 at one local patch; three points define a "
            "manual zero plane."
        ),
        equations=(
            "point reference:\n"
            "  ref = mean_finite(z[y-p:y+p+1, x-p:x+p+1])\n"
            "  z' = z - ref\n\n"
            "three-point plane reference:\n"
            "  sample heights h_k from finite patches around clicked points (x_k, y_k)\n"
            "  fit z_ref(x, y) = a*x + b*y + c through the three samples\n"
            "  z'(x, y) = z(x, y) - z_ref(x, y)",
        ),
        details=(
            "The set-zero plane tool stores three clicked pixel positions and samples "
            "small finite-valued patches around them. It is a manual reference "
            "operation, separate from automatic background fitting.",
            "The correction is applied to the whole image; zero markers can be hidden "
            "without changing the processing state.",
        ),
        cautions=(
            "A zero plane is only as good as the picked reference points. Picking "
            "points on adsorbates, crashes, or steps can tilt the whole image.",
        ),
    ),
    _DefinitionEntry(
        title="Image arithmetic",
        params=(
            "operation = add | subtract | multiply | divide",
            "operand = constant | image | generated",
            "ROI scope",
        ),
        summary=(
            "Applies numeric arithmetic to the image, using either a scalar, another "
            "same-shaped image plane, or a deterministic generated pattern."
        ),
        equations=(
            "constant operand:\n"
            "  add/subtract: z' = z +/- value_si\n"
            "  multiply:     z' = z * factor\n"
            "  divide:       z' = z / factor, factor != 0\n\n"
            "image or generated operand o:\n"
            "  z' = z + o  or  z' = z - o\n\n"
            "generated operands include checkerboard, ramp_x, ramp_y,\n"
            "speckle, and impulse_grid patterns.",
        ),
        details=(
            "Image operands must match the current image shape. Generated operands are "
            "created in the current image shape and recorded through their pattern "
            "parameters.",
            "When image arithmetic is launched with active-area ROI scope, the full "
            "operation is computed and only pixels inside the ROI mask are copied "
            "back into the result.",
        ),
        cautions=(
            "Arithmetic changes physical data values directly. Use display range or "
            "colormap controls when only visual contrast should change.",
        ),
    ),
    _DefinitionEntry(
        title="Thresholding and bit-depth conversion",
        params=("lower", "upper", "mode = clip | binarize", "bits = 8 | 16"),
        summary=(
            "Applies value thresholds or quantizes finite image values to a smaller "
            "number of intensity levels while preserving NaN pixels."
        ),
        equations=(
            "threshold clip:\n"
            "  z'_i,j = NaN when z_i,j < lower or z_i,j > upper\n"
            "  z'_i,j = z_i,j otherwise\n\n"
            "threshold binarize:\n"
            "  z'_i,j = 1 when finite(z_i,j) and lower <= z_i,j <= upper\n"
            "  z'_i,j = 0 when finite(z_i,j) and outside the band\n"
            "  z'_i,j = NaN when z_i,j is non-finite\n\n"
            "quantize:\n"
            "  q = round((clip(z, vmin, vmax) - vmin) * (2^bits - 1)/(vmax - vmin))\n"
            "  z' = q * (vmax - vmin)/(2^bits - 1) + vmin",
        ),
        details=(
            "Threshold values are in the image's physical data units. Binarize mode "
            "returns numeric 0/1 values rather than a display-only mask.",
            "Bit-depth conversion still returns float64 data, but finite values are "
            "restricted to the selected number of levels. If vmin/vmax are not "
            "provided, a robust percentile band is used.",
        ),
        cautions=(
            "Thresholding can discard pixels by turning them into NaN. Quantization is "
            "not reversible and can hide subtle height variation.",
        ),
    ),
    _DefinitionEntry(
        title="Geometric transforms and resampling",
        params=(
            "flip",
            "rotate 90/180/270",
            "rotate arbitrary",
            "shear",
            "scale image",
        ),
        summary=(
            "Changes image geometry by exact array transforms or interpolation-based "
            "resampling, updating scan extent rules where the canvas size changes."
        ),
        equations=(
            "lossless transforms:\n"
            "  flip_horizontal, flip_vertical, rotate_90_cw,\n"
            "  rotate_180, rotate_270_cw map pixels exactly\n\n"
            "arbitrary rotation:\n"
            "  z' = scipy.ndimage.rotate(z, angle, reshape=True, order=order)\n"
            "  out-of-input pixels and invalid interpolated regions become NaN\n\n"
            "shear:\n"
            "  [x'; y'] = [[1, shear_x], [shear_y, 1]] * [x; y]\n"
            "  sampled by inverse affine interpolation with expanded canvas\n\n"
            "scale:\n"
            "  z' = zoom(z, (new_height/Ny, new_width/Nx), order=order)",
        ),
        details=(
            "Right-angle rotations and flips transform existing ROI geometry exactly. "
            "Scale preserves physical scan extent while changing pixel density.",
            "Arbitrary rotation, shear, and expanded affine corrections preserve pixel "
            "size and grow the physical displayed extent with the output shape.",
        ),
        cautions=(
            "Arbitrary rotation invalidates existing ROI geometry; invalidated ROIs "
            "are removed from the displayed ROI set. Interpolated transforms can "
            "change local pixel values.",
        ),
    ),
    _DefinitionEntry(
        title="FFT-derived correction tools",
        params=(
            "mains pickup",
            "inverse FFT selections",
            "affine lattice correction",
        ),
        summary=(
            "Applies data-changing corrections from FFT-domain selections or fitted "
            "lattice geometry."
        ),
        equations=(
            "mains pickup suppression:\n"
            "  predict harmonics from scan_speed, scan_range, mains_frequency\n"
            "  optionally snap each predicted peak or streak to bright FFT power\n"
            "  apply symmetric Gaussian notches and inverse transform\n\n"
            "inverse FFT selection filter:\n"
            "  build an ellipse mask M(k) from selected FFT regions\n"
            "  mode remove_selected: keep (1 - M) * F(k)\n"
            "  mode keep_selected:   keep M(k) * F(k)\n"
            "  z' = real(ifft2(filtered F))\n\n"
            "affine lattice correction:\n"
            "  measured centred pixel coordinate u maps to ideal coordinate v = A*u\n"
            "  output pixels inverse-map through A^-1 and sample z by interpolation",
        ),
        details=(
            "Mains pickup can remove spot-like harmonics or fast-axis streaks. Inverse "
            "FFT selections can create either a corrected image or a new result image, "
            "depending on the dialog action.",
            "Affine lattice correction is usually produced by FFT or real-space "
            "lattice fitting. It may expand the canvas and fill outside-input regions "
            "with NaN or a configured background value.",
        ),
        cautions=(
            "FFT-domain corrections can remove real periodic signal as well as "
            "artifacts. Inspect previews before applying them to quantitative data.",
        ),
    ),
    _DefinitionEntry(
        title="Linear undistortion",
        params=("shear_x", "scale_y"),
        summary=(
            "Applies an affine drift or creep correction by inverse-mapping each "
            "output pixel to a bilinearly interpolated input location."
        ),
        equations=(
            "for output pixel (y, x):\n"
            "  src_y = y / scale_y\n"
            "  src_x = x - shear_x * y / max(Ny - 1, 1)\n"
            "  z'(y, x) = bilinear_sample(z, src_y, src_x)\n\n"
            "shear_x is the total horizontal shift in pixels accumulated from\n"
            "y=0 to y=Ny-1; scale_y is a dimensionless vertical scale ratio.",
        ),
        details=(
            "Positive shear_x means the accumulated horizontal correction grows across "
            "the slow-scan direction. scale_y corrects a y/x pixel-size mismatch.",
        ),
        cautions=(
            "Interpolation changes local pixel values. The operation is appropriate for "
            "geometric distortion, not for background levelling.",
        ),
    ),
    _DefinitionEntry(
        title="Forward/backward scan blending",
        params=("weight",),
        summary=(
            "Blends a forward scan plane with a left-right mirrored backward "
            "scan plane so both directions align to the same physical pixels."
        ),
        equations=(
            "b_mirror[i,j] = bwd[i, Nx - 1 - j]\n"
            "z'_i,j = weight * fwd_i,j + (1 - weight) * b_mirror_i,j\n"
            "if one side is non-finite and the other is finite, use the finite side",
        ),
        details=(
            "A weight of 0.5 is a symmetric average. Higher weights favour the forward scan.",
        ),
        cautions=(
            "The two planes must already represent the same image shape and scan area.",
        ),
    ),
)


_ROI_REFERENCE_ENTRIES: tuple[_DefinitionEntry, ...] = (
    _DefinitionEntry(
        title="ROI model and selection state",
        params=("pixel coordinates", "ROISet", "active ROI", "dock selection"),
        summary=(
            "ROIs are per-image objects stored in pixel coordinates and persisted in "
            "a sidecar next to the scan. Many viewer actions use the dock selection "
            "first, then fall back to the active ROI."
        ),
        equations=(
            "coordinate convention:\n"
            "  x = image column, y = image row\n"
            "  origin = top-left pixel, +x right, +y down\n\n"
            "ROI kinds:\n"
            "  area = rectangle | ellipse | polygon | freehand | multipolygon\n"
            "  line = two endpoints plus optional averaging width\n"
            "  point = one fixed pixel coordinate\n\n"
            "action context:\n"
            "  selected ROI(s) in ROI Manager dock win when present\n"
            "  otherwise use ROISet.active_roi_id\n"
            "  sidecar path = <scan stem>.rois.json",
        ),
        details=(
            "The ROI Manager can select multiple ROIs for operations such as combine "
            "or step height. Canvas click selection sets the active ROI.",
            "Measurements record ROI identity and name at creation time, so results "
            "can remain meaningful even if the ROI is later renamed or deleted.",
        ),
        cautions=(
            "If processing history references a ROI that no longer exists, interactive "
            "display warns or pauses and export can abort rather than silently using "
            "the wrong mask.",
        ),
    ),
    _DefinitionEntry(
        title="Drawing and pan tools",
        params=("pan", "rectangle", "ellipse", "polygon", "freehand", "line", "point"),
        summary=(
            "ROI drawing tools create one ROI and then return to pan mode. Pan mode "
            "handles navigation, hover hints, ROI selection, and active-ROI dragging."
        ),
        equations=(
            "pan mode:\n"
            "  drag blank image = pan scroll area\n"
            "  middle mouse = pan from any tool context\n"
            "  Ctrl+scroll = zoom\n"
            "  click ROI = select it; click active ROI = prepare to move or resize\n\n"
            "drawing completion:\n"
            "  rectangle/ellipse/line: drag, release to finish\n"
            "  point: click once to place\n"
            "  polygon: click vertices, Enter or double-click to close\n"
            "  freehand: drag path, release to finish\n"
            "  Escape: cancel active drawing preview and return to pan",
        ),
        details=(
            "Rectangle and ellipse require a minimum non-zero drawn size. Polygon and "
            "freehand ROIs require at least three points; incomplete shapes are "
            "discarded cleanly.",
            "The status bar and ROI item tooltips describe the current action. Hover "
            "highlight in pan mode shows which ROI a click will select.",
        ),
        cautions=(
            "Drawing mode captures clicks for drawing. Return to pan before trying to "
            "move or right-click existing ROIs.",
        ),
    ),
    _DefinitionEntry(
        title="Editing existing ROIs",
        params=("rename", "delete", "copy/paste", "duplicate", "move", "resize"),
        summary=(
            "Editing is explicit: only the active ROI can be dragged or resized on "
            "the canvas, while the ROI Manager and context menus expose object "
            "actions."
        ),
        equations=(
            "selection and movement:\n"
            "  click non-active ROI -> active_roi_id = roi.id\n"
            "  drag active ROI -> translate geometry by rounded (dx, dy)\n"
            "  release drag -> persist one geometry update to sidecar\n\n"
            "resize handles:\n"
            "  rectangle: nw ne se sw n e s w\n"
            "  ellipse: n e s w\n"
            "  line: p1 p2\n"
            "  Shift while dragging rectangle/ellipse handles preserves aspect ratio\n\n"
            "copy/duplicate:\n"
            "  copy keeps the ROI in memory\n"
            "  paste or duplicate creates a new ROI shifted by 10 px",
        ),
        details=(
            "Line ROI width is stored in the line geometry and controls the "
            "perpendicular averaging swath used by line-profile calculations.",
            "Delete/Backspace removes the active ROI from the canvas; the ROI Manager "
            "can delete one or more selected ROIs.",
        ),
        cautions=(
            "Polygon, freehand, point, and multipolygon ROIs currently do not have "
            "resize handles. They can still be moved, renamed, copied, or deleted.",
        ),
    ),
    _DefinitionEntry(
        title="Area ROI actions",
        params=("mask/filter scope", "invert", "combine", "FFT", "histogram", "measure"),
        summary=(
            "Area ROIs provide masks for local filters, region analysis, geometric "
            "ROI algebra, and measurement workflows."
        ),
        equations=(
            "area mask kinds:\n"
            "  rectangle, ellipse, polygon, freehand, multipolygon -> boolean mask\n"
            "  line and point are rejected for area-only actions\n\n"
            "ROI filters only:\n"
            "  eligible local filters = smooth, high-pass, edge, Fourier filter,\n"
            "                           FFT soft-border, arithmetic\n"
            "  processed_full = operation(z)\n"
            "  z'_inside_roi = processed_full_inside_roi\n"
            "  z'_outside_roi = z_outside_roi\n\n"
            "geometry algebra:\n"
            "  invert = image bounds minus ROI\n"
            "  combine modes = union | intersection | difference | xor",
        ),
        details=(
            "Area ROI context-menu actions include setting filter scope, invert, STM "
            "background fit from ROI, histogram, FFT this region, ROI statistics, and "
            "feature maxima detection.",
            "Step height requires exactly two selected area ROIs and records the "
            "mean-height difference between them.",
        ),
        cautions=(
            "ROI filter scope affects only eligible local filters. Background and "
            "scan-line corrections remain whole-image unless their own dialog has a "
            "specific ROI fit option.",
        ),
    ),
    _DefinitionEntry(
        title="Line ROI actions",
        params=("profile", "distance", "periodicity", "width", "endpoints"),
        summary=(
            "Line ROIs drive profile, distance, and periodicity workflows. The active "
            "line ROI also keeps the live line-profile panel in sync."
        ),
        equations=(
            "line geometry:\n"
            "  p1 = (x1, y1), p2 = (x2, y2)\n"
            "  width_px = max(1, stored width)\n\n"
            "profile sampling:\n"
            "  s runs from 0 to physical distance between p1 and p2\n"
            "  z(s) samples along the line\n"
            "  width_px > 1 averages finite pixels in a perpendicular swath\n\n"
            "line actions:\n"
            "  drag active line = translate both endpoints\n"
            "  drag p1/p2 handle = move one endpoint\n"
            "  estimate periodicity = analyse current line-profile signal",
        ),
        details=(
            "Line context menus expose show profile, add profile measurement, estimate "
            "periodicity, and set line width. The Measurements panel can add profile "
            "summaries and profile deltas.",
            "Distance/ruler measurement uses the active line ROI and reports physical "
            "length using the scan calibration.",
        ),
        cautions=(
            "A selected line ROI is not valid for area-only actions such as ROI "
            "statistics, histogram, FFT region, or filter masking.",
        ),
    ),
    _DefinitionEntry(
        title="Point ROI actions",
        params=("point marker", "copy coordinates", "point sources"),
        summary=(
            "Point ROIs mark individual pixel locations. They can be copied as "
            "coordinates and used as point sources by downstream feature-analysis "
            "tools."
        ),
        equations=(
            "point geometry:\n"
            "  point = (x, y) in pixel coordinates\n"
            "  copy action writes 'x, y' to the clipboard\n\n"
            "point-source collection:\n"
            "  selected point ROIs become one labelled point source\n"
            "  all point ROIs can also be offered as a point source\n"
            "  physical coordinates = (x * pixel_size_x_m, y * pixel_size_y_m)",
        ),
        details=(
            "pair-correlation, feature-to-lattice, and point-mask/FFT workflows can "
            "consume point sources from feature maxima or point ROIs.",
            "Point ROIs are fixed-screen-size markers on the canvas so they stay "
            "visible while zooming.",
        ),
        cautions=(
            "Point ROIs do not define an area mask. They are intentionally disabled "
            "for ROI filters, STM background fit masks, histogram, and ROI statistics.",
        ),
    ),
    _DefinitionEntry(
        title="Tool interactions and persistence",
        params=("sidecar save", "processing references", "geometric transforms"),
        summary=(
            "ROI edits save immediately to the per-image sidecar, and downstream "
            "tools resolve ROI references at the moment they run."
        ),
        equations=(
            "persistence:\n"
            "  ROI created / moved / resized / renamed / deleted\n"
            "    -> ROISet updated\n"
            "    -> canvas and dock refreshed\n"
            "    -> <scan stem>.rois.json saved\n\n"
            "transform rules:\n"
            "  flips and right-angle rotations transform ROI geometry exactly\n"
            "  crop shifts/clips ROIs and drops ROIs outside the crop\n"
            "  rotate_arbitrary invalidates existing ROIs\n\n"
            "STM background from ROI:\n"
            "  area ROI mask limits profile fitting pixels\n"
            "  correction is applied to the full image",
        ),
        details=(
            "The quick toolbar Mask action sets the Processing panel to 'ROI filters "
            "only' for the selected-or-active area ROI. Invert creates a new inverted "
            "area ROI and sets ROI filter scope to the inverted area when invoked from "
            "the active ROI path.",
            "The quick Simple background button remains a whole-image first-order "
            "plane subtraction. Use dedicated background controls when fit/apply/exclude "
            "ROI behaviour is required.",
        ),
        cautions=(
            "Changing ROI geometry after a processing step was configured can change "
            "future replays of that processing if the step references the ROI by ID.",
        ),
    ),
)


def _hex_to_rgb(value: object, default: str) -> tuple[int, int, int]:
    text = str(value or default).strip()
    if not text.startswith("#"):
        text = default
    text = text.lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        text = default.lstrip("#")
    try:
        return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return tuple(int(default.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))


def _is_light_color(value: object) -> bool:
    r, g, b = _hex_to_rgb(value, "#ffffff")
    channels = [c / 255.0 for c in (r, g, b)]
    linear = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        for c in channels
    ]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return luminance >= 0.55


def _definitions_palette(theme: Mapping[str, object] | None = None) -> dict[str, str]:
    theme = theme or {}
    bg = str(theme.get("bg") or "#ffffff")
    is_light = _is_light_color(bg)
    if is_light:
        return {
            "bg": bg,
            "text": "#111827",
            "muted": "#4b5563",
            "title": "#5b21b6",
            "heading": "#0f5ea8",
            "param": "#166534",
            "keyword": "#b42318",
            "note": "#92400e",
            "border": "#cbd5e1",
            "equation_bg": "#eef2f7",
            "equation_fg": "#111827",
        }
    return {
        "bg": bg,
        "text": str(theme.get("fg") or "#e5e7eb"),
        "muted": str(theme.get("sub_fg") or "#a6adc8"),
        "title": "#d8b4fe",
        "heading": str(theme.get("accent_bg") or "#89b4fa"),
        "param": str(theme.get("ok_fg") or "#a6e3a1"),
        "keyword": str(theme.get("err_fg") or "#f38ba8"),
        "note": str(theme.get("warn_fg") or "#fab387"),
        "border": str(theme.get("sep") or "#45475a"),
        "equation_bg": str(theme.get("entry_bg") or "#111827"),
        "equation_fg": str(theme.get("fg") or "#f9fafb"),
    }


def _entry_id(title: str) -> str:
    chars: list[str] = []
    prev_dash = False
    for ch in title.lower():
        if ch.isalnum():
            chars.append(ch)
            prev_dash = False
        elif not prev_dash:
            chars.append("-")
            prev_dash = True
    return "".join(chars).strip("-")


def _render_params(params: tuple[str, ...]) -> str:
    if not params:
        return ""
    rendered = ", ".join(
        f'<span class="param">{escape(param)}</span>' for param in params
    )
    return f'<p class="sub">Params: {rendered}</p>'


def _render_entry(entry: _DefinitionEntry, *, block_label: str = "Operation") -> str:
    blocks = [f'<div class="entry" id="{_entry_id(entry.title)}">']
    blocks.append(f"<h2>{escape(entry.title)}</h2>")
    blocks.append(_render_params(entry.params))
    blocks.append(f"<p>{escape(entry.summary)}</p>")
    for equation in entry.equations:
        blocks.append(f'<p class="label">{escape(block_label)}</p>')
        blocks.append(f'<pre class="equation">{escape(equation)}</pre>')
    for detail in entry.details:
        blocks.append(f"<p>{escape(detail)}</p>")
    for caution in entry.cautions:
        blocks.append(f'<p class="note">{escape(caution)}</p>')
    blocks.append("</div>")
    return "\n".join(blocks)


def _render_reference_html(
    *,
    title: str,
    intro: str,
    entries: tuple[_DefinitionEntry, ...],
    theme: Mapping[str, object] | None = None,
    block_label: str = "Operation",
) -> str:
    p = _definitions_palette(theme)
    rendered_entries = "\n<hr/>\n".join(
        _render_entry(entry, block_label=block_label) for entry in entries
    )
    return f"""
<style>
  body {{
      font-family: Helvetica, Arial, sans-serif;
      font-size: 13px;
      color: {p["text"]};
      background: {p["bg"]};
      margin: 0;
      padding: 0;
  }}
  h1 {{
      font-size: 20px;
      color: {p["title"]};
      margin: 0 0 8px 0;
      font-weight: 700;
  }}
  h2 {{
      font-size: 16px;
      color: {p["heading"]};
      margin: 18px 0 4px 0;
      font-weight: 700;
  }}
  p {{
      margin: 4px 0 8px 0;
      line-height: 1.45;
  }}
  .intro {{
      color: {p["muted"]};
  }}
  .sub {{
      color: {p["muted"]};
      font-style: italic;
      margin: 0 0 6px 0;
  }}
  .param {{
      color: {p["param"]};
      font-family: Menlo, Consolas, monospace;
      font-style: normal;
  }}
  .label {{
      color: {p["keyword"]};
      font-weight: 700;
      margin-top: 8px;
      margin-bottom: 3px;
  }}
  .equation {{
      color: {p["equation_fg"]};
      background-color: {p["equation_bg"]};
      border: 1px solid {p["border"]};
      padding: 8px;
      margin: 4px 0 9px 0;
      font-family: Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.35;
      white-space: pre-wrap;
  }}
  .note {{
      color: {p["note"]};
  }}
  hr {{
      border: none;
      border-top: 1px solid {p["border"]};
      margin: 16px 0;
  }}
</style>
<body>
<h1>{escape(title)}</h1>
<p class="intro">{escape(intro)}</p>
<hr/>
{rendered_entries}
</body>
"""


def render_definitions_html(theme: Mapping[str, object] | None = None) -> str:
    """Return theme-aware HTML for the processing definitions dialog."""
    return _render_reference_html(
        title="Processing Algorithm Reference",
        intro=(
            "Each step transforms the raw height data in physical units. The "
            "equations below describe the operation applied to finite float64 "
            "image data before display scaling or colour-map clipping."
        ),
        entries=_DEFINITION_ENTRIES,
        theme=theme,
    )


def render_roi_reference_html(theme: Mapping[str, object] | None = None) -> str:
    """Return theme-aware HTML for ROI actions and tool interactions."""
    return _render_reference_html(
        title="ROI Actions Reference",
        intro=(
            "ROI actions depend on selection state, ROI kind, and the active "
            "viewer tool. The behaviour blocks below describe when each action is "
            "available, which ROI context it uses, and how it affects downstream "
            "processing or measurement tools."
        ),
        entries=_ROI_REFERENCE_ENTRIES,
        theme=theme,
        block_label="Behaviour",
    )


_DEFINITIONS_HTML = render_definitions_html()
_ROI_REFERENCE_HTML = render_roi_reference_html()


class _HtmlReferencePanel(QWidget):
    """Scrollable HTML reference panel."""

    def __init__(self, t: dict, html: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        palette = _definitions_palette(t)
        inner = QTextEdit()
        inner.setReadOnly(True)
        inner.setHtml(html)
        inner.setStyleSheet(f"""
            QTextEdit {{
                background-color: {palette["bg"]};
                color: {palette["text"]};
                border: none;
                padding: 16px;
            }}
        """)
        inner.viewport().setStyleSheet(f"background-color: {palette['bg']};")

        scroll.setWidget(inner)
        lay.addWidget(scroll)
        self._text_edit = inner


class _DefinitionsPanel(_HtmlReferencePanel):
    """Scrollable reference panel listing processing algorithms."""

    def __init__(self, t: dict, parent=None):
        super().__init__(t, render_definitions_html(t), parent)


class _ROIReferencePanel(_HtmlReferencePanel):
    """Scrollable reference panel listing ROI actions and interactions."""

    def __init__(self, t: dict, parent=None):
        super().__init__(t, render_roi_reference_html(t), parent)


class _DefinitionsDialog(QDialog):
    """Closeable utility window for processing and ROI definitions/help."""

    def __init__(self, t: dict, parent=None, *, initial_tab: str = "processing"):
        super().__init__(parent)
        self.setWindowTitle("ProbeFlow Definitions")
        self.resize(820, 680)
        self.setModal(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget(self)
        self._panel = _DefinitionsPanel(t, self)
        self._roi_panel = _ROIReferencePanel(t, self)
        self._tabs.addTab(self._panel, "Processing")
        self._tabs.addTab(self._roi_panel, "ROI Actions")
        lay.addWidget(self._tabs)
        self.set_reference_tab(initial_tab)

    def set_reference_tab(self, tab: str) -> None:
        """Switch to the named reference tab."""
        key = str(tab or "processing").lower().replace("-", "_")
        index = 1 if key in {"roi", "roi_actions", "roi_reference"} else 0
        self._tabs.setCurrentIndex(index)

    def current_reference_tab(self) -> str:
        """Return the stable key for the currently selected reference tab."""
        return "roi" if self._tabs.currentIndex() == 1 else "processing"
