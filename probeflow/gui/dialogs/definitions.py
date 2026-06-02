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
            "An STM image is built one row at a time as the tip sweeps back and "
            "forth. Sometimes the tip glitches part-way along a row and leaves a "
            "short bright or dark streak that is not real surface. This tool finds "
            "those short damaged stretches by comparing each row with the rows just "
            "above and below it, then repairs only the bad stretch by copying from "
            "its healthy neighbours. Everything else in the image is left exactly "
            "as measured."
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
            "Two modes find the bad stretches. 'step' looks for a sudden jump up "
            "and a matching jump back down, which is what a glitch streak looks "
            "like. 'mad' simply flags pixels that sit far from the typical value "
            "for that row. 'Polarity' tells it whether you are chasing bright "
            "(too-high) or dark (too-low) streaks. The threshold is measured in "
            "robust noise units (MAD), so a value of 3 means 'about three times "
            "the usual row noise'.",
            "Minimum segment length ignores single-pixel speckle, so only genuine "
            "stretches are touched. Maximum adjacent bad lines is a safety brake: "
            "if a whole block of neighbouring rows is damaged there are no healthy "
            "neighbours to copy from, so the tool leaves that block alone rather "
            "than guessing.",
            "Use 'Preview detection' first — it highlights what would be repaired "
            "without changing the data. Only 'Apply' edits the image, and even "
            "then only the accepted stretches change.",
        ),
        cautions=(
            "This fixes local streaks, not whole-row level differences. Do not use "
            "it to flatten real terrace steps or step edges — those are genuine "
            "surface features, and row levelling (below) is the right tool if you "
            "want to remove row-to-row offsets.",
        ),
    ),
    _DefinitionEntry(
        title="Row alignment",
        params=("method = median | mean | linear",),
        summary=(
            "Because each scan line is recorded separately, neighbouring rows "
            "often sit at slightly different heights, giving the image a streaky, "
            "venetian-blind look. Row alignment removes that by shifting every row "
            "to a common level — subtracting each row's own typical height. The "
            "'linear' option also removes a gentle tilt within each row. This is "
            "usually the first cleanup step, done before background subtraction or "
            "filtering."
        ),
        equations=(
            "median: z'_i,j = z_i,j - median_j(z_i,j)\n"
            "mean:   z'_i,j = z_i,j - mean_j(z_i,j)\n"
            "linear: fit a_i*x_j + b_i to finite pixels in row i, x_j in [-1, 1]\n"
            "        z'_i,j = z_i,j - (a_i*x_j + b_i)",
        ),
        details=(
            "'Median' is the safe default: it asks 'what is the middle height in "
            "this row?', and a few unusual pixels (an adsorbed molecule, a tip "
            "crash, a bright defect) cannot drag that answer around. 'Mean' "
            "averages every pixel, so it is only suitable when rows are clean. "
            "'Linear' fits and removes a straight slope across each row, which "
            "helps when the rows are also tilted, not just offset.",
        ),
        cautions=(
            "If a row is dominated by a real sloped feature — say a wide terrace "
            "that rises across the whole row — alignment can subtract part of that "
            "genuine signal along with the offset. When in doubt, compare before "
            "and after.",
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
            "Real scans usually sit on a smooth, unwanted backdrop — an overall "
            "tilt from the sample not being perfectly flat, or a gentle bow from "
            "the scanner. This tool models that backdrop as a smooth surface (a "
            "polynomial), fits it to the image, and subtracts it, so the features "
            "you care about are left sitting on a flat, level background. Order 1 "
            "removes a flat tilt (a plane); higher orders remove gentle curvature "
            "or bowing."
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
            "The quick Plane/background button just removes a first-order tilt "
            "(B(x,y) = c00 + c10*x + c01*y) from the whole image, which is the "
            "most common case.",
            "The three ROI options give you fine control. 'Fit ROI' fits the "
            "background using only pixels inside a region — handy when one clean "
            "patch represents the true background. 'Exclude ROI' does the "
            "opposite: it leaves tall features (islands, molecules) out of the fit "
            "so they do not pull the surface upward. 'Apply ROI' limits which "
            "pixels actually get the subtraction. Fit and exclude change what the "
            "model learns from; apply changes what it edits.",
            "'Step tolerance' is for stepped surfaces: it drops steep pixels (step "
            "edges) from the fit so the background follows the flat terraces "
            "instead of bending across the steps — but only while enough flat "
            "pixels remain to solve the fit.",
        ),
        cautions=(
            "A polynomial keeps curving outside the region it was fitted to. If the "
            "fit ROI is small or not typical of the whole image, the subtraction "
            "can invent curvature where there was none. Prefer a fit region that "
            "samples genuinely flat background across the scan.",
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
            "STM images often drift slowly from the top of the scan to the bottom "
            "— the surface appears to ramp or bow in the slow-scan direction (the "
            "direction the tip steps from one row to the next) because of thermal "
            "drift or piezo creep. This tool measures one representative height per "
            "row, builds a smooth profile down the image, and subtracts it from "
            "every row, flattening that top-to-bottom trend while leaving the "
            "side-to-side detail intact. It is the STM-specific complement to the "
            "polynomial background above, which works in both directions at once."
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
            "Choose the 'model' to match the drift you see. 'linear' and "
            "'poly2/poly3' remove a straight or gently curved trend. 'low_pass' "
            "smooths the per-row profile and removes whatever is left — flexible "
            "but gentle. The 'piezo_creep' and 'sqrt_creep' models have the "
            "characteristic curved shape of scanner creep right after a large tip "
            "move, where the drift is fast at first and then settles.",
            "If you draw an area ROI and fit from it, only pixels inside that ROI "
            "are used to measure each row's height — useful for fitting on clean "
            "background while ignoring tall features. The correction is still "
            "applied to the whole image so nothing is left out.",
            "Two helpers refine the fit. 'Jump threshold' ignores large sudden "
            "steps in the profile (for example a terrace edge) so they do not bend "
            "the fitted trend. 'Preserve level' adds the background's average "
            "height back afterwards, so the corrected image keeps a sensible "
            "absolute height instead of hovering around zero.",
            "Technical note on the creep models: the logarithm and square-root "
            "shapes have a sharp kink at one point, so the fit pins that kink just "
            "before the first scan row, keeping it off the measured image; the eps "
            "term is only there to stop the logarithm blowing up numerically.",
        ),
        cautions=(
            "'line_by_line' gives every row its own independent background value, "
            "which is powerful but blunt: if the features you care about vary down "
            "the slow-scan direction, this mode can flatten them away along with "
            "the drift. Reach for it only when the row-to-row background really is "
            "independent.",
        ),
    ),
    _DefinitionEntry(
        title="Gaussian smoothing",
        params=("sigma_px",),
        summary=(
            "Smooths the image by blending each pixel with its neighbours, "
            "weighting nearby pixels more than distant ones (a Gaussian, or "
            "bell-curve, weighting). This softens random pixel noise so faint, "
            "broad features stand out. 'sigma_px' is the blur radius in pixels — "
            "larger values blur more."
        ),
        equations=(
            "G_sigma(x,y) = exp(-(x^2 + y^2) / (2*sigma_px^2))\n"
            "M_i,j = 1 for finite pixels, 0 otherwise\n"
            "z'_i,j = (G_sigma * (M*z))_i,j / (G_sigma * M)_i,j",
        ),
        details=(
            "Gaps (NaN pixels) are handled carefully: the blur is renormalised so "
            "a missing measurement cannot smear its emptiness into good "
            "neighbouring pixels, and the original gaps are put back afterwards.",
        ),
        cautions=(
            "Smoothing trades detail for a cleaner look. Too large a sigma will "
            "wash out atomic corrugation and other fine structure you may want to "
            "keep — start small and increase only as needed.",
        ),
    ),
    _DefinitionEntry(
        title="Gaussian high-pass",
        params=("sigma_px",),
        summary=(
            "The mirror image of smoothing: it makes a blurred copy of the image "
            "(the broad, slowly-varying part) and subtracts it, leaving only the "
            "fine, fast-changing detail. This is a quick way to flatten an uneven "
            "background and pop out small features like atoms or edges. 'sigma_px' "
            "sets the scale — anything broader than this blur radius is removed."
        ),
        equations=(
            "background = GaussianSmooth(z, sigma_px)\n"
            "z' = z - background",
        ),
        details=(
            "Think of it as 'keep the small stuff, throw away the big stuff'. "
            "Choosing sigma is choosing the boundary between the two. Gaps (NaNs) "
            "are handled the same careful way as Gaussian smoothing.",
        ),
        cautions=(
            "It cannot tell wanted broad topography from unwanted background — both "
            "are 'big stuff' and both get removed. After a high-pass you can no "
            "longer read true heights of large features; use it for visualising "
            "fine detail, not for quantitative step heights.",
        ),
    ),
    _DefinitionEntry(
        title="FFT soft-border filtering",
        params=("mode = low_pass | high_pass", "cutoff", "border_frac"),
        summary=(
            "A Fourier transform (FFT) re-describes the image as a sum of waves: "
            "smooth, broad variations are 'low spatial frequencies' near the "
            "centre, and fine, closely-spaced detail is 'high frequencies' further "
            "out. Filtering in this domain lets you keep or remove features by "
            "their size. Low-pass keeps the broad waves (smooths); high-pass keeps "
            "the fine waves (sharpens). The 'soft border' first fades the image "
            "edges to a common level, because the FFT secretly assumes the image "
            "tiles edge-to-edge, and a hard jump between opposite edges would "
            "create ripples across the result."
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
            "'cutoff' is the size boundary between 'broad' and 'fine', and "
            "'border_frac' is how much of the edge gets faded. The sequence is: "
            "fade the edges, transform to the wave (Fourier) domain, keep or "
            "remove waves on the chosen side of the cutoff, transform back, then "
            "undo the edge fade so the middle of the image stays at its true "
            "height.",
            "Undoing that edge fade is the main difference from the simpler "
            "Fourier radial filter below, which leaves its taper in place. That "
            "makes this version better when edge heights matter.",
        ),
        cautions=(
            "A very small cutoff or a very large border fraction can over-smooth "
            "the image and exaggerate edge artifacts. Change one setting at a time "
            "and watch the preview.",
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
            "The straightforward version of FFT filtering: keep the broad waves "
            "(low-pass, smooths) or the fine waves (high-pass, sharpens) using a "
            "circular cutoff in the Fourier domain. The optional Hann or Hamming "
            "'window' gently fades the image toward its edges before transforming, "
            "which softens ripple artifacts but is not undone afterwards. Choose "
            "this for a quick filter; choose the soft-border version when keeping "
            "edge heights exactly right matters."
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
            "The average height (the mean) is subtracted before the FFT in both "
            "modes and added back after the inverse FFT, so the result keeps its "
            "original height reference rather than drifting to zero. As a memory "
            "aid: low-pass keeps the big, broad structure; high-pass keeps the "
            "small, fine detail.",
        ),
        cautions=(
            "A hard circular cutoff can leave faint ripples (ringing) around sharp "
            "features, especially with no window. And because the Hann/Hamming "
            "taper is left in place here, heights near the image border are "
            "slightly suppressed — use the soft-border filter if that matters.",
        ),
    ),
    _DefinitionEntry(
        title="Periodic notch filtering",
        params=("peaks = (dx, dy)", "radius_px"),
        summary=(
            "A repeating pattern in the image — an atomic lattice, or unwanted "
            "electrical interference — shows up as bright spots in the FFT, one "
            "pair of spots per repeating wave. This tool lets you pick those spots "
            "and dim them, which removes that one periodic pattern from the image "
            "while leaving everything else. It is the precise way to kill a "
            "specific stripe or ripple without blurring the whole picture."
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
            "Each pattern produces two spots, symmetric about the FFT centre, and "
            "the tool always dims both together — that symmetry is what keeps the "
            "filtered image real (an ordinary height map rather than complex "
            "numbers). 'radius_px' sets how wide a notch to cut around each spot: "
            "wider removes the pattern more completely but also touches nearby "
            "detail.",
        ),
        cautions=(
            "Removing the lattice spots can make defects and adsorbates much easier "
            "to see — but you are deleting real periodic signal. Do not notch out "
            "the lattice and then try to measure that same lattice; keep an "
            "unfiltered copy for quantitative work.",
        ),
    ),
    _DefinitionEntry(
        title="Edge detection",
        params=("method = laplacian | log | dog", "sigma", "sigma2"),
        summary=(
            "Highlights where the height changes sharply — the rims of islands, "
            "step edges, and feature outlines — by responding to curvature in the "
            "image rather than to height itself. The result is not a height map: "
            "it is an 'edge map' where flat areas are near zero and edges light "
            "up, with opposite signs on the two sides of an edge. Useful for "
            "seeing shapes and boundaries clearly."
        ),
        equations=(
            "laplacian: z' = Laplacian(z)\n"
            "LoG:       z' = Laplacian(G_sigma * z)\n"
            "DoG:       z' = (G_sigma * z) - (G_sigma2 * z), sigma2 >= sigma + 0.1",
        ),
        details=(
            "The three methods differ in how much they smooth first. Plain "
            "'laplacian' reacts to the rawest detail (and the most noise). 'LoG' "
            "(Laplacian of Gaussian) blurs by 'sigma' before taking the edge "
            "response, so it ignores pixel noise and finds edges at a chosen "
            "scale. 'DoG' (Difference of Gaussians) subtracts two blurs and "
            "behaves like a band-pass — with 'sigma2' broader than 'sigma', "
            "compact bright bumps come out positive. Gaps are filled for the "
            "calculation and restored afterwards.",
        ),
        cautions=(
            "These respond to the rate of change of the rate of change, so they "
            "amplify noise. If a plain Laplacian looks like static, switch to LoG "
            "or DoG and increase the smoothing scale.",
        ),
    ),
    _DefinitionEntry(
        title="Manual zero reference",
        params=("set_zero_point", "set_zero_plane_points", "patch"),
        summary=(
            "Lets you choose, by clicking, what counts as 'zero height' in the "
            "image. STM height values have no absolute meaning on their own, so "
            "this is how you set a reference you trust. Click one point to make "
            "that spot the new zero (everything is measured relative to it), or "
            "click three points on what should be a flat surface to define a "
            "level reference plane and remove an overall tilt by hand."
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
            "To be robust against noise, the tool averages a small patch of pixels "
            "around each click rather than trusting one pixel. The three-point "
            "version fits a flat plane through the three sampled heights and "
            "subtracts it. This is a deliberate, manual choice of reference — quite "
            "different from the automatic background fits, which guess the "
            "background from the data.",
            "The correction applies to the whole image. The little markers showing "
            "your clicked points can be hidden for a clean view without undoing the "
            "correction.",
        ),
        cautions=(
            "The result is only as good as the points you pick. If you click on a "
            "molecule, a tip crash, or a step edge instead of true flat surface, "
            "you will tilt or offset the whole image. Pick clean, flat spots that "
            "really should be at the same height.",
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
            "Does basic maths on the image — add, subtract, multiply, or divide. "
            "The second operand can be a single number (shift or rescale every "
            "height), another image of the same size (for example, subtract one "
            "scan from another to see what changed), or a built-in test pattern. "
            "It is a general-purpose building block rather than a specific "
            "correction."
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
            "An image operand has to be the same size as the current scan. The "
            "generated patterns (checkerboard, ramps, speckle, impulse grid) are "
            "made to fit and are recorded by their settings, so the step can be "
            "replayed exactly later.",
            "If you run it with an area ROI as the scope, the maths is computed for "
            "the whole image but only the pixels inside the ROI are kept — a handy "
            "way to apply an operation to just one region.",
        ),
        cautions=(
            "This rewrites the actual measured values. If you only want the image "
            "to look brighter or higher-contrast, leave the data alone and use the "
            "display-range sliders or colour map instead.",
        ),
    ),
    _DefinitionEntry(
        title="Thresholding and bit-depth conversion",
        params=("lower", "upper", "mode = clip | binarize", "bits = 8 | 16"),
        summary=(
            "Two related tools. Thresholding selects pixels by height: 'clip' "
            "keeps only pixels inside a height band (the rest become gaps), and "
            "'binarize' turns the image into a simple yes/no map of 1 inside the "
            "band and 0 outside — useful for isolating islands or molecules above "
            "a height. Bit-depth conversion rounds the heights onto a fixed number "
            "of levels (256 for 8-bit, 65536 for 16-bit), as if re-digitising the "
            "image more coarsely."
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
            "The lower/upper limits are real height values in the image's units, "
            "not slider positions, so you are choosing physical heights. Binarize "
            "produces genuine 0/1 numbers in the data (not just a coloured "
            "overlay), so the result can feed later steps.",
            "Bit-depth conversion still stores the data as normal numbers; it just "
            "limits how many distinct height levels remain. If you do not set the "
            "range, a robust automatic range is used.",
        ),
        cautions=(
            "Both throw information away. Clipping permanently turns out-of-band "
            "pixels into gaps, and coarse bit depth cannot be undone — it can hide "
            "subtle height differences. Keep an unprocessed copy if you might need "
            "the full data later.",
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
            "Reorients or reshapes the image: flip it, rotate it, shear it, or "
            "resample it to a different pixel count. Some of these just rearrange "
            "existing pixels and are perfectly lossless (flips and right-angle "
            "rotations); others have to invent in-between pixels by interpolation "
            "(arbitrary rotation, shear, rescaling), which slightly alters values. "
            "The program keeps the physical scale bar correct as the image size "
            "changes."
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
            "Flips and 90-degree rotations also move any ROIs you have drawn to "
            "match, exactly. Rescaling changes how many pixels describe the same "
            "physical area (more or fewer dots per nanometre) without changing the "
            "real size of the scan.",
            "Arbitrary rotation and shear keep each pixel's physical size but grow "
            "the canvas to fit the tilted image, filling the new corners with gaps.",
        ),
        cautions=(
            "An arbitrary rotation no longer lines up with your existing ROIs, so "
            "those are dropped — re-draw them afterwards. And any interpolated "
            "transform nudges pixel values slightly, so avoid stacking several if "
            "you need quantitative heights.",
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
            "A set of advanced corrections that act through the Fourier (FFT) "
            "domain or a fitted lattice. 'Mains pickup' removes the regular "
            "ripples from 50/60 Hz electrical interference. 'Inverse FFT "
            "selection' lets you draw around features in the FFT and either delete "
            "or keep only those, rebuilding the image from what remains. 'Affine "
            "lattice correction' gently warps the image so a measured atomic "
            "lattice matches its ideal geometry, undoing drift-induced distortion."
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
            "Mains pickup predicts where the interference should appear from the "
            "scan speed and mains frequency, and can snap to the actual bright "
            "spots before removing them. Inverse-FFT selections can either replace "
            "the current image or produce a separate result image, depending on "
            "the button you press.",
            "Affine lattice correction normally comes from fitting the lattice in "
            "the FFT or in real space. Like an arbitrary rotation, it can enlarge "
            "the canvas and fill the new edges with gaps or a chosen background "
            "value.",
        ),
        cautions=(
            "Because these work in the FFT, they can delete genuine periodic signal "
            "along with the artifacts. Always check the preview, and keep an "
            "uncorrected copy before using the output for measurements.",
        ),
    ),
    _DefinitionEntry(
        title="Linear undistortion",
        params=("shear_x", "scale_y"),
        summary=(
            "Corrects simple, steady distortion from thermal drift or a "
            "pixel-size mismatch between the x and y axes. 'shear_x' slants the "
            "image to undo a sideways drift that built up from the top of the scan "
            "to the bottom; 'scale_y' stretches or squashes vertically to make the "
            "two axes consistent. It rebuilds the image by sampling the original "
            "at the shifted positions."
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
            "A positive 'shear_x' means the sideways correction grows steadily "
            "from the first row to the last — matching drift that accumulates as "
            "the slow scan proceeds. 'scale_y' is a simple vertical stretch factor "
            "to fix a y-versus-x pixel-size mismatch.",
        ),
        cautions=(
            "This reshapes geometry, not height. It interpolates, so it slightly "
            "changes pixel values — use it to straighten distorted shapes, not to "
            "level or flatten the background.",
        ),
    ),
    _DefinitionEntry(
        title="Forward/backward scan blending",
        params=("weight",),
        summary=(
            "The tip scans each line twice — once left-to-right (forward) and once "
            "right-to-left (backward). Combining the two cuts random noise and can "
            "reveal where the surface or tip changed between passes. Because the "
            "backward pass runs in reverse, it is flipped left-to-right first so "
            "the same physical point lines up, then the two are averaged."
        ),
        equations=(
            "b_mirror[i,j] = bwd[i, Nx - 1 - j]\n"
            "z'_i,j = weight * fwd_i,j + (1 - weight) * b_mirror_i,j\n"
            "if one side is non-finite and the other is finite, use the finite side",
        ),
        details=(
            "'weight' sets the balance: 0.5 is an even average of both passes, "
            "while higher values lean toward the forward scan. Where one pass has a "
            "gap but the other has a valid pixel, the valid one is kept.",
        ),
        cautions=(
            "The forward and backward planes must be the same size and cover the "
            "same scan area, otherwise they cannot be aligned point-for-point.",
        ),
    ),
)


_ROI_REFERENCE_ENTRIES: tuple[_DefinitionEntry, ...] = (
    _DefinitionEntry(
        title="ROI model and selection state",
        params=("pixel coordinates", "ROISet", "active ROI", "dock selection"),
        summary=(
            "Each ROI belongs to one image and is remembered between sessions in a "
            "small companion file saved next to the scan, so your regions are still "
            "there when you reopen it. Two ideas matter throughout: the 'active' "
            "ROI (the one currently highlighted, which canvas editing acts on) and "
            "the ROI Manager 'selection' (one or more ROIs ticked in the list). "
            "When you run a tool, it uses the dock selection if you have one, "
            "otherwise it falls back to the active ROI."
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
            "Use the ROI Manager when an action needs several ROIs at once — for "
            "example combining shapes, or measuring a step height between two "
            "regions. Clicking a shape on the image simply makes it the active ROI.",
            "When you take a measurement, the ROI's name and identity are stored "
            "with the result, so the number still makes sense later even if you "
            "rename or delete that ROI.",
        ),
        cautions=(
            "If a saved processing step points at an ROI you have since deleted, "
            "the program will warn you or pause rather than quietly use the wrong "
            "region — and an export may stop instead of producing a misleading "
            "result.",
        ),
    ),
    _DefinitionEntry(
        title="Drawing and pan tools",
        params=("pan", "rectangle", "ellipse", "polygon", "freehand", "line", "point"),
        summary=(
            "Pick a drawing tool, draw one ROI, and the viewer hands control back "
            "to the everyday 'pan' tool automatically. Pan mode is where you spend "
            "most of your time: it pans and zooms the image, shows a hint about "
            "what is under the cursor, lets you click an ROI to select it, and lets "
            "you drag the active ROI to move it."
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
            "A rectangle or ellipse has to be dragged to a real size (a single "
            "click will not make one), and polygons and freehand shapes need at "
            "least three points. If you stop short, the half-drawn shape is simply "
            "discarded — nothing is left behind.",
            "Watch the status bar and the tooltip that appears as you hover: they "
            "tell you what the current click will do. In pan mode an ROI lights up "
            "when the cursor is over it, and that highlight is a promise — clicking "
            "selects exactly the ROI that is highlighted.",
        ),
        cautions=(
            "While a drawing tool is active, clicks make new shapes instead of "
            "selecting existing ones. Switch back to pan (or press Escape) before "
            "you try to move or right-click an ROI you already drew.",
        ),
    ),
    _DefinitionEntry(
        title="Editing existing ROIs",
        params=("rename", "delete", "copy/paste", "duplicate", "move", "resize"),
        summary=(
            "Editing follows a simple select-then-edit rule: your first click "
            "selects an ROI (it becomes the active one and shows its handles), and "
            "only then can you drag it or its handles to reshape it. This means a "
            "stray click never accidentally drags the wrong shape. Renaming, "
            "deleting, copying, and duplicating live in the ROI Manager and the "
            "right-click menu."
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
            "Handles are the small squares on the active ROI: corners and edges "
            "for rectangles, four points for ellipses, the two endpoints for "
            "lines. Hold Shift while dragging a rectangle or ellipse corner to keep "
            "its proportions. A line's 'width' is part of its shape and sets how "
            "wide a strip the line profile averages over.",
            "Delete or Backspace removes the active ROI; the ROI Manager can delete "
            "several selected ROIs at once. Copy then paste (or duplicate) makes a "
            "fresh copy offset by a few pixels so it is easy to grab.",
        ),
        cautions=(
            "Polygon, freehand, point, and multipolygon ROIs do not have resize "
            "handles yet — you can still move, rename, copy, or delete them, but "
            "not reshape them by dragging.",
        ),
    ),
    _DefinitionEntry(
        title="Area ROI actions",
        params=("mask/filter scope", "invert", "combine", "FFT", "histogram", "measure"),
        summary=(
            "Area ROIs (rectangles, ellipses, polygons, freehand, multipolygon) "
            "enclose a patch of the image, which the program turns into a mask — a "
            "yes/no map of which pixels are inside. That mask is what lets you "
            "apply a filter to just one region, measure statistics inside it, take "
            "an FFT of only that area, or combine regions with set operations. They "
            "are the workhorse ROI for analysing part of a scan."
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
            "Right-click an area ROI for its actions: limit a filter to this region "
            "('ROI filters only'), invert it, fit an STM background from it, show a "
            "histogram or FFT of just this region, compute statistics, or find "
            "feature peaks inside it.",
            "Step height is a two-ROI measurement: select exactly two area ROIs and "
            "it reports the average height difference between them — handy for "
            "measuring a terrace or island step.",
            "Area ROIs also drive per-region display. In the View tab, set "
            "'Contrast scope' to 'Active ROI' and the brightness/contrast sliders "
            "then adjust only the active region, so a scan with a bright area and a "
            "dim area can show both well at once; 'Hide ROI overlays' there lets "
            "you study the result without the outlines in the way.",
        ),
        cautions=(
            "'ROI filters only' affects just the local filters listed above. "
            "Whole-image steps like background and scan-line correction still cover "
            "the entire image unless their own dialog offers an ROI fit option.",
        ),
    ),
    _DefinitionEntry(
        title="Line ROI actions",
        params=("profile", "distance", "periodicity", "width", "endpoints"),
        summary=(
            "A line ROI is two endpoints with an optional width. Its main job is "
            "the height profile: a graph of surface height along the line, which "
            "you use to measure step heights, feature widths, and spacings. While a "
            "line is active, the profile panel under the image updates live as you "
            "move it, so you can drag the line and watch the cross-section change."
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
            "  click a line = make it the active line\n"
            "  drag active line = translate both endpoints\n"
            "  drag p1/p2 handle = move one endpoint\n"
            "  estimate periodicity = analyse current line-profile signal",
        ),
        details=(
            "Giving the line a width greater than one pixel averages a strip "
            "perpendicular to the line, which smooths a noisy profile. Right-click "
            "for actions: show the profile, save it as a measurement, estimate a "
            "repeat spacing (periodicity) from the profile, or set the line width. "
            "The ruler/distance tool reports the line's true physical length using "
            "the scan calibration.",
        ),
        cautions=(
            "A line is not an area, so area-only actions — region statistics, "
            "histogram, FFT of a region, filter masking — are unavailable while a "
            "line is selected. Draw an area ROI for those.",
        ),
    ),
    _DefinitionEntry(
        title="Point ROI actions",
        params=("point marker", "copy coordinates", "point sources"),
        summary=(
            "A point ROI marks a single spot — the position of an atom, a defect, "
            "or any feature you want to record. You can copy its coordinates, and a "
            "collection of points can feed the feature-analysis tools that need a "
            "list of locations. Think of points as labelled pins rather than "
            "regions."
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
            "Several analyses can take a set of points as input — pair-correlation "
            "(how features are spaced relative to each other), matching features to "
            "a lattice, and point-based masking or FFT. Those points can come from "
            "your point ROIs or from automatically detected feature peaks.",
            "On screen the markers stay the same size as you zoom, so they remain "
            "easy to see whether you are zoomed in or out.",
        ),
        cautions=(
            "A point has no area, so it cannot act as a region mask. Area-only "
            "tools — ROI filters, STM background fit masks, histograms, region "
            "statistics — are deliberately turned off for points.",
        ),
    ),
    _DefinitionEntry(
        title="Tool interactions and persistence",
        params=("sidecar save", "processing references", "geometric transforms"),
        summary=(
            "Every ROI change — create, move, resize, rename, delete — is saved "
            "straight away to the per-image companion file, so you never have to "
            "remember to save. Tools that refer to an ROI look it up at the moment "
            "they run, which means the result reflects the ROI as it is then, not "
            "as it was when you set the tool up."
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
            "The toolbar 'Mask' shortcut switches the Processing panel to 'ROI "
            "filters only' for your chosen area ROI, and 'Invert' makes a new ROI "
            "covering everything outside the current one and scopes filtering to "
            "that inverted area.",
            "The quick 'Simple background' button is always a whole-image flat-tilt "
            "subtraction. When you need the fit/apply/exclude-ROI options, open the "
            "dedicated background dialog instead.",
        ),
        cautions=(
            "Because steps look up their ROI when they run, moving or reshaping an "
            "ROI after you configured a step will change what that step does the "
            "next time it is replayed. If you want a step frozen, avoid editing the "
            "ROI it depends on.",
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
            "This guide explains what each processing step does, when you would "
            "reach for it, and what to watch out for. Every step works on the "
            "real measured height values (in physical units such as metres or "
            "amps), not on the colours you see on screen — changing the "
            "colour map or contrast never alters the data, but these operations "
            "do. Each entry has a plain-language summary, the exact formula the "
            "program uses, practical notes, and cautions. NaN means a pixel with "
            "no valid measurement; most steps carry those gaps through untouched."
        ),
        entries=_DEFINITION_ENTRIES,
        theme=theme,
    )


def render_roi_reference_html(theme: Mapping[str, object] | None = None) -> str:
    """Return theme-aware HTML for ROI actions and tool interactions."""
    return _render_reference_html(
        title="ROI Actions Reference",
        intro=(
            "A region of interest (ROI) is a shape you draw on the image to mark "
            "an area, a line, or a point you want to work with — for measuring, "
            "filtering, or analysing just part of a scan. This guide explains the "
            "kinds of ROI, how to draw and edit them, and what each kind can do. "
            "What happens when you click depends on three things: which ROI (if "
            "any) is currently selected, what kind of ROI it is, and which tool is "
            "active in the viewer. The notes below spell out each of those cases."
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
