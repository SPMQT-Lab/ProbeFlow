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
    in_practice: str = ""
    details: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()


@dataclass(frozen=True)
class _HowToEntry:
    """A step-by-step walkthrough for one common task."""

    title: str
    goal: str
    steps: tuple[str, ...]
    notes: tuple[str, ...] = ()
    tips: tuple[str, ...] = ()


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
            "forth. Sometimes the tip glitches and a stretch of a row reads too "
            "bright or too dark — anything from a short spark to a long, faint "
            "streak that is not real surface. This tool finds those damaged "
            "stretches by comparing each row with the rows just above and below "
            "it, then repairs only the bad stretch by copying from its healthy "
            "neighbours. Everything else in the image is left exactly as measured."
        ),
        in_practice=(
            "Click 'Preview detection' first, then lower the threshold until only "
            "the real streaks light up. Use 'mad' for long, faint lines and 'step' "
            "for short, sharp scars, and set 'polarity' to bright or dark to match "
            "the streaks you see."
        ),
        equations=(
            "MADN(v) = 1.4826 * median(|v - median(v)|)\n"
            "b_i[j] = per-column median over nearby finite row samples\n"
            "r_i[j] = z_i[j] - b_i[j]   (residual vs neighbouring rows)\n\n"
            "segment convention:\n"
            "  S = [j0, j1) spans columns k where j0 <= k < j1\n"
            "  reject S when length(S) < min_segment_length_px\n\n"
            "step mode (sharp, short scars):\n"
            "  d_i[j] = r_i[j+1] - r_i[j]\n"
            "  cutoff = threshold_mad * MADN(d)\n"
            "  bright segment S=(j0:j1) requires d_i[j0-1] > cutoff and d_i[j1-1] < -cutoff\n"
            "  dark segment reverses those signs\n\n"
            "mad mode (extended bright/dark lines, matched filter):\n"
            "  s_i = smooth_along_row(r_i, window ~ 11 px)\n"
            "  cutoff = threshold_mad * MADN(s)\n"
            "  bright: s_i[j] > cutoff;   dark: s_i[j] < -cutoff\n"
            "  bridge sub-threshold gaps <= ~4 px, then keep a run S only when\n"
            "    length(S) >= min_segment_length_px and median(s on S) clears cutoff\n"
            "  (no per-row median subtraction, and no upper length cap)\n\n"
            "repair:\n"
            "  skip a vertical stack of column-overlapping bad segments taller than\n"
            "    max_adjacent_bad_lines consecutive rows\n"
            "  above/below neighbours must be finite on S and not marked bad on S\n"
            "  z'_i,S = 0.5 * (z_above,S + z_below,S) when both neighbours are valid\n"
            "  z'_i,S = nearest valid neighbour segment when only one side is valid",
        ),
        details=(
            "Two modes find the bad stretches. 'step' looks for a sudden jump up "
            "and a matching jump back down — the signature of a short, sharp scar. "
            "'mad' is built for extended lines: it compares each row to its "
            "neighbours and, after smoothing along the row so a long faint streak "
            "rises above the pixel noise, flags runs that stay too bright (or too "
            "dark). 'Polarity' tells it whether you are chasing bright (too-high) "
            "or dark (too-low) streaks. The threshold is measured in robust noise "
            "units (MAD), so a value of 3 means 'about three times the usual "
            "noise'.",
            "Minimum segment length ignores single-pixel speckle, so only genuine "
            "stretches are touched. Maximum adjacent bad lines is a safety brake: "
            "where many neighbouring rows are bad in the same columns there is no "
            "healthy neighbour to copy from, so the tool leaves that overlapping "
            "block alone rather than guessing — but streaks on adjacent rows in "
            "different parts of the image are still repaired independently.",
            "Use 'Preview detection' first — it highlights what would be repaired "
            "without changing the data. Only 'Apply' edits the image, and even "
            "then only the accepted stretches change.",
        ),
        cautions=(
            "It repairs stretches that disagree with the neighbouring rows, so do "
            "not point it at genuine surface features — real terraces and step "
            "edges would be 'corrected' away. For systematic row-to-row level "
            "offsets, use row levelling (below) instead.",
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
        in_practice=(
            "Reach for this first, almost every time. Leave it on 'median' unless "
            "the rows are very clean; switch to 'linear' when the rows are tilted, "
            "not just offset."
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
        in_practice=(
            "Start with order 1 (a flat tilt) — it fixes most scans. Go higher "
            "only for visible bowing, and draw a 'fit ROI' on clean background (or "
            "'exclude ROI' over tall features) so islands don't drag the fit."
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
        in_practice=(
            "Use it when the image ramps or bows from top to bottom. Try 'linear' "
            "or 'low_pass' first; reach for a 'creep' model right after a big tip "
            "move. Fit from a clean-background ROI when tall features are present."
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
            "broad features stand out. 'sigma_px' is the Gaussian width (its "
            "standard deviation) in pixels — larger values blur more."
        ),
        in_practice=(
            "Use a small sigma (about 0.5–1.5 px) to calm pixel noise without "
            "erasing detail. The kernel reaches roughly ±4 sigma, so sigma is a "
            "width, not a hard edge — start small and increase only if needed."
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
            "sets the scale — anything broader than this width is removed."
        ),
        in_practice=(
            "Use it to flatten an uneven background and bring out small features. "
            "Pick a sigma larger than the features you want to keep. Afterwards the "
            "heights of big features are no longer trustworthy — it is for "
            "visualising, not for measuring step heights."
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
        in_practice=(
            "Low-pass to smooth, high-pass to sharpen. Start near the default "
            "cutoff and change one setting at a time while watching the preview; "
            "raise 'border_frac' only if you see ripples near the edges."
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
        in_practice=(
            "The quick FFT filter: low-pass to smooth, high-pass to sharpen. Add a "
            "'hanning' window to soften ringing; switch to the soft-border version "
            "when heights near the image border must stay exact."
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
        in_practice=(
            "Use it to remove one repeating ripple — mains interference or a "
            "lattice — by clicking its bright FFT spots. Keep 'radius_px' small so "
            "you only touch the spot, and keep an unfiltered copy if you still need "
            "to measure that pattern."
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
        in_practice=(
            "Use it to outline islands and steps, not to read heights. On noisy "
            "data prefer 'LoG' or 'DoG' over the plain Laplacian, and raise 'sigma' "
            "to choose the feature size you want to highlight."
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
        title="Advanced edge detection (Canny / Sobel–Scharr)",
        params=(
            "method = Canny | Sobel/Scharr",
            "sigma",
            "low / high threshold (percentile or absolute)",
            "preset",
            "output = overlay | new image | mask | ROI(s)",
        ),
        summary=(
            "A dedicated edge-finding tool (opened from 'Advanced Edge "
            "Detection…' on the Process tab) that turns edges into something you "
            "can act on — a clean outline, a mask, or ROIs — rather than just a "
            "picture. 'Canny' traces thin, connected edge lines; 'Sobel/Scharr' "
            "gives a continuous gradient (how steep the surface is at each "
            "pixel). Use it to outline islands, grains, or step edges and feed "
            "them to the mask/ROI tools."
        ),
        in_practice=(
            "Pick a Canny preset (e.g. 'Step edges / islands'), watch the live "
            "preview, then send the result to a mask or ROIs with the output "
            "buttons. Raise 'sigma' on noisy scans; raise the thresholds to keep "
            "only the strongest edges."
        ),
        equations=(
            "Canny (skimage):\n"
            "  1. Gaussian-smooth the image with sigma (in px)\n"
            "  2. gradient magnitude + non-maximum suppression -> thin ridges\n"
            "  3. hysteresis: keep ridge pixels >= high threshold (strong) and\n"
            "     pixels >= low threshold that connect to a strong edge\n"
            "  thresholds are percentiles of the gradient magnitude inside the\n"
            "  valid region (or absolute values) -> boolean edge mask\n\n"
            "Sobel / Scharr:\n"
            "  gx, gy = Sobel|Scharr derivative kernels\n"
            "  magnitude = sqrt(gx^2 + gy^2)   (or x, y, or orientation atan2(gy, gx))\n"
            "  optional: mask = magnitude >= percentile(magnitude, threshold)",
        ),
        details=(
            "This is the analysis cousin of the 'Edge detection' display filter "
            "above: instead of replacing the image, it produces a boolean edge "
            "map you can overlay, open as a new image, store as the active mask "
            "layer, or convert to ROIs for measuring. Canny's two thresholds give "
            "hysteresis — a high bar to start an edge and a lower bar to continue "
            "it — which traces faint but real boundaries without lighting up "
            "noise. Percentile thresholds are the robust default because they "
            "adapt to each channel's units.",
            "Restricting the detector to an ROI computes its thresholds from "
            "inside that region only, so background pixels outside do not dilute "
            "the statistics.",
        ),
        cautions=(
            "Edge maps are a derived overlay, not height data — measure on the "
            "image, not the edge picture. Too small a sigma or too low a threshold "
            "fragments edges and picks up noise; too large merges or misses them. "
            "Tune against the preview.",
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
        in_practice=(
            "Use it to set a trustworthy zero by hand: one click for a point zero, "
            "or three clicks on a surface that should be flat for a level plane. "
            "Pick clean, flat spots — not molecules, tip crashes, or step edges."
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
        in_practice=(
            "A building block: subtract one scan from another to see what changed, "
            "or add/scale by a constant. To only brighten the view, use the display "
            "sliders instead — this rewrites the real measured values."
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
        in_practice=(
            "Use 'clip' to keep only a height band, or 'binarize' to make a 1/0 "
            "mask of everything above a height (handy for counting islands). Both "
            "throw data away, so keep an unprocessed copy."
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
        in_practice=(
            "Flips and right-angle rotations are exact and lossless; arbitrary "
            "rotation, shear and rescale interpolate and slightly change values, so "
            "use them for presentation rather than fine measurement. The scale bar "
            "is kept correct automatically."
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
        in_practice=(
            "Advanced FFT corrections, usually driven from the FFT viewer. Always "
            "check the preview and keep an uncorrected copy — because they work in "
            "the FFT they can remove genuine periodic signal along with the "
            "artifact."
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
        in_practice=(
            "Use it to straighten a steady sideways drift ('shear_x') or to fix an "
            "x-versus-y pixel-size mismatch ('scale_y'). It reshapes geometry, not "
            "height — don't use it to level or flatten."
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
        in_practice=(
            "Average the forward and backward passes (weight 0.5) to cut random "
            "noise. The two planes must be the same size and cover the same scan "
            "area so they line up point-for-point."
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
        params=("pixel coordinates", "ROISet", "active ROI", "list selection"),
        summary=(
            "Each ROI belongs to one image and is remembered between sessions in a "
            "small companion file saved next to the scan, so your regions are still "
            "there when you reopen it. Two ideas matter throughout: the 'active' "
            "ROI (the one currently highlighted, which canvas editing acts on) and "
            "the ROI Manager 'selection' (one or more ROIs ticked in the ROI/Mask tab's "
            "list). When you run a tool, it uses that selection if you have one, "
            "otherwise it falls back to the active ROI."
        ),
        in_practice=(
            "Draw a region and most tools act on it. Tick several ROIs in the "
            "ROI/Mask tab to act on them all at once; with none ticked, the "
            "highlighted (active) ROI is used. Regions are saved automatically."
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
            "  selected ROI(s) in the ROI Manager list when present\n"
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
        title="Drawing tools and the cursor",
        params=("cursor", "rectangle", "ellipse", "line", "point", "polygon", "freehand"),
        summary=(
            "Pick a drawing tool, draw one ROI, and the viewer returns to the "
            "cursor automatically. The cursor (↖) is the default tool: click an ROI "
            "to select it, drag to pan, and Ctrl+scroll to zoom — no need to think "
            "about it."
        ),
        in_practice=(
            "Press a letter then draw — R rectangle, E ellipse, L line, P point, "
            "G polygon, F freehand — and the viewer snaps back to the cursor. Press "
            "Escape to cancel a half-drawn shape."
        ),
        equations=(
            "shortcuts (press the key, then draw):\n"
            "  R rectangle · E ellipse · L line · P point · G polygon · F freehand\n\n"
            "drawing completion:\n"
            "  rectangle/ellipse/line: drag, release to finish\n"
            "  point: click once to place\n"
            "  polygon: click vertices, Enter or double-click to close\n"
            "  freehand: drag path, release to finish\n"
            "  Escape: cancel the drawing and return to the cursor",
        ),
        details=(
            "A rectangle or ellipse has to be dragged to a real size (a single "
            "click will not make one), and polygons and freehand shapes need at "
            "least three points. If you stop short, the half-drawn shape is simply "
            "discarded — nothing is left behind.",
            "With the cursor, an ROI lights up when you hover over it, and that "
            "highlight is a promise: clicking selects exactly the ROI that is "
            "highlighted.",
        ),
        cautions=(
            "While a drawing tool is active, clicks make new shapes instead of "
            "selecting existing ones. Press Escape (or pick the cursor) before you "
            "move or right-click an ROI you already drew.",
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
        in_practice=(
            "Click once to select (handles appear), then drag the shape or a handle "
            "to reshape it. Hold Shift on a corner to keep proportions. Rename, "
            "delete, copy and duplicate from the right-click menu or ROI Manager."
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
        in_practice=(
            "Draw an area, then right-click it to filter just that region, measure "
            "statistics, take an FFT, or combine shapes. Select exactly two areas "
            "for a step-height measurement between them."
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
        in_practice=(
            "Draw a line across a step or feature to read its height profile in the "
            "panel below, and drag it to watch the cross-section update. Increase "
            "the line width to average out noise."
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
            "The measurements a line produces — Line profile (and Δ), Distance, "
            "Angle, and Line periodicity — are described in full in the "
            "Measurements tab.",
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
        in_practice=(
            "Drop points on features you want to record or feed to the feature "
            "tools (pair-correlation, lattice matching). Copy a point's coordinates "
            "from its right-click menu. Points can't act as a region mask."
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
        in_practice=(
            "You never have to save ROIs — they are written next to the scan as you "
            "go. Just remember that a step which references an ROI will follow that "
            "ROI if you move or reshape it later, so leave it be once a step depends "
            "on it."
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


_MEASUREMENT_ENTRIES: tuple[_DefinitionEntry, ...] = (
    _DefinitionEntry(
        title="Distance",
        params=("line ROI", "length_m", "dx_m", "dy_m", "angle_deg"),
        summary=(
            "Measures the straight-line distance between two points using a line "
            "ROI, in real units from the scan calibration. It also reports the "
            "horizontal and vertical components (Δx, Δy) and the line's angle from "
            "horizontal. Use it for feature sizes, spacings, and how far apart two "
            "things are."
        ),
        in_practice=(
            "Draw a line ROI across the gap you want, then take the Distance "
            "measurement. The length is calibrated, so it is a real nanometre "
            "distance, not pixels."
        ),
        equations=(
            "from line endpoints (x1, y1) -> (x2, y2) in pixels:\n"
            "  dx_m = (x2 - x1) * pixel_size_x_m\n"
            "  dy_m = (y2 - y1) * pixel_size_y_m\n"
            "  length_m = sqrt(dx_m^2 + dy_m^2)\n"
            "  angle_deg = atan2(|dy_m|, |dx_m|)   (from the horizontal)",
        ),
        details=(
            "The two axes are scaled by their own pixel sizes before the length is "
            "computed, so distances are correct even when the scan has "
            "non-square pixels.",
        ),
        cautions=(
            "The number is only as good as where you place the endpoints. Zoom in "
            "and snap them to the real feature edges; a line drawn a few pixels off "
            "changes the reading.",
        ),
    ),
    _DefinitionEntry(
        title="Angle",
        params=("two line ROIs", "angle_deg in [0, 90]"),
        summary=(
            "Measures the angle between two directions — for example two step "
            "edges, or two lattice rows — by drawing two line ROIs. The result is "
            "the acute angle between them, always reported between 0 and 90 "
            "degrees."
        ),
        in_practice=(
            "Draw two line ROIs along the directions you care about, select both, "
            "and take the Angle measurement. Direction is what matters, not which "
            "way you drew each line."
        ),
        equations=(
            "line vectors a and b in physical units (scaled by pixel size):\n"
            "  cos(theta) = (a . b) / (|a| * |b|)\n"
            "  angle_deg = acos(clamp(cos(theta), -1, 1))\n"
            "  if angle_deg > 90: angle_deg = 180 - angle_deg",
        ),
        details=(
            "Because only the directions matter, the angle is folded into 0–90°: "
            "drawing a line the other way round gives the same answer.",
        ),
        cautions=(
            "Very short lines make the direction uncertain — a one-pixel wobble at "
            "the ends swings the angle. Draw each line as long as the feature "
            "allows.",
        ),
    ),
    _DefinitionEntry(
        title="Line profile (and Δ)",
        params=("line ROI", "width", "length", "delta_y", "delta_x"),
        summary=(
            "Reads out the surface height along a line as a graph — the "
            "cross-section you use to measure step heights, feature widths, and "
            "spacings. Drop two markers on the graph to read the height difference "
            "(Δy) and horizontal separation (Δx) between them."
        ),
        in_practice=(
            "Draw a line across a step or feature; the profile updates live in the "
            "panel below. Increase the line width to average out noise, and use the "
            "two markers to read a step height (Δy) directly."
        ),
        equations=(
            "sample height along the line:\n"
            "  s runs from 0 to the physical length of the line\n"
            "  z(s) = image sampled along the line (bilinear)\n"
            "  width > 1 px: average finite pixels in a perpendicular swath\n\n"
            "two-marker delta:\n"
            "  delta_x = |s_2 - s_1|   (physical distance along the line)\n"
            "  delta_y = z(s_2) - z(s_1)   (height difference)",
        ),
        details=(
            "A width greater than one pixel averages a strip perpendicular to the "
            "line, which smooths a noisy profile while keeping the same length "
            "axis. The length axis is calibrated, so spacings read directly in "
            "nanometres.",
            "The line itself is a line ROI: how to draw it, move its endpoints, "
            "and set its averaging width is covered under 'Line ROI actions' in "
            "the ROI Actions tab. Distance and Angle (above) also use line ROIs.",
        ),
        cautions=(
            "Averaging over a wide swath blurs sloped or curved features — keep the "
            "width small when the step you are measuring is short or tilted.",
        ),
    ),
    _DefinitionEntry(
        title="Line periodicity",
        params=(
            "line profile",
            "method = autocorrelation | peak_spacing | fft",
            "period_m",
        ),
        summary=(
            "Estimates the repeat spacing of a regular pattern sampled along a line "
            "profile — for instance the period of an atomic row or a standing-wave "
            "ripple. It reports one characteristic period (and how many repeats fit "
            "along the line)."
        ),
        in_practice=(
            "Draw a line along the periodic direction (several repeats long), then "
            "estimate periodicity from the line-profile tools. A longer line gives "
            "a more reliable period."
        ),
        equations=(
            "detrend the profile z(s), then by method:\n"
            "  autocorrelation: C(lag) = sum_s z(s) z(s + lag);\n"
            "                   period = first strong off-zero peak in C\n"
            "  peak_spacing:    period = median spacing of detected profile peaks\n"
            "  fft:             period = 1 / (dominant spatial frequency of z(s))\n\n"
            "n_periods = line_length / period",
        ),
        details=(
            "Autocorrelation (the default) is the most robust on noisy data: it "
            "asks 'how far must I shift the profile for it to line up with itself "
            "again?'. The FFT method is sharpest when the pattern is clean and "
            "spans many repeats.",
        ),
        cautions=(
            "A line that covers only one or two repeats cannot pin a period down. "
            "Make sure the line spans several periods, and keep it parallel to the "
            "pattern, not across it.",
        ),
    ),
    _DefinitionEntry(
        title="ROI statistics",
        params=(
            "area ROI",
            "mean_height",
            "median_height",
            "std_height",
            "rms_roughness",
            "area",
        ),
        summary=(
            "Summarises the heights inside an area ROI: the average and middle "
            "height, the spread, the surface roughness, the min/max, and the "
            "physical area. Use it to characterise a patch — how rough a terrace "
            "is, how tall an island sits, how much area a phase covers."
        ),
        in_practice=(
            "Draw an area ROI over the patch you care about and read its "
            "statistics. Level the image first (row align / background) so heights "
            "are measured against a flat reference."
        ),
        equations=(
            "over finite pixels z inside the ROI mask:\n"
            "  mean_height   = mean(z)\n"
            "  median_height = median(z)\n"
            "  std_height    = std(z)\n"
            "  rms_roughness = sqrt(mean((z - mean(z))^2))   (Sq)\n"
            "  peak_to_peak  = max(z) - min(z)\n"
            "  area = (number of selected pixels) * pixel_size_x_m * pixel_size_y_m",
        ),
        details=(
            "'RMS roughness' (Sq) is the root-mean-square height deviation from the "
            "mean — the standard single-number measure of how rough a surface is. "
            "Non-finite (gap) pixels are ignored, and the area counts only the "
            "selected finite pixels.",
        ),
        cautions=(
            "Heights are relative to whatever reference the current processing "
            "leaves in place. A residual tilt or background inflates roughness and "
            "shifts the mean — level the surface before trusting these numbers.",
        ),
    ),
    _DefinitionEntry(
        title="Step height",
        params=("two area ROIs", "height_difference"),
        summary=(
            "Measures the height difference between two flat regions — the classic "
            "way to read a terrace or island step. Draw one ROI on the upper level "
            "and one on the lower, and it reports the difference between their "
            "average heights."
        ),
        in_practice=(
            "Place two area ROIs on the flat areas either side of the step (not on "
            "the step itself), select both, and take Step height. Averaging over a "
            "patch beats reading two single pixels."
        ),
        equations=(
            "over finite pixels in each ROI:\n"
            "  mean_a = mean(z in ROI A)\n"
            "  mean_b = mean(z in ROI B)\n"
            "  height_difference = mean_b - mean_a",
        ),
        details=(
            "Using the mean over a whole region (rather than two clicked points) "
            "averages away pixel noise, giving a much more stable step height. The "
            "per-ROI medians and standard deviations are also recorded.",
        ),
        cautions=(
            "Both regions must sit on genuinely flat terrace, not on the step face "
            "or on adsorbates. A tilt across the image biases the difference — "
            "level first, and keep the two ROIs close to the step.",
        ),
    ),
    _DefinitionEntry(
        title="Feature maxima",
        params=(
            "threshold_mode = above | below | between",
            "threshold_low / high",
            "min_distance_px",
        ),
        summary=(
            "Automatically finds the bright peaks (or dark pits) in the image — the "
            "positions of atoms, molecules, or islands — and drops a point at each "
            "one. The detected points become a list you can count, measure, or feed "
            "to the pair-correlation and lattice tools."
        ),
        in_practice=(
            "Set the polarity (above for bright maxima, below for dark minima), a "
            "height threshold, and a minimum spacing so each feature is counted "
            "once. Preview, then convert the peaks to point ROIs."
        ),
        equations=(
            "keep a pixel as a candidate when it passes the threshold:\n"
            "  above:   z >= threshold_low\n"
            "  below:   z <= threshold_high\n"
            "  between: threshold_low <= z <= threshold_high\n"
            "local maxima are then thinned so no two are closer than\n"
            "  min_distance_px (one detection per feature)\n"
            "n_points = number of detected features",
        ),
        details=(
            "The minimum-distance rule stops a single broad feature from being "
            "counted many times — only the strongest pixel within that radius "
            "survives. 'below' mode detects pits/minima by the same logic with the "
            "sign flipped.",
        ),
        cautions=(
            "Too low a threshold or too small a spacing counts noise as features; "
            "too high misses real ones. Tune against the preview, and remember the "
            "threshold is in the image's height units, which shift if you reprocess.",
        ),
    ),
    _DefinitionEntry(
        title="Point mask / FFT",
        params=("point set", "dominant_frequency"),
        summary=(
            "Takes a set of points (your point ROIs or detected feature maxima), "
            "stamps them onto a blank image, and Fourier-transforms that — turning "
            "an arrangement of points into its repeating spacings and directions. "
            "Bright spots in the result reveal the dominant lattice spacing of the "
            "points."
        ),
        in_practice=(
            "Detect or place the points first, then run Point mask / FFT. Bright "
            "off-centre spots mark the main repeat directions; their distance from "
            "the centre gives the spacing (spacing = 1 / frequency)."
        ),
        equations=(
            "M(x, y) = 1 at each point, 0 elsewhere\n"
            "F(qx, qy) = |fftshift(fft2(M))|\n"
            "qx, qy from fftfreq with the physical pixel size (cycles per length)\n"
            "dominant spacing = 1 / |q| of the brightest off-centre peak",
        ),
        details=(
            "Because it transforms only the point positions (not the height data), "
            "it isolates how the features are arranged from how tall they are — a "
            "clean way to see order in a scatter of detections.",
        ),
        cautions=(
            "A handful of points gives a noisy, hard-to-read transform; it needs "
            "many well-detected features to show clear spots. Stray or missed "
            "detections smear the pattern.",
        ),
    ),
    _DefinitionEntry(
        title="Pair correlation",
        params=("point set", "nn_median", "g(r)"),
        summary=(
            "Describes how a set of points is arranged relative to each other: it "
            "measures every point's distance to its nearest neighbour and builds "
            "the pair-correlation function g(r), which shows at what separations "
            "points tend to sit. It is the standard way to quantify ordering, "
            "spacing, and clustering."
        ),
        in_practice=(
            "Detect features or place point ROIs, then run Pair correlation. The "
            "nearest-neighbour median is a quick characteristic spacing; peaks in "
            "g(r) mark preferred separations (an ordered lattice gives sharp "
            "peaks)."
        ),
        equations=(
            "pairwise distances d_ij = |r_i - r_j| (physical units)\n"
            "nearest-neighbour: nn_i = min_{j != i} d_ij;  report median(nn_i)\n\n"
            "g(r): histogram all d_ij into radial bins, then normalise by the\n"
            "  number of pairs, the point density, and the bin's annulus area,\n"
            "  with an edge correction for pairs cut off by the ROI boundary",
        ),
        details=(
            "g(r) is built so that a completely random arrangement averages to 1; "
            "values above 1 mean points are more likely than random at that "
            "separation (a preferred spacing), below 1 means less likely. The edge "
            "correction stops the finite ROI from artificially suppressing long "
            "distances.",
        ),
        cautions=(
            "Reliable statistics need many points; a few detections give a noisy "
            "g(r). Missed or spurious features distort the nearest-neighbour "
            "distance, so check the detection first.",
        ),
    ),
    _DefinitionEntry(
        title="Feature → lattice",
        params=("point set", "ideal lattice", "rms_displacement_m"),
        summary=(
            "Compares detected features against an ideal, perfectly regular lattice "
            "and measures how far each one is displaced from where it 'should' be. "
            "The single-number result — the RMS displacement — quantifies disorder, "
            "strain, or distortion in an otherwise periodic arrangement."
        ),
        in_practice=(
            "Detect the features, define or fit the ideal lattice, then run "
            "Feature-to-lattice. A small RMS displacement means a well-ordered "
            "lattice; a large one flags strain or disorder."
        ),
        equations=(
            "match each feature to its nearest ideal lattice site (within a radius)\n"
            "displacement d_k = |feature_k - matched_site_k|\n"
            "rms_displacement = sqrt(mean(d_k^2))   over matched features\n"
            "reported in pixels and in metres",
        ),
        details=(
            "Only features that fall within the match radius of a lattice site are "
            "counted, so a few stray detections do not dominate the result. The RMS "
            "displacement is the root-mean-square of how far the real features sit "
            "from the ideal grid.",
        ),
        cautions=(
            "The answer depends on the ideal lattice you compare against — a wrong "
            "lattice constant or orientation inflates the displacement. Make sure "
            "the reference lattice matches the real one before reading disorder "
            "from this number.",
        ),
    ),
)


_HOWTO_ENTRIES: tuple[_HowToEntry, ...] = (
    _HowToEntry(
        title="Open an image and flatten it",
        goal="Load a scan and remove the basic tilt and drift so features stand out.",
        steps=(
            "Browse to your data folder and double-click a scan thumbnail to open "
            "it in the image viewer.",
            "Go to the Process tab and set 'Align rows' to Median to remove the "
            "row-to-row streaks, then click 'Apply processing'.",
            "Click 'STM Background...' to open the background dialog, and pick a "
            "model that matches the top-to-bottom drift you see (start with Linear "
            "or Poly2).",
            "Watch the live preview — the corrected image and the fitted background "
            "update as you change settings. Use 'Jump threshold' to ignore step "
            "edges and 'Preserve level' to keep a sensible absolute height.",
            "When the preview looks flat, click Apply. The correction is added to "
            "the image's processing pipeline and the main view updates.",
        ),
        notes=(
            "For a quick one-click flatten, use the Simple background button "
            "instead — it removes a first-order plane. STM Background is the better "
            "choice for STM's slow-scan drift and creep.",
            "Order matters: align rows first, then subtract a background.",
        ),
        tips=(
            "Every step is reversible — Undo/Redo step through changes, and Reset "
            "reloads the original on-disk data.",
        ),
    ),
    _HowToEntry(
        title="Move around the image: zoom, pan, colour, channels",
        goal="Navigate the scan and change how it looks, without changing the data.",
        steps=(
            "Ctrl+scroll (Cmd+scroll on a Mac) to zoom in and out.",
            "Drag a blank part of the image, or hold the middle mouse button, to "
            "pan around.",
            "Change the 'Colormap' dropdown at the top to recolour the image for "
            "clarity or contrast.",
            "Use the 'Channel' selector to switch between recorded channels, such "
            "as the forward and backward scans.",
        ),
        notes=(
            "Zoom, pan, colour map, and channel are all display choices — they "
            "never alter the measured height values.",
        ),
    ),
    _HowToEntry(
        title="Create, select, and delete ROIs",
        goal="Mark regions, lines, or points on the image and manage them.",
        steps=(
            "Pick a drawing tool (rectangle, ellipse, line, point, polygon, or "
            "freehand) and draw one shape; the viewer returns to the cursor "
            "automatically.",
            "Click a shape to select it — it lights up as you hover so you can see "
            "what a click will pick — and the selected ROI shows handles you can "
            "drag to reshape it.",
            "Open the ROI/Mask tab, which holds the ROI Manager, to see every ROI, "
            "rename them, set which one is active, or select several at once.",
            "To delete, select an ROI and press Delete or Backspace, or use Delete "
            "in the ROI Manager (which can remove several selected ROIs at once).",
        ),
        notes=(
            "ROIs are saved automatically in a small file next to the scan, so they "
            "are still there when you reopen the image.",
        ),
        tips=(
            "First click selects, the next interaction edits — a stray click never "
            "drags the wrong shape.",
        ),
    ),
    _HowToEntry(
        title="Measure a height profile along a line",
        goal="Read heights and distances across a feature, and save the profile.",
        steps=(
            "Choose the Line tool and drag a line across the feature; the tool "
            "returns to the cursor and the new line becomes the active one.",
            "Read the profile panel below the image — it plots height versus "
            "distance along the line and updates live as you drag the line or its "
            "endpoints.",
            "To smooth a noisy profile, right-click the line and set a larger "
            "width (or change it in the ROI Manager); the profile then averages a "
            "strip that wide, perpendicular to the line.",
            "To save it, right-click the line and choose 'Export line profile as "
            "CSV...'.",
        ),
        notes=(
            "The CSV begins with comment lines for the file name, bias (mV) and "
            "setpoint current, then a header row naming the two columns (Distance "
            "and Height, with their units), then the distance/value pairs — ready "
            "to open in a plotting program.",
        ),
        tips=(
            "Click a different line to switch which profile is shown; the panel "
            "always follows the active line.",
        ),
    ),
    _HowToEntry(
        title="Measure a distance or angle",
        goal="Read a physical length or an angle directly off the image.",
        steps=(
            "For a distance, draw a Line across the two points; its length is "
            "reported in real units (such as nanometres) from the scan "
            "calibration.",
            "For an angle, use the angle tool and click three points; the angle at "
            "the middle point is reported.",
        ),
        notes=(
            "These use the scan's calibration, so results are physical lengths and "
            "angles, not pixel counts.",
        ),
    ),
    _HowToEntry(
        title="Measure a step height between two regions",
        goal="Find the height difference between two terraces, or an island and "
             "its substrate.",
        steps=(
            "Draw an area ROI on each level — one on the upper terrace, one on the "
            "lower.",
            "In the ROI Manager, select both ROIs (Ctrl/Cmd-click the second to "
            "add it).",
            "Open the Measurements menu and choose 'Add step height from selected "
            "ROIs'.",
            "The result, added to the Measurements panel, is the difference between "
            "the two regions' average heights.",
        ),
        tips=(
            "Use flat, representative patches on each level — avoid step edges and "
            "adsorbates, which would bias the average.",
        ),
    ),
    _HowToEntry(
        title="Show two regions at once (per-region contrast)",
        goal="View a scan where one area is bright and another is dim without "
             "either contrast washing the other out.",
        steps=(
            "Draw an area ROI around each region (for example, the top half and "
            "the bottom half of a split scan).",
            "In the View tab, set 'Contrast scope' to 'Active ROI'.",
            "Click a region to make it active, then move the brightness/contrast "
            "sliders — they now affect only that region.",
            "Repeat for the other region. The image composites both, each scaled "
            "on its own.",
            "Tick 'Hide ROI overlays' to study the combined image without the "
            "outlines in the way.",
        ),
        notes=(
            "'Whole image' is the normal mode; switch back to it to adjust the "
            "global contrast again.",
        ),
        tips=(
            "This is ideal for a scan that was split mid-acquisition, where the two "
            "halves need different background and contrast.",
        ),
    ),
    _HowToEntry(
        title="Filter just one region",
        goal="Smooth or sharpen only part of the image, leaving the rest as "
             "measured.",
        steps=(
            "Draw an area ROI around the region you want to filter.",
            "In the Process tab, set the scope to 'ROI filters only'.",
            "Choose an eligible filter — smoothing, high-pass, edge detection, or "
            "an FFT filter — and apply it.",
            "Only the pixels inside the ROI change; everything outside is "
            "untouched.",
        ),
        notes=(
            "Whole-image steps like background and scan-line correction still cover "
            "the whole image; only the local filters listed above respect the ROI "
            "scope.",
        ),
    ),
    _HowToEntry(
        title="Fix scan-line glitches (bad-line correction)",
        goal="Remove bright or dark streaks — short sparks or long faint lines — "
             "left when the tip glitches along a row.",
        steps=(
            "In the Process tab, open the bad-line controls and choose a method "
            "and a polarity (bright or dark) to match the streaks you see: 'mad' "
            "for long faint lines, 'step' for short sharp scars.",
            "Click 'Preview detection' to highlight what would be repaired — "
            "nothing is changed yet.",
            "Adjust the threshold (measured in robust noise units) until only the "
            "genuine streaks are caught.",
            "Click Apply to repair just those stretches from their healthy "
            "neighbouring rows.",
        ),
        notes=(
            "This patches streaks that differ from the rows around them; it will "
            "not flatten real step edges or terraces. Use row alignment or a "
            "background fit for those.",
        ),
    ),
    _HowToEntry(
        title="Set a height zero reference",
        goal="Choose, by clicking, what counts as zero height in the image.",
        steps=(
            "In the Process tab, click 'Set zero plane'.",
            "Click one point to make that spot zero, or click three points on a "
            "surface that should be flat to define a level reference plane.",
            "The image is re-levelled against your chosen reference.",
        ),
        notes=(
            "A small patch around each click is averaged, so noise on a single "
            "pixel cannot throw off the reference. The markers can be hidden ('Hide "
            "Points') without undoing the correction.",
        ),
        tips=(
            "Pick clean, flat spots that really should be at the same height; "
            "clicking on a molecule, a tip crash, or a step will tilt the whole "
            "image.",
        ),
    ),
    _HowToEntry(
        title="Correct lattice distortion with the FFT viewer",
        goal="Straighten drift-distorted atomic rows using the reciprocal-space "
             "(FFT) lattice.",
        steps=(
            "With the image open, click 'FFT viewer...' (Process tab under "
            "Advanced, or the Measurements menu). The FFT opens on the Inspect "
            "tab.",
            "On Inspect, use the Min/Max/Brightness/Contrast sliders to bring the "
            "sharp Bragg spots out against the background — raise the minimum until "
            "mostly the bright lattice spots remain.",
            "Switch to the Grid tab and drag the g1 and g2 handles onto a pair of "
            "Bragg spots so the reciprocal grid matches the lattice.",
            "To check against a known material, tick 'Show shell rings' and enter "
            "the real-space lattice spacing under 'Known structure'; the overlay "
            "rings show where that lattice's Bragg spots should fall, confirming "
            "your grid.",
            "Switch to the Correction tab and click 'Preview' to see the "
            "de-distorted image — the main view updates live. Use 'Clear preview' "
            "to revert while you experiment.",
            "When satisfied, click 'Apply correction'. This adds an affine lattice "
            "correction to the image's processing pipeline and updates the main "
            "image.",
            "Close the FFT viewer. The corrected image stays in the main viewer "
            "with the correction applied.",
        ),
        notes=(
            "Bragg spots are the bright dots in the FFT — each pair marks one "
            "repeating lattice direction, and their positions encode the lattice "
            "spacing and angle. 'Show shell rings' draws the radii those spots "
            "should sit at for a known lattice, so a correct grid lines up with "
            "the rings.",
            "The Correction tab also exposes the ideal (target) lattice, the "
            "interpolation choice, and a suggested piezo calibration if you need "
            "finer control. Make sure the target lattice matches the spacing you "
            "set for the Bragg rings / known structure.",
        ),
        tips=(
            "If the spots are hard to see, push the minimum slider up and the "
            "contrast high: the lattice spots are sharp and bright, while the noise "
            "is broad and dim.",
        ),
    ),
    _HowToEntry(
        title="Undo, redo, and reset processing",
        goal="Step back through changes, or return all the way to the original "
             "data.",
        steps=(
            "After any processing step, use the Undo button (Process tab) to go "
            "back one step, and Redo to reapply it.",
            "Click 'Reset' to discard all processing and reload the raw, on-disk "
            "data for the current image.",
        ),
        notes=(
            "Display settings such as contrast and colour map are separate from "
            "processing — resetting the processing does not change your colour "
            "choices.",
        ),
    ),
    _HowToEntry(
        title="Export your processed image (data and provenance)",
        goal="Save the corrected image, and understand what is stored with it.",
        steps=(
            "Open the Export tab.",
            "To save a picture for a figure, choose PNG: this bakes in the current "
            "colour map, brightness/contrast, and (optionally) the scale bar. It is "
            "for presentation, not re-analysis.",
            "To save the data, export to a data format such as .sxm: this keeps the "
            "real height values so the scan can be reopened and analysed.",
            "Leave 'provenance' enabled to also write a JSON sidecar next to the "
            "file.",
        ),
        notes=(
            "The JSON provenance records the original source file, the full list of "
            "processing steps you applied (with their settings and the program "
            "version), the display settings (colour map and contrast), a "
            "timestamp, any warnings, and your ROIs — everything needed to "
            "reproduce the result.",
            "A processed export also carries a warning noting it is not raw "
            "instrument data, so it can never be mistaken for the original.",
        ),
        tips=(
            "Keep the JSON next to the exported file; it is your record of exactly "
            "how the image was made.",
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
    if entry.in_practice:
        blocks.append(
            f'<p class="lead"><b>In practice:</b> {escape(entry.in_practice)}</p>'
        )
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
    rendered_entries = "\n<hr/>\n".join(
        _render_entry(entry, block_label=block_label) for entry in entries
    )
    return _reference_document(
        title=title, intro=intro, rendered_entries=rendered_entries, theme=theme,
    )


def _reference_document(
    *,
    title: str,
    intro: str,
    rendered_entries: str,
    theme: Mapping[str, object] | None = None,
) -> str:
    """Wrap pre-rendered entry HTML in the shared, themed document shell."""
    p = _definitions_palette(theme)
    return f"""
<style>
  body {{
      font-family: Helvetica, Arial;
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
  .lead {{
      color: {p["heading"]};
      background-color: {p["equation_bg"]};
      border-left: 3px solid {p["keyword"]};
      padding: 6px 10px;
      margin: 6px 0 9px 0;
      line-height: 1.45;
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
  ol {{
      margin: 4px 0 9px 0;
      padding-left: 22px;
      line-height: 1.5;
  }}
  li {{
      margin: 3px 0;
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


_PARTICLE_STATISTICS_ENTRIES: tuple[_DefinitionEntry, ...] = (
    _DefinitionEntry(
        title="What is a null model?",
        params=(),
        summary=(
            "A null model is the 'boring explanation' you test your points against - "
            "a precise recipe for what the positions would look like if nothing "
            "interesting were going on. The most common one is complete randomness: "
            "points dropped independently, with no clustering, no preferred spacing, "
            "and no preference for any location. Particle Statistics asks whether "
            "your real points could plausibly have come from that recipe. If they "
            "could, you have no evidence for structure; if they clearly could not, "
            "that is evidence something real is shaping the pattern."
        ),
        in_practice=(
            "Choose the null model that captures the 'boring' explanation you want to "
            "rule out, then run the comparison. Start with homogeneous Poisson "
            "(pure randomness) unless you have a specific alternative in mind."
        ),
        equations=(),
        details=(
            "Think of it as a strawman on purpose: you set up the simplest "
            "explanation, then see whether your data can knock it down. Rejecting the "
            "null ('inconsistent') is the informative outcome; failing to reject it "
            "('consistent') just means this data does not provide evidence against it.",
            "Different null models encode different boring explanations - pure "
            "randomness, randomness with a minimum spacing, or randomness biased "
            "toward a measured feature - so the model you pick defines exactly what "
            "'no structure' means for your question.",
        ),
        cautions=(
            "A null model is a baseline to argue against, not a description of your "
            "sample. 'Consistent with the null' is not proof that the null is true - "
            "only that this data cannot rule it out.",
        ),
    ),
    _DefinitionEntry(
        title="How a comparison works (null model + envelope test)",
        params=(
            "model = homogeneous_poisson | hard_core_random | measured_feature_poisson",
            "simulations (n_sim)",
            "random_seed",
            "alpha = 0.05",
        ),
        summary=(
            "Particle Statistics never judges a pattern in isolation. It measures a "
            "statistic on your points, then measures the same statistic on many "
            "random patterns drawn from a chosen null model on the identical "
            "region, and asks whether your observed curve looks like one of those "
            "null draws. The spread of the null draws is the envelope; the verdict "
            "is a single global test, not a glance at the band."
        ),
        in_practice=(
            "Pick a null model, set the number of simulations (more gives a smoother "
            "envelope and a finer smallest p-value), and run. Use the same region "
            "for the observed statistic and every simulation."
        ),
        equations=(
            "Observed statistic T_obs(r); null draws T_1(r) … T_n(r) on the same region.\n"
            "\n"
            "Pointwise band (display only): at each r, [low, high] = 2.5th-97.5th\n"
            "percentile of the simulated T_k(r).\n"
            "\n"
            "Global verdict - extreme rank length (ERL; Myllymaki et al. 2017):\n"
            "  pool = {T_obs, T_1, ..., T_n}     (n+1 functions, exchangeable under H0)\n"
            "  R_k(r) = min( #{j: T_j(r) >= T_k(r)}, #{j: T_j(r) <= T_k(r)} )\n"
            "          (two-sided pointwise rank; smaller = more extreme)\n"
            "  sort each function's ranks ascending, compare lexicographically\n"
            "  p = (1 + #{k>=1 : T_k at least as extreme as T_obs}) / (n_sim + 1)\n"
            "  verdict = inconsistent if p <= alpha (0.05), else consistent",
        ),
        details=(
            "Why uncorrected statistics still give a correct test: the SAME estimator "
            "- with the same finite-region edge bias - is applied to the observed "
            "pattern and to every null simulation on the identical region or mask. "
            "The ERL test only asks whether the observed curve is exchangeable with "
            "the simulated ones, so any bias they share cancels out. AdStat's "
            "calibration test confirms the test holds its nominal ~5% false-positive "
            "rate under a true null.",
            "The pointwise band is for the eye only. Because it is computed at each r "
            "separately, some points fall outside it a few percent of the time even "
            "under a true null - so the verdict comes from the one global ERL test, "
            "not from counting out-of-band points.",
            "The smallest possible p-value is 1/(n_sim+1): with 19 simulations the "
            "best you can claim is p = 0.05; use 99 or more to resolve p <= 0.01.",
        ),
        cautions=(
            "Comparing one pattern against several models (or reading several "
            "statistics) is multiple testing, and ProbeFlow does not correct for it. "
            "Treat a single rejection among many comparisons cautiously.",
        ),
    ),
    _DefinitionEntry(
        title="Null model: Homogeneous Poisson (complete spatial randomness)",
        params=("N fixed to the observed count",),
        summary=(
            "The baseline 'no structure' model: every point is placed independently "
            "and uniformly across the allowed region, at one average density. It is "
            "the reference for clustering and spacing questions alike."
        ),
        in_practice=(
            "Start here. If the pattern is already consistent with random placement, "
            "the more specific models rarely add anything."
        ),
        equations=(
            "Simulate: N points placed independently and uniformly over the region\n"
            "(mask-aware - only allowed pixels are used).\n"
            "Intensity (density): lambda = N / A_region\n"
            "\n"
            "This is a binomial process: the count is conditioned on the observed N\n"
            "(not Poisson-distributed N), which is exactly the right question - 'are\n"
            "THESE N points randomly placed?'.",
        ),
        cautions=(
            "It assumes one homogeneous density over the whole region (stationarity). "
            "A strong large-scale coverage gradient violates that assumption and can "
            "masquerade as clustering; restrict the analysis region or mask to a "
            "uniform area first.",
        ),
    ),
    _DefinitionEntry(
        title="Null model: Hard-core random (minimum separation)",
        params=(
            "hard_core_radius (centre-to-centre minimum, nm)",
            "attempt_limit",
        ),
        summary=(
            "Random placement, but with a forbidden minimum separation - a simple "
            "model of particles that cannot overlap or sit too close (excluded "
            "volume, site blocking)."
        ),
        in_practice=(
            "Use it when nearest-neighbour distances show a clear minimum spacing and "
            "you want to test whether plain exclusion explains the pattern."
        ),
        equations=(
            "Simple sequential inhibition (SSI / random sequential adsorption):\n"
            "  repeat: draw a uniform candidate; accept iff its distance to every\n"
            "  already-accepted point >= r_hc; stop at N accepted or attempt_limit.\n"
            "\n"
            "Each accepted centre carries an exclusion disk of radius r_hc/2; these\n"
            "disks never overlap. Packing fraction phi = N * pi (r_hc/2)^2 / A.\n"
            "SSI jams (cannot fit more) near phi ~ 0.547.",
        ),
        cautions=(
            "SSI is a non-equilibrium process: it is NOT the thermodynamic "
            "equilibrium hard-disk gas. 'Consistent with hard-core' means consistent "
            "with this sequential-exclusion null, not with a Gibbs hard-disk model.",
            "If placement cannot fit N points before the attempt limit, lower the "
            "radius or the count - a failure here means the requested density exceeds "
            "what exclusion allows.",
        ),
    ),
    _DefinitionEntry(
        title="Null model: Measured-feature Poisson (association)",
        params=(
            "feature_layer (independently measured)",
            "kernel_sigma_nm (sigma)",
            "feature_weight (w1)",
            "background_weight (w0)",
        ),
        summary=(
            "Tests whether points follow an independently measured feature - for "
            "example, whether adsorbates sit preferentially near step edges. The "
            "feature defines a non-uniform expected density."
        ),
        in_practice=(
            "Provide a feature set that was measured separately from the particles "
            "(e.g. step traces). Choose sigma to match how far the influence reaches."
        ),
        equations=(
            "Inhomogeneous Poisson intensity from a Gaussian kernel around the\n"
            "measured features f_i:\n"
            "  lambda(x) = w0 + w1 * SUM_i exp( - d(x, f_i)^2 / (2 sigma^2) )\n"
            "  d = distance to a point feature, or to the nearest point on a line\n"
            "      segment.\n"
            "Simulate: sample N points with probability proportional to lambda(x)\n"
            "(evaluated on an adaptive grid, then jittered within a cell).",
        ),
        cautions=(
            "The feature layer must be measured independently of the particles being "
            "tested. Using the particles - or anything derived from them - as their "
            "own feature layer is circular and manufactures a false association.",
        ),
    ),
    _DefinitionEntry(
        title="Statistic: Pair correlation g(r)",
        params=(
            "pair_bin_width_nm",
            "pair_max_radius_nm",
            "edge_correction = none | translation",
        ),
        summary=(
            "At each separation r, are there more or fewer neighbour pairs than "
            "random placement would give? The most sensitive first look at "
            "short-range clustering or avoidance."
        ),
        in_practice=(
            "A bump above 1 at small r is clustering; a dip below 1 near zero is "
            "spacing/avoidance. Read it together with the model envelope, not against "
            "the 1.0 line alone."
        ),
        equations=(
            "Count unordered pairs (i<j) into distance shells [r0, r1):\n"
            "  shell area A_shell = pi (r1^2 - r0^2),  bin centre r = (r0+r1)/2\n"
            "Expected pairs under CSR:  E = 1/2 * N * lambda * A_shell,  lambda = N/A\n"
            "  g(r) = observed_pairs(r) / E\n"
            "  g(r) = 1 random ;  > 1 clustering ;  < 1 spacing/avoidance\n"
            "\n"
            "Optional translation edge weight for a pair separated by (dx, dy):\n"
            "  w = A_region / [ (W - |dx|) (H - |dy|) ]",
        ),
        cautions=(
            "Uncorrected g(r) is biased high near the boundary; the matched "
            "simulations carry the same bias so the verdict stays valid, but read "
            "absolute g(r) values at r approaching half the field size with care.",
        ),
    ),
    _DefinitionEntry(
        title="Statistic: Nearest-neighbour distribution",
        params=("nn_bin_width_nm", "nn_max_distance_nm"),
        summary=(
            "The distribution of each point's distance to its single closest "
            "neighbour - the clearest first test of spacing versus clumping."
        ),
        in_practice=(
            "Clustering pushes weight toward short distances; exclusion/spacing piles "
            "the distribution up near a minimum distance."
        ),
        equations=(
            "For each point i:  d_NN(i) = min over j != i of || x_i - x_j ||\n"
            "Histogram d_NN over bins, normalised to a fraction (counts / N).\n"
            "\n"
            "CSR reference distribution (intensity lambda):\n"
            "  G(r) = 1 - exp( - lambda * pi * r^2 )",
        ),
        details=(
            "The maximum distance is chosen from the point density so the histogram "
            "is not truncated for sparse fields. The empirical distribution is "
            "compared to the simulations, not to the closed-form G(r).",
        ),
    ),
    _DefinitionEntry(
        title="Statistic: Ripley's L",
        params=("derived radii", "edge_correction = none | translation"),
        summary=(
            "A cumulative, variance-stabilised count of how many neighbours fall "
            "within each distance - sensitive to clustering or regularity across a "
            "range of scales at once."
        ),
        in_practice=(
            "L(r) - r above zero is clustering, below zero is regularity. Because it "
            "is cumulative, read where the curve first leaves the envelope."
        ),
        equations=(
            "K(r) = (A_region / [N(N-1)]) * 2 * C(r),\n"
            "  C(r) = number of unordered pairs with distance <= r\n"
            "L(r) = sqrt( K(r) / pi )\n"
            "Under CSR:  K(r) = pi r^2  =>  L(r) - r = 0\n"
            "  L(r) - r > 0 clustering ;  < 0 regularity/spacing",
        ),
        cautions=(
            "Being cumulative, it accumulates structure across scales - an excursion "
            "at large r can reflect structure that really lives at smaller r. Use "
            "g(r) to localise the scale.",
        ),
    ),
    _DefinitionEntry(
        title="Statistic: Cluster sizes",
        params=("cluster_radius_nm (link distance)",),
        summary=(
            "Groups points that chain together within a link distance and reports how "
            "many singletons, pairs, triples, and larger groups there are."
        ),
        in_practice=(
            "An excess of large groups versus the null is clustering. Choose the link "
            "distance from the scale of grouping you care about."
        ),
        equations=(
            "Build a graph: link i-j iff || x_i - x_j || <= r_link.\n"
            "Clusters = connected components (single linkage; links are transitive,\n"
            "so a chain of close points forms one cluster).\n"
            "Report the histogram of component sizes.",
        ),
        cautions=(
            "The link distance is an analysis threshold, not a physical capture "
            "radius. Single linkage can chain a sparse bridge of points into one "
            "large cluster - compare the size histogram to the simulated null, not to "
            "an absolute expectation.",
        ),
    ),
    _DefinitionEntry(
        title="Local-order checks: psi4 / psi6 and angular g(r, theta) (opt-in)",
        params=(
            "neighbour_radius_nm (~1.35 x median nearest-neighbour distance)",
            "symmetry n = 4 | 6",
            "pair_angle_bin_width_deg",
        ),
        summary=(
            "Do neighbours sit at lattice-like angles - square (psi4) or triangular "
            "(psi6)? These answer a different question from the randomness checks and "
            "are off by default; tick 'Include local-order checks' to compute them."
        ),
        in_practice=(
            "Reach for these only when you suspect ordered packing or registry. "
            "Values of |psi_n| near 1 mean strong local n-fold order."
        ),
        equations=(
            "Per point i, over its neighbours j within the cutoff radius:\n"
            "  psi_n(i) = (1 / k_i) * SUM_j exp( i * n * theta_ij ),\n"
            "    theta_ij = atan2(dy, dx)\n"
            "  |psi_n| = 0 isotropic ... 1 perfect n-fold local order\n"
            "Report the histogram of |psi_n| (psi6 triangular, psi4 square).\n"
            "\n"
            "Angular pair map g(r, theta): pair counts in (distance, direction)\n"
            "bins, with directions folded to [0, 180) since a pair has no arrow.",
        ),
        details=(
            "Validated by known-answer tests: a triangular lattice rejects on psi6 "
            "and a square lattice on psi4, while random points stay consistent "
            "(tests/test_adstat_validation.py).",
        ),
        cautions=(
            "Sensitive to the neighbour cutoff and to edges: there is no boundary "
            "correction, so points near the region edge have truncated "
            "neighbourhoods and read as less ordered. A high |psi_n| suggests local "
            "order; it is not proof of a crystal or a specific adsorption site.",
        ),
    ),
    _DefinitionEntry(
        title="Reading verdicts and limitations",
        params=(),
        summary=(
            "How to read a result and what it does - and does not - prove."
        ),
        equations=(),
        details=(
            "'Consistent with the model' means the null was not rejected at alpha. It "
            "is not positive proof there is no structure: a small or noisy set may "
            "simply lack the power to detect an effect.",
            "'Inconsistent with the model' is evidence the pattern departs from that "
            "null - not proof of any particular physical mechanism.",
            "Pooling several independent images of the same condition is the "
            "practical way to gain statistical power.",
        ),
        cautions=(
            "This is the newest and least user-tested part of ProbeFlow and may "
            "contain mistakes; verify important results independently (the maturity "
            "note at the top of this reference says the same).",
            "Hard-core compares against a sequential-exclusion null, not an "
            "equilibrium hard-disk gas; no correction is applied for comparing "
            "several models or statistics - treat an isolated rejection cautiously.",
        ),
    ),
)


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


def render_measurements_html(theme: Mapping[str, object] | None = None) -> str:
    """Return theme-aware HTML for the image-measurements reference."""
    return _render_reference_html(
        title="Measurements Reference",
        intro=(
            "These are the measurements you can take from the Measure tab. Every "
            "one works on the real, calibrated data — physical heights and "
            "distances in metres/nanometres and angles in degrees — not on the "
            "colours on screen. Each result is added to the Measure-tab table with "
            "its units, kept with the ROI it came from, and exported with the "
            "image. Each entry below has a plain-language summary, when to reach "
            "for it, the formula the program uses, and cautions. Level the image "
            "first (row alignment / background) so heights are measured against a "
            "flat reference."
        ),
        entries=_MEASUREMENT_ENTRIES,
        theme=theme,
        block_label="Computation",
    )


def _render_howto_entry(entry: _HowToEntry) -> str:
    blocks = [f'<div class="entry" id="{_entry_id(entry.title)}">']
    blocks.append(f"<h2>{escape(entry.title)}</h2>")
    blocks.append(f'<p class="sub">{escape(entry.goal)}</p>')
    if entry.steps:
        blocks.append('<p class="label">Steps</p>')
        items = "".join(f"<li>{escape(step)}</li>" for step in entry.steps)
        blocks.append(f"<ol>{items}</ol>")
    for note in entry.notes:
        blocks.append(f"<p>{escape(note)}</p>")
    for tip in entry.tips:
        blocks.append(f'<p class="note">Tip: {escape(tip)}</p>')
    blocks.append("</div>")
    return "\n".join(blocks)


def render_howto_html(theme: Mapping[str, object] | None = None) -> str:
    """Return theme-aware HTML for the step-by-step how-to guides."""
    rendered_entries = "\n<hr/>\n".join(
        _render_howto_entry(entry) for entry in _HOWTO_ENTRIES
    )
    return _reference_document(
        title="How-to Guides",
        intro=(
            "Short, numbered walkthroughs for the most common tasks — opening and "
            "flattening a scan, drawing ROIs and line profiles, measuring, "
            "filtering, correcting lattice distortion in the FFT viewer, and "
            "exporting. Each guide lists the steps in order, with notes on what you "
            "get and tips to avoid common mistakes. For the underlying maths, see "
            "the Processing and ROI Actions tabs."
        ),
        rendered_entries=rendered_entries,
        theme=theme,
    )


def render_particle_statistics_html(theme: Mapping[str, object] | None = None) -> str:
    """Return theme-aware HTML for the Particle Statistics reference."""
    return _render_reference_html(
        title="Particle Statistics Reference",
        intro=(
            "Particle Statistics asks a spatial question: are these point positions "
            "consistent with a simple null model, or do they show clustering, "
            "spacing, or association with an independently measured feature? It never "
            "judges a pattern alone - it compares the observed statistic with the "
            "same statistic measured on many random patterns drawn from a chosen "
            "model on the identical region. Each entry below gives a plain-language "
            "summary, the exact estimator or algorithm the program uses, practical "
            "notes, and cautions. Maturity note: this is the newest and least "
            "user-tested part of ProbeFlow and may contain mistakes - treat verdicts "
            "as exploratory and verify important results independently. Calculations "
            "are powered by the AdStat engine."
        ),
        entries=_PARTICLE_STATISTICS_ENTRIES,
        theme=theme,
    )


_DEFINITIONS_HTML = render_definitions_html()
_ROI_REFERENCE_HTML = render_roi_reference_html()
_MEASUREMENTS_HTML = render_measurements_html()
_HOWTO_HTML = render_howto_html()
_PARTICLE_STATISTICS_HTML = render_particle_statistics_html()


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


class _MeasurementsPanel(_HtmlReferencePanel):
    """Scrollable reference panel describing the image measurements."""

    def __init__(self, t: dict, parent=None):
        super().__init__(t, render_measurements_html(t), parent)


class _HowToPanel(_HtmlReferencePanel):
    """Scrollable panel with step-by-step how-to walkthroughs."""

    def __init__(self, t: dict, parent=None):
        super().__init__(t, render_howto_html(t), parent)


class _ParticleStatisticsPanel(_HtmlReferencePanel):
    """Scrollable reference panel describing the Particle Statistics models."""

    def __init__(self, t: dict, parent=None):
        super().__init__(t, render_particle_statistics_html(t), parent)


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
        self._howto_panel = _HowToPanel(t, self)
        self._panel = _DefinitionsPanel(t, self)
        self._measurements_panel = _MeasurementsPanel(t, self)
        self._roi_panel = _ROIReferencePanel(t, self)
        self._particle_stats_panel = _ParticleStatisticsPanel(t, self)
        self._tabs.addTab(self._howto_panel, "How-to")
        self._tabs.addTab(self._panel, "Processing")
        self._tabs.addTab(self._measurements_panel, "Measurements")
        self._tabs.addTab(self._roi_panel, "ROI Actions")
        self._tabs.addTab(self._particle_stats_panel, "Particle Statistics")
        lay.addWidget(self._tabs)
        self.set_reference_tab(initial_tab)

    # Stable tab keys -> tab index (How-to first as the friendliest landing).
    _TAB_INDEX = {
        "howto": 0,
        "processing": 1,
        "measurements": 2,
        "roi": 3,
        "particle_statistics": 4,
    }

    def set_reference_tab(self, tab: str) -> None:
        """Switch to the named reference tab."""
        key = str(tab or "processing").lower().replace("-", "_")
        if key in {"roi", "roi_actions", "roi_reference"}:
            key = "roi"
        elif key in {"howto", "how_to", "guides", "guide"}:
            key = "howto"
        elif key in {"measurements", "measurement", "measure"}:
            key = "measurements"
        elif key in {
            "particle_statistics",
            "particle",
            "particles",
            "stats",
            "statistics",
        }:
            key = "particle_statistics"
        else:
            key = "processing"
        self._tabs.setCurrentIndex(self._TAB_INDEX[key])

    def current_reference_tab(self) -> str:
        """Return the stable key for the currently selected reference tab."""
        index = self._tabs.currentIndex()
        for key, idx in self._TAB_INDEX.items():
            if idx == index:
                return key
        return "processing"
