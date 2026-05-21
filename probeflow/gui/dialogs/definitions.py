"""Processing definitions/help dialog for ProbeFlow."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Mapping

from PySide6.QtWidgets import QDialog, QFrame, QScrollArea, QTextEdit, QVBoxLayout, QWidget


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
            "b_i[j] = median of nearby valid rows at column j\n"
            "r_i[j] = z_i[j] - b_i[j]\n\n"
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
        params=("order = 1..4", "fit ROI", "apply ROI", "exclude ROI", "step tolerance"),
        summary=(
            "Fits a two-dimensional polynomial background on selected finite "
            "pixels, then subtracts the fitted surface from the whole image or "
            "from an apply ROI."
        ),
        equations=(
            "normalised coordinates: x, y in [-1, 1]\n"
            "B(x, y) = sum c_pq * x^p * y^q, for p + q <= order\n"
            "least squares on fit pixels: min_c sum_fit (z(x,y) - B(x,y))^2\n"
            "z'(x, y) = z(x, y) - B(x, y)",
        ),
        details=(
            "The viewer's Plane/background button uses a first-order plane by default: "
            "B(x,y) = c00 + c10*x + c01*y.",
            "Fit ROI and exclude ROI affect only the pixels used to estimate the "
            "background. Apply ROI controls which pixels are modified.",
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
            "background_i,j = B_i + L_i(x_j) when x prefit is enabled\n"
            "z'_i,j = z_i,j - background_i,j + reference(background)",
        ),
        details=(
            "Active ROI limits where the line profile is estimated, but subtraction is "
            "still applied to the full image.",
            "The optional jump threshold removes large discontinuities in the profile "
            "before fitting. Preserve level adds back the median or mean of the fitted "
            "background so the corrected image keeps a familiar height reference.",
            "For piezo and sqrt creep models, d is constrained before the scan start "
            "so the singularity stays off the measured image.",
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
            "Low-pass keeps broad structure and suppresses high-frequency detail. "
            "High-pass suppresses low-frequency structure but adds the original "
            "finite-pixel mean back to keep the output on a comparable height level.",
        ),
        cautions=(
            "Hard circular cutoffs can create ringing, especially when no window is used.",
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
        ),
        cautions=(
            "Second-derivative filters are noise-sensitive. Use LoG or DoG when raw "
            "Laplacian contrast is too harsh.",
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
            "  z'(y, x) = bilinear_sample(z, src_y, src_x)",
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


def _render_entry(entry: _DefinitionEntry) -> str:
    blocks = [f'<div class="entry" id="{_entry_id(entry.title)}">']
    blocks.append(f"<h2>{escape(entry.title)}</h2>")
    blocks.append(_render_params(entry.params))
    blocks.append(f"<p>{escape(entry.summary)}</p>")
    for equation in entry.equations:
        blocks.append('<p class="label">Operation</p>')
        blocks.append(f'<pre class="equation">{escape(equation)}</pre>')
    for detail in entry.details:
        blocks.append(f"<p>{escape(detail)}</p>")
    for caution in entry.cautions:
        blocks.append(f'<p class="note">{escape(caution)}</p>')
    blocks.append("</div>")
    return "\n".join(blocks)


def render_definitions_html(theme: Mapping[str, object] | None = None) -> str:
    """Return theme-aware HTML for the processing definitions dialog."""
    p = _definitions_palette(theme)
    entries = "\n<hr/>\n".join(_render_entry(entry) for entry in _DEFINITION_ENTRIES)
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
<h1>Processing Algorithm Reference</h1>
<p class="intro">Each step transforms the raw height data in physical units. The equations
below describe the operation applied to finite float64 image data before display scaling
or colour-map clipping.</p>
<hr/>
{entries}
</body>
"""


_DEFINITIONS_HTML = render_definitions_html()


class _DefinitionsPanel(QWidget):
    """Scrollable reference panel listing processing algorithms."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        palette = _definitions_palette(t)
        inner = QTextEdit()
        inner.setReadOnly(True)
        inner.setHtml(render_definitions_html(t))
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


class _DefinitionsDialog(QDialog):
    """Closeable utility window for processing definitions/help."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ProbeFlow Definitions")
        self.resize(820, 680)
        self.setModal(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._panel = _DefinitionsPanel(t, self)
        lay.addWidget(self._panel)
