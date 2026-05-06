"""Processing definitions/help dialog for ProbeFlow."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QFrame, QScrollArea, QTextEdit, QVBoxLayout, QWidget


_DEFINITIONS_HTML = """
<style>
  body  { font-family: Helvetica, Arial, sans-serif; font-size: 13px;
          color: #cdd6f4; background: transparent; margin: 0; padding: 0; }
  h1    { font-size: 15px; color: #cba6f7; margin: 0 0 12px 0; }
  h2    { font-size: 13px; color: #89b4fa; margin: 18px 0 2px 0; }
  .sub  { font-size: 11px; color: #a6adc8; font-style: italic; margin: 0 0 4px 0; }
  p     { margin: 3px 0 6px 0; line-height: 1.45; }
  .kw   { color: #f38ba8; font-family: monospace; }
  .param{ color: #a6e3a1; font-family: monospace; }
  .note { color: #fab387; }
  hr    { border: none; border-top: 1px solid #45475a; margin: 14px 0; }
</style>
<body>
<h1>Processing Algorithm Reference</h1>
<p>Each step transforms the raw height data. Steps are applied in the order
listed in the viewer's processing panel. All functions operate on
float64 arrays in physical metres — no display-unit clipping involved.</p>
<hr/>

<h2>remove_bad_lines</h2>
<p class="sub">Params: <span class="param">method</span> = step | mad,
<span class="param">polarity</span> = bright | dark,
<span class="param">threshold_mad</span>,
<span class="param">min_segment_length_px</span>,
<span class="param">max_adjacent_bad_lines</span></p>
<p><b>step:</b> Compares each fast-scan row with neighbouring rows, then finds
paired positive/negative jumps along the row.  Only the segment between the
jumps is corrected.</p>
<p><b>mad:</b> Compares each fast-scan row with neighbouring rows and flags
contiguous outlier segments in the row residual.  This is more direct for
plateau-like partial defects.</p>
<p>Both methods repair a detected segment from local neighbouring scan lines;
pixels outside detected segments remain unchanged.  Preview detection in the
viewer is non-destructive.</p>

<h2>Bad scan-line segment</h2>
<p>A short damaged part of a fast-scan line. ProbeFlow corrects the segment
only, not the whole row or column.</p>
<h2>Threshold</h2>
<p>Detection sensitivity. A higher value detects fewer, more obvious artifacts.
A lower value detects more candidate artifacts. It is not a pixel length.</p>
<h2>Minimum segment length (px)</h2>
<p>The shortest run of neighbouring pixels along the fast-scan direction that
can be treated as a bad segment.</p>
<h2>Maximum adjacent bad lines</h2>
<p>The largest number of neighbouring scan lines that ProbeFlow will attempt to
repair as a local bad-line artifact. Broader damaged regions are skipped
because local interpolation becomes unreliable.</p>
<h2>Bright bad segment</h2>
<p>A segment that is higher or brighter than nearby scan lines.</p>
<h2>Dark bad segment</h2>
<p>A segment that is lower or darker than nearby scan lines.</p>
<h2>Preview detection</h2>
<p>Shows candidate bad segments without modifying the image.</p>
<h2>Apply correction</h2>
<p>Repairs the currently detected and accepted bad segments; skipped unsafe
groups remain unchanged.</p>

<hr/>

<h2>align_rows</h2>
<p class="sub">Params: <span class="param">method</span> = median | mean | linear</p>
<p>Removes per-row DC offsets — the most common first step for raw STM data,
where each scan line has an independent height datum due to thermal drift or
tip jumps between lines.</p>
<p><b>median:</b> Subtracts each row's median.  Robust to tip crashes and
outlier pixels within a row.  <b>mean:</b> Subtracts each row's mean — faster
but sensitive to outliers.  <b>linear:</b> Fits and subtracts a first-order
polynomial (slope + offset) per row, correcting both offset and tilt within
each scan line.</p>

<hr/>

<h2>STM Background</h2>
<p class="sub">Params: <span class="param">fit_region</span>,
<span class="param">line_statistic</span>, <span class="param">model</span>,
<span class="param">linear_x_first</span>, <span class="param">blur_length</span>,
<span class="param">jump_threshold</span></p>
<p>Estimates one background value per fast-scan line, fits or smooths that
scan-line profile, then subtracts the fitted background from the whole image.
This is distinct from ROI-scoped processing: the fit region determines where
the background is estimated, but subtraction is applied to the full image.</p>
<h2>Scan-line profile</h2>
<p>The one-dimensional background estimate, one value per image row.</p>
<h2>Line statistic</h2>
<p>The row value used for the scan-line profile. Median is robust against
adsorbates, pits, and spikes; mean follows all selected pixels.</p>
<h2>Linear fit in x first</h2>
<p>Optionally fits and removes a straight x-direction slope from each scan line
before estimating the y-direction background profile.</p>
<h2>Piezo creep</h2>
<p>A future nonlinear background model for slow scanner relaxation, based on a
logarithmic creep-like curve. It is not exposed in the first ProbeFlow STM
Background dialog until robust constrained fitting is available.</p>
<h2>Piezo creep + x^2 / x^3</h2>
<p>Piezo creep models with additional polynomial terms. These are useful for
more complex slow-scan drift, but should be previewed carefully once enabled.</p>
<h2>Sqrt creep</h2>
<p>A future nonlinear background model based on a square-root creep-like curve.</p>
<h2>Low-pass background</h2>
<p>Smooths the scan-line profile using the blur length. Larger blur lengths
produce a slower, smoother background.</p>
<h2>Line-by-line background</h2>
<p>Uses the raw scan-line profile directly as the background. This is
aggressive and should be previewed carefully.</p>
<h2>Fit region</h2>
<p>Whole image uses all finite pixels. Active ROI uses only the selected area
ROI to estimate the profile, then applies the subtraction to the full image.</p>
<h2>Preview background</h2>
<p>Shows the fitted background image without modifying the data.</p>
<h2>Preview corrected image</h2>
<p>Shows the proposed corrected image before applying the processing step.</p>

<hr/>

<h2>smooth (gaussian_smooth)</h2>
<p class="sub">Params: <span class="param">sigma_px</span> (default 1.0)</p>
<p>Isotropic 2-D Gaussian blur.  NaN pixels are handled by weighted
normalisation (a NaN never propagates into its neighbours).  Typical STM
values: 0.5–2&nbsp;px.  Equivalent to ImageJ's Gaussian blur on float data.</p>

<hr/>

<h2>gaussian_high_pass</h2>
<p class="sub">Params: <span class="param">sigma_px</span> (default 8.0)</p>
<p>Subtracts a Gaussian-blurred version of the image from itself, retaining
only high-spatial-frequency detail.  Output = original &minus; blur(original).
Equivalent to the ImageJ "Highpass" plugin (M.&nbsp;Schmid): the ImageJ version
adds an integer offset for byte/short images; for float data the offset is zero,
so the algorithms are identical.</p>

<hr/>

<h2>fft_soft_border</h2>
<p class="sub">Params: <span class="param">mode</span> low_pass|high_pass,
<span class="param">cutoff</span> [0–1], <span class="param">border_frac</span></p>
<p>FFT-based frequency filter with a Tukey-tapered border.  Before
transforming, pixels within <span class="param">border_frac</span> of any edge
are smoothly ramped to the image mean, eliminating the wrap-around
discontinuity that causes ringing artefacts in DFT-based filters.  After the
inverse FFT, the taper is compensated so the image interior is preserved.
The <span class="kw">low_pass</span> mode keeps frequencies inside a radial
cutoff (fraction of Nyquist); <span class="kw">high_pass</span> keeps
frequencies outside.</p>
<p class="note">The name is inherited from the ImageJ FFT_Soft_Border plugin
(M.&nbsp;Schmid), but the operations differ: the ImageJ plugin computed only
the forward FFT spectrum for display.  This implementation is a complete
forward+filter+inverse pipeline returning a filtered image.</p>

<hr/>

<h2>fourier_filter</h2>
<p class="sub">Params: <span class="param">mode</span> low_pass|high_pass,
<span class="param">cutoff</span>, <span class="param">window</span> hanning|hamming|none</p>
<p>Global radial FFT filter without border compensation.  A 2-D Hanning (or
Hamming) window is applied before transforming to reduce edge discontinuities.
The frequency-domain filter is a hard circular cutoff.  The mean is preserved
for low-pass; removed for high-pass.</p>

<hr/>

<h2>periodic_notch_filter</h2>
<p class="sub">Params: <span class="param">peaks</span> list of (dx,dy),
<span class="param">radius_px</span></p>
<p>Suppresses selected periodic FFT peaks and their Hermitian conjugates using
Gaussian notches.  Peaks are specified as integer pixel offsets from the
centred FFT origin.  Used to remove lattice periodicity from topography so that
defects and adsorbates stand out.</p>
<p class="note">Complementary to (not a port of) the ImageJ Periodic_Filter
(M.&nbsp;Schmid), which <em>extracts</em> the periodic component by convolution
with a lattice-frequency kernel.  Notch removal is the standard workflow for
defect imaging; the ImageJ bandpass extraction is useful for lattice
characterisation.</p>

<hr/>

<h2>linear_undistort</h2>
<p class="sub">Params: <span class="param">shear_x</span> (px),
<span class="param">scale_y</span></p>
<p>Affine drift/creep correction.  <span class="param">shear_x</span> is the
total horizontal pixel drift accumulated over the slow-scan height (positive =
right drift).  <span class="param">scale_y</span> corrects a y/x pixel-size
mismatch.  Each output pixel is bilinearly interpolated from the input.
Equivalent to the ImageJ Linear_Undistort plugin (M.&nbsp;Schmid);
parameterisation differs: ImageJ uses shear angle in degrees and a y/x ratio,
with the conversion: shear_x&nbsp;=&nbsp;tan(angle)&nbsp;&times;&nbsp;scale_y&nbsp;&times;&nbsp;(Ny&minus;1).</p>

<hr/>

<h2>blend_forward_backward</h2>
<p class="sub">Params: <span class="param">weight</span> (default 0.5)</p>
<p>Blends a forward scan plane with a left-right-mirrored backward scan plane.
The backward scan is automatically flipped before blending because it is
recorded right-to-left in the fast-scan direction.  This differs from the
ImageJ Blend_Images plugin (M.&nbsp;Schmid), which does a generic weighted
sum without flipping — the flip is required for correct physical alignment of
STM forward/backward pairs.</p>

<hr/>

<h2>edge_detect</h2>
<p class="sub">Params: <span class="param">method</span> laplacian|log|dog,
<span class="param">sigma</span>, <span class="param">sigma2</span></p>
<p><b>laplacian:</b> Discrete second-derivative operator — enhances sharp
edges and atomic corrugation peaks.  <b>log (Laplacian of Gaussian):</b>
Pre-smooths with a Gaussian of width <span class="param">sigma</span> before
applying the Laplacian; reduces noise sensitivity.  <b>dog (Difference of
Gaussians):</b> Difference between two Gaussians of width
<span class="param">sigma</span> and <span class="param">sigma2</span> — a
band-pass approximation to the LoG, useful for isolating features of a
specific spatial scale.</p>
</body>
"""


class _DefinitionsPanel(QWidget):
    """Scrollable reference panel listing every processing algorithm."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        inner = QTextEdit()
        inner.setReadOnly(True)
        inner.setHtml(_DEFINITIONS_HTML)
        inner.setStyleSheet(f"""
            QTextEdit {{
                background-color: {t.get('bg', '#1e1e2e')};
                border: none;
                padding: 16px;
            }}
        """)
        inner.document().setDefaultStyleSheet(
            f"body {{ color: {t.get('fg', '#cdd6f4')}; }}"
        )

        scroll.setWidget(inner)
        lay.addWidget(scroll)


class _DefinitionsDialog(QDialog):
    """Closeable utility window for processing definitions/help."""

    def __init__(self, t: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ProbeFlow Definitions")
        self.resize(760, 640)
        self.setModal(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._panel = _DefinitionsPanel(t, self)
        lay.addWidget(self._panel)
