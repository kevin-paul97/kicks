"use client";

import { Latex } from "@/components/ui/latex";

/* ------------------------------------------------------------------ */
/*  Helpers                                                             */
/* ------------------------------------------------------------------ */

function Section({
  id,
  number,
  title,
  children,
}: {
  id: string;
  number: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-24">
      <h2 className="text-2xl font-bold tracking-tight mb-4 flex items-center gap-3">
        <span className="text-xs font-mono text-muted-foreground bg-muted px-2 py-1 rounded">
          {String(number).padStart(2, "0")}
        </span>
        {title}
      </h2>
      <div className="space-y-5">{children}</div>
    </section>
  );
}

function Legend({ rows }: { rows: [string, string, string, string][] }) {
  return (
    <div className="overflow-x-auto my-4">
      <table className="w-full text-sm border border-border rounded-lg overflow-hidden">
        <thead>
          <tr className="bg-muted/50 text-left">
            <th className="px-3 py-2 font-semibold">Symbol</th>
            <th className="px-3 py-2 font-semibold">Type</th>
            <th className="px-3 py-2 font-semibold">Description</th>
            <th className="px-3 py-2 font-semibold">Value / Unit</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([sym, type, desc, val], i) => (
            <tr key={i} className="border-t border-border">
              <td className="px-3 py-2 font-mono">
                <Latex>{sym}</Latex>
              </td>
              <td className="px-3 py-2 text-muted-foreground">{type}</td>
              <td className="px-3 py-2">{desc}</td>
              <td className="px-3 py-2 text-muted-foreground font-mono">
                {val}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Eq({ children }: { children: string }) {
  return (
    <div className="rounded-xl border border-border bg-card/60 backdrop-blur px-6 py-5 my-4 overflow-x-auto">
      <Latex block>{children}</Latex>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  TOC entries                                                        */
/* ------------------------------------------------------------------ */

const TOC = [
  { id: "stft", label: "Short-Time Fourier Transform" },
  { id: "mel", label: "Mel Filterbank & Log-Mel Spectrogram" },
  { id: "mel-clamp", label: "BigVGAN Log-Mel Normalization" },
  { id: "lufs", label: "LUFS Loudness Normalization" },
  { id: "biquad", label: "Biquad Filter Mathematics" },
  { id: "envelope", label: "Envelope Extraction" },
  { id: "autocorrelation", label: "Autocorrelation Loop Detection" },
  { id: "cosine-fade", label: "Cosine Fade-Out" },
  { id: "griffinlim", label: "Griffin-LIM Phase Reconstruction" },
  { id: "postprocess", label: "Post-Processing & Summary" },
];

/* ------------------------------------------------------------------ */
/*  Page                                                                */
/* ------------------------------------------------------------------ */

export default function MathsPage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-3xl px-5 py-12 sm:py-20 space-y-14">
        {/* ---- Header ---- */}
        <header className="space-y-4">
          <h1 className="text-4xl sm:text-5xl font-black tracking-tighter bg-gradient-to-r from-amber-400 via-rose-400 to-sky-400 bg-clip-text text-transparent">
            Signal Processing Mathematics
          </h1>
          <p className="text-muted-foreground leading-relaxed max-w-2xl">
            A thorough mathematical derivation of every signal-processing
            operation in the Kicks pipeline &mdash; from the STFT and mel
            filterbank through LUFS loudness normalisation, biquad filtering,
            autocorrelation-based loop detection, cosine fade-out, and Griffin-LIM
            phase reconstruction.
          </p>
        </header>

        {/* ---- Notation ---- */}
        <div className="rounded-xl border border-border bg-card/60 backdrop-blur px-6 py-5 space-y-2 text-sm">
          <p className="font-semibold mb-2">Notation conventions</p>
          <ul className="list-disc list-inside space-y-1 text-muted-foreground">
            <li>
              <Latex>{"x[n]"}</Latex> denotes a discrete-time signal indexed by sample{" "}
              <Latex>{"n"}</Latex>.
            </li>
            <li>
              Bold uppercase for matrices:{" "}
              <Latex>{"\\mathbf{X}, \\mathbf{M}, \\mathbf{F}"}</Latex>.
            </li>
            <li>
              <Latex>{"j = \\sqrt{-1}"}</Latex> is the imaginary unit.
            </li>
            <li>
              <Latex>{"\\mathbb{C}, \\mathbb{R}, \\mathbb{N}"}</Latex> denote
              the complex, real, and natural numbers respectively.
            </li>
            <li>
              <Latex>{"\\operatorname{Re}\\{\\cdot\\}, \\operatorname{Im}\\{\\cdot\\}"}</Latex> denote real and imaginary parts.
            </li>
          </ul>
        </div>

        {/* ---- Table of contents ---- */}
        <nav className="space-y-1">
          <p className="text-xs font-semibold tracking-widest uppercase text-muted-foreground mb-2">
            Contents
          </p>
          <ol className="grid grid-cols-1 sm:grid-cols-2 gap-1 text-sm">
            {TOC.map((t, i) => (
              <li key={t.id}>
                <a
                  href={`#${t.id}`}
                  className="hover:text-primary transition-colors"
                >
                  <span className="text-muted-foreground font-mono mr-2">
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {t.label}
                </a>
              </li>
            ))}
          </ol>
        </nav>

        {/* ================================================================
            SECTION 1 — Short-Time Fourier Transform
        ================================================================ */}
        <Section id="stft" number={1} title="Short-Time Fourier Transform">

          <p className="text-muted-foreground">
            The Short-Time Fourier Transform (STFT) decomposes a discrete-time
            signal into overlapping windowed segments and computes the DFT of
            each segment, producing a time-frequency representation. Kicks uses
            the STFT as the first step of the mel spectrogram computation
            (via BigVGAN&apos;s <Latex>{"\\texttt{mel\\_spectrogram}"}</Latex>)
            and also inside the Griffin-LIM vocoder for iterative phase
            estimation.
          </p>

          <p className="font-medium mt-2">Definition</p>
          <Eq>{
            "X[m, k] = \\sum_{n = -\\infty}^{\\infty} x[n] \\, w[n - mH] \\, e^{-j 2\\pi k n / N}"
          }</Eq>

          <p>
            where <Latex>{"x[n]"}</Latex> is the input signal,{" "}
            <Latex>{"w[n]"}</Latex> is the analysis window,{" "}
            <Latex>{"H"}</Latex> is the hop length (frame stride in samples),
            <Latex>{"N"}</Latex> is the FFT size, and the sum is taken over the
            support of the window (zero outside). In practice the summation
            runs over <Latex>{"n = mH, \\ldots, mH + N - 1"}</Latex> for a
            finite-length window of size <Latex>{"N"}</Latex>.
          </p>

          <p className="font-medium mt-2">Hann window</p>
          <Eq>{
            "w[n] = 0.5\\, \\left[ 1 - \\cos\\!\\left( \\frac{2\\pi n}{N - 1} \\right) \\right], \\qquad n = 0, 1, \\ldots, N - 1"
          }</Eq>

          <p>
            The Hann window tapers smoothly to zero at both endpoints,
            reducing spectral leakage (side-lobe rejection ~ -31 dB compared to
            a rectangular window). The window satisfies the constant-overlap-add
            (COLA) property when <Latex>{"H = N/4"}</Latex> (75% overlap),
            ensuring perfect reconstruction in the absence of modification.
          </p>

          <p className="font-medium mt-2">Overlap ratio</p>
          <Eq>{
            "\\text{Overlap} = 1 - \\frac{H}{N} = 1 - \\frac{256}{1024} = 0.75"
          }</Eq>

          <p>
            With <Latex>{"N = 1024"}</Latex> and <Latex>{"H = 256"}</Latex>,
            each frame overlaps its neighbour by 768 samples. This high overlap
            provides dense time resolution and reduces artefacts from the
            analysis window.
          </p>

          <p className="font-medium mt-2">STFT configuration</p>
          <Legend
            rows={[
              ["N = N_{\\text{FFT}}", "Scalar", "FFT size", "1024"],
              ["H", "Scalar", "Hop length (frame stride)", "256 samples"],
              ["W = N_{\\text{win}}", "Scalar", "Window length", "1024 samples"],
              ["N_{\\text{bins}} = N/2 + 1", "Scalar", "Number of unique frequency bins (onesided)", "513"],
              ["T = \\lfloor L / H \\rfloor", "Scalar", "Number of time frames for length L", "256 (for L=65536)"],
              ["f_k = k \\cdot f_s / N", "Scalar", "Centre frequency of bin k (Hz)", "0, 43.07, 86.13, ... , 22050"],
            ]}
          />

          <p className="font-medium mt-2">Magnitude spectrogram</p>
          <Eq>{
            "|X[m, k]| = \\sqrt{\\operatorname{Re}\\{X[m,k]\\}^2 + \\operatorname{Im}\\{X[m,k]\\}^2 + \\varepsilon}"
          }</Eq>

          <p>
            where <Latex>{"\\varepsilon = 10^{-9}"}</Latex> is a tiny constant
            to ensure numerical stability at zero. BigVGAN uses the magnitude
            spectrogram (not the power spectrogram), which preserves the linear
            amplitude scale. This choice influences the dynamic range of
            subsequent mel-band summation.
          </p>

          <p className="font-medium mt-2">Reflection padding</p>
          <Eq>{
            "x_{\\text{padded}}[n + (N - H)/2] = x[\\;|n|\\;], \\qquad n = -(N-H)/2, \\ldots, -1"
          }</Eq>

          <p>
            Before computing the STFT, BigVGAN pads the signal by reflecting
            <Latex>{"(N - H)/2 = 384"}</Latex> samples at both boundaries.
            This symmetric extension reduces edge artefacts without introducing
            a discontinuity at the signal boundary.
          </p>
        </Section>

        {/* ================================================================
            SECTION 2 — Mel Filterbank & Log-Mel Spectrogram
        ================================================================ */}
        <Section id="mel" number={2} title="Mel Filterbank &amp; Log-Mel Spectrogram">

          <p className="text-muted-foreground">
            The mel scale maps physical frequency (Hz) to a perceptual pitch
            scale. BigVGAN uses the Slaney mel scale (linear below 1 kHz,
            logarithmic above) via librosa. The mel spectrogram is formed by
            multiplying the magnitude STFT by a triangular filterbank whose
            centre frequencies are uniformly spaced on the mel axis.
          </p>

          <p className="font-medium mt-2">Slaney mel scale (Hz to mel)</p>
          <Eq>{
            "\\text{mel}(f) = \\begin{cases} \n" +
            "    \\dfrac{f}{\\;f_{\\text{sp}}\\;} , & f \\leq 1000\\text{ Hz} \\\\[6pt]\n" +
            "    \\text{mel}(1000) + \\dfrac{\\ln(f / 1000)}{\\alpha} , & f > 1000\\text{ Hz}\n" +
            "\\end{cases}"
          }</Eq>

          <Legend
            rows={[
              ["f_{\\text{sp}} = 200/3", "Scalar", "Slope of the linear region", "66.67 mel/Hz"],
              ["\\alpha = \\ln(6.4) / 27", "Scalar", "Log-stepping constant", "~0.0687"],
              ["\\text{mel}(1000)", "Scalar", "Offset at the transition point", "15 mel"],
            ]}
          />

          <p className="font-medium mt-2">HTK mel scale (for reference)</p>
          <Eq>{
            "\\text{mel}_{\\text{HTK}}(f) = 2595 \\cdot \\log_{10}\\left(1 + \\frac{f}{700}\\right)"
          }</Eq>

          <p>
            The Slaney scale (used by BigVGAN) differs from the HTK scale
            below 1 kHz where it is linear, matching the empirical observation
            that pitch perception is approximately linear at low frequencies.
            The HTK scale is used only by torchaudio&apos;s
            <Latex>{"\\texttt{melscale\\_fbanks}"}</Latex> for the Griffin-LIM
            pseudo-inverse.
          </p>

          <p className="font-medium mt-2">Triangular filterbank construction</p>

          <p>
            Let <Latex>{"f_{\\text{min}} = 0"}</Latex> Hz and
            <Latex>{"f_{\\text{max}} = f_s / 2 = 22050"}</Latex> Hz.
            We compute <Latex>{"M + 2"}</Latex> mel-spaced centre frequencies:
          </p>

          <Eq>{
            "\\text{mel}_i = \\text{mel}(f_{\\text{min}}) + \\frac{i}{M+1}\\bigl(\\text{mel}(f_{\\text{max}}) - \\text{mel}(f_{\\text{min}})\\bigr), \\quad i = 0, \\ldots, M+1"
          }</Eq>

          <p>
            These are converted back to Hz via the inverse Slaney mapping to
            give <Latex>{"\\{f_0, f_1, \\ldots, f_{M+1}\\}"}</Latex>.
            The <Latex>{"i"}</Latex>-th triangular filter
            <Latex>{"H_i(f)"}</Latex> for <Latex>{"i = 1, \\ldots, M"}</Latex>
            is:
          </p>

          <Eq>{
            "H_i(f_k) = \\begin{cases}\n" +
            "    0, & f_k \\leq f_{i-1} \\\\[4pt]\n" +
            "    \\dfrac{f_k - f_{i-1}}{f_i - f_{i-1}}, & f_{i-1} < f_k \\leq f_i \\\\[8pt]\n" +
            "    \\dfrac{f_{i+1} - f_k}{f_{i+1} - f_i}, & f_i < f_k \\leq f_{i+1} \\\\[4pt]\n" +
            "    0, & f_k > f_{i+1}\n" +
            "\\end{cases}"
          }</Eq>

          <p>
            where <Latex>{"f_k = k \\cdot f_s / N"}</Latex> for
            <Latex>{"k = 0, \\ldots, N_{\\text{bins}} - 1"}</Latex> are the
            centre frequencies of the STFT bins. With Slaney normalisation,
            each filter is scaled by{" "}
            <Latex>{"2 / (f_{i+1} - f_{i-1})"}</Latex> to achieve approximate
            constant energy per channel.
          </p>

          <p className="font-medium mt-2">Mel filterbank matrix</p>
          <Eq>{
            "\\mathbf{M} \\in \\mathbb{R}^{M \\times N_{\\text{bins}}}, \\quad M_{i,k} = H_i(f_k)"
          }</Eq>

          <Legend
            rows={[
              ["M", "Scalar", "Number of mel bands", "128"],
              ["N_{\\text{bins}} = N/2 + 1", "Scalar", "Number of STFT frequency bins", "513"],
            ]}
          />

          <p className="font-medium mt-2">Mel spectrogram (magnitude-domain)</p>
          <Eq>{
            "\\mathbf{S}_{\\text{mel}}[i, m] = \\sum_{k=0}^{N_{\\text{bins}}-1} M_{i,k} \\, |X[m, k]| \\in \\mathbb{R}^{128 \\times 256}"
          }</Eq>

          <p>
            Each mel band output is a weighted sum of nearby FFT-bin magnitudes.
            Because BigVGAN uses the magnitude spectrogram (rather than the
            power spectrogram), the summation is linear in amplitude, which
            preserves transient onsets better than the squared equivalent.
          </p>

          <p className="font-medium mt-2">Log-amplitude compression</p>
          <Eq>{
            "\\mathbf{S}_{\\text{log}}[i, m] = \\ln\\!\\bigl(\\max(\\mathbf{S}_{\\text{mel}}[i, m],\\; \\epsilon)\\bigr), \\qquad \\epsilon = 10^{-5}"
          }</Eq>

          <p>
            The natural logarithm compresses the wide dynamic range of the
            mel spectrogram (often 60+ dB) into a range suitable for neural
            network processing. The clamp at <Latex>{"\\epsilon = 10^{-5}"}</Latex>
            prevents <Latex>{"\\ln(0)"}</Latex>:
          </p>

          <Legend
            rows={[
              ["\\ln(\\epsilon) = \\ln(10^{-5})", "Scalar", "Silence floor value", "-11.5129"],
              ["\\epsilon", "Scalar", "Magnitude clamping threshold", "10^{-5}"],
              ["\\ln(\\cdot)", "Function", "Natural logarithm", "—"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 3 — BigVGAN Log-Mel Normalization
        ================================================================ */}
        <Section id="mel-clamp" number={3} title="BigVGAN Log-Mel Normalization">

          <p className="text-muted-foreground">
            Before feeding the log-mel spectrogram into the VAE,
            it is clamped and affine-transformed to the unit interval{" "}
            <Latex>{"[0, 1]"}</Latex>. This fixed-bounds normalisation uses
            pre-computed endpoints derived from BigVGAN&apos;s output range.
          </p>

          <p className="font-medium mt-2">Clamping</p>
          <Eq children={"\\tilde{S}_{ij} = \\max\\bigl(\\min(S_{\\text{log}}[i,m],\\, s_{\\max}),\\, s_{\\min}\\bigr)"} />

          <p>
            where <Latex>{"s_{\\min} = \\ln(10^{-5}) = -11.5129"}</Latex> is the
            silence floor (any bin below the clamp threshold maps to this value)
            and <Latex>{"s_{\\max} = 2.5"}</Latex> is a headroom ceiling chosen
            to be above the observed dataset maximum (~2.23 nats).
          </p>

          <p className="font-medium mt-2">Affine map to [0, 1]</p>
          <Eq children={"\\hat{S} = \\frac{\\tilde{S} - s_{\\min}}{s_{\\max} - s_{\\min}}"}/>

          <p>
            The denominator is the width of the valid range:
          </p>

          <Eq>{
            "\\Delta s = s_{\\max} - s_{\\min} = 2.5 - (-11.5129) = 14.0129"
          }</Eq>

          <p>
            This maps the BigVGAN log-mel output range approximately
            <Latex>{"[-11.5, 2.5] \\to [0, 1]"}</Latex>. The normalised
            spectrogram <Latex>{"\\hat{\\mathbf{S}}"}</Latex> is what the VAE
            encoder receives as input and the decoder is trained to reproduce.
          </p>

          <p className="font-medium mt-2">Denormalisation (inverse transform)</p>
          <Eq>{
            "\\mathbf{S}_{\\text{log}} = \\hat{\\mathbf{S}} \\cdot \\Delta s + s_{\\min} = \\hat{\\mathbf{S}} \\cdot 14.0129 - 11.5129"
          }</Eq>

          <p>
            The <Latex>{"\\texttt{denormalize}"}</Latex> static method on
            <Latex>{"\\texttt{KickDataset}"}</Latex> applies this inverse affine
            transform, restoring the log-mel representation before passing it to
            the vocoder.
          </p>

          <Legend
            rows={[
              ["s_{\\min}", "Scalar", "Silence floor = ln(1e-5)", "-11.5129"],
              ["s_{\\max}", "Scalar", "Headroom ceiling (above max ~2.23)", "2.5"],
              ["\\Delta s", "Scalar", "Range width", "14.0129"],
              ["\\hat{S}_{ij}", "Scalar", "Normalised bin (VAE input)", "[0, 1]"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 4 — LUFS Loudness Normalization
        ================================================================ */}
        <Section id="lufs" number={4} title="LUFS Loudness Normalization">

          <p className="text-muted-foreground">
            Before the STFT, each audio sample is normalised to a target
            integrated loudness of <Latex>{"-14"}</Latex> LUFS (Loudness Units
            relative to Full Scale). This ensures consistent perceptual
            loudness across the dataset, which is critical for stable VAE
            training. The implementation uses the
            <Latex>{"\\texttt{pyloudnorm}"}</Latex> library, which follows
            ITU-R BS.1770-4.
          </p>

          <p className="font-medium mt-2">Stage 1: K-weighting pre-filter</p>
          <p>
            The input signal is filtered through two cascaded filters:
          </p>
          <Eq>{
            "H_{\\text{shelf}}(z) = \\frac{1 + \\mu z^{-1}}{1 - \\lambda z^{-1}}, \\qquad\n" +
            "H_{\\text{high}}(z) = \\frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}"
          }</Eq>

          <p>
            The shelf filter is a first-order high-shelf with a gain of +4 dB
            at high frequencies (approximating the ear&apos;s increased
            sensitivity in the 2&ndash;4 kHz range), and the second stage is
            a second-order highpass Butterworth at about 38 Hz (removing
            subsonic content that should not contribute to perceived loudness).
            Let <Latex>{"z_i[n]"}</Latex> be the filtered output for channel
            <Latex>{"i"}</Latex>.
          </p>

          <p className="font-medium mt-2">Stage 2: Mean-square summation with channel weighting</p>
          <Eq>{
            "L_{\\text{k}} = -0.691 + 10 \\cdot \\log_{10}\\left( \\sum_{i=1}^{C} G_i \\cdot \\frac{1}{N} \\sum_{n=1}^{N} z_i^2[n] \\right)"
          }</Eq>

          <Legend
            rows={[
              ["G_i", "Scalar", "Channel weight for channel i", "1.0 (L/R), 1.0 (C), 1.41 (LFE)"],
              ["z_i[n]", "Vector", "K-weighted filtered signal, channel i", "\\mathbb{R}^N"],
              ["C", "Scalar", "Number of channels", "1 (mono) or 2"],
              ["N", "Scalar", "Number of samples", "65 536"],
              ["-0.691", "Scalar", "Alignment constant (dB)", "—"],
            ]}
          />

          <p className="font-medium mt-2">Stage 3: Gating (absolute + relative)</p>
          <p>
            Only audio segments above a gating threshold contribute to the
            integrated loudness measurement. The ITU standard defines a
            two-stage gate:
          </p>

          <Eq>{
            "\\text{Gate 1:}\\quad \\text{block } t \\text{ if } L_t < -70\\text{ LUFS (absolute gate)}"
          }</Eq>
          <Eq>{
            "\\text{Gate 2:}\\quad \\text{block } t \\text{ if } L_t < L_{\\text{rel}} + 10\\text{ (relative gate)}"
          }</Eq>

          <p>
            where <Latex>{"L_t"}</Latex> is the loudness of the
            <Latex>{"t"}</Latex>-th block (typically 400 ms) and
            <Latex>{"L_{\\text{rel}}"}</Latex> is the loudness measured
            after applying Gate 1. Blocks below the relative threshold
            (<Latex>{"L_{\\text{rel}} - 10"}</Latex> LUFS) are excluded
            from the final integration.
          </p>

          <p className="font-medium mt-2">Stage 4: Gain adjustment</p>
          <Eq>{
            "x_{\\text{norm}}[n] = x[n] \\cdot 10^{\\,(L_{\\text{target}} - L_{\\text{measured}})\\,/\\,20}"
          }</Eq>

          <Legend
            rows={[
              ["L_{\\text{target}}", "Scalar", "Target integrated loudness", "-14 LUFS"],
              ["L_{\\text{measured}}", "Scalar", "Measured loudness of input", "dB LUFS"],
            ]}
          />

          <p className="font-medium mt-2">Stage 5: Hard clipping</p>
          <Eq>{
            "x_{\\text{final}}[n] = \\max(\\min(x_{\\text{norm}}[n],\\, 1.0),\\, -1.0)"
          }</Eq>

          <p>
            After gain adjustment, samples that would exceed full-scale
            are clipped to <Latex>{"\\pm 1.0"}</Latex>. The
            <Latex>{"\\texttt{pyloudnorm}"}</Latex> library issues a warning
            when clipping occurs, but the process is generally safe when
            samples are normalised from a reasonable listening level to
            <Latex>{"-14"}</Latex> LUFS.
          </p>
        </Section>

        {/* ================================================================
            SECTION 5 — Biquad Filter Mathematics
        ================================================================ */}
        <Section id="biquad" number={5} title="Biquad Filter Mathematics">

          <p className="text-muted-foreground">
            The strip command uses cascaded biquad filters for two purposes:
            isolating the low-frequency kick envelope (lowpass at 200 Hz) and
            detecting high-frequency onsets from hi-hat or snare hits (highpass
            at 2000 Hz). Both filters are designed as Butterworth and applied
            through torchaudio&apos;s biquad functions.
          </p>

          <p className="font-medium mt-2">General biquad transfer function</p>
          <Eq>{
            "H(z) = \\frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}"
          }</Eq>

          <p>
            In the time domain, this corresponds to the linear difference
            equation:
          </p>
          <Eq>{
            "a_0 \\, y[n] = b_0 \\, x[n] + b_1 \\, x[n-1] + b_2 \\, x[n-2] - a_1 \\, y[n-1] - a_2 \\, y[n-2]"
          }</Eq>

          <p>
            Normalising so that <Latex>{"a_0 = 1"}</Latex>:
          </p>
          <Eq>{
            "y[n] = b_0 \\, x[n] + b_1 \\, x[n-1] + b_2 \\, x[n-2] - a_1 \\, y[n-1] - a_2 \\, y[n-2]"
          }</Eq>

          <p className="font-medium mt-2">Butterworth lowpass design</p>

          <p>
            A 2nd-order Butterworth lowpass filter with cutoff
            <Latex>{"f_c"}</Latex> Hz is designed via the bilinear transform.
            The continuous-time prototype is:
          </p>

          <Eq>{
            "|H_a(j\\Omega)|^2 = \\frac{1}{1 + (\\Omega / \\Omega_c)^{2n}}, \\quad n=2"
          }</Eq>

          <p>
            where <Latex>{"\\Omega_c = 2\\pi f_c"}</Latex> is the analog cutoff
            frequency. Pre-warping for the bilinear transform:
          </p>

          <Eq>{
            "\\omega_c = 2\\pi \\frac{f_c}{f_s}, \\quad\n" +
            "\\Omega_a = \\frac{2}{T} \\tan\\!\\left(\\frac{\\omega_c}{2}\\right)"
          }</Eq>

          <p>
            The pre-warped analog frequency <Latex>{"\\Omega_a"}</Latex> is used
            to compute the discrete-time coefficients{" "}
            <Latex>{"(b_0, b_1, b_2, a_1, a_2)"}</Latex> via the bilinear
            transform <Latex>{"s \\leftarrow (2/T)(1 - z^{-1})/(1 + z^{-1})"}</Latex>.
          </p>

          <p className="font-medium mt-2">Cascaded application</p>
          <Eq>{
            "H_{\\text{cascade}}(z) = H(z)^2"
          }</Eq>

          <p>
            The filter is applied twice sequentially (order=2 in the code),
            meaning the output of the first biquad section feeds the input of
            the second identical section. This produces an effective 4th-order
            rolloff:
          </p>

          <Eq>{
            "|H_{\\text{cascade}}(j\\omega)|_{\\text{dB}} = 2 \\times 20 \\log_{10}|H(e^{j\\omega})| \\approx -40 \\text{ dB/decade}"
          }</Eq>

          <p>
            or equivalently at the asymptotic slope of -24 dB/octave. The 200 Hz
            lowpass aggressively removes mid/high-frequency content, leaving
            only the kick&apos;s fundamental and first few harmonics. The 2000 Hz
            highpass does the opposite, passing only the high-frequency
            components typical of hi-hat and snare hits.
          </p>

          <Legend
            rows={[
              ["(b_0, b_1, b_2)", "Scalars", "Feed-forward coefficients", "Depends on f_c, f_s, Q"],
              ["(a_0, a_1, a_2)", "Scalars", "Feed-back coefficients", "a_0 = 1 normalised"],
              ["f_c", "Scalar", "Cutoff frequency", "200 Hz or 2000 Hz"],
              ["f_s", "Scalar", "Sample rate", "44100 Hz"],
              ["n = order", "Scalar", "Number of cascaded passes", "2"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 6 — Envelope Extraction
        ================================================================ */}
        <Section id="envelope" number={6} title="Envelope Extraction">

          <p className="text-muted-foreground">
            After filtering, the signal envelope is extracted via
            full-wave rectification followed by a moving-average smoother.
            This envelope is used for onset/decay detection in the kick
            stripping pipeline.
          </p>

          <p className="font-medium mt-2">Full-wave rectification</p>
          <Eq>{
            "e_{\\text{rect}}[n] = |y[n]|"
          }</Eq>

          <p>
            The absolute value discards phase information, converting the
            oscillatory filtered signal into a unipolar representation whose
            amplitude follows the signal&apos;s instantaneous energy.
          </p>

          <p className="font-medium mt-2">Moving-average smoothing (boxcar filter)</p>
          <Eq>{
            "e[n] = \\frac{1}{W} \\sum_{i=0}^{W-1} e_{\\text{rect}}[n - i]"
          }</Eq>

          <Legend
            rows={[
              ["W = \\lfloor f_s \\cdot \\tau / 1000 \\rfloor", "Scalar", "Window length in samples", "220 (for 5 ms at 44.1 kHz)"],
              ["\\tau", "Scalar", "Smoothing time constant", "5 ms"],
              ["f_s", "Scalar", "Sample rate", "44100 Hz"],
            ]}
          />

          <p>
            This is a finite-impulse-response (FIR) filter with a rectangular
            kernel of length <Latex>{"W"}</Latex>:
          </p>

          <Eq>{
            "e = e_{\\text{rect}} \\ast h, \\quad h[n] = \\begin{cases} 1/W, & 0 \\leq n < W \\\\ 0, & \\text{otherwise} \\end{cases}"
          }</Eq>

          <p>
            The moving average is implemented via
            <Latex>{"\\texttt{np.convolve(env, kernel, mode='same')}"}</Latex>,
            which computes the full discrete convolution and then crops to the
            original signal length:
          </p>

          <Eq>{
            "(e_{\\text{rect}} \\ast h)[n] = \\sum_{m=-\\infty}^{\\infty} e_{\\text{rect}}[m] \\, h[n - m]"
          }</Eq>

          <p>
            The frequency response of the moving-average filter is a
            Dirichlet kernel (periodic sinc):
          </p>

          <Eq>{
            "H(e^{j\\omega}) = \\frac{1}{W} \\cdot e^{-j\\omega (W-1)/2} \\cdot \\frac{\\sin(\\omega W / 2)}{\\sin(\\omega / 2)}"
          }</Eq>

          <p>
            The first null occurs at <Latex>{"f = f_s / W \\approx 200"}</Latex> Hz
            for <Latex>{"W = 220"}</Latex>, so the moving average strongly
            attenuates fluctuations above ~200 Hz while preserving the slow
            amplitude modulation of the kick envelope.
          </p>
        </Section>

        {/* ================================================================
            SECTION 7 — Autocorrelation Loop Detection
        ================================================================ */}
        <Section id="autocorrelation" number={7} title="Autocorrelation Loop Detection">

          <p className="text-muted-foreground">
            To distinguish single kicks from cyclical drum loops, the strip
            command computes the normalised autocorrelation of the RMS
            energy envelope. A single kick has a monotonically decaying
            envelope whose autocorrelation falls off; a loop has periodic
            energy peaks producing a strong autocorrelation at the loop
            period.
          </p>

          <p className="font-medium mt-2">RMS energy envelope</p>
          <p>
            The signal is divided into non-overlapping frames of length
            <Latex>{"L = \\lfloor f_s \\cdot \\tau_{\\text{frame}} / 1000 \\rfloor"}</Latex>:
          </p>

          <Eq>{
            "E[m] = \\sqrt{ \\frac{1}{L} \\sum_{n = mL}^{(m+1)L - 1} x^2[n] }, \\qquad m = 0, 1, \\ldots, M-1"
          }</Eq>

          <Legend
            rows={[
              ["\\tau_{\\text{frame}}", "Scalar", "Frame duration", "10 ms"],
              ["L", "Scalar", "Frame length in samples", "441 (for 10 ms at 44.1 kHz)"],
              ["M = \\lfloor N / L \\rfloor", "Scalar", "Number of frames", "148 (for N = 65536)"],
              ["E[m]", "Vector", "RMS envelope", "\\mathbb{R}^M"],
            ]}
          />

          <p className="font-medium mt-2">Mean-subtraction and normalisation</p>
          <Eq>{
            "\\tilde{E}[m] = E[m] - \\bar{E}, \\quad \\bar{E} = \\frac{1}{M} \\sum_{m=0}^{M-1} E[m]"
          }</Eq>
          <Eq>{
            "\\sigma_E^2 = \\sum_{m=0}^{M-1} \\tilde{E}[m]^2"
          }</Eq>

          <p className="font-medium mt-2">Autocorrelation via the Wiener-Khinchin theorem</p>

          <p>
            The autocorrelation can be computed directly in the time domain as:
          </p>

          <Eq children={"R[k] = \\frac{1}{M} \\sum_{m=0}^{M-1-k} \\tilde{E}[m] \\, \\tilde{E}[m + k], \\qquad k = 0, \\ldots, M-1"} />

          <p>
            For computational efficiency, the code uses the Wiener-Khinchin
            theorem, which states that the autocorrelation is the inverse
            Fourier transform of the power spectral density:
          </p>

          <Eq>{
            "R[k] = \\mathcal{F}^{-1}\\bigl( |\\mathcal{F}(\\tilde{E})|^2 \\bigr)[k]"
          }</Eq>

          <p>
            where <Latex>{"\\mathcal{F}"}</Latex> denotes the DFT. The
            implementation:
          </p>

          <Eq>{
            "N_{\\text{FFT}} = 2^{\\lceil \\log_2(2M) \\rceil} \\quad\\text{(next power of 2)}"
          }</Eq>
          <Eq>{
            "\\mathcal{E}[\\ell] = \\sum_{m=0}^{M-1} \\tilde{E}[m] \\, e^{-j 2\\pi \\ell m / N_{\\text{FFT}}} \\quad\\text{(zero-padded RFFT)}"
          }</Eq>
          <Eq>{
            "R[k] = \\frac{1}{N_{\\text{FFT}}} \\sum_{\\ell=0}^{N_{\\text{FFT}}-1} |\\mathcal{E}[\\ell]|^2 \\, e^{j 2\\pi \\ell k / N_{\\text{FFT}}} \\quad\\text{(IRFFT, truncated to M)}"
          }</Eq>

          <p className="font-medium mt-2">Normalised autocorrelation</p>
          <Eq>{
            "\\hat{R}[k] = \\frac{R[k]}{R[0]} = \\frac{R[k]}{\\sigma_E^2}"
          }</Eq>

          <p>
            This ensures <Latex>{"\\hat{R}[0] = 1"}</Latex> and
            <Latex>{"\\hat{R}[k] \\in [-1, 1]"}</Latex>. For a purely stochastic
            signal, <Latex>{"\\hat{R}[k] \\to 0"}</Latex> as
            <Latex>{"k \\to \\infty"}</Latex>.
          </p>

          <p className="font-medium mt-2">Loop decision rule</p>
          <Eq>{
            "k_{\\min} = \\left\\lceil \\frac{\\tau_{\\min}}{\\tau_{\\text{frame}}} \\right\\rceil = \\left\\lceil \\frac{100\\text{ ms}}{10\\text{ ms}} \\right\\rceil = 10"
          }</Eq>
          <Eq>{
            "\\text{IsLoop} = \\begin{cases}\n" +
            "    \\text{true}, & \\max\\{\\,\\hat{R}[k] \\mid k \\geq k_{\\min}\\,\\} > \\theta \\\\\n" +
            "    \\text{false}, & \\text{otherwise}\n" +
            "\\end{cases}"
          }</Eq>

          <Legend
            rows={[
              ["\\tau_{\\min}", "Scalar", "Minimum lag to examine (avoids kick body)", "100 ms"],
              ["k_{\\min}", "Scalar", "Minimum lag in frames", "10"],
              ["\\theta", "Scalar", "Autocorrelation threshold", "0.3"],
            ]}
          />

          <p className="font-medium mt-2">Why this works</p>
          <p>
            A single kick&apos;s envelope <Latex>{"E[m]"}</Latex> is
            approximately a one-sided exponential decay:
          </p>
          <Eq>{
            "E_{\\text{kick}}[m] \\approx A \\, e^{-\\alpha m}, \\quad m \\geq 0"
          }</Eq>

          <p>
            whose autocorrelation is:
          </p>
          <Eq>{
            "\\hat{R}_{\\text{kick}}[k] = e^{-\\alpha k}"
          }</Eq>

          <p>
            This decays monotonically from 1 and stays below threshold for
            <Latex>{"k \\geq k_{\\min}"}</Latex> when <Latex>{"\\alpha > 0"}</Latex>.
            A loop, by contrast, has a periodic envelope:
          </p>
          <Eq>{
            "E_{\\text{loop}}[m] \\approx E_{\\text{kick}}[m \\bmod P]"
          }</Eq>

          <p>
            where <Latex>{"P"}</Latex> is the loop period in frames.
            Its autocorrelation peaks at <Latex>{"k = P, 2P, \\ldots"}</Latex>,
            producing values well above 0.3.
          </p>
        </Section>

        {/* ================================================================
            SECTION 8 — Cosine Fade-Out
        ================================================================ */}
        <Section id="cosine-fade" number={8} title="Cosine Fade-Out">

          <p className="text-muted-foreground">
            After the kick endpoint is detected, a half-cosine fade is applied
            to smoothly transition the tail to silence, avoiding an audible
            click from an abrupt truncation.
          </p>

          <p className="font-medium mt-2">Fade window</p>
          <Eq>{
            "\\text{fade}[n] = \\frac{1}{2}\\left(1 + \\cos\\!\\left(\\pi \\, \\frac{n}{L}\\right)\\right), \\qquad n = 0, 1, \\ldots, L-1"
          }</Eq>

          <Legend
            rows={[
              ["L = \\lfloor f_s \\cdot \\tau_{\\text{fade}} / 1000 \\rfloor", "Scalar", "Fade length in samples", "441 (for 10 ms at 44.1 kHz)"],
              ["\\tau_{\\text{fade}}", "Scalar", "Fade duration", "10 ms"],
              ["n", "Scalar", "Sample index within fade region", "0, ..., L-1"],
            ]}
          />

          <p className="font-medium mt-2">Properties of the half-cosine window</p>

          <p>
            The fade window is a raised cosine over half a period. Its
            endpoints have zero derivative:
          </p>

          <Eq>{
            "\\text{fade}[0] = 1, \\quad \\text{fade}[L-1] = 0"
          }</Eq>

          <Eq>{
            "\\left. \\frac{d\\,\\text{fade}}{dn} \\right|_{n=0} = -\\frac{\\pi}{2L}\\sin(0) = 0, \\quad\n" +
            "\\left. \\frac{d\\,\\text{fade}}{dn} \\right|_{n=L-1} = -\\frac{\\pi}{2L}\\sin\\!\\left(\\pi\\frac{L-1}{L}\\right) \\approx 0"
          }</Eq>

          <p>
            The zero derivatives at both ends ensure a C<Latex>{}^1</Latex>
            continuous transition, avoiding audible discontinuities.
          </p>

          <p className="font-medium mt-2">Application</p>
          <Eq>{
            "y[n] = \\begin{cases}\n" +
            "    x[n], & n < N_{\\text{start}} \\\\[4pt]\n" +
            "    x[n] \\cdot \\text{fade}[n - N_{\\text{start}}], & N_{\\text{start}} \\leq n < N_{\\text{end}} \\\\[4pt]\n" +
            "    0, & n \\geq N_{\\text{end}}\n" +
            "\\end{cases}"
          }</Eq>

          <Legend
            rows={[
              ["N_{\\text{start}} = N_{\\text{end}} - L", "Scalar", "Start of fade region", "end - fade_samples"],
              ["N_{\\text{end}}", "Scalar", "End of fade (= kick endpoint)", "Detected decay sample"],
            ]}
          />

          <p className="font-medium mt-2">Relationship to the Hann window</p>
          <p>
            The half-cosine fade is exactly one half of a Hann window. A full
            Hann window of length <Latex>{"2L"}</Latex> is:
          </p>
          <Eq>{
            "w_{\\text{Hann}}[n] = 0.5\\left(1 - \\cos\\!\\left(\\frac{2\\pi n}{2L - 1}\\right)\\right)"
          }</Eq>

          <p>
            For <Latex>{"n = 0, \\ldots, L-1"}</Latex> with
            <Latex>{"2L \\gg 1"}</Latex>:
          </p>
          <Eq>{
            "w_{\\text{Hann}}[n] \\approx 0.5\\left(1 - \\cos\\!\\left(\\frac{\\pi n}{L}\\right)\\right) = 0.5\\left(1 + \\cos\\!\\left(\\pi\\frac{(L - n)}{L}\\right)\\right)"
          }</Eq>

          <p>
            So the fade window used is the second (descending) half of a
            Hann window, time-reversed — equivalent to a quarter-cycle of a
            cosine squared.
          </p>
        </Section>

        {/* ================================================================
            SECTION 9 — Griffin-LIM Phase Reconstruction
        ================================================================ */}
        <Section id="griffinlim" number={9} title="Griffin-LIM Phase Reconstruction">

          <p className="text-muted-foreground">
            The VAE decoder produces a normalised log-mel spectrogram. To
            convert this back to audio, we need both magnitude and phase.
            The BigVGAN neural vocoder learns phase implicitly. The Griffin-LIM
            vocoder uses an iterative algorithm to estimate phase from
            magnitude alone.
          </p>

          <p className="font-medium mt-2">Step 1: Log-mel to linear magnitude</p>

          <p>
            First, the normalised spectrogram is denormalised and exponentiated:
          </p>

          <Eq>{
            "\\mathbf{S}_{\\text{mel}} = \\exp(\\hat{\\mathbf{S}} \\cdot \\Delta s + s_{\\min}) \\in \\mathbb{R}^{128 \\times T}"
          }</Eq>

          <p>
            This gives the mel-band magnitudes (non-log). The inverse mel
            transform must then recover a linear-frequency magnitude spectrogram
            <Latex>{"\\mathbf{X} \\in \\mathbb{R}^{513 \\times T}"}</Latex>
            such that:
          </p>

          <Eq>{
            "\\mathbf{S}_{\\text{mel}} \\approx \\mathbf{M}^{\\top} \\, \\mathbf{X}"
          }</Eq>

          <p>
            where <Latex>{"\\mathbf{M} \\in \\mathbb{R}^{513 \\times 128}"}</Latex>
            is the mel filterbank matrix used by torchaudio
            (<Latex>{"\\texttt{melscale\\_fbanks}"}</Latex> with HTK scale).
            This is an underdetermined system (513 unknowns per time frame,
            128 equations), so we seek the least-squares solution.
          </p>

          <p className="font-medium mt-2">Step 2: Pseudo-inverse of the mel filterbank</p>

          <p>
            The Moore-Penrose pseudo-inverse of <Latex>{"\\mathbf{M}^{\\top}"}</Latex>
            (size 128 x 513) is computed once at initialisation:
          </p>

          <Eq>{
            "\\mathbf{M}^{\\dagger} = \\operatorname{pinv}(\\mathbf{M}^{\\top}) \\in \\mathbb{R}^{513 \\times 128}"
          }</Eq>

          <p>
            The pseudo-inverse is defined via the singular value decomposition.
            Let <Latex>{"\\mathbf{A} = \\mathbf{M}^{\\top} \\in \\mathbb{R}^{128 \\times 513}"}</Latex>:
          </p>

          <Eq>{
            "\\mathbf{A} = \\mathbf{U} \\, \\boldsymbol{\\Sigma} \\, \\mathbf{V}^{\\top}"
          }</Eq>

          <p>
            where <Latex>{"\\mathbf{U} \\in \\mathbb{R}^{128 \\times 128}"}</Latex>,
            <Latex>{"\\boldsymbol{\\Sigma} \\in \\mathbb{R}^{128 \\times 513}"}</Latex>
            is diagonal, and{" "}
            <Latex>{"\\mathbf{V} \\in \\mathbb{R}^{513 \\times 513}"}</Latex>.
            Then:
          </p>

          <Eq>{
            "\\mathbf{A}^{\\dagger} = \\mathbf{V} \\, \\boldsymbol{\\Sigma}^{\\dagger} \\, \\mathbf{U}^{\\top} \\in \\mathbb{R}^{513 \\times 128}"
          }</Eq>

          <p>
            where <Latex>{"\\boldsymbol{\\Sigma}^{\\dagger}"}</Latex> replaces
            each non-zero singular value <Latex>{"\\sigma_i"}</Latex> with
            <Latex>{"1 / \\sigma_i"}</Latex>. The estimated linear magnitude
            spectrogram is then:
          </p>

          <Eq>{
            "\\hat{\\mathbf{X}} = \\mathbf{M}^{\\dagger} \\, \\mathbf{S}_{\\text{mel}} \\in \\mathbb{R}^{513 \\times T}"
          }</Eq>

          <p>
            This solves the least-squares problem:
          </p>

          <Eq>{
            "\\hat{\\mathbf{X}} = \\arg\\min_{\\mathbf{X}} \\; \\|\\mathbf{M}^{\\top} \\mathbf{X} - \\mathbf{S}_{\\text{mel}}\\|_F^2"
          }</Eq>

          <p>
            After applying the pseudo-inverse, the linear magnitude is clamped
            non-negative and a small noise floor is added to aid convergence:
          </p>

          <Eq>{
            "\\tilde{X}[k, m] = \\max(\\hat{X}[k, m],\\, 0) + 10^{-4}"
          }</Eq>

          <p className="font-medium mt-2">Step 3: Griffin-LIM iterative phase estimation</p>

          <p>
            We have the magnitude spectrogram{" "}
            <Latex>{"|\\mathbf{X}| = \\tilde{\\mathbf{X}}"}</Latex> but lack
            phase <Latex>{"\\angle \\mathbf{X}"}</Latex>. Griffin and Lim (1984)
            showed that the signal whose STFT magnitude best matches a given
            target can be found by iteratively projecting between the
            time-domain and STFT-domain, each time imposing the known magnitude
            while retaining the estimated phase.
          </p>

          <p>
            <strong>Algorithm</strong>. Let <Latex>{"\\mathcal{G}"}</Latex>
            denote the STFT operator (analysis) and{" "}
            <Latex>{"\\mathcal{G}^{-1}"}</Latex> the inverse STFT (synthesis).
            Initialize with random phase:
          </p>

          <Eq>{
            "\\phi_0[k, m] \\sim \\mathcal{U}(-\\pi, \\pi)"
          }</Eq>

          <p>
            This produces the initial complex spectrogram:
          </p>

          <Eq>{
            "\\hat{\\mathbf{X}}_0[k, m] = |X[k, m]| \\, e^{j \\phi_0[k, m]}"
          }</Eq>

          <p>
            For each iteration <Latex>{"i = 1, \\ldots, N"}</Latex>:
          </p>

          <Eq>{
            "x_i[n] = \\mathcal{G}^{-1}\\left(\\hat{\\mathbf{X}}_{i-1}\\right) \\quad \\text{(ISTFT to time domain)}"
          }</Eq>
          <Eq>{
            "\\mathbf{Y}_i = \\mathcal{G}(x_i) \\quad \\text{(STFT back)}"
          }</Eq>

          <p>
            With momentum (torchaudio&apos;s fast Griffin-Lim with
            <Latex>{"\\texttt{momentum}=0.99"}</Latex>):
          </p>

          <Eq>{
            "\\hat{\\mathbf{X}}_i[k, m] = |X[k, m]| \\, \\frac{\\mathbf{Y}_i[k, m] - \\alpha \\, \\hat{\\mathbf{X}}_{i-1}[k, m]}{|\\mathbf{Y}_i[k, m] - \\alpha \\, \\hat{\\mathbf{X}}_{i-1}[k, m]|}"
          }</Eq>

          <p>
            where <Latex>{"\\alpha = \\texttt{momentum} / (1 + \\texttt{momentum}) \\approx 0.4975"}</Latex>
            is the internal momentum coefficient. The standard (non-momentum)
            update simply replaces the magnitude of the re-synthesised STFT
            with the target:
          </p>

          <Eq>{
            "\\hat{\\mathbf{X}}_i[k, m] = |X[k, m]| \\, \\frac{\\mathbf{Y}_i[k, m]}{|\\mathbf{Y}_i[k, m]|}"
          }</Eq>

          <p>
            After <Latex>{"N = 64"}</Latex> iterations, the final waveform is
            synthesised one last time:
          </p>

          <Eq>{
            "x_{\\text{final}}[n] = \\mathcal{G}^{-1}\\left(|X| \\, e^{j \\angle \\mathbf{Y}_N}\\right)"
          }</Eq>

          <p className="font-medium mt-2">Convergence property</p>

          <p>
            Griffin and Lim proved that each iteration reduces the Frobenius
            norm of the spectrogram magnitude error:
          </p>

          <Eq>{
            "\\bigl\\| \\, |\\mathcal{G}(x_i)| - |X| \\, \\bigr\\|_F \\leq \\bigl\\| \\, |\\mathcal{G}(x_{i-1})| - |X| \\, \\bigr\\|_F"
          }</Eq>

          <p>
            The algorithm converges to a stationary point of the objective:
          </p>

          <Eq>{
            "J(x) = \\bigl\\| \\, |\\mathcal{G}(x)| - |X| \\, \\bigr\\|_F^2"
          }</Eq>

          <p>
            However, the solution is not guaranteed to be globally optimal
            (the objective is non-convex due to the magnitude operator), and
            different random phase initialisations can lead to different
            results. In practice, 64 iterations with momentum
            (<Latex>{"\\texttt{momentum}=0.99"}</Latex>) produces
            perceptually transparent reconstruction for percussive sounds.
          </p>

          <Legend
            rows={[
              ["\\mathcal{G}", "Operator", "STFT (analysis)", "N=1024, H=256, Hann window"],
              ["\\mathcal{G}^{-1}", "Operator", "ISTFT (synthesis)", "Overlap-add"],
              ["N_{\\text{iter}}", "Scalar", "Number of iterations", "64"],
              ["\\alpha", "Scalar", "Internal momentum coefficient", "~0.4975"],
              ["\\phi_0", "Matrix", "Initial random phase", "\\mathcal{U}(-\\pi, \\pi)"],
            ]}
          />
        </Section>

        {/* ================================================================
            SECTION 10 — Post-Processing & Summary
        ================================================================ */}
        <Section id="postprocess" number={10} title="Post-Processing &amp; Parameter Summary">

          <p className="text-muted-foreground">
            After the vocoder produces a raw waveform, a final post-processing
            chain removes DC offset and ultrasonic content, then peak-normalises
            the output.
          </p>

          <p className="font-medium mt-2">Biquad highpass (DC removal)</p>
          <Eq>{
            "H_{\\text{HP}}(z) : f_c = 25\\text{ Hz}"
          }</Eq>

          <p>
            Removes subsonic rumble and any DC offset introduced by the
            vocoder. Applied via <Latex>{"\\texttt{highpass\\_biquad}"}</Latex>
            (biquad Butterworth, 2nd order).
          </p>

          <p className="font-medium mt-2">Biquad lowpass (anti-aliasing)</p>
          <Eq>{
            "H_{\\text{LP}}(z) : f_c = 20\\,000\\text{ Hz}"
          }</Eq>

          <p>
            Removes any aliasing artefacts above 20 kHz that may have been
            introduced by the neural vocoder or the Griffin-LIM iteration.
          </p>

          <p className="font-medium mt-2">Peak normalisation</p>
          <Eq>{
            "x_{\\text{out}}[n] = \\frac{x_{\\text{filt}}[n]}{\\|\\mathbf{x}_{\\text{filt}}\\|_\\infty + \\varepsilon}, \\qquad \\|\\mathbf{x}\\|_\\infty = \\max_n |x[n]|"
          }</Eq>

          <p>
            Scales the waveform so its maximum absolute value is 1.0, ensuring
            maximum dynamic range without clipping.
          </p>

          <p className="font-medium mt-2">Complete pipeline summary</p>

          <p className="text-sm text-muted-foreground mb-3">
            The full signal flow, with each transform and its mathematical
            characterisation:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-xs border border-border rounded-lg overflow-hidden">
              <thead>
                <tr className="bg-muted/50 text-left">
                  <th className="px-3 py-2 font-semibold">#</th>
                  <th className="px-3 py-2 font-semibold">Stage</th>
                  <th className="px-3 py-2 font-semibold">Domain</th>
                  <th className="px-3 py-2 font-semibold">Key Parameters</th>
                  <th className="px-3 py-2 font-semibold">Source</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">1</td>
                  <td className="px-3 py-2">LUFS normalisation</td>
                  <td className="px-3 py-2">Time</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"L_{\\text{target}} = -14"}</Latex> LUFS
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\texttt{dataset.py}"}</Latex>
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">2</td>
                  <td className="px-3 py-2">STFT (Hann window)</td>
                  <td className="px-3 py-2">Time &rarr; TF</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"N=1024, H=256, 75\\% overlap"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    BigVGAN
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">3</td>
                  <td className="px-3 py-2">Magnitude spectrogram</td>
                  <td className="px-3 py-2">TF</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"|X[k,m]|"}</Latex> (linear)
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    BigVGAN
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">4</td>
                  <td className="px-3 py-2">Mel filterbank</td>
                  <td className="px-3 py-2">TF</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"128 bands, Slaney scale, norm"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    BigVGAN
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">5</td>
                  <td className="px-3 py-2">Log compression</td>
                  <td className="px-3 py-2">TF</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\ln(\\max(\\cdot, 10^{-5}))"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    BigVGAN
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">6</td>
                  <td className="px-3 py-2">[0, 1] normalisation</td>
                  <td className="px-3 py-2">TF</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"[-11.51, 2.5] \\to [0, 1]"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\texttt{dataset.py}"}</Latex>
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">7</td>
                  <td className="px-3 py-2">VAE encode/decode</td>
                  <td className="px-3 py-2">Latent</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"d = 128"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\texttt{model.py}"}</Latex>
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">8a</td>
                  <td className="px-3 py-2">Denormalise + exp</td>
                  <td className="px-3 py-2">TF</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    Inverse affine of step 6
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\texttt{dataset.py}"}</Latex>
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">8b</td>
                  <td className="px-3 py-2">Pseudo-inverse mel</td>
                  <td className="px-3 py-2">TF</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\mathbf{M}^{\\dagger} \\in \\mathbb{R}^{513\\times 128}"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\texttt{vocoder.py} (Griffin-LIM only)"}</Latex>
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">9</td>
                  <td className="px-3 py-2">Griffin-LIM (64 iter)</td>
                  <td className="px-3 py-2">TF &rarr; Time</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\texttt{momentum}=0.99, N=64"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\texttt{vocoder.py} (Griffin-LIM only)"}</Latex>
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">10</td>
                  <td className="px-3 py-2">Biquad HP + LP</td>
                  <td className="px-3 py-2">Time</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"f_c = 25 Hz, f_c = 20 kHz"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\texttt{vocoder.py}"}</Latex>
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">11</td>
                  <td className="px-3 py-2">Peak normalisation</td>
                  <td className="px-3 py-2">Time</td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\|\\cdot\\|_\\infty \\to 1.0"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    <Latex>{"\\texttt{vocoder.py}"}</Latex>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <p className="font-medium mt-4">Kick stripping pipeline (pre-processing)</p>

          <p className="text-sm text-muted-foreground mb-3">
            In parallel with the main generation pipeline, the
            <Latex>{"\\texttt{\\_strip\\_cmd.py}"}</Latex> module applies its
            own signal chain for isolating single kicks from loop recordings:
          </p>

          <div className="overflow-x-auto my-4">
            <table className="w-full text-xs border border-border rounded-lg overflow-hidden">
              <thead>
                <tr className="bg-muted/50 text-left">
                  <th className="px-3 py-2 font-semibold">#</th>
                  <th className="px-3 py-2 font-semibold">Stage</th>
                  <th className="px-3 py-2 font-semibold">Math</th>
                  <th className="px-3 py-2 font-semibold">Parameters</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">S1</td>
                  <td className="px-3 py-2">RMS envelope</td>
                  <td className="px-3 py-2">
                    <Latex>{"E[m] = \\sqrt{\\frac{1}{L}\\sum x^2}"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    10 ms frames
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">S2</td>
                  <td className="px-3 py-2">FFT autocorrelation</td>
                  <td className="px-3 py-2">
                    <Latex>{"\\hat{R}[k] = \\frac{\\mathcal{F}^{-1}|\\mathcal{F}(E)|^2}{R[0]}"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    Skip first 100 ms lag
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">S3</td>
                  <td className="px-3 py-2">Loop decision</td>
                  <td className="px-3 py-2">
                    <Latex>{"\\max\\hat{R} > 0.3 \\implies"}</Latex> loop
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    Threshold = 0.3
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">S4</td>
                  <td className="px-3 py-2">Cascaded lowpass</td>
                  <td className="px-3 py-2">
                    <Latex>{"H(z)^2, f_c=200 Hz"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    Butterworth, order 2+2
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">S5</td>
                  <td className="px-3 py-2">Envelope extraction</td>
                  <td className="px-3 py-2">
                    <Latex>{"|y| \\ast h_{\\text{boxcar}}"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    5 ms moving average
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">S6</td>
                  <td className="px-3 py-2">Onset/decay detect</td>
                  <td className="px-3 py-2">
                    Peak backtrack + forward threshold
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    5% onset, 1% decay threshold
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">S7</td>
                  <td className="px-3 py-2">Cascaded highpass</td>
                  <td className="px-3 py-2">
                    <Latex>{"H(z)^2, f_c=2000 Hz"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    For hi-hat/snare detection
                  </td>
                </tr>
                <tr className="border-t border-border">
                  <td className="px-3 py-2 font-mono">S8</td>
                  <td className="px-3 py-2">Cosine fade-out</td>
                  <td className="px-3 py-2">
                    <Latex>{"0.5(1 + \\cos(\\pi n/L))"}</Latex>
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    10 ms fade
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </Section>

        {/* ---- Footer ---- */}
        <footer className="pt-8 border-t border-border text-center text-xs text-muted-foreground/50">
          &copy; {new Date().getFullYear()} Kevin Paul Klaiber
        </footer>
      </div>
    </div>
  );
}
