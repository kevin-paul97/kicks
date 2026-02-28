"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Slider } from "@/components/ui/slider";

const API = "/api";

const SLIDER_COLORS = [
  { accent: "#a78bfa", glow: "rgba(167,139,250,0.35)" },
  { accent: "#f472b6", glow: "rgba(244,114,182,0.35)" },
  { accent: "#34d399", glow: "rgba(52,211,153,0.35)" },
  { accent: "#fbbf24", glow: "rgba(251,191,36,0.35)" },
];

interface SliderConfig {
  id: number;
  name: string;
  min: number;
  max: number;
  default: number;
  step: number;
}

/* ── Animated waveform background ── */
function WaveBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    let w = 0,
      h = 0;

    function resize() {
      w = canvas!.width = window.innerWidth;
      h = canvas!.height = window.innerHeight;
    }
    resize();
    window.addEventListener("resize", resize);

    function draw(t: number) {
      ctx.clearRect(0, 0, w, h);

      const waves = [
        { color: "rgba(167,139,250,0.07)", speed: 0.0003, amp: 80, freq: 0.003, yOff: 0.35 },
        { color: "rgba(244,114,182,0.05)", speed: 0.0005, amp: 60, freq: 0.004, yOff: 0.45 },
        { color: "rgba(52,211,153,0.04)", speed: 0.0004, amp: 50, freq: 0.005, yOff: 0.55 },
        { color: "rgba(251,191,36,0.03)", speed: 0.0006, amp: 40, freq: 0.006, yOff: 0.65 },
      ];

      for (const wave of waves) {
        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let x = 0; x <= w; x += 3) {
          const y =
            h * wave.yOff +
            Math.sin(x * wave.freq + t * wave.speed) * wave.amp +
            Math.sin(x * wave.freq * 0.5 + t * wave.speed * 1.3) * wave.amp * 0.5;
          ctx.lineTo(x, y);
        }
        ctx.lineTo(w, h);
        ctx.closePath();
        ctx.fillStyle = wave.color;
        ctx.fill();
      }

      // Subtle radial glow at center
      const grd = ctx.createRadialGradient(w / 2, h * 0.38, 0, w / 2, h * 0.38, w * 0.5);
      grd.addColorStop(0, "rgba(167,139,250,0.06)");
      grd.addColorStop(0.5, "rgba(244,114,182,0.02)");
      grd.addColorStop(1, "transparent");
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, w, h);

      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  );
}

/* ── Mini waveform visualiser ── */
function WaveformVis({ audioRef }: { audioRef: React.RefObject<HTMLAudioElement | null> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const animRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const audio = audioRef.current;
    if (!canvas || !audio) return;

    function initAudio() {
      if (ctxRef.current) return;
      const actx = new AudioContext();
      const analyser = actx.createAnalyser();
      analyser.fftSize = 256;
      const source = actx.createMediaElementSource(audio!);
      source.connect(analyser);
      analyser.connect(actx.destination);
      ctxRef.current = actx;
      analyserRef.current = analyser;
      sourceRef.current = source;
    }

    function draw() {
      const ctx = canvas!.getContext("2d")!;
      const w = canvas!.width;
      const h = canvas!.height;
      ctx.clearRect(0, 0, w, h);

      if (analyserRef.current) {
        const bufLen = analyserRef.current.frequencyBinCount;
        const data = new Uint8Array(bufLen);
        analyserRef.current.getByteFrequencyData(data);

        const barW = w / bufLen;
        for (let i = 0; i < bufLen; i++) {
          const v = data[i] / 255;
          const barH = v * h * 0.9;

          const gradient = ctx.createLinearGradient(0, h, 0, h - barH);
          gradient.addColorStop(0, "rgba(167,139,250,0.8)");
          gradient.addColorStop(0.5, "rgba(244,114,182,0.6)");
          gradient.addColorStop(1, "rgba(52,211,153,0.4)");
          ctx.fillStyle = gradient;

          const x = i * barW;
          ctx.beginPath();
          ctx.roundRect(x + 0.5, h - barH, Math.max(barW - 1, 1), barH, 2);
          ctx.fill();
        }
      } else {
        // Idle state: draw a subtle sine wave
        const t = Date.now() * 0.002;
        ctx.beginPath();
        ctx.strokeStyle = "rgba(167,139,250,0.2)";
        ctx.lineWidth = 1.5;
        for (let x = 0; x < w; x++) {
          const y = h / 2 + Math.sin(x * 0.04 + t) * 8 + Math.sin(x * 0.02 + t * 0.7) * 5;
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      animRef.current = requestAnimationFrame(draw);
    }

    audio.addEventListener("play", initAudio);
    animRef.current = requestAnimationFrame(draw);

    return () => {
      audio.removeEventListener("play", initAudio);
      cancelAnimationFrame(animRef.current);
    };
  }, [audioRef]);

  return (
    <canvas
      ref={canvasRef}
      width={480}
      height={80}
      className="w-full rounded-xl"
      style={{ height: 80 }}
    />
  );
}

export default function Home() {
  const [sliders, setSliders] = useState<SliderConfig[]>([]);
  const [values, setValues] = useState<number[]>([]);
  const [status, setStatus] = useState("Loading...");
  const playerRef = useRef<HTMLAudioElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const blobUrlRef = useRef<string | null>(null);
  const initialGenDone = useRef(false);

  useEffect(() => {
    fetch(`${API}/config`)
      .then((r) => r.json())
      .then((data) => {
        setSliders(data.sliders);
        setValues(data.sliders.map((s: SliderConfig) => s.default));
        setStatus("");
      })
      .catch(() => setStatus("Cannot connect to backend"));
  }, []);

  const generate = useCallback((vals: number[]) => {
    if (vals.length === 0) return;
    setStatus("Generating...");
    const params = vals.map((v, i) => `pc${i + 1}=${v}`).join("&");
    fetch(`${API}/generate?${params}`)
      .then((r) => r.blob())
      .then((blob) => {
        if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
        const url = URL.createObjectURL(blob);
        blobUrlRef.current = url;
        const player = playerRef.current;
        if (player) {
          player.src = url;
          player.play().catch(() => {});
        }
        setStatus("");
      })
      .catch(() => setStatus("Error generating"));
  }, []);

  useEffect(() => {
    if (values.length > 0 && !initialGenDone.current) {
      initialGenDone.current = true;
      generate(values);
    }
  }, [values, generate]);

  const handleSliderChange = (index: number, newValue: number[]) => {
    const updated = [...values];
    updated[index] = newValue[0];
    setValues(updated);

    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => generate(updated), 150);
  };

  return (
    <>
      <WaveBackground />

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center p-6 sm:p-8">
        <div className="w-full max-w-lg">
          {/* Header */}
          <div className="mb-10 text-center">
            <h1 className="text-5xl sm:text-6xl font-black tracking-tighter mb-3 bg-gradient-to-r from-violet-400 via-pink-400 to-emerald-400 bg-clip-text text-transparent">
              Kick Synth
            </h1>
            <p className="text-muted-foreground text-sm sm:text-base leading-relaxed max-w-sm mx-auto">
              Neural kick drum synthesizer powered by a variational autoencoder.
              Shape your sound with four latent dimensions.
            </p>
          </div>

          {/* Main Card */}
          <div className="rounded-2xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-xl p-6 sm:p-8 space-y-7 shadow-2xl shadow-violet-500/5">
            {/* Sliders */}
            <div className="space-y-5">
              {sliders.map((s, i) => {
                const color = SLIDER_COLORS[i % SLIDER_COLORS.length];
                return (
                  <div key={s.id} className="space-y-2.5">
                    <div className="flex items-center justify-between">
                      <label
                        className="text-sm font-semibold tracking-wide uppercase"
                        style={{ color: color.accent }}
                      >
                        {s.name}
                      </label>
                      <span className="font-mono text-xs tabular-nums text-muted-foreground bg-white/5 px-2.5 py-1 rounded-lg">
                        {values[i]?.toFixed(2)}
                      </span>
                    </div>
                    <div
                      className="slider-colored"
                      style={
                        {
                          "--slider-accent": color.accent,
                          "--slider-glow": color.glow,
                        } as React.CSSProperties
                      }
                    >
                      <Slider
                        min={s.min}
                        max={s.max}
                        step={s.step}
                        value={[values[i] ?? s.default]}
                        onValueChange={(v) => handleSliderChange(i, v)}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3">
              <button
                onClick={() => {
                  if (!sliders.length) return;
                  const randomized = sliders.map((s) => {
                    const mid = (s.min + s.max) / 2;
                    const range = s.max - s.min;
                    const gauss =
                      (Math.random() + Math.random() + Math.random()) / 3 - 0.5;
                    return Math.max(s.min, Math.min(s.max, mid + gauss * range));
                  });
                  setValues(randomized);
                  if (debounceRef.current) clearTimeout(debounceRef.current);
                  debounceRef.current = setTimeout(
                    () => generate(randomized),
                    150
                  );
                }}
                className="flex-1 py-2.5 rounded-xl border border-violet-500/30 bg-violet-500/10 text-sm font-semibold tracking-wide uppercase text-violet-300 hover:bg-violet-500/20 hover:border-violet-500/50 transition-all duration-200"
              >
                Randomise
              </button>
              <button
                onClick={() => {
                  if (!blobUrlRef.current) return;
                  const a = document.createElement("a");
                  a.href = blobUrlRef.current;
                  a.download = "kick.wav";
                  a.click();
                }}
                className="flex-1 py-2.5 rounded-xl border border-emerald-500/30 bg-emerald-500/10 text-sm font-semibold tracking-wide uppercase text-emerald-300 hover:bg-emerald-500/20 hover:border-emerald-500/50 transition-all duration-200"
              >
                Download WAV
              </button>
            </div>

            {/* Divider */}
            <div className="border-t border-white/[0.06]" />

            {/* Waveform Visualiser */}
            <div className="space-y-3">
              <label className="text-sm font-semibold tracking-wide uppercase text-muted-foreground">
                Output
              </label>
              <div className="rounded-xl bg-black/30 border border-white/[0.06] p-3 space-y-3">
                <WaveformVis audioRef={playerRef} />
                <audio
                  ref={playerRef}
                  controls
                  className="w-full audio-player"
                />
              </div>
            </div>

            {/* Status */}
            <div className="h-5 flex items-center justify-center gap-2">
              {status && (
                <>
                  {status === "Generating..." && (
                    <div className="size-3 rounded-full border-2 border-violet-400/40 border-t-violet-400 animate-spin" />
                  )}
                  <p className="text-xs text-muted-foreground">{status}</p>
                </>
              )}
            </div>
          </div>

          {/* Footer */}
          <div className="mt-8 text-center space-y-2">
            <a
              href="/cluster"
              className="block text-xs text-violet-400/70 hover:text-violet-300 transition-colors"
            >
              View Cluster Visualization
            </a>
            <p className="text-xs text-muted-foreground/50">
              &copy; {new Date().getFullYear()} Kevin Paul Klaiber
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
