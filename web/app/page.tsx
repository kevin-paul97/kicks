"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Slider } from "@/components/ui/slider";

const API = "http://localhost:8080";

interface SliderConfig {
  id: number;
  name: string;
  min: number;
  max: number;
  default: number;
  step: number;
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
    <div className="min-h-screen flex flex-col items-center justify-center p-6 sm:p-8">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight mb-3">
            Kick Synth
          </h1>
          <p className="text-muted-foreground text-sm sm:text-base leading-relaxed max-w-sm mx-auto">
            Neural kick drum synthesizer powered by a variational autoencoder.
            Shape your sound with four latent dimensions.
          </p>
        </div>

        {/* Main Card */}
        <div className="rounded-2xl border border-border/60 bg-card/50 backdrop-blur-sm p-6 sm:p-8 space-y-8">
          {/* Sliders */}
          <div className="space-y-6">
            {sliders.map((s, i) => (
              <div key={s.id} className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium tracking-wide uppercase text-foreground/80">
                    {s.name}
                  </label>
                  <span className="font-mono text-xs tabular-nums text-muted-foreground bg-muted/50 px-2 py-0.5 rounded-md">
                    {values[i]?.toFixed(2)}
                  </span>
                </div>
                <Slider
                  min={s.min}
                  max={s.max}
                  step={s.step}
                  value={[values[i] ?? s.default]}
                  onValueChange={(v) => handleSliderChange(i, v)}
                />
              </div>
            ))}
          </div>

          {/* Divider */}
          <div className="border-t border-border/40" />

          {/* Audio Player */}
          <div className="space-y-3">
            <label className="text-sm font-medium tracking-wide uppercase text-foreground/80">
              Output
            </label>
            <audio ref={playerRef} controls className="w-full audio-player" />
          </div>

          {/* Status */}
          <div className="h-5 flex items-center justify-center gap-2">
            {status && (
              <>
                {status === "Generating..." && (
                  <div className="size-3 rounded-full border-2 border-muted-foreground/40 border-t-foreground animate-spin" />
                )}
                <p className="text-xs text-muted-foreground">{status}</p>
              </>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-xs text-muted-foreground/60">
            &copy; {new Date().getFullYear()} Kevin Paul Klaiber
          </p>
        </div>
      </div>
    </div>
  );
}
