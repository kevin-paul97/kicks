"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent } from "@/components/ui/card";

const API = "http://localhost:8080";

interface SliderConfig {
  id: number;
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
    <div className="min-h-screen flex items-center justify-center p-8">
      <Card className="w-full max-w-md">
        <CardContent className="pt-6 space-y-6">
          <h1 className="text-2xl font-bold text-center tracking-tight">
            Kick Synth
          </h1>

          <div className="space-y-5">
            {sliders.map((s, i) => (
              <div key={s.id} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">PC{s.id}</span>
                  <span className="font-mono text-muted-foreground">
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

          {status && (
            <p className="text-sm text-center text-muted-foreground">
              {status}
            </p>
          )}

          <audio ref={playerRef} controls className="w-full" />
        </CardContent>
      </Card>
    </div>
  );
}
