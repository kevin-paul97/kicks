"use client";

import { useEffect, useState, useRef, useMemo, useCallback, Suspense } from "react";
import { useTheme } from "next-themes";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Html, useCursor } from "@react-three/drei";
import * as THREE from "three";
import { ThemeToggle } from "@/components/theme-toggle";

/* ─── Types ─── */

interface Sample {
  sample_idx: number;
  filename: string;
  original_path: string;
  cluster: number;
  pc1: number;
  pc2: number;
  pc3: number;
  descriptors: {
    sub: number;
    punch: number;
    click: number;
    bright: number;
    decay: number;
  };
  probs: number[];
}

interface ClusterProfile {
  count: number;
  sub: number;
  punch: number;
  click: number;
  bright: number;
  decay: number;
}

interface PCName {
  name: string;
  descriptor: string | null;
  correlation: number;
}

interface ClusterData {
  pca_variance_explained: number[];
  pca_source?: string;
  n_clusters: number;
  samples: Sample[];
  pc_names?: PCName[];
  pca_loadings?: Record<string, Record<string, number>>;
  pc_descriptor_correlations?: Record<string, Record<string, number>>;
  cluster_profiles?: Record<string, ClusterProfile>;
  descriptor_stats?: Record<string, { mean: number; std: number; min: number; max: number }>;
}

type DescriptorKey = "sub" | "punch" | "click" | "bright" | "decay";
type ColorMode = "cluster" | DescriptorKey;

const DESCRIPTOR_KEYS: DescriptorKey[] = ["sub", "punch", "click", "bright", "decay"];

const DESCRIPTOR_LABELS: Record<DescriptorKey, string> = {
  sub: "Sub",
  punch: "Punch",
  click: "Click",
  bright: "Bright",
  decay: "Decay",
};

const CLUSTER_COLORS = [
  "#a78bfa", "#f472b6", "#34d399", "#fbbf24",
  "#60a5fa", "#f87171", "#c084fc", "#22d3ee",
  "#fb923c", "#a3e635",
];

const PC_COLORS = ["#f472b6", "#34d399", "#a78bfa"];
const PC_LABELS_SHORT = ["PC1", "PC2", "PC3"];

/* ─── Utilities ─── */

function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  const sx = Math.sqrt(x.reduce((a, b) => a + (b - mx) ** 2, 0) / n);
  const sy = Math.sqrt(y.reduce((a, b) => a + (b - my) ** 2, 0) / n);
  if (sx === 0 || sy === 0) return 0;
  return x.reduce((a, b, i) => a + (b - mx) * (y[i] - my), 0) / (n * sx * sy);
}

function descriptorColor(value: number): string {
  const t = Math.max(0, Math.min(1, value));
  if (t < 0.5) {
    const s = t / 0.5;
    return `rgb(${Math.round(60 + s * 60)},${Math.round(80 + s * 80)},${Math.round(160 + s * 40)})`;
  } else {
    const s = (t - 0.5) / 0.5;
    return `rgb(${Math.round(120 + s * 135)},${Math.round(160 + s * 40)},${Math.round(200 - s * 150)})`;
  }
}

function correlationColor(v: number): string {
  const abs = Math.min(Math.abs(v), 1);
  const alpha = 0.12 + abs * 0.6;
  if (v >= 0) return `rgba(52, 211, 153, ${alpha})`;
  return `rgba(248, 113, 113, ${alpha})`;
}

function correlationTextColor(v: number, isDark = true): string {
  if (isDark) return v >= 0 ? "#6ee7b7" : "#fca5a5";
  return v >= 0 ? "#047857" : "#dc2626";
}

function sampleColor(sample: Sample, mode: ColorMode, stats?: ClusterData["descriptor_stats"]): string {
  if (mode === "cluster") {
    return CLUSTER_COLORS[sample.cluster % CLUSTER_COLORS.length];
  }
  const val = sample.descriptors[mode];
  let normalized = val;
  if (stats && stats[mode]) {
    const { min, max } = stats[mode];
    normalized = max > min ? (val - min) / (max - min) : 0.5;
  }
  return descriptorColor(normalized);
}

/* ─── Shared Styles ─── */

const card = "rounded-2xl border border-border bg-card/50 backdrop-blur-sm";
const cardHover = `${card} hover:bg-card/70 transition-colors`;
const sectionTitle = "text-[11px] font-semibold text-muted-foreground uppercase tracking-widest";
const pill = "px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all duration-150";
const pillActive = `${pill} bg-accent text-accent-foreground shadow-sm`;
const pillInactive = `${pill} text-muted-foreground hover:text-foreground hover:bg-accent/50`;

/* ─── Section: Variance Explained Bar ─── */

function VarianceBar({ variance, pcNames }: { variance: number[]; pcNames: PCName[] }) {
  const total = variance.reduce((a, b) => a + b, 0);

  return (
    <div className={`${card} p-5 space-y-3`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-foreground/80">Variance Explained</span>
        <span className="font-mono text-xs text-muted-foreground bg-muted px-2.5 py-0.5 rounded-md">
          {(total * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-4 rounded-full overflow-hidden flex bg-muted gap-px">
        {variance.map((v, i) => (
          <div
            key={i}
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${(v / total) * 100}%`,
              backgroundColor: PC_COLORS[i],
              opacity: 0.75,
            }}
            title={`${pcNames[i]?.name ?? PC_LABELS_SHORT[i]}: ${(v * 100).toFixed(1)}%`}
          />
        ))}
      </div>
      <div className="flex gap-5 text-xs">
        {variance.map((v, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: PC_COLORS[i], boxShadow: '0 0 0 2px var(--border)' }} />
            <span className="text-foreground/80 font-medium">{pcNames[i]?.name ?? PC_LABELS_SHORT[i]}</span>
            <span className="font-mono text-muted-foreground">{(v * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Section: PC Interpretation Cards ─── */

function PCCard({
  pcIndex,
  variance,
  correlations,
  loadings,
  pcName,
}: {
  pcIndex: number;
  variance: number;
  correlations: Record<string, number>;
  loadings?: Record<string, number>;
  pcName: PCName;
}) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  // Prefer loadings (direct descriptor weights) over correlations when PCA is on descriptors
  const values = loadings || correlations;
  const sorted = Object.entries(values).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.01);

  return (
    <div className={`${card} p-5 space-y-4`}>
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2.5">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: PC_COLORS[pcIndex], boxShadow: `0 0 0 2px ${PC_COLORS[pcIndex]}40` }}
            />
            <span className="text-[15px] font-bold text-foreground">{pcName.name}</span>
          </div>
          <div className="text-[11px] text-muted-foreground mt-1 ml-[22px] font-mono">
            PC{pcIndex + 1} {loadings ? "· loadings" : "· correlations"}
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-mono font-black tracking-tight" style={{ color: PC_COLORS[pcIndex] }}>
            {(variance * 100).toFixed(1)}
            <span className="text-sm font-medium opacity-60">%</span>
          </div>
        </div>
      </div>

      <div className="space-y-2">
        {sorted.map(([desc, corr]) => {
          const width = (Math.abs(corr) / maxAbs) * 100;
          const isPositive = corr >= 0;
          return (
            <div key={desc} className="flex items-center gap-2 text-xs">
              <span className="w-12 text-muted-foreground text-right font-medium shrink-0">
                {DESCRIPTOR_LABELS[desc as DescriptorKey] || desc}
              </span>
              <div className="flex-1 h-5 relative rounded-md bg-muted/50 overflow-hidden">
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
                <div
                  className="absolute top-1 bottom-1 rounded-sm transition-all duration-500"
                  style={{
                    width: `${width / 2}%`,
                    left: isPositive ? "50%" : `${50 - width / 2}%`,
                    background: isPositive
                      ? "linear-gradient(to right, rgba(52,211,153,0.4), rgba(52,211,153,0.7))"
                      : "linear-gradient(to left, rgba(248,113,113,0.4), rgba(248,113,113,0.7))",
                  }}
                />
              </div>
              <span
                className="w-12 font-mono text-right shrink-0 font-medium"
                style={{ color: correlationTextColor(corr, isDark) }}
              >
                {corr > 0 ? "+" : ""}{corr.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── Section: Correlation Heatmap ─── */

function CorrelationHeatmap({
  correlations,
  pcNames,
}: {
  correlations: Record<string, Record<string, number>>;
  pcNames: PCName[];
}) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  return (
    <div className={`${card} p-5 space-y-4`}>
      <h3 className={sectionTitle}>Correlation Matrix</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left text-muted-foreground pb-3 pr-4 font-normal" />
              {["pc1", "pc2", "pc3"].map((pc, i) => (
                <th key={pc} className="pb-3 px-2 font-bold text-center text-[13px]" style={{ color: PC_COLORS[i] }}>
                  {pcNames[i]?.name ?? pc.toUpperCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {DESCRIPTOR_KEYS.map((desc) => (
              <tr key={desc}>
                <td className="text-foreground/80 pr-4 py-1.5 font-medium text-[13px]">
                  {DESCRIPTOR_LABELS[desc]}
                </td>
                {["pc1", "pc2", "pc3"].map((pc) => {
                  const v = correlations[pc]?.[desc] ?? 0;
                  return (
                    <td key={pc} className="px-2 py-1.5 text-center">
                      <div
                        className="rounded-lg px-3 py-2 font-mono font-semibold text-[13px] transition-colors"
                        style={{
                          backgroundColor: correlationColor(v),
                          color: Math.abs(v) > 0.25 ? (isDark ? "#fff" : "#000") : correlationTextColor(v, isDark),
                        }}
                      >
                        {v > 0 ? "+" : ""}{v.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ─── Section: 2D Scatter Plot ─── */

function ScatterPlot({
  samples,
  xKey,
  yKey,
  xLabel,
  yLabel,
  colorMode,
  stats,
  selectedIdx,
  onSelect,
}: {
  samples: Sample[];
  xKey: "pc1" | "pc2" | "pc3";
  yKey: "pc1" | "pc2" | "pc3";
  xLabel: string;
  yLabel: string;
  colorMode: ColorMode;
  stats?: ClusterData["descriptor_stats"];
  selectedIdx: number | null;
  onSelect: (idx: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  const padding = 44;

  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const xs = samples.map((s) => s[xKey]);
    const ys = samples.map((s) => s[yKey]);
    const xPad = (Math.max(...xs) - Math.min(...xs)) * 0.08;
    const yPad = (Math.max(...ys) - Math.min(...ys)) * 0.08;
    return {
      xMin: Math.min(...xs) - xPad,
      xMax: Math.max(...xs) + xPad,
      yMin: Math.min(...ys) - yPad,
      yMax: Math.max(...ys) + yPad,
    };
  }, [samples, xKey, yKey]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const plotW = w - padding * 2;
    const plotH = h - padding * 2;

    const toX = (v: number) => padding + ((v - xMin) / (xMax - xMin)) * plotW;
    const toY = (v: number) => padding + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    ctx.clearRect(0, 0, w, h);

    // Theme-aware canvas colors
    const bgFill = isDark ? "rgba(255,255,255,0.01)" : "rgba(0,0,0,0.02)";
    const gridStroke = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.06)";
    const borderStroke = isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.08)";
    const labelFill = isDark ? "rgba(255,255,255,0.5)" : "rgba(0,0,0,0.5)";
    const tickFill = isDark ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.3)";
    const selectionStroke = isDark ? "#fff" : "#000";

    // Plot background
    ctx.fillStyle = bgFill;
    ctx.beginPath();
    ctx.roundRect(padding, padding, plotW, plotH, 6);
    ctx.fill();

    // Grid
    ctx.strokeStyle = gridStroke;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = padding + (plotW * i) / 4;
      const y = padding + (plotH * i) / 4;
      ctx.beginPath(); ctx.moveTo(x, padding); ctx.lineTo(x, padding + plotH); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(padding, y); ctx.lineTo(padding + plotW, y); ctx.stroke();
    }

    // Border
    ctx.strokeStyle = borderStroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(padding, padding, plotW, plotH, 6);
    ctx.stroke();

    // Points
    for (const sample of samples) {
      const x = toX(sample[xKey]);
      const y = toY(sample[yKey]);
      const isSelected = sample.sample_idx === selectedIdx;
      const isHovered = sample.sample_idx === hoveredIdx;
      const color = sampleColor(sample, colorMode, stats);

      ctx.beginPath();
      ctx.arc(x, y, isSelected ? 5.5 : isHovered ? 4.5 : 2.5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = isSelected ? 1 : isHovered ? 0.95 : 0.7;
      ctx.fill();

      if (isSelected) {
        ctx.globalAlpha = 1;
        ctx.strokeStyle = selectionStroke;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // Axis labels
    ctx.fillStyle = labelFill;
    ctx.font = "500 11px var(--font-geist-mono, monospace)";
    ctx.textAlign = "center";
    ctx.fillText(xLabel, w / 2, h - 4);

    ctx.save();
    ctx.translate(10, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // Ticks
    ctx.fillStyle = tickFill;
    ctx.font = "9px var(--font-geist-mono, monospace)";
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const xVal = xMin + ((xMax - xMin) * i) / 4;
      const yVal = yMin + ((yMax - yMin) * i) / 4;
      ctx.fillText(xVal.toFixed(1), padding + (plotW * i) / 4, padding + plotH + 14);
      ctx.textAlign = "right";
      ctx.fillText(yVal.toFixed(1), padding - 6, padding + plotH - (plotH * i) / 4 + 3);
      ctx.textAlign = "center";
    }
  }, [samples, xKey, yKey, xMin, xMax, yMin, yMax, colorMode, stats, selectedIdx, hoveredIdx, xLabel, yLabel, isDark]);

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const plotW = rect.width - padding * 2;
      const plotH = rect.height - padding * 2;

      let closest: Sample | null = null;
      let closestDist = Infinity;

      for (const sample of samples) {
        const x = padding + ((sample[xKey] - xMin) / (xMax - xMin)) * plotW;
        const y = padding + plotH - ((sample[yKey] - yMin) / (yMax - yMin)) * plotH;
        const dist = Math.hypot(mx - x, my - y);
        if (dist < closestDist && dist < 12) {
          closest = sample;
          closestDist = dist;
        }
      }
      if (closest) onSelect(closest.sample_idx);
    },
    [samples, xKey, yKey, xMin, xMax, yMin, yMax, onSelect]
  );

  const handleCanvasMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const plotW = rect.width - padding * 2;
      const plotH = rect.height - padding * 2;

      let closest: Sample | null = null;
      let closestDist = Infinity;

      for (const sample of samples) {
        const x = padding + ((sample[xKey] - xMin) / (xMax - xMin)) * plotW;
        const y = padding + plotH - ((sample[yKey] - yMin) / (yMax - yMin)) * plotH;
        const dist = Math.hypot(mx - x, my - y);
        if (dist < closestDist && dist < 12) {
          closest = sample;
          closestDist = dist;
        }
      }
      setHoveredIdx(closest?.sample_idx ?? null);
      canvas.style.cursor = closest ? "pointer" : "default";
    },
    [samples, xKey, yKey, xMin, xMax, yMin, yMax]
  );

  return (
    <div ref={containerRef} className={`${card} overflow-hidden`}>
      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height: 320 }}
        onClick={handleCanvasClick}
        onMouseMove={handleCanvasMove}
        onMouseLeave={() => setHoveredIdx(null)}
      />
    </div>
  );
}

/* ─── Section: Cluster Profile Card ─── */

function ClusterProfileCard({
  clusterIdx,
  profile,
  globalStats,
  onPlay,
  isPlaying,
}: {
  clusterIdx: number;
  profile: ClusterProfile;
  globalStats?: ClusterData["descriptor_stats"];
  onPlay: () => void;
  isPlaying: boolean;
}) {
  const color = CLUSTER_COLORS[clusterIdx % CLUSTER_COLORS.length];

  return (
    <div className={`${card} p-3.5 space-y-2.5`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color, boxShadow: `0 0 0 2px ${color}30` }} />
          <span className="text-sm font-semibold text-foreground">Cluster {clusterIdx}</span>
          <span className="text-[11px] text-muted-foreground font-mono bg-muted px-1.5 py-0.5 rounded">{profile.count}</span>
        </div>
        <button
          onClick={onPlay}
          className={`w-7 h-7 rounded-full flex items-center justify-center transition-all ${
            isPlaying
              ? "bg-accent text-foreground scale-110"
              : "bg-muted text-muted-foreground hover:text-foreground hover:bg-accent"
          }`}
        >
          {isPlaying ? (
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="4" width="4" height="16" />
              <rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            <svg className="w-3 h-3 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          )}
        </button>
      </div>

      <div className="space-y-1.5">
        {DESCRIPTOR_KEYS.map((dk) => {
          const val = profile[dk];
          const maxVal = globalStats?.[dk]?.max ?? 1;
          const width = Math.min((val / (maxVal || 1)) * 100, 100);
          return (
            <div key={dk} className="flex items-center gap-2 text-[11px]">
              <span className="w-10 text-muted-foreground text-right font-medium shrink-0">
                {DESCRIPTOR_LABELS[dk]}
              </span>
              <div className="flex-1 h-2.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${width}%`, backgroundColor: color, opacity: 0.75 }}
                />
              </div>
              <span className="w-8 text-muted-foreground font-mono text-right">{val.toFixed(2)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── Section: Sample Inspector ─── */

function SampleInspector({
  sample,
  waveform,
  isPlaying,
  onPlayPause,
  onClose,
  pcNames,
}: {
  sample: Sample;
  waveform: number[] | null;
  isPlaying: boolean;
  onPlayPause: () => void;
  onClose: () => void;
  pcNames: PCName[];
}) {
  return (
    <div className="rounded-2xl border border-border bg-gradient-to-b from-card/80 to-card/40 backdrop-blur-sm p-5 space-y-4 shadow-xl shadow-black/10 dark:shadow-black/20">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={onPlayPause}
            className="w-10 h-10 rounded-full bg-muted border border-border flex items-center justify-center hover:bg-accent transition-all active:scale-95 shrink-0"
          >
            {isPlaying ? (
              <svg className="w-4 h-4 text-foreground/80" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="4" width="4" height="16" />
                <rect x="14" y="4" width="4" height="16" />
              </svg>
            ) : (
              <svg className="w-4 h-4 text-foreground/80 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>
          <div>
            <div className="text-sm text-foreground font-semibold truncate max-w-[300px]">
              {sample.filename}
            </div>
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground mt-0.5">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: CLUSTER_COLORS[sample.cluster % CLUSTER_COLORS.length] }}
              />
              <span>Cluster {sample.cluster}</span>
              <span className="text-muted-foreground/60">|</span>
              <span className="font-mono">
                {pcNames[0]?.name ?? "PC1"}:{sample.pc1.toFixed(1)} {pcNames[1]?.name ?? "PC2"}:{sample.pc2.toFixed(1)} {pcNames[2]?.name ?? "PC3"}:{sample.pc3.toFixed(1)}
              </span>
            </div>
          </div>
        </div>
        <button
          onClick={onClose}
          className="w-8 h-8 rounded-full bg-muted flex items-center justify-center hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>
      </div>

      {waveform && (
        <div className="h-16 rounded-xl bg-muted border border-border overflow-hidden">
          <svg viewBox="0 0 200 64" className="w-full h-full" preserveAspectRatio="none">
            <defs>
              <linearGradient id="inspectorWaveGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#a78bfa" />
                <stop offset="50%" stopColor="#f472b6" />
                <stop offset="100%" stopColor="#34d399" />
              </linearGradient>
            </defs>
            {waveform.map((v, i) => (
              <line
                key={i}
                x1={i} y1={32 - v * 26} x2={i} y2={32 + v * 26}
                stroke="url(#inspectorWaveGradient)" strokeWidth="1.2" opacity="0.8"
              />
            ))}
          </svg>
        </div>
      )}

      <div className="flex gap-2">
        {DESCRIPTOR_KEYS.map((dk) => {
          const val = sample.descriptors[dk];
          return (
            <div key={dk} className="flex-1 bg-muted/50 rounded-xl p-2.5 text-center">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1">
                {DESCRIPTOR_LABELS[dk]}
              </div>
              <div className="text-sm font-mono font-bold text-foreground">
                {val.toFixed(3)}
              </div>
              <div className="mt-1.5 h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${Math.min(val * 100, 100)}%`,
                    background: `linear-gradient(to right, ${PC_COLORS[0]}, ${PC_COLORS[1]})`,
                    opacity: 0.7,
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── 3D Scene Components ─── */

function Point3D({
  coords,
  color,
  isSelected,
  onClick,
  selectionColor = "white",
}: {
  coords: { x: number; y: number; z: number };
  color: string;
  isSelected: boolean;
  onClick: () => void;
  selectionColor?: string;
}) {
  const [hovered, setHovered] = useState(false);
  useCursor(hovered);

  return (
    <group>
      {isSelected && (
        <mesh position={[coords.x, coords.y, coords.z]}>
          <sphereGeometry args={[0.1, 16, 16]} />
          <meshBasicMaterial color={selectionColor} transparent opacity={0.25} />
        </mesh>
      )}
      <mesh
        position={[coords.x, coords.y, coords.z]}
        onClick={onClick}
        onPointerOver={(e) => { e.stopPropagation(); setHovered(true); }}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[isSelected ? 0.06 : 0.035, 12, 12]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isSelected ? 0.8 : hovered ? 0.6 : 0.25}
          transparent
          opacity={0.65}
        />
      </mesh>
    </group>
  );
}

function Axes3D({ labels }: { labels: [string, string, string] }) {
  return (
    <group>
      <mesh position={[2, 0, 0]}><boxGeometry args={[4, 0.02, 0.02]} /><meshStandardMaterial color="#f472b6" /></mesh>
      <mesh position={[0, 2, 0]}><boxGeometry args={[0.02, 4, 0.02]} /><meshStandardMaterial color="#34d399" /></mesh>
      <mesh position={[0, 0, 2]}><boxGeometry args={[0.02, 0.02, 4]} /><meshStandardMaterial color="#a78bfa" /></mesh>
      <Html position={[2.2, 0, 0]}><div className="text-xs text-pink-400 font-mono font-medium">{labels[0]}</div></Html>
      <Html position={[0, 2.2, 0]}><div className="text-xs text-emerald-400 font-mono font-medium">{labels[1]}</div></Html>
      <Html position={[0, 0, 2.2]}><div className="text-xs text-violet-400 font-mono font-medium">{labels[2]}</div></Html>
    </group>
  );
}

function Scene3D({
  data,
  selectedIdx,
  setSelectedIdx,
  colorMode,
  spherize,
  pcNames,
}: {
  data: ClusterData;
  selectedIdx: number | null;
  setSelectedIdx: (idx: number | null) => void;
  colorMode: ColorMode;
  spherize: boolean;
  pcNames: PCName[];
}) {
  const { resolvedTheme } = useTheme();
  const gridColor = useMemo(
    () => new THREE.Color(resolvedTheme === "dark" ? 0xffffff : 0x000000),
    [resolvedTheme]
  );
  const gridOpacity = resolvedTheme === "dark" ? 0.06 : 0.12;

  const flips: [number, number, number] = [
    pcNames[0]?.correlation < 0 ? -1 : 1,
    pcNames[1]?.correlation < 0 ? -1 : 1,
    pcNames[2]?.correlation < 0 ? -1 : 1,
  ];

  const getCoords = useMemo(() => {
    if (!spherize) {
      return (s: Sample) => ({ x: s.pc1 * flips[0], y: s.pc2 * flips[1], z: s.pc3 * flips[2] });
    }
    const pcs = data.samples.map((s) => [s.pc1 * flips[0], s.pc2 * flips[1], s.pc3 * flips[2]]);
    const means = [0, 1, 2].map((i) => pcs.reduce((a, b) => a + b[i], 0) / pcs.length);
    const stds = [0, 1, 2].map((i) => Math.sqrt(pcs.reduce((a, b) => a + (b[i] - means[i]) ** 2, 0) / pcs.length));
    return (s: Sample) => ({
      x: (s.pc1 * flips[0] - means[0]) / (stds[0] || 1),
      y: (s.pc2 * flips[1] - means[1]) / (stds[1] || 1),
      z: (s.pc3 * flips[2] - means[2]) / (stds[2] || 1),
    });
  }, [data.samples, spherize, flips]);

  const axisLabels: [string, string, string] = [
    pcNames[0]?.name ?? "PC1",
    pcNames[1]?.name ?? "PC2",
    pcNames[2]?.name ?? "PC3",
  ];

  const gridPos: [number, number, number] = spherize ? [0, 0, 0] : [2, 0, 2];

  const sceneBg = useMemo(
    () => new THREE.Color(resolvedTheme === "dark" ? 0x08080c : 0xf5f5f5),
    [resolvedTheme]
  );

  return (
    <>
      <color attach="background" args={[sceneBg]} key={`bg-${resolvedTheme}`} />
      <ambientLight intensity={resolvedTheme === "dark" ? 0.5 : 0.7} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      <Axes3D labels={axisLabels} />
      <gridHelper
        key={`grid-${resolvedTheme}-${spherize}`}
        args={[8, 16, gridColor, gridColor]}
        position={gridPos}
        material-transparent
        material-opacity={gridOpacity}
        material-depthWrite={false}
      />
      {data.samples.map((sample) => (
        <Point3D
          key={sample.sample_idx}
          coords={getCoords(sample)}
          color={sampleColor(sample, colorMode, data.descriptor_stats)}
          isSelected={selectedIdx === sample.sample_idx}
          onClick={() => setSelectedIdx(selectedIdx === sample.sample_idx ? null : sample.sample_idx)}
          selectionColor={resolvedTheme === "dark" ? "white" : "black"}
        />
      ))}
      <OrbitControls enableDamping dampingFactor={0.05} />
    </>
  );
}

/* ─── Descriptor Distribution Histogram ─── */

function DescriptorDistribution({
  samples,
  descriptor,
  stats,
}: {
  samples: Sample[];
  descriptor: DescriptorKey;
  stats?: { mean: number; std: number; min: number; max: number };
}) {
  const bins = 20;
  const values = samples.map((s) => s.descriptors[descriptor]);
  const min = stats?.min ?? Math.min(...values);
  const max = stats?.max ?? Math.max(...values);
  const range = max - min || 1;

  const histogram = new Array(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(Math.floor(((v - min) / range) * bins), bins - 1);
    histogram[idx]++;
  }
  const maxCount = Math.max(...histogram);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <span className="text-foreground/80 font-semibold">{DESCRIPTOR_LABELS[descriptor]}</span>
        {stats && (
          <span className="text-muted-foreground font-mono">
            {stats.mean.toFixed(2)} +/- {stats.std.toFixed(2)}
          </span>
        )}
      </div>
      <div className="flex items-end gap-px h-12 rounded-lg bg-muted/30 p-1">
        {histogram.map((count, i) => (
          <div
            key={i}
            className="flex-1 rounded-t-sm transition-all"
            style={{
              height: `${(count / (maxCount || 1)) * 100}%`,
              backgroundColor: descriptorColor(i / bins),
              opacity: 0.75,
              minHeight: count > 0 ? 2 : 0,
            }}
          />
        ))}
      </div>
      <div className="flex justify-between text-[9px] text-muted-foreground font-mono">
        <span>{min.toFixed(2)}</span>
        <span>{max.toFixed(2)}</span>
      </div>
    </div>
  );
}

/* ─── Color Mode Pill Selector ─── */

function ColorModeSelector({ colorMode, setColorMode }: { colorMode: ColorMode; setColorMode: (m: ColorMode) => void }) {
  return (
    <div className="flex gap-1 bg-card/50 rounded-xl p-1 border border-border">
      {(["cluster", ...DESCRIPTOR_KEYS] as ColorMode[]).map((mode) => (
        <button
          key={mode}
          onClick={() => setColorMode(mode)}
          className={colorMode === mode ? pillActive : pillInactive}
        >
          {mode === "cluster" ? "Cluster" : DESCRIPTOR_LABELS[mode as DescriptorKey]}
        </button>
      ))}
    </div>
  );
}

/* ─── Main Page ─── */

export default function ClusterPage() {
  const [data, setData] = useState<ClusterData | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [colorMode, setColorMode] = useState<ColorMode>("cluster");
  const [view, setView] = useState<"analysis" | "3d">("analysis");
  const [spherize, setSpherize] = useState(false);
  const [waveform, setWaveform] = useState<number[] | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playingCluster, setPlayingCluster] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const correlations = useMemo(() => {
    if (!data) return {};
    if (data.pc_descriptor_correlations) return data.pc_descriptor_correlations;

    const result: Record<string, Record<string, number>> = {};
    for (const pc of ["pc1", "pc2", "pc3"] as const) {
      const pcVals = data.samples.map((s) => s[pc]);
      result[pc] = {};
      for (const dk of DESCRIPTOR_KEYS) {
        const dkVals = data.samples.map((s) => s.descriptors[dk]);
        result[pc][dk] = pearsonCorrelation(pcVals, dkVals);
      }
    }
    return result;
  }, [data]);

  const clusterProfiles = useMemo(() => {
    if (!data) return {};
    if (data.cluster_profiles) return data.cluster_profiles;

    const profiles: Record<string, ClusterProfile> = {};
    for (let k = 0; k < data.n_clusters; k++) {
      const cluster = data.samples.filter((s) => s.cluster === k);
      if (cluster.length === 0) continue;
      const profile: ClusterProfile = { count: cluster.length, sub: 0, punch: 0, click: 0, bright: 0, decay: 0 };
      for (const dk of DESCRIPTOR_KEYS) {
        profile[dk] = cluster.reduce((a, s) => a + s.descriptors[dk], 0) / cluster.length;
      }
      profiles[String(k)] = profile;
    }
    return profiles;
  }, [data]);

  const pcNames = useMemo((): PCName[] => {
    if (data?.pc_names) return data.pc_names;
    const used = new Set<string>();
    return ["pc1", "pc2", "pc3"].map((pc, i) => {
      const corrs = correlations[pc] || {};
      const sorted = Object.entries(corrs)
        .filter(([k]) => !used.has(k))
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      const top = sorted[0];
      if (top && Math.abs(top[1]) >= 0.15) {
        used.add(top[0]);
        return { name: DESCRIPTOR_LABELS[top[0] as DescriptorKey] || top[0], descriptor: top[0], correlation: top[1] };
      }
      return { name: `PC${i + 1}`, descriptor: null, correlation: 0 };
    });
  }, [data, correlations]);

  useEffect(() => {
    fetch("/api/cluster-data")
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedIdx === null || !data) {
      setWaveform(null);
      return;
    }

    const audioSrc = `/api/play?idx=${selectedIdx}`;

    if (audioRef.current) {
      audioRef.current.src = audioSrc;
      audioRef.current.play().catch(() => {});
      setIsPlaying(true);
      setPlayingCluster(null);
    }

    fetch(audioSrc)
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        const ctx = new AudioContext();
        return ctx.decodeAudioData(buf);
      })
      .then((audioBuf) => {
        const raw = audioBuf.getChannelData(0);
        const n = 200;
        const block = Math.floor(raw.length / n);
        const wf: number[] = [];
        for (let i = 0; i < n; i++) {
          let sum = 0;
          for (let j = 0; j < block; j++) sum += Math.abs(raw[i * block + j]);
          wf.push(sum / block);
        }
        const mx = Math.max(...wf);
        setWaveform(wf.map((v) => v / (mx || 1)));
      })
      .catch(() => setWaveform(null));
  }, [selectedIdx, data]);

  const handlePlayPause = () => {
    if (!audioRef.current) return;
    if (isPlaying) audioRef.current.pause();
    else audioRef.current.play();
    setIsPlaying(!isPlaying);
  };

  const handlePlayCluster = (k: number) => {
    setPlayingCluster(k);
    setSelectedIdx(null);
    if (audioRef.current) {
      audioRef.current.src = `/api/cluster-avg?cluster=${k}`;
      audioRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  };

  const selectedSample = data?.samples.find((s) => s.sample_idx === selectedIdx);

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex items-center gap-3">
          <div className="w-4 h-4 rounded-full border-2 border-muted-foreground/30 border-t-violet-400 animate-spin" />
          <span className="text-muted-foreground text-sm">Loading analysis data...</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center space-y-3">
          <div className="text-foreground/80 font-medium">No analysis data found</div>
          <code className="text-xs text-muted-foreground bg-muted px-4 py-2 rounded-xl inline-block">
            python cluster.py
          </code>
        </div>
      </div>
    );
  }

  const totalVariance = data.pca_variance_explained.reduce((a, b) => a + b, 0);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* ─── Header ─── */}
      <header className="border-b border-border bg-background/90 backdrop-blur-xl sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-foreground to-muted-foreground bg-clip-text text-transparent">
              PCA Analysis
            </h1>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-[11px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded-md">
                {data.samples.length} samples
              </span>
              <span className="text-[11px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded-md">
                {data.n_clusters} clusters
              </span>
              <span className="text-[11px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded-md">
                {(totalVariance * 100).toFixed(1)}% variance
              </span>
              {data.pca_source === "descriptors_zscore" && (
                <span className="text-[11px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded-md">
                  PCA on descriptors
                </span>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 bg-card/50 rounded-xl p-1 border border-border">
            <button
              onClick={() => setView("analysis")}
              className={view === "analysis" ? pillActive : pillInactive}
            >
              Analysis
            </button>
            <button
              onClick={() => setView("3d")}
              className={view === "3d" ? pillActive : pillInactive}
            >
              3D View
            </button>
          </div>
          <ThemeToggle />
          </div>
        </div>
      </header>

      {view === "3d" ? (
        <div className="h-[calc(100vh-73px)] relative">
          <div className="absolute top-4 left-4 z-10 flex items-center gap-2">
            <ColorModeSelector colorMode={colorMode} setColorMode={setColorMode} />
            <button
              onClick={() => setSpherize(!spherize)}
              className={spherize ? pillActive : `${pillInactive} bg-background/40 backdrop-blur-sm border border-border`}
            >
              Spherize
            </button>
          </div>
          <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
            <Suspense fallback={<Html center><div className="text-muted-foreground text-sm">Loading 3D...</div></Html>}>
              <Scene3D data={data} selectedIdx={selectedIdx} setSelectedIdx={setSelectedIdx} colorMode={colorMode} spherize={spherize} pcNames={pcNames} />
            </Suspense>
          </Canvas>
        </div>
      ) : (
        <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">

          <section>
            <VarianceBar variance={data.pca_variance_explained} pcNames={pcNames} />
          </section>

          <section className="space-y-3">
            <h2 className={sectionTitle}>Principal Components</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {data.pca_variance_explained.map((v, i) => (
                <PCCard
                  key={i}
                  pcIndex={i}
                  variance={v}
                  correlations={correlations[`pc${i + 1}`] || {}}
                  loadings={data.pca_loadings?.[`pc${i + 1}`]}
                  pcName={pcNames[i] || { name: `PC${i+1}`, descriptor: null, correlation: 0 }}
                />
              ))}
            </div>
          </section>

          <section>
            <CorrelationHeatmap correlations={correlations} pcNames={pcNames} />
          </section>

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className={sectionTitle}>PC Space</h2>
              <ColorModeSelector colorMode={colorMode} setColorMode={setColorMode} />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <ScatterPlot
                samples={data.samples}
                xKey="pc1" yKey="pc2"
                xLabel={`${pcNames[0]?.name ?? "PC1"} (PC1)`}
                yLabel={`${pcNames[1]?.name ?? "PC2"} (PC2)`}
                colorMode={colorMode}
                stats={data.descriptor_stats}
                selectedIdx={selectedIdx}
                onSelect={(idx) => setSelectedIdx(idx)}
              />
              <ScatterPlot
                samples={data.samples}
                xKey="pc1" yKey="pc3"
                xLabel={`${pcNames[0]?.name ?? "PC1"} (PC1)`}
                yLabel={`${pcNames[2]?.name ?? "PC3"} (PC3)`}
                colorMode={colorMode}
                stats={data.descriptor_stats}
                selectedIdx={selectedIdx}
                onSelect={(idx) => setSelectedIdx(idx)}
              />
              <ScatterPlot
                samples={data.samples}
                xKey="pc2" yKey="pc3"
                xLabel={`${pcNames[1]?.name ?? "PC2"} (PC2)`}
                yLabel={`${pcNames[2]?.name ?? "PC3"} (PC3)`}
                colorMode={colorMode}
                stats={data.descriptor_stats}
                selectedIdx={selectedIdx}
                onSelect={(idx) => setSelectedIdx(idx)}
              />
            </div>
          </section>

          {selectedSample && (
            <section>
              <SampleInspector
                sample={selectedSample}
                waveform={waveform}
                isPlaying={isPlaying}
                onPlayPause={handlePlayPause}
                onClose={() => { setSelectedIdx(null); setWaveform(null); }}
                pcNames={pcNames}
              />
            </section>
          )}

          <section className="space-y-3">
            <h2 className={sectionTitle}>Cluster Profiles</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
              {Object.entries(clusterProfiles)
                .sort(([a], [b]) => parseInt(a) - parseInt(b))
                .map(([k, profile]) => (
                  <ClusterProfileCard
                    key={k}
                    clusterIdx={parseInt(k)}
                    profile={profile}
                    globalStats={data.descriptor_stats}
                    onPlay={() => handlePlayCluster(parseInt(k))}
                    isPlaying={playingCluster === parseInt(k) && isPlaying}
                  />
                ))}
            </div>
          </section>

          <section className="space-y-3">
            <h2 className={sectionTitle}>Descriptor Distributions</h2>
            <div className={`grid grid-cols-2 md:grid-cols-5 gap-5 ${card} p-5`}>
              {DESCRIPTOR_KEYS.map((dk) => (
                <DescriptorDistribution
                  key={dk}
                  samples={data.samples}
                  descriptor={dk}
                  stats={data.descriptor_stats?.[dk]}
                />
              ))}
            </div>
          </section>

          <div className="pt-6 pb-10 text-center text-[11px] text-muted-foreground/60">
            Kick Drum PCA Analysis
          </div>
        </main>
      )}

      <audio
        ref={audioRef}
        className="hidden"
        onEnded={() => { setIsPlaying(false); setPlayingCluster(null); }}
      />
    </div>
  );
}
