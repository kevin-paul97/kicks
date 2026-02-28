"use client";

import { useEffect, useState, useRef, Suspense } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html, useCursor } from "@react-three/drei";
import * as THREE from "three";

interface Sample {
  sample_idx: number;
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

interface ClusterData {
  pca_variance_explained: number[];
  n_clusters: number;
  samples: Sample[];
}

const CLUSTER_COLORS = [
  "#a78bfa",
  "#f472b6",
  "#34d399",
  "#fbbf24",
  "#60a5fa",
  "#f87171",
  "#c084fc",
  "#22d3ee",
];

function Point({
  sample,
  isSelected,
  onClick,
  onHover,
}: {
  sample: Sample;
  isSelected: boolean;
  onClick: () => void;
  onHover: (hovering: boolean) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  
  useCursor(hovered);

  const color = CLUSTER_COLORS[sample.cluster % CLUSTER_COLORS.length];

  return (
    <mesh
      ref={meshRef}
      position={[sample.pc1, sample.pc2, sample.pc3]}
      onClick={onClick}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
        onHover(true);
      }}
      onPointerOut={(e) => {
        setHovered(false);
        onHover(false);
      }}
    >
      <sphereGeometry args={[isSelected ? 0.15 : 0.08, 16, 16]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={isSelected ? 0.8 : hovered ? 0.5 : 0.2}
        transparent
        opacity={0.9}
      />
    </mesh>
  );
}

function Axes() {
  return (
    <group>
      <mesh position={[2, 0, 0]}>
        <boxGeometry args={[4, 0.02, 0.02]} />
        <meshStandardMaterial color="#f472b6" />
      </mesh>
      <mesh position={[0, 2, 0]}>
        <boxGeometry args={[0.02, 4, 0.02]} />
        <meshStandardMaterial color="#34d399" />
      </mesh>
      <mesh position={[0, 0, 2]}>
        <boxGeometry args={[0.02, 0.02, 4]} />
        <meshStandardMaterial color="#a78bfa" />
      </mesh>
      <Html position={[2.2, 0, 0]}>
        <div className="text-xs text-pink-400 font-mono">PC1</div>
      </Html>
      <Html position={[0, 2.2, 0]}>
        <div className="text-xs text-emerald-400 font-mono">PC2</div>
      </Html>
      <Html position={[0, 0, 2.2]}>
        <div className="text-xs text-violet-400 font-mono">PC3</div>
      </Html>
    </group>
  );
}

function Scene({
  data,
  selectedIdx,
  setSelectedIdx,
}: {
  data: ClusterData;
  selectedIdx: number | null;
  setSelectedIdx: (idx: number | null) => void;
}) {
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      <Axes />
      {data.samples.map((sample) => (
        <Point
          key={sample.sample_idx}
          sample={sample}
          isSelected={selectedIdx === sample.sample_idx}
          onClick={() => setSelectedIdx(selectedIdx === sample.sample_idx ? null : sample.sample_idx)}
          onHover={() => {}}
        />
      ))}
      <OrbitControls enableDamping dampingFactor={0.05} />
    </>
  );
}

function Loading() {
  return (
    <Html center>
      <div className="text-violet-300 text-lg">Loading 3D visualization...</div>
    </Html>
  );
}

export default function ClusterPage() {
  const [data, setData] = useState<ClusterData | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    fetch("/api/cluster-data")
      .then((r) => r.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedIdx === null || !data) return;

    const sample = data.samples.find((s) => s.sample_idx === selectedIdx);
    if (!sample) return;

    const params = `pc1=${sample.pc1}&pc2=${sample.pc2}&pc3=${sample.pc3}`;
    fetch(`/api/generate?${params}`)
      .then((r) => r.blob())
      .then((blob) => {
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        if (audioRef.current) {
          audioRef.current.src = url;
          audioRef.current.play();
        }
      });
  }, [selectedIdx, data]);

  const selectedSample = data?.samples.find((s) => s.sample_idx === selectedIdx);

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-violet-300 text-lg">Loading cluster data...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="text-pink-400 text-lg">No cluster data found</div>
          <p className="text-muted-foreground">Run <code className="text-violet-300">python cluster.py</code> first</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="p-4 border-b border-white/10 flex items-center justify-between">
        <h1 className="text-xl font-bold bg-gradient-to-r from-violet-400 to-pink-400 bg-clip-text text-transparent">
          Kick Cluster Visualizer
        </h1>
        <div className="text-sm text-muted-foreground">
          {data.samples.length} samples · {data.n_clusters} clusters · PC1 {data.pca_variance_explained[0].toFixed(1)}% · PC2 {data.pca_variance_explained[1].toFixed(1)}% · PC3 {data.pca_variance_explained[2].toFixed(1)}%
        </div>
      </div>

      <div className="flex">
        <div className="flex-1 h-[calc(100vh-65px)]">
          <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
            <Suspense fallback={<Loading />}>
              <Scene data={data} selectedIdx={selectedIdx} setSelectedIdx={setSelectedIdx} />
            </Suspense>
          </Canvas>
        </div>

        <div className="w-80 border-l border-white/10 p-4 space-y-4 overflow-y-auto">
          <div>
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">
              Clusters
            </h2>
            <div className="flex flex-wrap gap-2">
              {Array.from({ length: data.n_clusters }).map((_, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5"
                >
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }}
                  />
                  <span className="text-sm">Cluster {i}</span>
                  <span className="text-xs text-muted-foreground">
                    ({data.samples.filter((s) => s.cluster === i).length})
                  </span>
                </div>
              ))}
            </div>
          </div>

          {selectedSample && (
            <div className="space-y-4 pt-4 border-t border-white/10">
              <div>
                <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">
                  Selected Sample #{selectedSample.sample_idx}
                </h2>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-pink-400">PC1</span>
                    <span className="font-mono">{selectedSample.pc1.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-emerald-400">PC2</span>
                    <span className="font-mono">{selectedSample.pc2.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-violet-400">PC3</span>
                    <span className="font-mono">{selectedSample.pc3.toFixed(3)}</span>
                  </div>
                </div>
              </div>

                <div>
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                    Descriptors
                  </h3>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(selectedSample.descriptors).map(([key, val]) => (
                      <div key={key} className="flex justify-between text-xs">
                        <span className="capitalize text-muted-foreground">{key}</span>
                        <span className="font-mono">{val.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                <audio ref={audioRef} controls className="w-full" />
                {audioUrl && (
                  <a
                    href={audioUrl}
                    download={`kick_${selectedSample.sample_idx}.wav`}
                    className="block text-center py-2 rounded-lg bg-emerald-500/20 text-emerald-300 text-sm hover:bg-emerald-500/30 transition-colors"
                  >
                    Download WAV
                  </a>
                )}
              </div>
            </div>
          )}

          {!selectedSample && (
            <div className="text-center text-muted-foreground text-sm py-8">
              Click a point to preview audio
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
