import { useState, useRef, Suspense, useEffect } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Search, Terminal, ArrowLeft, LoaderCircle } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import * as THREE from "three";

interface StarSystem {
  star_name: string;
  predictions: PredictionItem[];
}

function Star() {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <group>
      <mesh ref={meshRef}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial
          color="#ffd54f"
          emissive="#ff8a00"
          emissiveIntensity={1.2}
        />
      </mesh>
      <pointLight position={[0, 0, 0]} intensity={2} color="#FDB813" distance={30} decay={2} />
    </group>
  );
}

function Planet({ orbitRadius, size, color, orbitSpeed, isSelected, onClick }: {
  orbitRadius: number;
  size: number;
  color: string;
  orbitSpeed: number;
  isSelected: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * orbitSpeed;
    }
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.5;
    }
  });

  return (
    <group ref={groupRef}>
      <mesh ref={meshRef} position={[orbitRadius, 0, 0]} onClick={onClick}>
        <sphereGeometry args={[size, 32, 32]} />
        <meshStandardMaterial color={color} roughness={0.7} metalness={0.3} />
        {isSelected && (
          <mesh>
            <ringGeometry args={[size + 0.08, size + 0.12, 32]} />
            <meshBasicMaterial color="#ffffff" side={THREE.DoubleSide} transparent opacity={0.8} />
          </mesh>
        )}
      </mesh>
    </group>
  );
}

function OrbitPath({ radius }: { radius: number }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[radius - 0.02, radius + 0.02, 128]} />
      <meshBasicMaterial color="#333333" side={THREE.DoubleSide} transparent opacity={0.4} />
    </mesh>
  );
}

function StarSystemScene({ planets, selectedStarIdx, onSystemClick, onPlanetClick }: { 
  planets: any[]; 
  selectedStarIdx: number;
  onSystemClick: (idx: number) => void;
  onPlanetClick: (sysIdx: number, predIdx: number) => void;
}) {
  return (
    <>
      <ambientLight intensity={0.5} />
      <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
      <Star />
      {planets.map((planet, i) => (
        <OrbitPath key={`orbit-${i}`} radius={planet.orbitRadius} />
      ))}
      {planets.map((planet, i) => (
        <Planet
          key={planet.id}
          orbitRadius={planet.orbitRadius}
          size={planet.size}
          color={planet.color}
          orbitSpeed={planet.orbitSpeed}
          isSelected={selectedStarIdx === planet.systemIdx}
          onClick={() => onPlanetClick(planet.systemIdx, planet.predIdx)}
        />
      ))}
      <OrbitControls 
        enablePan={true} 
        enableZoom={true} 
        enableRotate={true} 
        minDistance={3} 
        maxDistance={25}
        autoRotate={false}
        makeDefault
      />
    </>
  );
}

const Visualizer = () => {
  const navigate = useNavigate();
  const [starSystems, setStarSystems] = useState<StarSystem[]>([]);
  const [selectedStarIdx, setSelectedStarIdx] = useState(0);
  const [selectedPlanetIdx, setSelectedPlanetIdx] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const response = await fetch("http://localhost:8000/api/history?limit=200&sort=timestamp&order=desc");
        if (!response.ok) throw new Error("Failed to fetch history");
        const data: HistoryResponse = await response.json();
        
        // Group predictions by star_name
        const grouped: { [key: string]: PredictionItem[] } = {};
        data.items.forEach((pred) => {
          if (!grouped[pred.star_name]) {
            grouped[pred.star_name] = [];
          }
          grouped[pred.star_name].push(pred);
        });
        
        const systems: StarSystem[] = Object.entries(grouped).map(([name, preds]) => ({
          star_name: name,
          predictions: preds,
        }));
        
        setStarSystems(systems);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
        console.error("Error fetching history:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  // Convert star systems to 3D planets/orbit objects
  const systems3d = starSystems.map((sys, sysIdx) =>
    sys.predictions.map((pred, predIdx) => ({
      id: pred.id,
      star_name: sys.star_name,
      name: `${sys.star_name} ${String.fromCharCode(98 + predIdx)}`, // kepler-10 b, c, d, etc
      mission: pred.mission,
      period: pred.period_days,
      probability: pred.percentage,
      orbitRadius: 2 + predIdx * 1.5,
      size: 0.08 + (pred.percentage / 100) * 0.2,
      color: pred.verdict === "TRANSIT_DETECTED" ? "#22c55e" : "#ef4444",
      orbitSpeed: 1 - predIdx * 0.2,
      systemIdx: sysIdx,
      predIdx: predIdx,
    }))
  ).flat();

  const selectedStar = starSystems[selectedStarIdx];
  const selectedPred = selectedStar?.predictions[selectedPlanetIdx];

  // Convert folded lightcurve to chart data
  const chartData = selectedPred?.folded_lightcurve
    ? selectedPred.folded_lightcurve.map((value, i) => ({
        phase: ((i / selectedPred.folded_lightcurve!.length) * 2 - 1).toFixed(3),
        flux: parseFloat(value.toFixed(5)),
      }))
    : [];

  if (loading) {
    return (
      <div className="h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-2">
          <LoaderCircle className="h-8 w-8 animate-spin text-foreground" />
          <p className="font-data text-sm text-muted-foreground">Loading predictions...</p>
        </div>
      </div>
    );
  }

  if (error || starSystems.length === 0) {
    return (
      <div className="h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <p className="font-data text-sm text-red-500">{error || "No predictions available"}</p>
          <Button onClick={() => navigate("/dashboard")}>Back to Dashboard</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-background flex">
      {/* 3D Canvas */}
      <div className="flex-1 relative">
        {/* Top bar */}
        <div className="absolute top-0 left-0 right-0 z-10 flex items-center gap-4 px-4 py-3 bg-background/80 backdrop-blur-sm border-b border-border">
          <Button variant="ghost" size="icon" onClick={() => navigate("/dashboard")} className="text-muted-foreground hover:text-foreground h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div className="flex items-center gap-2">
            <div onClick={() => navigate("/")} className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity">
              <Terminal className="h-4 w-4 text-foreground" />
              <span className="font-data text-xs font-semibold tracking-wide">COSMIK_AI</span>
            </div>
            <span className="font-data text-xs text-muted-foreground">/ VISUALIZER</span>
          </div>
          <div className="flex-1 max-w-sm ml-auto">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
              <Input
                placeholder="Search stars..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9 bg-card border-border font-data text-xs focus-visible:ring-1 focus-visible:ring-foreground rounded-md h-8"
              />
            </div>
          </div>
        </div>

        {/* WebGL Canvas */}
        <Canvas
          camera={{ position: [0, 8, 15], fov: 50 }}
          gl={{ antialias: true }}
          style={{ background: 'transparent' }}
        >
          <Suspense fallback={null}>
            <StarSystemScene 
              planets={systems3d} 
              selectedStarIdx={selectedStarIdx}
              onSystemClick={(idx) => {
                setSelectedStarIdx(idx);
                setSelectedPlanetIdx(0);
              }}
              onPlanetClick={(sysIdx, predIdx) => {
                setSelectedStarIdx(sysIdx);
                setSelectedPlanetIdx(predIdx);
              }}
            />
          </Suspense>
        </Canvas>

        {/* Instructions */}
        <div className="absolute bottom-4 left-4 font-data text-[10px] text-muted-foreground uppercase tracking-wider bg-background/60 backdrop-blur-sm px-3 py-2 rounded-md border border-border">
          Click star or planet • {starSystems.length} star systems
        </div>
      </div>

      {/* Right Sidebar */}
      <motion.aside
        initial={{ x: 100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.4 }}
        className="w-[380px] border-l border-border bg-background overflow-y-auto"
      >
        <div className="p-4 space-y-4">
          {selectedStar && selectedPred && (
            <>
              {/* System info */}
              <div className="panel p-4">
                <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider mb-1">Star System</p>
                <p className="font-data text-sm font-semibold">{selectedStar.star_name}</p>
                <p className="text-xs text-muted-foreground mt-1">{selectedStar.predictions.length} detection{selectedStar.predictions.length !== 1 ? 's' : ''}</p>
              </div>

              {/* Planet selector for this system */}
              {selectedStar.predictions.length > 1 && (
                <div className="flex flex-col gap-2">
                  <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider">Planets ({selectedPlanetIdx + 1}/{selectedStar.predictions.length})</p>
                  <div className="flex gap-2 flex-wrap">
                    {selectedStar.predictions.map((_, idx) => (
                      <button
                        key={idx}
                        onClick={() => setSelectedPlanetIdx(idx)}
                        className={`px-3 py-2 rounded-md font-data text-xs transition-colors ${
                          selectedPlanetIdx === idx
                            ? "bg-foreground text-background"
                            : "bg-card border border-border text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {String.fromCharCode(98 + idx).toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Star system browser */}
              <div className="flex flex-col gap-2">
                <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider">Browse Systems ({selectedStarIdx + 1}/{starSystems.length})</p>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setSelectedStarIdx(Math.max(0, selectedStarIdx - 1));
                      setSelectedPlanetIdx(0);
                    }}
                    disabled={selectedStarIdx === 0}
                  >
                    ← Prev System
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setSelectedStarIdx(Math.min(starSystems.length - 1, selectedStarIdx + 1));
                      setSelectedPlanetIdx(0);
                    }}
                    disabled={selectedStarIdx === starSystems.length - 1}
                  >
                    Next System →
                  </Button>
                </div>
              </div>

              {/* Prediction Stats */}
              <div className="panel p-4">
                <div className="flex items-center justify-between mb-4">
                  <p className="font-data text-xs uppercase tracking-wider">{selectedPred.verdict}</p>
                  <span className={`px-2 py-0.5 rounded-md font-data text-[10px] uppercase tracking-wider ${
                    selectedPred.verdict === "TRANSIT_DETECTED"
                      ? "bg-success/20 text-success"
                      : "bg-destructive/20 text-destructive"
                  }`}>
                    {selectedPred.percentage.toFixed(1)}% PROB
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: "Period (days)", value: selectedPred.period_days?.toFixed(3) ?? "—" },
                    { label: "Score", value: selectedPred.score.toFixed(4) },
                    { label: "Mission", value: selectedPred.mission },
                    { label: "Timestamp", value: new Date(selectedPred.timestamp).toLocaleTimeString() },
                  ].map((s) => (
                    <div key={s.label} className="bg-muted p-3 rounded-md">
                      <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider mb-1">{s.label}</p>
                      <p className="font-data text-xs font-semibold truncate">{s.value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Light Curve */}
              {chartData.length > 0 && (
                <div className="panel p-4">
                  <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider mb-4">Folded Light Curve</p>
                  <div className="bg-background rounded-md p-2 border border-border">
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="2 2" stroke="hsl(225 6% 16%)" />
                        <XAxis
                          dataKey="phase"
                          tick={{ fill: "hsl(220 5% 55%)", fontSize: 9, fontFamily: "JetBrains Mono" }}
                          tickLine={{ stroke: "hsl(225 6% 16%)" }}
                          axisLine={{ stroke: "hsl(225 6% 16%)" }}
                        />
                        <YAxis
                          tick={{ fill: "hsl(220 5% 55%)", fontSize: 9, fontFamily: "JetBrains Mono" }}
                          tickLine={{ stroke: "hsl(225 6% 16%)" }}
                          axisLine={{ stroke: "hsl(225 6% 16%)" }}
                          tickFormatter={(v) => v.toFixed(3)}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "hsl(225 6% 8%)",
                            border: "1px solid hsl(225 6% 16%)",
                            borderRadius: "6px",
                            fontFamily: "JetBrains Mono",
                            fontSize: 10,
                          }}
                          labelStyle={{ color: "hsl(220 5% 55%)" }}
                          itemStyle={{ color: "hsl(220 5% 90%)" }}
                        />
                        <Line
                          type="monotone"
                          dataKey="flux"
                          stroke={selectedPred.verdict === "TRANSIT_DETECTED" ? "#22c55e" : "#ef4444"}
                          strokeWidth={1.5}
                          dot={false}
                          activeDot={{ r: 3 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </motion.aside>
    </div>
  );
};

export default Visualizer;
