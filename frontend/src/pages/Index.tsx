import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Terminal, ArrowRight, Activity, Radar, Satellite, Search, Loader2, Clock, Zap, Target, CheckCircle2, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

interface HistoryItem {
  id: number;
  star_name: string;
  mission: string;
  score: number;
  percentage: number;
  period_days: number | null;
  transit_depth_estimate: number | null;
  verdict: string;
  timestamp: string;
  folded_lightcurve: number[] | null;
}

interface HistoryResponse {
  items: HistoryItem[];
  total: number;
}

interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string;
  model_auprc: number;
  total_predictions: number;
  uptime_seconds: number;
}

interface StatsResponse {
  total_analyzed: number;
  average_score: number;
  detection_rate: number;
  missions: Record<string, number>;
  above_threshold: number;
  below_threshold: number;
}

interface PredictionResult {
  star_name: string;
  mission: string;
  score: number;
  percentage: number;
  period_days: number | null;
  verdict: string;
  transit_depth_estimate: number | null;
  duration_estimate: number | null;
  num_datapoints: number | null;
  cached: boolean;
  processing_time_seconds: number;
  timestamp: string;
}

const MISSION_MAP: Record<string, string> = {
  kepler: "Kepler",
  k2: "K2",
  tess: "TESS",
};

const fallbackFluxSignalData = [
  { phase: 0, flux: 1.0012 },
  { phase: 0.1, flux: 1.0008 },
  { phase: 0.2, flux: 1.0001 },
  { phase: 0.3, flux: 0.9995 },
  { phase: 0.4, flux: 0.9987 },
  { phase: 0.5, flux: 0.9982 },
  { phase: 0.6, flux: 0.9988 },
  { phase: 0.7, flux: 0.9996 },
  { phase: 0.8, flux: 1.0003 },
  { phase: 0.9, flux: 1.0009 },
  { phase: 1, flux: 1.0013 },
];

const Index = () => {
  const navigate = useNavigate();
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [statsData, setStatsData] = useState<StatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedMission, setSelectedMission] = useState("kepler");
  const [threshold, setThreshold] = useState([50]);
  const [starQuery, setStarQuery] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

    const fetchTelemetry = async () => {
      setLoading(true);
      try {
        const [historyRes, healthRes, statsRes] = await Promise.allSettled([
          fetch(`${apiBase}/api/history?limit=120&offset=0`, { signal: controller.signal }),
          fetch(`${apiBase}/api/health`, { signal: controller.signal }),
          fetch(`${apiBase}/api/stats`, { signal: controller.signal }),
        ]);

        if (historyRes.status === "fulfilled" && historyRes.value.ok) {
          const historyJson = (await historyRes.value.json()) as HistoryResponse;
          setHistoryItems(historyJson.items ?? []);
          setHistoryTotal(historyJson.total ?? 0);
        }

        if (healthRes.status === "fulfilled" && healthRes.value.ok) {
          const healthJson = (await healthRes.value.json()) as HealthResponse;
          setHealth(healthJson);
        }

        if (statsRes.status === "fulfilled" && statsRes.value.ok) {
          const statsJson = (await statsRes.value.json()) as StatsResponse;
          setStatsData(statsJson);
        }
      } catch (err) {
        console.error("Homepage telemetry fetch failed:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchTelemetry();
    return () => controller.abort();
  }, []);

  const fluxSignalData = useMemo(() => {
    const source = historyItems.find((item) => Array.isArray(item.folded_lightcurve) && item.folded_lightcurve.length > 10);
    if (!source?.folded_lightcurve) {
      return fallbackFluxSignalData;
    }

    const curve = source.folded_lightcurve;
    const stride = Math.max(1, Math.floor(curve.length / 120));
    const sampled = curve.filter((_, idx) => idx % stride === 0);
    const denom = Math.max(1, sampled.length - 1);

    return sampled.map((flux, idx) => ({
      phase: Number((idx / denom).toFixed(3)),
      flux: Number((Number(flux) || 0).toFixed(6)),
    }));
  }, [historyItems]);

  const chartYDomain = useMemo(() => {
    const values = fluxSignalData.map((point) => point.flux).filter((v) => Number.isFinite(v));
    if (values.length === 0) return [0.9975, 1.002] as [number, number];
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const padding = Math.max(0.0003, (maxVal - minVal) * 0.2);
    return [Number((minVal - padding).toFixed(6)), Number((maxVal + padding).toFixed(6))] as [number, number];
  }, [fluxSignalData]);

  const detectionCount = useMemo(
    () => historyItems.filter((item) => item.verdict === "TRANSIT_DETECTED" || item.percentage >= 50).length,
    [historyItems],
  );

  const uniqueMissionList = useMemo(
    () => Array.from(new Set(historyItems.map((item) => item.mission).filter(Boolean))),
    [historyItems],
  );

  const statCards = [
    {
      icon: Activity,
      label: "Model Status",
      value: health?.model_loaded ? `ONLINE (${health.model_version})` : loading ? "CHECKING..." : "OFFLINE",
    },
    {
      icon: Radar,
      label: "Candidate Signals",
      value: detectionCount > 0 ? detectionCount.toLocaleString() : loading ? "..." : "0",
    },
    {
      icon: Satellite,
      label: "Mission Feeds",
      value: uniqueMissionList.length > 0 ? uniqueMissionList.slice(0, 3).join("/") : "Kepler/K2/TESS",
    },
  ];

  const totalPredictionsLabel = (health?.total_predictions ?? historyTotal).toLocaleString();

  const telemetryStats = [
    {
      label: "MODEL_ACCURACY",
      value: health ? `${(health.model_auprc * 100).toFixed(1)}%` : "—",
      sub: "AUPRC",
      icon: Target,
    },
    {
      label: "PROCESSED_TARGETS",
      value: statsData ? statsData.total_analyzed.toLocaleString() : "—",
      sub: "stars analysed",
      icon: Activity,
    },
    {
      label: "CANDIDATE_FLAGS",
      value: statsData ? statsData.above_threshold.toLocaleString() : "—",
      sub: `score ≥ ${threshold[0]}%`,
      icon: CheckCircle2,
    },
    {
      label: "FALSE_POSITIVES",
      value: statsData ? statsData.below_threshold.toLocaleString() : "—",
      sub: `score < ${threshold[0]}%`,
      icon: AlertTriangle,
    },
  ];

  const handleQueryArchive = async () => {
    if (!starQuery.trim()) return;

    const apiBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";
    setIsPredicting(true);
    setPredictionError(null);

    try {
      const res = await fetch(`${apiBase}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          star_name: starQuery.trim(),
          mission: MISSION_MAP[selectedMission] ?? "Kepler",
          threshold: threshold[0] / 100,
        }),
        signal: AbortSignal.timeout(195_000),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail ?? `Server error: ${res.status}`);
      }

      const data = (await res.json()) as PredictionResult;
      setPredictionResult(data);

      // Refresh dashboard telemetry after prediction.
      const [historyRes, healthRes, statsRes] = await Promise.allSettled([
        fetch(`${apiBase}/api/history?limit=120&offset=0`),
        fetch(`${apiBase}/api/health`),
        fetch(`${apiBase}/api/stats`),
      ]);

      if (historyRes.status === "fulfilled" && historyRes.value.ok) {
        const historyJson = (await historyRes.value.json()) as HistoryResponse;
        setHistoryItems(historyJson.items ?? []);
        setHistoryTotal(historyJson.total ?? 0);
      }

      if (healthRes.status === "fulfilled" && healthRes.value.ok) {
        const healthJson = (await healthRes.value.json()) as HealthResponse;
        setHealth(healthJson);
      }

      if (statsRes.status === "fulfilled" && statsRes.value.ok) {
        const statsJson = (await statsRes.value.json()) as StatsResponse;
        setStatsData(statsJson);
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "TimeoutError") {
        setPredictionError("Request timed out after 3 minutes. MAST may be slow, please try again.");
      } else if (err instanceof Error) {
        setPredictionError(err.message);
      } else {
        setPredictionError("An unexpected error occurred.");
      }
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border px-6 py-4">
        <div className="flex items-center gap-3">
          <Terminal className="h-5 w-5 text-foreground" />
          <span className="font-data text-sm font-semibold tracking-wide">COSMIK_AI</span>
        </div>
      </header>

      <section className="px-2 py-8 md:px-4 md:py-8">
        <div className="mx-auto max-w-[1720px]">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="grid gap-5 lg:grid-cols-[0.95fr_1.05fr]"
          >
            <div className="space-y-4">
              <p className="font-data text-[10px] text-muted-foreground uppercase tracking-widest mb-2">
                Astrophysics Data Pipeline v2.4.1
              </p>
              <h1 className="text-2xl md:text-[2rem] font-semibold leading-tight tracking-tight">
                Stellar Transit Analysis
              </h1>
              <div className="max-w-3xl text-sm text-muted-foreground leading-relaxed space-y-3">
                <p>
                  Compact command center for loading missions, running inference, and inspecting candidate transit signals in real time.
                </p>
                <p className="text-xs">
                  Missions: <span className="font-data text-foreground">Kepler</span>, <span className="font-data text-foreground">K2</span>, <span className="font-data text-foreground">TESS</span>.
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <Button
                  onClick={() => document.getElementById("inference-workbench")?.scrollIntoView({ behavior: "smooth", block: "start" })}
                  className="bg-foreground text-background hover:bg-foreground/90 font-data text-xs uppercase tracking-wider px-6 rounded-md"
                >
                  Run Inference
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
                <Button
                  variant="outline"
                  onClick={() => navigate("/visualizer")}
                  className="border-border text-muted-foreground hover:text-foreground hover:bg-card font-data text-xs uppercase tracking-wider px-6 rounded-md"
                >
                  Open Visualizer
                </Button>
              </div>

              <div className="grid gap-3 sm:grid-cols-3">
                {statCards.map((stat) => (
                  <div key={stat.label} className="panel p-4">
                    <stat.icon className="h-4 w-4 text-muted-foreground mb-2" />
                    <p className="font-data text-[10px] uppercase tracking-wider text-muted-foreground mb-1">{stat.label}</p>
                    <p className="font-data text-sm text-foreground truncate">{stat.value}</p>
                  </div>
                ))}
              </div>

              <div id="inference-workbench" className="panel p-4 md:p-5 space-y-4">
                <div>
                  <p className="font-data text-[10px] uppercase tracking-wider text-muted-foreground">Inference Workbench</p>
                  <p className="text-xs text-muted-foreground mt-1">Query MAST and run transit detection directly from this dashboard.</p>
                </div>

                <div className="space-y-2">
                  <Label className="font-data text-[10px] uppercase tracking-wider text-muted-foreground">Star Identifier</Label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
                    <Input
                      value={starQuery}
                      onChange={(e) => setStarQuery(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && !isPredicting && handleQueryArchive()}
                      placeholder="e.g. Kepler-10, TOI-700"
                      disabled={isPredicting}
                      className="pl-9 bg-background border-border font-data text-sm"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <Label className="font-data text-[10px] uppercase tracking-wider text-muted-foreground">Mission</Label>
                    <Select value={selectedMission} onValueChange={setSelectedMission} disabled={isPredicting}>
                      <SelectTrigger className="bg-background border-border font-data text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-card border-border">
                        <SelectItem value="kepler">Kepler</SelectItem>
                        <SelectItem value="k2">K2</SelectItem>
                        <SelectItem value="tess">TESS</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label className="font-data text-[10px] uppercase tracking-wider text-muted-foreground">Threshold</Label>
                    <div className="rounded-md border border-border bg-card/40 px-3 py-2">
                      <div className="flex items-center justify-between mb-2 font-data text-[10px] text-muted-foreground">
                        <span>Conservative</span>
                        <span className="text-foreground">{threshold[0]}%</span>
                        <span>Aggressive</span>
                      </div>
                      <Slider
                        value={threshold}
                        onValueChange={setThreshold}
                        min={30}
                        max={99}
                        step={1}
                        className="[&_[role=slider]]:bg-foreground [&_[role=slider]]:border-foreground"
                      />
                    </div>
                  </div>
                </div>

                <Button
                  className="w-full bg-foreground text-background hover:bg-foreground/90 font-data text-xs uppercase tracking-wider"
                  onClick={handleQueryArchive}
                  disabled={isPredicting || !starQuery.trim()}
                >
                  {isPredicting ? (
                    <>
                      <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                      Downloading from MAST...
                    </>
                  ) : (
                    <>
                      <Search className="mr-2 h-3 w-3" />
                      Query Archive
                    </>
                  )}
                </Button>

                {predictionError && (
                  <p className="text-xs text-destructive">{predictionError}</p>
                )}

                {predictionResult && (
                  <div className="rounded-md border border-border bg-card/40 p-3 space-y-3">
                    <div className="flex items-center justify-between gap-2">
                      <p className="font-data text-xs text-foreground truncate">{predictionResult.star_name}</p>
                      <span className={`font-data text-[10px] uppercase tracking-wider ${predictionResult.verdict === "TRANSIT_DETECTED" ? "text-green-400" : "text-amber-300"}`}>
                        {predictionResult.verdict.replace("_", " ")}
                      </span>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <div className="rounded-sm bg-background/50 border border-border/70 p-2">
                        <p className="font-data text-[10px] text-muted-foreground">Confidence</p>
                        <p className="font-data text-xs text-foreground">{predictionResult.percentage.toFixed(1)}%</p>
                      </div>
                      <div className="rounded-sm bg-background/50 border border-border/70 p-2">
                        <p className="font-data text-[10px] text-muted-foreground">Period</p>
                        <p className="font-data text-xs text-foreground">{predictionResult.period_days != null ? `${predictionResult.period_days.toFixed(3)} d` : "-"}</p>
                      </div>
                      <div className="rounded-sm bg-background/50 border border-border/70 p-2">
                        <p className="font-data text-[10px] text-muted-foreground">Runtime</p>
                        <p className="font-data text-xs text-foreground">{predictionResult.processing_time_seconds.toFixed(1)} s</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            <div className="space-y-5">
              <div className="panel p-4 md:p-5">
                <div className="flex items-center justify-between mb-3">
                  <p className="font-data text-[10px] uppercase tracking-wider text-muted-foreground">Transit Signal Preview</p>
                  <p className="font-data text-[10px] text-foreground">{loading ? "Syncing" : `${totalPredictionsLabel} Total Predictions`}</p>
                </div>

                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={fluxSignalData} margin={{ top: 6, right: 6, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="fluxFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="rgba(229,231,235,0.30)" />
                          <stop offset="100%" stopColor="rgba(229,231,235,0.03)" />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.18)" />
                      <XAxis dataKey="phase" tick={{ fill: "rgba(148,163,184,0.85)", fontSize: 10 }} tickLine={false} axisLine={false} />
                      <YAxis
                        tick={{ fill: "rgba(148,163,184,0.85)", fontSize: 10 }}
                        tickLine={false}
                        axisLine={false}
                        domain={chartYDomain}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "rgba(2,6,23,0.95)",
                          border: "1px solid rgba(148,163,184,0.35)",
                          borderRadius: "6px",
                          fontFamily: "JetBrains Mono",
                          fontSize: 10,
                        }}
                        labelStyle={{ color: "rgba(203,213,225,0.9)" }}
                        itemStyle={{ color: "rgba(241,245,249,0.95)" }}
                      />
                      <Area type="monotone" dataKey="flux" stroke="rgba(229,231,235,0.9)" strokeWidth={1.8} fill="url(#fluxFill)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="panel p-4 md:p-5">
                <div className="flex items-center justify-between mb-3">
                  <p className="font-data text-[10px] uppercase tracking-wider text-muted-foreground">System Telemetry</p>
                  <p className="font-data text-[10px] text-muted-foreground">live stats</p>
                </div>

                <div className="grid gap-3 grid-cols-2">
                  {telemetryStats.map((stat) => (
                    <div key={stat.label} className="rounded-md border border-border bg-card/40 p-3">
                      <div className="flex items-center justify-between mb-1">
                        <p className="font-data text-[10px] uppercase tracking-wider text-muted-foreground">{stat.label}</p>
                        <stat.icon className="h-3.5 w-3.5 text-muted-foreground" />
                      </div>
                      <p className="font-data text-lg text-foreground leading-none">{stat.value}</p>
                      <p className="font-data text-[10px] text-muted-foreground mt-1">{stat.sub}</p>
                    </div>
                  ))}
                </div>

                <div className="mt-4">
                  <p className="font-data text-[10px] uppercase tracking-wider text-muted-foreground mb-2">Mission Breakdown</p>
                  <div className="flex flex-wrap gap-2">
                    {statsData && Object.keys(statsData.missions).length > 0 ? (
                      Object.entries(statsData.missions).map(([mission, count]) => (
                        <div key={mission} className="rounded-md border border-border bg-background/50 px-3 py-1.5">
                          <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider">{mission}</p>
                          <p className="font-data text-sm text-foreground">{count}</p>
                        </div>
                      ))
                    ) : (
                      <p className="text-xs text-muted-foreground">No mission stats yet.</p>
                    )}
                  </div>
                </div>
              </div>

              <div className="panel p-4 md:p-5">
                <div className="flex items-center justify-between mb-3">
                  <p className="font-data text-[10px] uppercase tracking-wider text-muted-foreground">Detection Archive</p>
                  <p className="font-data text-[10px] text-muted-foreground">{historyTotal} records</p>
                </div>

                {historyItems.length === 0 ? (
                  <p className="text-xs text-muted-foreground">No detections yet. Run a query to populate the archive.</p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full font-data text-xs">
                      <thead>
                        <tr className="border-b border-border text-muted-foreground uppercase tracking-wider">
                          <th className="text-left py-2 px-2 font-medium">Star</th>
                          <th className="text-left py-2 px-2 font-medium">Mission</th>
                          <th className="text-left py-2 px-2 font-medium">Period</th>
                          <th className="text-left py-2 px-2 font-medium">Confidence</th>
                          <th className="text-left py-2 px-2 font-medium">Age</th>
                        </tr>
                      </thead>
                      <tbody>
                        {historyItems.slice(0, 8).map((row) => (
                          <tr key={row.id} className="border-b border-border/70 hover:bg-card/30 transition-colors">
                            <td className="py-2 px-2 text-foreground truncate max-w-[220px]">{row.star_name}</td>
                            <td className="py-2 px-2 text-muted-foreground">{row.mission}</td>
                            <td className="py-2 px-2 text-muted-foreground">{row.period_days != null ? row.period_days.toFixed(3) : "-"}</td>
                            <td className="py-2 px-2 text-foreground">{row.percentage.toFixed(1)}%</td>
                            <td className="py-2 px-2 text-muted-foreground">
                              <span className="inline-flex items-center gap-1"><Clock className="h-3 w-3" />{new Date(row.timestamp).toLocaleDateString()}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  <div className="rounded-md border border-border bg-card/40 p-3">
                    <p className="font-data text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Detection Rate</p>
                    <p className="font-data text-lg text-foreground">
                      {historyItems.length > 0 ? `${((detectionCount / historyItems.length) * 100).toFixed(1)}%` : "-"}
                    </p>
                  </div>
                  <div className="rounded-md border border-border bg-card/40 p-3">
                    <p className="font-data text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Mean Depth</p>
                    <p className="font-data text-lg text-foreground inline-flex items-center gap-1">
                      <Zap className="h-4 w-4" />
                      {historyItems.length > 0
                        ? `${(
                            (historyItems.reduce((sum, item) => sum + Math.abs(item.transit_depth_estimate ?? 0), 0) /
                              Math.max(1, historyItems.length)) *
                            100
                          ).toFixed(4)}%`
                        : "-"}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      <footer className="border-t border-border px-2 py-3 md:px-4">
        <div className="mx-auto max-w-[1720px] flex items-center justify-between">
          <p className="font-data text-xs text-muted-foreground">
            © 2024 Cosmik AI Research
          </p>
          <p className="font-data text-xs text-muted-foreground">
            Build: <span className="text-foreground">2024.03.15</span>
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
