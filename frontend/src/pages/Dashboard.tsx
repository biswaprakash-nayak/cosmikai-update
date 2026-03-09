import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import {
  Upload, Search, Sliders, BarChart3, Terminal, Eye,
  FileUp, CheckCircle2, AlertTriangle, Target, Activity
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const stats = [
  { label: "MODEL_ACCURACY", value: "94.7%", icon: Target },
  { label: "PROCESSED_TARGETS", value: "12,847", icon: Activity },
  { label: "CANDIDATE_FLAGS", value: "342", icon: CheckCircle2 },
  { label: "FALSE_POSITIVES", value: "18", icon: AlertTriangle },
];

const detectionHistory = [
  { star: "Kepler-10", planets: 2, confidence: 94.7, mission: "Kepler" },
  { star: "K2-18", planets: 1, confidence: 91.2, mission: "K2" },
  { star: "TOI-700", planets: 3, confidence: 88.5, mission: "TESS" },
  { star: "Kepler-442", planets: 1, confidence: 96.1, mission: "Kepler" },
  { star: "TOI-1338", planets: 1, confidence: 85.3, mission: "TESS" },
];

const Dashboard = () => {
  const navigate = useNavigate();
  const [threshold, setThreshold] = useState([65]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [features, setFeatures] = useState({
    transitDepth: true,
    orbitalPeriod: true,
    snr: false,
    transitDuration: true,
  });

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div onClick={() => navigate("/")} className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity">
            <Terminal className="h-5 w-5 text-foreground" />
            <span className="font-data text-sm font-semibold tracking-wide">COSMIK_AI</span>
          </div>
          <span className="font-data text-xs text-muted-foreground ml-4">/ DASHBOARD</span>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 font-data text-xs">
            <span className="h-2 w-2 rounded-full bg-success" />
            <span className="text-muted-foreground">MODEL_STATUS:</span>
            <span className="text-success">ACTIVE</span>
          </div>
          <Button variant="ghost" size="sm" onClick={() => navigate("/visualizer")} className="font-data text-xs uppercase tracking-wider text-muted-foreground hover:text-foreground">
            <Eye className="mr-2 h-4 w-4" /> Visualizer
          </Button>
        </div>
      </header>

      {/* Content */}
      <main className="p-6">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
          <Tabs defaultValue="input" className="space-y-6">
            <TabsList className="bg-card border border-border rounded-md p-1 h-auto">
              <TabsTrigger value="input" className="font-data text-xs uppercase tracking-wider data-[state=active]:bg-muted data-[state=active]:text-foreground rounded-md px-4 py-2">
                <Upload className="mr-2 h-3 w-3" /> FITS/CSV Ingestion
              </TabsTrigger>
              <TabsTrigger value="config" className="font-data text-xs uppercase tracking-wider data-[state=active]:bg-muted data-[state=active]:text-foreground rounded-md px-4 py-2">
                <Sliders className="mr-2 h-3 w-3" /> Model Parameters
              </TabsTrigger>
              <TabsTrigger value="stats" className="font-data text-xs uppercase tracking-wider data-[state=active]:bg-muted data-[state=active]:text-foreground rounded-md px-4 py-2">
                <BarChart3 className="mr-2 h-3 w-3" /> System Telemetry
              </TabsTrigger>
            </TabsList>

            {/* Data Input Tab */}
            <TabsContent value="input">
              <div className="grid gap-6 lg:grid-cols-2">
                {/* File Upload */}
                <div
                  className={`panel p-6 flex flex-col items-center justify-center min-h-[280px] transition-colors cursor-pointer ${isDragOver ? "border-foreground bg-muted" : "hover:border-muted-foreground"}`}
                  onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                  onDragLeave={() => setIsDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setIsDragOver(false); }}
                >
                  <FileUp className="h-8 w-8 text-muted-foreground mb-4" />
                  <p className="font-data text-xs uppercase tracking-wider mb-1">Drop Light Curve Files</p>
                  <p className="text-xs text-muted-foreground mb-4">FITS, CSV, TSV formats accepted</p>
                  <Button variant="outline" size="sm" className="font-data text-xs uppercase tracking-wider border-border hover:bg-muted rounded-md">
                    Browse Files
                  </Button>
                </div>

                {/* Query by Star */}
                <div className="panel p-6 space-y-6">
                  <div>
                    <p className="font-data text-xs uppercase tracking-wider text-foreground mb-1">Query by Star ID</p>
                    <p className="text-xs text-muted-foreground">Search mission archives for photometry data</p>
                  </div>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label className="font-data text-xs uppercase tracking-wider text-muted-foreground">Star Identifier</Label>
                      <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
                        <Input placeholder="e.g. Kepler-10, TOI-700" className="pl-9 bg-background border-border font-data text-sm focus-visible:ring-1 focus-visible:ring-foreground rounded-md" />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label className="font-data text-xs uppercase tracking-wider text-muted-foreground">Mission</Label>
                      <Select defaultValue="kepler">
                        <SelectTrigger className="bg-background border-border font-data text-sm rounded-md">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-card border-border rounded-md">
                          <SelectItem value="kepler" className="font-data text-sm">Kepler</SelectItem>
                          <SelectItem value="k2" className="font-data text-sm">K2</SelectItem>
                          <SelectItem value="tess" className="font-data text-sm">TESS</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Button className="w-full bg-foreground text-background hover:bg-foreground/90 font-data text-xs uppercase tracking-wider rounded-md">
                      <Search className="mr-2 h-3 w-3" /> Query Archive
                    </Button>
                  </div>
                </div>
              </div>
            </TabsContent>

            {/* Model Config Tab */}
            <TabsContent value="config">
              <div className="grid gap-6 lg:grid-cols-2">
                <div className="panel p-6 space-y-6">
                  <div>
                    <p className="font-data text-xs uppercase tracking-wider text-foreground mb-1">Detection Threshold</p>
                    <p className="text-xs text-muted-foreground">Configure sensitivity of transit detection</p>
                  </div>
                  <div className="space-y-4">
                    <div className="flex justify-between font-data text-xs">
                      <span className="text-muted-foreground">Conservative</span>
                      <span className="text-foreground">{threshold[0]}%</span>
                      <span className="text-muted-foreground">Aggressive</span>
                    </div>
                    <Slider value={threshold} onValueChange={setThreshold} min={30} max={99} step={1} className="[&_[role=slider]]:bg-foreground [&_[role=slider]]:border-foreground" />
                  </div>
                  <div className="space-y-2">
                    <Label className="font-data text-xs uppercase tracking-wider text-muted-foreground">Model Type</Label>
                    <Select defaultValue="rf">
                      <SelectTrigger className="bg-background border-border font-data text-sm rounded-md">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-card border-border rounded-md">
                        <SelectItem value="rf" className="font-data text-sm">Random Forest</SelectItem>
                        <SelectItem value="xgb" className="font-data text-sm">XGBoost</SelectItem>
                        <SelectItem value="nn" className="font-data text-sm">Neural Network</SelectItem>
                        <SelectItem value="svm" className="font-data text-sm">SVM</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="panel p-6 space-y-6">
                  <div>
                    <p className="font-data text-xs uppercase tracking-wider text-foreground mb-1">Feature Extraction</p>
                    <p className="text-xs text-muted-foreground">Select features for model input</p>
                  </div>
                  <div className="space-y-3">
                    {[
                      { key: "transitDepth", label: "TRANSIT_DEPTH (δ)" },
                      { key: "orbitalPeriod", label: "ORBITAL_PERIOD (P)" },
                      { key: "snr", label: "SIGNAL_TO_NOISE (SNR)" },
                      { key: "transitDuration", label: "TRANSIT_DURATION (T₁₄)" },
                    ].map((f) => (
                      <label key={f.key} className="flex items-center gap-3 p-3 bg-background border border-border rounded-md hover:border-muted-foreground cursor-pointer transition-colors">
                        <Checkbox
                          checked={features[f.key as keyof typeof features]}
                          onCheckedChange={(checked) => setFeatures((p) => ({ ...p, [f.key]: !!checked }))}
                          className="border-border data-[state=checked]:bg-foreground data-[state=checked]:border-foreground rounded-sm"
                        />
                        <span className="font-data text-xs tracking-wide">{f.label}</span>
                      </label>
                    ))}
                  </div>
                  <Button className="w-full bg-foreground text-background hover:bg-foreground/90 font-data text-xs uppercase tracking-wider rounded-md">
                    Apply Configuration
                  </Button>
                </div>
              </div>
            </TabsContent>

            {/* Stats Tab */}
            <TabsContent value="stats">
              <div className="space-y-6">
                <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
                  {stats.map((stat, i) => (
                    <motion.div
                      key={stat.label}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.05 }}
                      className="panel p-4"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-data text-[10px] text-muted-foreground uppercase tracking-wider">{stat.label}</span>
                        <stat.icon className="h-3 w-3 text-muted-foreground" />
                      </div>
                      <p className="font-data text-2xl font-semibold">{stat.value}</p>
                    </motion.div>
                  ))}
                </div>

                <div className="panel p-6">
                  <p className="font-data text-xs uppercase tracking-wider text-foreground mb-4">Detection History</p>
                  <div className="overflow-x-auto">
                    <table className="w-full font-data text-xs">
                      <thead>
                        <tr className="border-b border-border text-muted-foreground uppercase tracking-wider">
                          <th className="text-left py-3 px-4 font-medium">Star_ID</th>
                          <th className="text-left py-3 px-4 font-medium">Mission</th>
                          <th className="text-left py-3 px-4 font-medium">Planets</th>
                          <th className="text-left py-3 px-4 font-medium">Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {detectionHistory.map((row) => (
                          <tr key={row.star} className="border-b border-border hover:bg-muted/50 transition-colors">
                            <td className="py-3 px-4 text-foreground">{row.star}</td>
                            <td className="py-3 px-4">
                              <span className="bg-muted px-2 py-0.5 rounded-md text-[10px] uppercase tracking-wider">{row.mission}</span>
                            </td>
                            <td className="py-3 px-4">{row.planets}</td>
                            <td className="py-3 px-4">
                              <span className={row.confidence > 90 ? "text-success" : "text-warning"}>{row.confidence}%</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </motion.div>
      </main>
    </div>
  );
};

export default Dashboard;
