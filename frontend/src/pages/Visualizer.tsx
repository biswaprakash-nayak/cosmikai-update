import { useState, useRef, Suspense, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Search, ArrowLeft, LoaderCircle, Pause, Play } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Billboard, OrbitControls, Stars, Text } from "@react-three/drei";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import * as THREE from "three";

interface StarSystem {
  star_name: string;
  predictions: PredictionItem[];
}

interface PredictionItem {
  id: number;
  star_name: string;
  planet_name?: string;
  mission: string;
  score: number;
  percentage: number;
  period_days: number | null;
  planet_radius_earth?: number | null;
  semi_major_axis_au?: number | null;
  transit_depth_estimate?: number | null;
  verdict: string;
  timestamp: string;
  folded_lightcurve: number[] | null;
}

interface HistoryResponse {
  items: PredictionItem[];
  total: number;
}

interface StarDetails {
  star_name: string;
  source: string;
  ra: number | null;
  dec: number | null;
  gaia_id: string | null;
  tic_id: string | null;
  teff: number | null;
  radius: number | null;
  mass: number | null;
  logg: number | null;
  distance: number | null;
  vmag: number | null;
  tmag: number | null;
  found: boolean;
}

type CameraPreset = "cinematic" | "tactical" | "close" | "free";

const PLANET_COLORS = [
  "#22c55e",
  "#60a5fa",
  "#f59e0b",
  "#a78bfa",
  "#f97316",
  "#14b8a6",
  "#e879f9",
  "#f43f5e",
];

const SOLAR_PLANET_COLORS: Record<string, string> = {
  mercury: "#9d9386",
  venus: "#d5a774",
  earth: "#4f8ddf",
  mars: "#b9653f",
  jupiter: "#d0b28a",
  saturn: "#d5c39c",
  uranus: "#5b8fff",
  neptune: "#4a76d3",
};

const getPlanetColor = (starName: string, planetName: string | undefined, fallbackColor: string) => {
  if (starName.toLowerCase() === SOLAR_DEMO_STAR_NAME && planetName) {
    return SOLAR_PLANET_COLORS[planetName.toLowerCase()] ?? fallbackColor;
  }
  return fallbackColor;
};

const solarSizeFromRadiusEarth = (radiusEarth: number) => {
  const scaled = 0.09 + Math.pow(clamp(radiusEarth, 0.3, 13), 0.7) * 0.055;
  return clamp(scaled, 0.085, 0.32);
};

const getPlanetOrbitRadius = (
  starName: string,
  planet: PredictionItem,
  fallbackRadius: number,
) => {
  if (starName.toLowerCase() === SOLAR_DEMO_STAR_NAME && planet.semi_major_axis_au != null) {
    return 2.0 + planet.semi_major_axis_au * 1.25;
  }
  return fallbackRadius;
};

const getPlanetRenderSize = (
  starName: string,
  planet: PredictionItem,
  stellarRadiusRsun: number | null | undefined,
  fallbackScorePct: number,
) => {
  if (starName.toLowerCase() === SOLAR_DEMO_STAR_NAME && planet.planet_radius_earth != null) {
    return solarSizeFromRadiusEarth(planet.planet_radius_earth);
  }
  return estimatePlanetRenderSize(planet.transit_depth_estimate, stellarRadiusRsun, fallbackScorePct);
};

const SOLAR_DEMO_STAR_NAME = "solar system";
const RSUN_TO_REARTH = 109.076;

const solarDepthFromRadiusEarth = (radiusEarth: number) => {
  const rpOverRstar = radiusEarth / RSUN_TO_REARTH;
  return rpOverRstar * rpOverRstar;
};

const SOLAR_DEMO_PREDICTIONS: PredictionItem[] = [
  { id: -1, star_name: SOLAR_DEMO_STAR_NAME, planet_name: "Mercury", mission: "Demo", score: 0.99, percentage: 99, period_days: 87.969, planet_radius_earth: 0.383, semi_major_axis_au: 0.387, transit_depth_estimate: solarDepthFromRadiusEarth(0.383), verdict: "TRANSIT_DETECTED", timestamp: "2026-01-01T00:00:00Z", folded_lightcurve: null },
  { id: -2, star_name: SOLAR_DEMO_STAR_NAME, planet_name: "Venus", mission: "Demo", score: 0.99, percentage: 99, period_days: 224.701, planet_radius_earth: 0.949, semi_major_axis_au: 0.723, transit_depth_estimate: solarDepthFromRadiusEarth(0.949), verdict: "TRANSIT_DETECTED", timestamp: "2026-01-01T00:00:00Z", folded_lightcurve: null },
  { id: -3, star_name: SOLAR_DEMO_STAR_NAME, planet_name: "Earth", mission: "Demo", score: 0.99, percentage: 99, period_days: 365.256, planet_radius_earth: 1.0, semi_major_axis_au: 1.0, transit_depth_estimate: solarDepthFromRadiusEarth(1.0), verdict: "TRANSIT_DETECTED", timestamp: "2026-01-01T00:00:00Z", folded_lightcurve: null },
  { id: -4, star_name: SOLAR_DEMO_STAR_NAME, planet_name: "Mars", mission: "Demo", score: 0.99, percentage: 99, period_days: 686.980, planet_radius_earth: 0.532, semi_major_axis_au: 1.524, transit_depth_estimate: solarDepthFromRadiusEarth(0.532), verdict: "TRANSIT_DETECTED", timestamp: "2026-01-01T00:00:00Z", folded_lightcurve: null },
  { id: -5, star_name: SOLAR_DEMO_STAR_NAME, planet_name: "Jupiter", mission: "Demo", score: 0.99, percentage: 99, period_days: 4332.589, planet_radius_earth: 11.209, semi_major_axis_au: 5.204, transit_depth_estimate: solarDepthFromRadiusEarth(11.209), verdict: "TRANSIT_DETECTED", timestamp: "2026-01-01T00:00:00Z", folded_lightcurve: null },
  { id: -6, star_name: SOLAR_DEMO_STAR_NAME, planet_name: "Saturn", mission: "Demo", score: 0.99, percentage: 99, period_days: 10759.220, planet_radius_earth: 9.449, semi_major_axis_au: 9.582, transit_depth_estimate: solarDepthFromRadiusEarth(9.449), verdict: "TRANSIT_DETECTED", timestamp: "2026-01-01T00:00:00Z", folded_lightcurve: null },
  { id: -7, star_name: SOLAR_DEMO_STAR_NAME, planet_name: "Uranus", mission: "Demo", score: 0.99, percentage: 99, period_days: 30688.500, planet_radius_earth: 4.007, semi_major_axis_au: 19.201, transit_depth_estimate: solarDepthFromRadiusEarth(4.007), verdict: "TRANSIT_DETECTED", timestamp: "2026-01-01T00:00:00Z", folded_lightcurve: null },
  { id: -8, star_name: SOLAR_DEMO_STAR_NAME, planet_name: "Neptune", mission: "Demo", score: 0.99, percentage: 99, period_days: 60182.000, planet_radius_earth: 3.883, semi_major_axis_au: 30.047, transit_depth_estimate: solarDepthFromRadiusEarth(3.883), verdict: "TRANSIT_DETECTED", timestamp: "2026-01-01T00:00:00Z", folded_lightcurve: null },
];

const SOLAR_DEMO_STAR_DETAILS: StarDetails = {
  star_name: SOLAR_DEMO_STAR_NAME,
  source: "Solar System",
  ra: 286.13,
  dec: 63.87,
  gaia_id: null,
  tic_id: null,
  teff: 5772,
  radius: 1.0,
  mass: 1.0,
  logg: 4.44,
  distance: 0.0,
  vmag: -26.74,
  tmag: -26.74,
  found: true,
};

const SOLAR_DEMO_SYSTEM: StarSystem = {
  star_name: SOLAR_DEMO_STAR_NAME,
  predictions: SOLAR_DEMO_PREDICTIONS,
};

const clamp = (value: number, minValue: number, maxValue: number) =>
  Math.min(maxValue, Math.max(minValue, value));

const phaseSeedFromPlanet = (planetId: number, idx: number) => {
  const seed = Math.abs((planetId + 97) * 12.9898 + (idx + 1) * 78.233);
  const normalized = seed - Math.floor(seed);
  return normalized * Math.PI * 2;
};

const orbitArcPath = (cx: number, cy: number, radius: number, startAngleRad: number, endAngleRad: number) => {
  let start = startAngleRad;
  let end = endAngleRad;
  while (end < start) {
    end += Math.PI * 2;
  }
  const startX = cx + radius * Math.cos(start);
  const startY = cy + radius * Math.sin(start);
  const endX = cx + radius * Math.cos(end);
  const endY = cy + radius * Math.sin(end);
  const largeArcFlag = end - start > Math.PI ? 1 : 0;
  return `M ${startX} ${startY} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY}`;
};

const temperatureToStarColor = (teff: number | null | undefined) => {
  if (!teff || !Number.isFinite(teff)) {
    return "#ffcc66";
  }
  if (teff < 3800) return "#ff8e45";
  if (teff < 5000) return "#ffb869";
  if (teff < 6100) return "#ffd47f";
  if (teff < 7000) return "#fff3de";
  if (teff < 8200) return "#d7e6ff";
  return "#b8d2ff";
};

const radiusToStarScale = (radiusRsun: number | null | undefined) => {
  if (!radiusRsun || !Number.isFinite(radiusRsun)) {
    return 1.0;
  }
  const scaled = 0.9 + Math.pow(radiusRsun, 0.55) * 0.7;
  return clamp(scaled, 0.7, 2.8);
};

const temperatureToLightIntensity = (teff: number | null | undefined) => {
  if (!teff || !Number.isFinite(teff)) {
    return 2.0;
  }
  const normalized = clamp((teff - 3200) / 5200, 0, 1.25);
  return 1.55 + normalized * 1.5;
};

const estimatePlanetRadiusEarth = (
  transitDepth: number | null | undefined,
  stellarRadiusRsun: number | null | undefined,
) => {
  if (
    transitDepth == null ||
    !Number.isFinite(transitDepth) ||
    stellarRadiusRsun == null ||
    !Number.isFinite(stellarRadiusRsun) ||
    transitDepth <= 0 ||
    stellarRadiusRsun <= 0
  ) {
    return null;
  }

  // Transit depth approximation: delta ~= (Rp/R*)^2
  const depthFraction = clamp(transitDepth, 0, 1);
  const rpOverRstar = Math.sqrt(depthFraction);
  const rpRsun = stellarRadiusRsun * rpOverRstar;
  const rsunToRearth = 109.076;
  const rpRearth = rpRsun * rsunToRearth;
  return Number.isFinite(rpRearth) ? rpRearth : null;
};

const estimatePlanetRenderSize = (
  transitDepth: number | null | undefined,
  stellarRadiusRsun: number | null | undefined,
  fallbackScorePct: number,
) => {
  const rpRearth = estimatePlanetRadiusEarth(transitDepth, stellarRadiusRsun);
  if (rpRearth == null) {
    return 0.08 + (fallbackScorePct / 100) * 0.2;
  }

  // Smooth nonlinear scaling so extreme values do not dominate the scene.
  const scaled = 0.06 + Math.pow(clamp(rpRearth, 0.3, 25), 0.55) * 0.035;
  return clamp(scaled, 0.08, 0.34);
};

function Star({ onClick, isPaused, radius, color, lightIntensity }: { onClick?: () => void; isPaused: boolean; radius: number; color: string; lightIntensity: number }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const coronaRef = useRef<THREE.Mesh>(null);
  const starSpinRef = useRef(0);
  const pulseTimeRef = useRef(0);
  const secondaryLightColor = useMemo(() => {
    const mixed = new THREE.Color(color).lerp(new THREE.Color("#ffffff"), 0.58);
    return `#${mixed.getHexString()}`;
  }, [color]);
  
  useFrame((_, delta) => {
    if (meshRef.current) {
      if (!isPaused) {
        starSpinRef.current += delta * 0.1;
        meshRef.current.rotation.y = starSpinRef.current;
      }
    }
    if (coronaRef.current) {
      if (!isPaused) {
        pulseTimeRef.current += delta;
      }
      const pulse = 1 + Math.sin(pulseTimeRef.current * 1.1) * 0.03;
      coronaRef.current.scale.setScalar(pulse);
    }
  });

  return (
    <group>
      <mesh
        ref={meshRef}
        onPointerDown={(event) => {
          event.stopPropagation();
          onClick?.();
        }}
      >
        <sphereGeometry args={[radius, 64, 64]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <mesh ref={coronaRef}>
        <sphereGeometry args={[radius * 1.07, 48, 48]} />
        <meshBasicMaterial color={color} transparent opacity={0.08} side={THREE.DoubleSide} />
      </mesh>
      <pointLight position={[0, 0, 0]} intensity={lightIntensity * 42} color={color} distance={0} decay={1.25} />
      <pointLight position={[0, 0, 0]} intensity={lightIntensity * 10} color={secondaryLightColor} distance={0} decay={1.35} />
    </group>
  );
}

function Planet({ orbitRadius, size, color, orbitSpeed, orbitPhaseStart, isSelected, isPaused, planetName, showLabel, atmosphereColor, hasAtmosphere, hasRing, onClick, onPositionUpdate }: {
  orbitRadius: number;
  size: number;
  color: string;
  orbitSpeed: number;
  orbitPhaseStart?: number;
  isSelected: boolean;
  isPaused: boolean;
  planetName?: string;
  showLabel?: boolean;
  atmosphereColor?: string;
  hasAtmosphere?: boolean;
  hasRing?: boolean;
  onClick: (position: THREE.Vector3) => void;
  onPositionUpdate?: (position: THREE.Vector3) => void;
}) {
  const { camera } = useThree();
  const meshRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);
  const labelRef = useRef<THREE.Group>(null);
  const orbitAngleRef = useRef(orbitPhaseStart ?? Math.random() * Math.PI * 2);
  const planetSpinRef = useRef(Math.random() * Math.PI * 2);
  useFrame((_, delta) => {
    if (isPaused) return;
    if (groupRef.current) {
      orbitAngleRef.current += delta * orbitSpeed;
      groupRef.current.rotation.y = orbitAngleRef.current;
    }
    if (meshRef.current) {
      planetSpinRef.current += delta * 0.5;
      meshRef.current.rotation.y = planetSpinRef.current;
      meshRef.current.getWorldPosition(_planetWorldPosition);
      if (labelRef.current) {
        // Keep labels readable across zoom levels without exploding in close shots.
        const distance = camera.position.distanceTo(_planetWorldPosition);
        const labelScale = THREE.MathUtils.clamp(distance * 0.03, 0.62, 2.25);
        labelRef.current.scale.setScalar(labelScale);
      }
      if (onPositionUpdate) {
        onPositionUpdate(_planetWorldPosition);
      }
    }
  });

  return (
    <group ref={groupRef}>
      <mesh
        ref={meshRef}
        position={[orbitRadius, 0, 0]}
        onPointerDown={(event) => {
          event.stopPropagation();
          event.object.getWorldPosition(_planetClickWorldPosition);
          onClick(_planetClickWorldPosition);
        }}
      >
        <sphereGeometry args={[size, 32, 32]} />
        <meshStandardMaterial color={color} roughness={0.88} metalness={0.02} envMapIntensity={0.2} />
        {showLabel && planetName && (
          <Billboard ref={labelRef} position={[0, size + Math.max(0.14, size * 0.45), 0]} follow>
            <Text
              fontSize={0.1}
              color={isSelected ? "#e6f6ff" : "#b5d9f0"}
              anchorX="center"
              anchorY="bottom"
              outlineWidth={0.012}
              outlineColor="#020611"
              maxWidth={2.6}
            >
              {planetName}
            </Text>
          </Billboard>
        )}
        {(hasAtmosphere ?? false) && (
          <mesh>
            <sphereGeometry args={[size * 1.09, 28, 28]} />
            <meshPhongMaterial
              color={atmosphereColor ?? color}
              transparent
              opacity={0.18}
              depthWrite={false}
              side={THREE.DoubleSide}
            />
          </mesh>
        )}
        {(hasRing ?? false) && (
          <mesh rotation={[Math.PI * 0.48, 0, 0]}>
            <ringGeometry args={[size * 1.35, size * 2.1, 64]} />
            <meshBasicMaterial color="#c9b488" side={THREE.DoubleSide} transparent opacity={0.62} />
          </mesh>
        )}
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

const _planetWorldPosition = new THREE.Vector3();
const _planetClickWorldPosition = new THREE.Vector3();
const _starWorldPosition = new THREE.Vector3(0, 0, 0);
const _cameraRadialDirection = new THREE.Vector3();
const _cameraInlineOffset = new THREE.Vector3();

function OrbitPath({ radius, isSelected, level, maxLevel }: { radius: number; isSelected: boolean; level: number; maxLevel: number }) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((_, delta) => {
    if (!meshRef.current) return;
    const geometry = meshRef.current.geometry as THREE.RingGeometry;
    const targetInner = Math.max(0.02, radius - 0.02);
    const targetOuter = radius + 0.02;
    const currentParams = geometry.parameters as { innerRadius: number; outerRadius: number };
    const step = 1 - Math.exp(-delta * 4.0);

    const nextInner = THREE.MathUtils.lerp(currentParams.innerRadius, targetInner, step);
    const nextOuter = THREE.MathUtils.lerp(currentParams.outerRadius, targetOuter, step);

    if (
      Math.abs(nextInner - currentParams.innerRadius) > 1e-4 ||
      Math.abs(nextOuter - currentParams.outerRadius) > 1e-4
    ) {
      geometry.dispose();
      meshRef.current.geometry = new THREE.RingGeometry(nextInner, nextOuter, 128);
    }
  });

  return (
    <mesh ref={meshRef} rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[radius - 0.02, radius + 0.02, 128]} />
      <meshBasicMaterial
        color={isSelected ? "#9ee8ff" : "#2a394b"}
        side={THREE.DoubleSide}
        transparent
        opacity={isSelected ? 0.75 : 0.16 + ((maxLevel - level) / Math.max(maxLevel, 1)) * 0.24}
      />
    </mesh>
  );
}

function AutoZoom({
  orbitExtent,
  focusTarget,
  focusKey,
  trackTarget,
  cameraPreset,
  cameraOffset,
  activeRef,
  controlsRef,
}: {
  orbitExtent: number;
  focusTarget: THREE.Vector3;
  focusKey: number;
  trackTarget: boolean;
  cameraPreset: CameraPreset;
  cameraOffset: THREE.Vector3;
  activeRef: React.MutableRefObject<boolean>;
  controlsRef: React.MutableRefObject<any>;
}) {
  const { camera } = useThree();
  const targetPositionRef = useRef(new THREE.Vector3(0, 0, 0));
  const lookAtTargetRef = useRef(new THREE.Vector3(0, 0, 0));
  const prevTrackedTargetRef = useRef(new THREE.Vector3(0, 0, 0));
  const trackingInitializedRef = useRef(false);
  const trackDeltaRef = useRef(new THREE.Vector3(0, 0, 0));
  const isCinematicLock = trackTarget && cameraPreset === "cinematic";

  useEffect(() => {
    if (cameraPreset === "free") {
      activeRef.current = false;
      return;
    }
    if (trackTarget) {
      if (cameraPreset === "cinematic") {
        _cameraRadialDirection.copy(focusTarget).sub(_starWorldPosition);
        if (_cameraRadialDirection.lengthSq() < 1e-6) {
          _cameraRadialDirection.set(0, 0, 1);
        }
        _cameraRadialDirection.normalize();
        _cameraInlineOffset.copy(_cameraRadialDirection).multiplyScalar(Math.max(0.2, cameraOffset.z));
        _cameraInlineOffset.y += cameraOffset.y;
        targetPositionRef.current.copy(focusTarget).add(_cameraInlineOffset);
      } else {
        targetPositionRef.current.copy(focusTarget).add(cameraOffset);
      }
      prevTrackedTargetRef.current.copy(focusTarget);
      trackingInitializedRef.current = false;
    } else {
      const extent = Math.max(orbitExtent, 3);
      const targetDistance = Math.min(80, Math.max(10, extent * 1.55 + 4));
      const targetHeight = Math.min(35, Math.max(7, extent * 0.32 + 3));
      targetPositionRef.current.set(0, targetHeight, targetDistance).add(cameraOffset.clone().multiplyScalar(0.42));
    }
    lookAtTargetRef.current.copy(isCinematicLock ? _starWorldPosition : focusTarget);
    activeRef.current = true;
  }, [camera, orbitExtent, focusTarget, focusKey, trackTarget, cameraOffset, cameraPreset, isCinematicLock]);

  useFrame((_, delta) => {
    if (cameraPreset === "free") return;
    if (!activeRef.current && !trackTarget) return;

    if (trackTarget) {
      if (trackingInitializedRef.current) {
        trackDeltaRef.current.copy(focusTarget).sub(prevTrackedTargetRef.current);
        camera.position.add(trackDeltaRef.current);
        if (controlsRef.current && !isCinematicLock) {
          controlsRef.current.target.add(trackDeltaRef.current);
        }
      }
      prevTrackedTargetRef.current.copy(focusTarget);
      trackingInitializedRef.current = true;

      if (!activeRef.current) {
        if (isCinematicLock) {
          camera.lookAt(_starWorldPosition);
          if (controlsRef.current) {
            controlsRef.current.target.copy(_starWorldPosition);
          }
        }
        if (controlsRef.current) {
          controlsRef.current.update();
        }
        camera.updateProjectionMatrix();
        return;
      }

      if (cameraPreset === "cinematic") {
        _cameraRadialDirection.copy(focusTarget).sub(_starWorldPosition);
        if (_cameraRadialDirection.lengthSq() < 1e-6) {
          _cameraRadialDirection.set(0, 0, 1);
        }
        _cameraRadialDirection.normalize();
        _cameraInlineOffset.copy(_cameraRadialDirection).multiplyScalar(Math.max(0.2, cameraOffset.z));
        _cameraInlineOffset.y += cameraOffset.y;
        targetPositionRef.current.copy(focusTarget).add(_cameraInlineOffset);
      } else {
        targetPositionRef.current.copy(focusTarget).add(cameraOffset);
      }
    }

    const step = 1 - Math.exp(-delta * 2.8);
    camera.position.lerp(targetPositionRef.current, step);
    lookAtTargetRef.current.copy(isCinematicLock ? _starWorldPosition : focusTarget);
    camera.lookAt(lookAtTargetRef.current);
    if (controlsRef.current) {
      controlsRef.current.target.lerp(lookAtTargetRef.current, step);
      controlsRef.current.update();
    }
    camera.updateProjectionMatrix();

    if (trackTarget && activeRef.current && camera.position.distanceTo(targetPositionRef.current) < 0.05) {
      camera.position.copy(targetPositionRef.current);
      if (controlsRef.current) {
        controlsRef.current.target.copy(isCinematicLock ? _starWorldPosition : focusTarget);
        controlsRef.current.update();
      }
      activeRef.current = false;
      return;
    }

    if (!trackTarget && camera.position.distanceTo(targetPositionRef.current) < 0.05) {
      camera.position.copy(targetPositionRef.current);
      camera.lookAt(focusTarget);
      if (controlsRef.current) {
        controlsRef.current.target.copy(focusTarget);
        controlsRef.current.update();
      }
      activeRef.current = false;
    }
  });

  return null;
}

function StarSystemScene({ planets, selectedStarIdx, selectedPlanetIdx, orbitExtent, selectedPlanetFocus, focusOnStar, focusKey, isAnimationPaused, trackPlanet, showPlanetNames, cameraPreset, cameraPresetOffset, starRadius, starColor, starLightIntensity, onSystemClick, onPlanetClick, onUserCameraControl }: { 
  planets: any[]; 
  selectedStarIdx: number;
  selectedPlanetIdx: number;
  orbitExtent: number;
  selectedPlanetFocus: THREE.Vector3;
  focusOnStar: boolean;
  focusKey: number;
  isAnimationPaused: boolean;
  trackPlanet: boolean;
  showPlanetNames: boolean;
  cameraPreset: CameraPreset;
  cameraPresetOffset: THREE.Vector3;
  starRadius: number;
  starColor: string;
  starLightIntensity: number;
  onSystemClick: (idx: number) => void;
  onPlanetClick: (sysIdx: number, predIdx: number) => void;
  onUserCameraControl: () => void;
}) {
  const autoZoomActiveRef = useRef(false);
  const controlsRef = useRef<any>(null);
  const activeFocusTarget = focusOnStar ? _starWorldPosition : selectedPlanetFocus;

  return (
    <>
      <fog attach="fog" args={["#111418", 45, 180]} />
      <ambientLight intensity={0} />
      <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={isAnimationPaused ? 0 : 1} />
      <Stars radius={180} depth={140} count={1800} factor={1.4} saturation={0} fade speed={isAnimationPaused ? 0 : 0.22} />
      <AutoZoom
        orbitExtent={orbitExtent}
        focusTarget={activeFocusTarget}
        focusKey={focusKey}
        trackTarget={trackPlanet && !focusOnStar && cameraPreset !== "free"}
        cameraPreset={cameraPreset}
        cameraOffset={cameraPresetOffset}
        activeRef={autoZoomActiveRef}
        controlsRef={controlsRef}
      />
      <Star
        onClick={() => onSystemClick(selectedStarIdx)}
        isPaused={isAnimationPaused}
        radius={starRadius}
        color={starColor}
        lightIntensity={starLightIntensity}
      />
      {planets.map((planet, i) => (
        <OrbitPath
          key={`orbit-${i}`}
          radius={planet.orbitRadius}
          isSelected={!focusOnStar && selectedPlanetIdx === i}
          level={i}
          maxLevel={Math.max(1, planets.length - 1)}
        />
      ))}
      {planets.map((planet, i) => (
        <Planet
          key={planet.id}
          orbitRadius={planet.orbitRadius}
          size={planet.size}
          color={planet.color}
          orbitSpeed={planet.orbitSpeed}
          orbitPhaseStart={planet.orbitPhaseStart}
          isSelected={!focusOnStar && selectedStarIdx === planet.systemIdx && selectedPlanetIdx === planet.predIdx}
          isPaused={isAnimationPaused}
          planetName={planet.displayName}
          showLabel={showPlanetNames}
          hasAtmosphere={planet.hasAtmosphere}
          atmosphereColor={planet.atmosphereColor}
          hasRing={planet.hasRing}
          onPositionUpdate={!focusOnStar && planet.predIdx === selectedPlanetIdx ? (position) => {
            selectedPlanetFocus.copy(position);
            if (controlsRef.current) {
              controlsRef.current.target.copy(position);
              controlsRef.current.update();
            }
          } : undefined}
          onClick={(position) => {
            selectedPlanetFocus.copy(position);
            if (controlsRef.current) {
              controlsRef.current.target.copy(position);
              controlsRef.current.update();
            }
            onPlanetClick(planet.systemIdx, planet.predIdx);
          }}
        />
      ))}
      <OrbitControls 
        ref={controlsRef}
        enablePan={true} 
        enableZoom={true} 
        enableRotate={true} 
        minDistance={0.8} 
        maxDistance={Math.max(120, orbitExtent * 3.5)}
        autoRotate={false}
        onStart={() => {
          autoZoomActiveRef.current = false;
          onUserCameraControl();
        }}
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
  const [isAnimationPaused, setIsAnimationPaused] = useState(false);
  const [trackPlanet, setTrackPlanet] = useState(true);
  const [focusOnStar, setFocusOnStar] = useState(true);
  const [focusKey, setFocusKey] = useState(0);
  const selectedPlanetFocusRef = useRef(new THREE.Vector3(0, 0, 0));
  const [starDetailsCache, setStarDetailsCache] = useState<Record<string, StarDetails>>({
    [SOLAR_DEMO_STAR_NAME]: SOLAR_DEMO_STAR_DETAILS,
  });
  const [starDetailsLoading, setStarDetailsLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  const [highlightedSuggestionIdx, setHighlightedSuggestionIdx] = useState(0);
  const [showPlanetNames, setShowPlanetNames] = useState(true);
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("cinematic");
  const [minimapTime, setMinimapTime] = useState(0);
  const suppressNextControlStartRef = useRef(false);

  const cameraPresetOffset =
    cameraPreset === "tactical"
      ? new THREE.Vector3(0, 7.2, 8.8)
      : cameraPreset === "close"
        ? new THREE.Vector3(0, 0.24, 0.62)
        : new THREE.Vector3(0, 2.2, 4.6);

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
        
        const systems: StarSystem[] = Object.entries(grouped).map(([name, preds]) => {
          // Keep planets ordered from closest to farthest by orbital period.
          const sortedPredictions = [...preds].sort((a, b) => {
            const pa = a.period_days ?? Number.POSITIVE_INFINITY;
            const pb = b.period_days ?? Number.POSITIVE_INFINITY;
            return pa - pb;
          });

          return {
            star_name: name,
            predictions: sortedPredictions,
          };
        });
        
        const withoutSolarDemo = systems.filter((sys) => sys.star_name.toLowerCase() !== SOLAR_DEMO_STAR_NAME);
        setStarSystems([SOLAR_DEMO_SYSTEM, ...withoutSolarDemo]);
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

  useEffect(() => {
    setMinimapTime(0);
  }, [selectedStarIdx]);

  useEffect(() => {
    if (isAnimationPaused) return;
    const timer = window.setInterval(() => {
      setMinimapTime((time) => time + 0.05);
    }, 50);
    return () => window.clearInterval(timer);
  }, [isAnimationPaused]);

  const normalizedSearch = searchTerm.trim().toLowerCase();
  const visibleSystemIndices = starSystems.reduce<number[]>((acc, system, idx) => {
    if (!normalizedSearch || system.star_name.toLowerCase().includes(normalizedSearch)) {
      acc.push(idx);
    }
    return acc;
  }, []);

  const selectedVisibleIdx = visibleSystemIndices.indexOf(selectedStarIdx);
  const searchSuggestions = normalizedSearch
    ? visibleSystemIndices.slice(0, 8).map((idx) => ({ idx, name: starSystems[idx].star_name }))
    : [];
  const showSuggestions = isSearchFocused && normalizedSearch.length > 0;

  const selectSystemFromSearch = (systemIdx: number) => {
    suppressNextControlStartRef.current = true;
    setSelectedStarIdx(systemIdx);
    setSelectedPlanetIdx(0);
    setFocusOnStar(true);
    selectedPlanetFocusRef.current.set(0, 0, 0);
    setFocusKey((k) => k + 1);
  };

  useEffect(() => {
    setHighlightedSuggestionIdx(0);
  }, [normalizedSearch]);

  useEffect(() => {
    if (visibleSystemIndices.length === 0) return;
    if (selectedVisibleIdx === -1) {
      selectSystemFromSearch(visibleSystemIndices[0]);
    }
  }, [visibleSystemIndices, selectedVisibleIdx]);

  const selectedStar = starSystems[selectedStarIdx];
  const selectedPred = selectedStar?.predictions[selectedPlanetIdx];
  const selectedStarDetails = selectedStar ? starDetailsCache[selectedStar.star_name] : undefined;
  const starRadius = radiusToStarScale(selectedStarDetails?.radius);
  const starColor = temperatureToStarColor(selectedStarDetails?.teff);
  const starLightIntensity = temperatureToLightIntensity(selectedStarDetails?.teff);
  const selectedPlanetRadiusEarth = estimatePlanetRadiusEarth(
    selectedPred?.transit_depth_estimate,
    selectedStarDetails?.radius,
  );

  useEffect(() => {
    const starName = selectedStar?.star_name?.trim();
    if (!starName) return;
    if (starDetailsCache[starName]) return;

    let cancelled = false;

    const fetchStarDetails = async () => {
      try {
        setStarDetailsLoading(true);
        const response = await fetch(`http://localhost:8000/api/star-details?star_name=${encodeURIComponent(starName)}`);
        if (!response.ok) {
          throw new Error(`Failed to fetch star details (${response.status})`);
        }
        const details: StarDetails = await response.json();
        if (!cancelled) {
          setStarDetailsCache((prev) => ({ ...prev, [starName]: details }));
        }
      } catch (detailsError) {
        if (!cancelled) {
          console.error("Error fetching MAST star details:", detailsError);
        }
      } finally {
        if (!cancelled) {
          setStarDetailsLoading(false);
        }
      }
    };

    fetchStarDetails();

    return () => {
      cancelled = true;
    };
  }, [selectedStar, starDetailsCache]);

  const planetCount = selectedStar?.predictions.length ?? 0;
  const validSystemPeriods = (selectedStar?.predictions ?? [])
    .map((pred) => pred.period_days)
    .filter((p): p is number => Number.isFinite(p) && p > 0);
  const minSystemPeriodDays = validSystemPeriods.length > 0 ? Math.min(...validSystemPeriods) : null;
  const orbitRadiusMin = 2.8;
  const orbitRadiusMax = 34.0;
  const orbitSpacingMin = 2.4;
  const orbitUnitsPerAu = 6.6;
  const orbitBaseOffset = 1.8;
  const earthPeriodDays = 365.25;

  const periodToOrbitRadius = (period: number | null | undefined, index: number) => {
    if (Number.isFinite(period) && (period as number) > 0) {
      // Kepler-style approximation in display units: a/AU ~= (P/365d)^(2/3)
      const periodDays = period as number;
      const semiMajorAxisAu = Math.pow(periodDays / earthPeriodDays, 2 / 3);
      const rawRadius = orbitBaseOffset + semiMajorAxisAu * orbitUnitsPerAu;
      return Math.min(orbitRadiusMax, Math.max(orbitRadiusMin, rawRadius));
    }

    if (planetCount <= 1) {
      return (orbitRadiusMin + orbitRadiusMax) / 2;
    }

    return orbitRadiusMin + (index / (planetCount - 1)) * (orbitRadiusMax - orbitRadiusMin);
  };

  const periodToOrbitSpeed = (period: number | null | undefined, index: number) => {
    if (Number.isFinite(period) && (period as number) > 0) {
      // Kepler-style angular speed: omega ~ 1 / period.
      // Scale by system minimum period so relative speed differences are obvious.
      const periodDays = period as number;
      const periodRatio = minSystemPeriodDays != null ? periodDays / minSystemPeriodDays : periodDays / earthPeriodDays;
      const rawSpeed = 1.35 / Math.pow(Math.max(periodRatio, 1e-6), 0.92);
      return clamp(rawSpeed, 0.04, 1.65);
    }

    if (planetCount <= 1) {
      return 0.8;
    }

    const normalized = index / (planetCount - 1);
    return 1.15 - normalized * 0.8;
  };

  // Convert only the selected star system into 3D planets/orbit objects.
  const selectedSystem3d = selectedStar
    ? selectedStar.predictions
        .map((pred, predIdx) => {
          const rawOrbitRadius = getPlanetOrbitRadius(
            selectedStar.star_name,
            pred,
            periodToOrbitRadius(pred.period_days, predIdx),
          );
          const fallbackColor = PLANET_COLORS[predIdx % PLANET_COLORS.length];
          const resolvedColor = getPlanetColor(selectedStar.star_name.toLowerCase(), pred.planet_name, fallbackColor);
          return {
            id: pred.id,
            star_name: selectedStar.star_name,
            name: pred.planet_name ?? `${selectedStar.star_name} ${String.fromCharCode(98 + predIdx)}`,
            mission: pred.mission,
            period: pred.period_days,
            probability: pred.percentage,
            orbitRadius: rawOrbitRadius,
            size: getPlanetRenderSize(selectedStar.star_name, pred, selectedStarDetails?.radius, pred.percentage),
            color: resolvedColor,
            hasAtmosphere: ["venus", "earth", "jupiter", "saturn", "uranus", "neptune"].includes((pred.planet_name ?? "").toLowerCase()),
            atmosphereColor:
              (pred.planet_name ?? "").toLowerCase() === "earth"
                ? "#8fd9ff"
                : (pred.planet_name ?? "").toLowerCase() === "uranus"
                  ? "#6ca6ff"
                  : resolvedColor,
            hasRing: (pred.planet_name ?? "").toLowerCase() === "saturn",
            orbitSpeed: periodToOrbitSpeed(pred.period_days, predIdx),
            orbitPhaseStart: phaseSeedFromPlanet(pred.id, predIdx),
            systemIdx: selectedStarIdx,
            predIdx: predIdx,
            planetName: pred.planet_name,
            displayName:
              selectedStar.star_name.toLowerCase() === SOLAR_DEMO_STAR_NAME
                ? (pred.planet_name ?? `${selectedStar.star_name} ${String.fromCharCode(98 + predIdx)}`)
                : `Planet ${String.fromCharCode(65 + predIdx)}`,
          };
        })
        .sort((a, b) => a.orbitRadius - b.orbitRadius)
        .map((planet) => ({ ...planet }))
        .map((planet, idx, arr) => {
          if (idx === 0 && arr.length > 1) {
            const count = arr.length;
            const maxGapCapacity = (orbitRadiusMax - orbitRadiusMin) / Math.max(1, count - 1);
            const effectiveSpacing = Math.min(orbitSpacingMin, maxGapCapacity);

            // Forward pass: enforce ascending radii with minimum spacing.
            arr[0].orbitRadius = clamp(
              arr[0].orbitRadius,
              orbitRadiusMin,
              orbitRadiusMax - effectiveSpacing * (count - 1),
            );
            for (let i = 1; i < count; i += 1) {
              const minAllowed = arr[i - 1].orbitRadius + effectiveSpacing;
              arr[i].orbitRadius = Math.max(arr[i].orbitRadius, minAllowed);
            }

            // Backward pass if the outermost orbit overflowed max bound.
            if (arr[count - 1].orbitRadius > orbitRadiusMax) {
              arr[count - 1].orbitRadius = orbitRadiusMax;
              for (let i = count - 2; i >= 0; i -= 1) {
                const maxAllowed = arr[i + 1].orbitRadius - effectiveSpacing;
                arr[i].orbitRadius = Math.min(arr[i].orbitRadius, maxAllowed);
              }
            }

            // Final guard to keep the innermost orbit inside the minimum bound.
            if (arr[0].orbitRadius < orbitRadiusMin) {
              arr[0].orbitRadius = orbitRadiusMin;
              for (let i = 1; i < count; i += 1) {
                arr[i].orbitRadius = Math.max(arr[i].orbitRadius, arr[i - 1].orbitRadius + effectiveSpacing);
              }
            }
          }
          return planet;
        })
        .sort((a, b) => a.predIdx - b.predIdx)
    : [];
  const selectedSystemOrbitExtent = selectedSystem3d.reduce(
    (maxRadius, planet) => Math.max(maxRadius, planet.orbitRadius),
    3,
  );
  const selectedSystem3dByOrbit = useMemo(
    () => [...selectedSystem3d].sort((a, b) => a.orbitRadius - b.orbitRadius),
    [selectedSystem3d],
  );
  const targetIconMinPx = 10;
  const targetIconMaxPx = 18;
  const targetTrackStartPct = 2;
  const targetTrackEndPct = 95;
  const targetTrackWidthPx = 252;
  const targetIconGapPx = 2;
  const selectedSystemPlanetSizes = selectedSystem3dByOrbit.map((planet) => planet.size);
  const minSystemPlanetSize = selectedSystemPlanetSizes.length > 0 ? Math.min(...selectedSystemPlanetSizes) : 0;
  const maxSystemPlanetSize = selectedSystemPlanetSizes.length > 0 ? Math.max(...selectedSystemPlanetSizes) : 0;
  const maxSystemOrbitRadius = selectedSystem3dByOrbit.length > 0 ? selectedSystem3dByOrbit[selectedSystem3dByOrbit.length - 1].orbitRadius : 0;
  const getTargetIconSizePx = (planet: (typeof selectedSystem3d)[number]) => {
    const rawSize = planet?.size;
    if (rawSize == null || !Number.isFinite(rawSize)) {
      return 13;
    }
    if (Math.abs(maxSystemPlanetSize - minSystemPlanetSize) < 1e-6) {
      return Math.round((targetIconMinPx + targetIconMaxPx) / 2);
    }
    const normalized = (rawSize - minSystemPlanetSize) / (maxSystemPlanetSize - minSystemPlanetSize);
    return Math.round(targetIconMinPx + normalized * (targetIconMaxPx - targetIconMinPx));
  };
  const getTargetOrbitPositionPct = (planet: (typeof selectedSystem3d)[number]) => {
    const orbitRadius = planet?.orbitRadius;
    if (orbitRadius == null || !Number.isFinite(orbitRadius)) {
      return (targetTrackStartPct + targetTrackEndPct) / 2;
    }
    if (maxSystemOrbitRadius <= 0) {
      return (targetTrackStartPct + targetTrackEndPct) / 2;
    }
    const normalized = clamp(orbitRadius / maxSystemOrbitRadius, 0, 1);
    return targetTrackStartPct + normalized * (targetTrackEndPct - targetTrackStartPct);
  };
  const starmapTargetLayout = useMemo(() => {
    const raw = selectedSystem3dByOrbit.map((planet, idx) => ({
      idx,
      planet,
      sizePx: getTargetIconSizePx(planet),
      desiredPct: getTargetOrbitPositionPct(planet),
      leftPct: getTargetOrbitPositionPct(planet),
    }));

    if (raw.length <= 1) {
      return raw;
    }

    const layout = raw.map((entry) => ({ ...entry }));
    const availableWidthPx = ((targetTrackEndPct - targetTrackStartPct) / 100) * targetTrackWidthPx;

    const minimumRequiredWidthPx = (entries: typeof layout) => {
      let required = 0;
      for (let i = 1; i < entries.length; i += 1) {
        const prev = entries[i - 1];
        const current = entries[i];
        required += ((prev.sizePx + current.sizePx) / 2) + targetIconGapPx;
      }
      return required;
    };

    // If spacing is physically impossible, shrink icon diameters slightly for layout stability.
    const needed = minimumRequiredWidthPx(layout);
    if (needed > availableWidthPx) {
      const scale = clamp(availableWidthPx / needed, 0.62, 1);
      for (let i = 0; i < layout.length; i += 1) {
        layout[i].sizePx = Math.max(7, Math.round(layout[i].sizePx * scale));
      }
    }

    // Pass 1: honor desired linear positions while enforcing minimum gap.
    layout[0].leftPct = clamp(layout[0].desiredPct, targetTrackStartPct, targetTrackEndPct);
    for (let i = 1; i < layout.length; i += 1) {
      const prev = layout[i - 1];
      const current = layout[i];
      const minGapPct = ((((prev.sizePx + current.sizePx) / 2) + targetIconGapPx) / targetTrackWidthPx) * 100;
      const minAllowed = prev.leftPct + minGapPct;
      current.leftPct = Math.max(current.desiredPct, minAllowed);
    }

    // If tail overflows, shift everything left equally, then re-enforce gaps in reverse.
    const overflowPct = layout[layout.length - 1].leftPct - targetTrackEndPct;
    if (overflowPct > 0) {
      for (let i = 0; i < layout.length; i += 1) {
        layout[i].leftPct -= overflowPct;
      }

      layout[layout.length - 1].leftPct = Math.min(layout[layout.length - 1].leftPct, targetTrackEndPct);
      for (let i = layout.length - 2; i >= 0; i -= 1) {
        const next = layout[i + 1];
        const current = layout[i];
        const minGapPct = ((((next.sizePx + current.sizePx) / 2) + targetIconGapPx) / targetTrackWidthPx) * 100;
        current.leftPct = Math.min(current.leftPct, next.leftPct - minGapPct);
      }
    }

    layout[0].leftPct = Math.max(layout[0].leftPct, targetTrackStartPct);
    return layout;
  }, [
    selectedSystem3dByOrbit,
    maxSystemOrbitRadius,
    minSystemPlanetSize,
    maxSystemPlanetSize,
  ]);

  // Convert folded lightcurve to chart data
  const chartData = selectedPred?.folded_lightcurve
    ? selectedPred.folded_lightcurve.map((value, i) => ({
        phase: ((i / selectedPred.folded_lightcurve!.length) * 2 - 1).toFixed(3),
        flux: parseFloat(value.toFixed(5)),
      }))
    : [];
  const selectedPlanetLetter = String.fromCharCode(65 + selectedPlanetIdx);
  const selectedPlanetName = selectedPred?.planet_name ?? `Planet ${selectedPlanetLetter}`;
  const focusDesignation = focusOnStar ? "STAR" : selectedPlanetLetter;
  const focusTargetLabel = focusOnStar
    ? `${selectedStar?.star_name ?? "Unknown"} STAR`
    : selectedPlanetName;

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
    <div className="h-screen bg-background relative overflow-hidden">
      <Canvas
        camera={{ position: [0, 8, 15], fov: 50 }}
        gl={{ antialias: true }}
        style={{ background: "#111418" }}
      >
        <Suspense fallback={null}>
          <StarSystemScene
            planets={selectedSystem3d}
            selectedStarIdx={selectedStarIdx}
            selectedPlanetIdx={selectedPlanetIdx}
            orbitExtent={selectedSystemOrbitExtent}
            selectedPlanetFocus={selectedPlanetFocusRef.current}
            focusOnStar={focusOnStar}
            focusKey={focusKey}
            isAnimationPaused={isAnimationPaused}
            trackPlanet={trackPlanet}
            showPlanetNames={showPlanetNames}
            cameraPreset={cameraPreset}
            cameraPresetOffset={cameraPresetOffset}
            starRadius={starRadius}
            starColor={starColor}
            starLightIntensity={starLightIntensity}
            onSystemClick={(idx) => {
              suppressNextControlStartRef.current = true;
              setSelectedStarIdx(idx);
              setSelectedPlanetIdx(0);
              setFocusOnStar(true);
              setTrackPlanet(false);
              setCameraPreset((prev) => (prev === "free" ? "cinematic" : prev));
              selectedPlanetFocusRef.current.set(0, 0, 0);
              setFocusKey((k) => k + 1);
            }}
            onPlanetClick={(sysIdx, predIdx) => {
              suppressNextControlStartRef.current = true;
              setSelectedStarIdx(sysIdx);
              setSelectedPlanetIdx(predIdx);
              setFocusOnStar(false);
              setTrackPlanet(true);
              setFocusKey((k) => k + 1);
            }}
            onUserCameraControl={() => {
              if (suppressNextControlStartRef.current) {
                suppressNextControlStartRef.current = false;
                return;
              }
              setCameraPreset((prev) => (prev === "free" ? prev : "free"));
            }}
          />
        </Suspense>
      </Canvas>

      <div className="absolute left-2 top-1/2 -translate-y-1/2 z-10 hidden md:block">
        <div className="rotate-[-90deg] origin-left font-data text-[9px] tracking-[0.35em] text-gray-200/60">STARMAP</div>
      </div>

      <motion.div
        initial={{ x: -40, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.35 }}
        className="absolute top-4 left-4 z-20 w-[320px] max-w-[calc(100vw-1.5rem)] space-y-3 pb-5"
      >
        <div className="bg-neutral-950/84 border border-neutral-400/28 rounded-sm p-3 backdrop-blur-sm shadow-[0_0_26px_rgba(163,163,163,0.08)]">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" onClick={() => navigate("/dashboard")} className="h-7 w-7 text-neutral-200/85 hover:text-neutral-100 hover:bg-neutral-400/12">
                <ArrowLeft className="h-3.5 w-3.5" />
              </Button>
              <div onClick={() => navigate("/")} className="flex items-center gap-1.5 cursor-pointer hover:opacity-80 transition-opacity">
                <span className="font-data text-[10px] tracking-[0.18em] text-neutral-100">CosmikAi</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsAnimationPaused((v) => !v)}
                className="h-7 px-2 font-data text-[10px] border-neutral-400/30 bg-neutral-900/88 text-neutral-100 hover:bg-neutral-700/30"
              >
                {isAnimationPaused ? <Play className="h-3 w-3 mr-1" /> : <Pause className="h-3 w-3 mr-1" />}
                {isAnimationPaused ? "Resume" : "Pause"}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowPlanetNames((v) => !v)}
                className="h-7 px-2 font-data text-[10px] border-neutral-400/30 bg-neutral-900/88 text-neutral-100 hover:bg-neutral-700/30"
              >
                {showPlanetNames ? "Names On" : "Names Off"}
              </Button>
            </div>
          </div>

          <div className="relative mb-3">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3 w-3 text-neutral-200/50" />
            <Input
              placeholder="Search stars..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setHighlightedSuggestionIdx(0);
              }}
              onFocus={() => setIsSearchFocused(true)}
              onBlur={() => {
                window.setTimeout(() => setIsSearchFocused(false), 120);
              }}
              onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                if (!showSuggestions || searchSuggestions.length === 0) return;

                if (e.key === "ArrowDown") {
                  e.preventDefault();
                  setHighlightedSuggestionIdx((prev) => Math.min(searchSuggestions.length - 1, prev + 1));
                  return;
                }
                if (e.key === "ArrowUp") {
                  e.preventDefault();
                  setHighlightedSuggestionIdx((prev) => Math.max(0, prev - 1));
                  return;
                }
                if (e.key === "Enter") {
                  e.preventDefault();
                  const selectedSuggestion = searchSuggestions[highlightedSuggestionIdx] ?? searchSuggestions[0];
                  if (selectedSuggestion) {
                    setSearchTerm(selectedSuggestion.name);
                    selectSystemFromSearch(selectedSuggestion.idx);
                    setIsSearchFocused(false);
                  }
                  return;
                }
                if (e.key === "Escape") {
                  setIsSearchFocused(false);
                }
              }}
              className="pl-9 h-8 rounded-sm bg-neutral-900/85 border-neutral-400/22 text-neutral-50 placeholder:text-neutral-300/40 font-data text-[11px]"
            />
            {showSuggestions && (
              <div className="absolute left-0 right-0 mt-1 rounded-sm border border-neutral-400/26 bg-neutral-950/96 backdrop-blur-sm shadow-[0_12px_30px_rgba(0,0,0,0.45)] overflow-hidden z-30">
                {searchSuggestions.length === 0 ? (
                  <p className="px-3 py-2 font-data text-[10px] text-neutral-300/55">No matching systems</p>
                ) : (
                  searchSuggestions.map((suggestion, suggestionIdx) => (
                    <button
                      key={suggestion.idx}
                      type="button"
                      onPointerDown={(event) => {
                        event.preventDefault();
                        setSearchTerm(suggestion.name);
                        selectSystemFromSearch(suggestion.idx);
                        setIsSearchFocused(false);
                      }}
                      className={`w-full text-left px-3 py-2 font-data text-[11px] border-b border-neutral-400/12 last:border-b-0 transition-colors ${
                        suggestionIdx === highlightedSuggestionIdx
                          ? "bg-neutral-500/18 text-neutral-50"
                          : "text-neutral-200/85 hover:bg-neutral-500/12"
                      }`}
                    >
                      {suggestion.name}
                    </button>
                  ))
                )}
              </div>
            )}
          </div>
          {normalizedSearch && (
            <p className="mt-1 font-data text-[10px] text-neutral-300/55">
              {visibleSystemIndices.length} match{visibleSystemIndices.length !== 1 ? "es" : ""}
            </p>
          )}

          <div className="border-y border-gray-300/20 py-2 mt-2 mb-2">
            <div className="flex items-end justify-between">
              <div>
                <p className="font-data text-[8px] tracking-[0.18em] text-gray-200/55 uppercase">System</p>
                <p className="font-data text-[26px] leading-none text-gray-50 lowercase">{selectedStar?.star_name ?? "no match"}</p>
              </div>
              <div className="text-right">
                <p className="font-data text-[8px] tracking-[0.16em] text-gray-200/55 uppercase">Designation</p>
                <p className="font-data text-base text-amber-300 leading-none">{focusDesignation}</p>
              </div>
            </div>
            <p className="font-data text-[10px] text-gray-100/55 mt-1">{selectedStar?.predictions.length ?? 0} detections</p>
          </div>

          <div className="flex items-start justify-between">
            <div>
              <p className="font-data text-[9px] tracking-[0.2em] text-gray-200/70">FOCUS</p>
              <p className="font-data text-sm tracking-wide text-gray-50 leading-tight">{focusTargetLabel}</p>
            </div>
            <div className="text-right">
              <p className="font-data text-[9px] tracking-[0.16em] text-gray-200/60">TRACK</p>
              <p className="font-data text-sm text-gray-100 leading-none">{trackPlanet ? "ON" : "OFF"}</p>
            </div>
          </div>

          <div className="mt-3 rounded-sm border border-neutral-400/18 bg-neutral-900/75 px-2 py-2">
            <div className="mb-1 flex justify-end">
              <p className="font-data text-[9px] text-gray-100/60 uppercase tracking-wider">Starmap Targets</p>
            </div>
            <div className="relative h-5">
              <div className="absolute left-3 right-2 top-1/2 h-px bg-gray-300/20" />
              <button
                onClick={() => {
                  suppressNextControlStartRef.current = true;
                  setFocusOnStar(true);
                  setTrackPlanet(false);
                  setCameraPreset((prev) => (prev === "free" ? "cinematic" : prev));
                  selectedPlanetFocusRef.current.set(0, 0, 0);
                  setFocusKey((k) => k + 1);
                }}
                className={`absolute left-0 top-1/2 -translate-y-1/2 z-10 h-7 w-3.5 rounded-r-full border border-l-0 transition-all ${
                  focusOnStar
                    ? "border-amber-200 shadow-[0_0_10px_rgba(251,191,36,0.45)]"
                    : "border-gray-200/30 opacity-85 hover:opacity-100"
                }`}
                style={{ backgroundColor: starColor }}
                title="Center on star"
              />
              {starmapTargetLayout.map(({ planet, idx, sizePx, leftPct }) => (
                <button
                  key={planet.id}
                  onClick={() => {
                    suppressNextControlStartRef.current = true;
                    setSelectedPlanetIdx(planet.predIdx);
                    setFocusOnStar(false);
                    setTrackPlanet(true);
                    setFocusKey((k) => k + 1);
                  }}
                  className={`absolute top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 rounded-full border transition-transform ${!focusOnStar && selectedPlanetIdx === planet.predIdx ? "scale-110 border-white" : "border-gray-200/30"}`}
                  style={{
                    left: `${leftPct}%`,
                    backgroundColor: planet.color ?? PLANET_COLORS[idx % PLANET_COLORS.length],
                    width: `${sizePx}px`,
                    height: `${sizePx}px`,
                    zIndex: 20 + idx,
                  }}
                  title={planet.displayName ?? planet.planetName ?? `Planet ${String.fromCharCode(65 + idx)}`}
                />
              ))}
            </div>
          </div>

          <div className="mt-3 border-t border-gray-300/20 pt-2 flex items-center justify-between">
            <p className="font-data text-[10px] text-neutral-200/70 uppercase tracking-wider">Probability</p>
            <p className="font-data text-[10px] text-gray-50">{selectedPred?.percentage.toFixed(1)}%</p>
          </div>
          <div className="mt-1 h-1.5 rounded-full bg-gray-800 overflow-hidden">
            <div className="h-full bg-gray-300" style={{ width: `${Math.max(4, selectedPred?.percentage ?? 0)}%` }} />
          </div>
        </div>

        <div className="bg-neutral-950/78 border border-neutral-400/20 rounded-sm p-3 backdrop-blur-sm">
          <div className="mb-2 flex items-center justify-between">
            <p className="font-data text-[10px] uppercase tracking-wider text-gray-100/75">Telemetry</p>
            {focusOnStar && selectedStarDetails?.source && (
              <p className="font-data text-[9px] text-gray-100/55">Source: {selectedStarDetails.source}</p>
            )}
          </div>
          {focusOnStar ? (
            <>
              {starDetailsLoading && !selectedStarDetails ? (
                <p className="font-data text-[10px] text-gray-100/60">Loading star details...</p>
              ) : (
                <div className="grid grid-cols-2 gap-2 mb-1">
                  {[
                    { label: "RA", value: selectedStarDetails?.ra != null ? selectedStarDetails.ra.toFixed(5) : "-" },
                    { label: "Dec", value: selectedStarDetails?.dec != null ? selectedStarDetails.dec.toFixed(5) : "-" },
                    { label: "Teff", value: selectedStarDetails?.teff != null ? `${selectedStarDetails.teff.toFixed(0)} K` : "-" },
                    { label: "Radius", value: selectedStarDetails?.radius != null ? `${selectedStarDetails.radius.toFixed(2)} Rsun` : "-" },
                    { label: "Mass", value: selectedStarDetails?.mass != null ? `${selectedStarDetails.mass.toFixed(2)} Msun` : "-" },
                    { label: "Distance", value: selectedStarDetails?.distance != null ? `${selectedStarDetails.distance.toFixed(1)} pc` : "-" },
                    { label: "GAIA", value: selectedStarDetails?.gaia_id ?? "-" },
                    { label: "TIC", value: selectedStarDetails?.tic_id ?? "-" },
                  ].map((item) => (
                    <div key={item.label} className="rounded-sm border border-neutral-400/16 bg-neutral-900/75 px-2 py-1.5">
                      <p className="font-data text-[9px] text-gray-100/55 uppercase tracking-wider">{item.label}</p>
                      <p className="font-data text-[11px] text-gray-50 truncate">{item.value}</p>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2 mb-3">
                {[
                  { label: "Period", value: selectedPred?.period_days?.toFixed(3) ?? "-" },
                  { label: "Score", value: selectedPred?.score.toFixed(4) ?? "-" },
                  {
                    label: "Radius Est",
                    value: selectedPlanetRadiusEarth != null ? `${selectedPlanetRadiusEarth.toFixed(2)} Re` : "-",
                  },
                  { label: "Body", value: selectedPlanetName },
                  { label: "Mission", value: selectedPred?.mission ?? "-" },
                  { label: "Status", value: selectedPred?.verdict === "TRANSIT_DETECTED" ? "Transit" : "No Transit" },
                ].map((item) => (
                  <div key={item.label} className="rounded-sm border border-neutral-400/16 bg-neutral-900/75 px-2 py-1.5">
                    <p className="font-data text-[9px] text-gray-100/55 uppercase tracking-wider">{item.label}</p>
                    <p className="font-data text-[11px] text-gray-50 truncate">{item.value}</p>
                  </div>
                ))}
              </div>

              {chartData.length > 0 && (
                <div className="rounded-sm border border-neutral-400/16 bg-neutral-900/75 p-2">
                  <ResponsiveContainer width="100%" height={130}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="2 2" stroke="rgba(156,163,175,0.20)" />
                      <XAxis
                        dataKey="phase"
                        tick={{ fill: "rgba(209,213,219,0.65)", fontSize: 8, fontFamily: "JetBrains Mono" }}
                        tickLine={false}
                        axisLine={{ stroke: "rgba(156,163,175,0.22)" }}
                      />
                      <YAxis
                        tick={{ fill: "rgba(209,213,219,0.65)", fontSize: 8, fontFamily: "JetBrains Mono" }}
                        tickLine={false}
                        axisLine={{ stroke: "rgba(156,163,175,0.22)" }}
                        tickFormatter={(v) => v.toFixed(3)}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "rgba(17,24,39,0.95)",
                          border: "1px solid rgba(156,163,175,0.30)",
                          borderRadius: "4px",
                          fontFamily: "JetBrains Mono",
                          fontSize: 10,
                        }}
                        labelStyle={{ color: "rgba(209,213,219,0.72)" }}
                        itemStyle={{ color: "rgba(243,244,246,1)" }}
                      />
                      <Line
                        type="monotone"
                        dataKey="flux"
                        stroke={selectedPred?.verdict === "TRANSIT_DETECTED" ? "#d1d5db" : "#9ca3af"}
                        strokeWidth={1.4}
                        dot={false}
                        activeDot={{ r: 2.5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </div>

        <div className="bg-neutral-950/74 border border-neutral-400/20 rounded-sm p-2 backdrop-blur-sm">
          <p className="font-data text-[10px] tracking-wider text-gray-100/75 uppercase mb-1">System Map</p>
          <div className="relative mx-auto mb-1 aspect-square w-full max-w-[15rem] rounded-full border border-neutral-400/16 bg-neutral-900/60 overflow-hidden">
            <div
              className="absolute inset-0 rounded-full"
              style={{
                background:
                  "radial-gradient(circle at center, rgba(75,85,99,0.30) 0%, rgba(55,65,81,0.22) 48%, rgba(17,24,39,0.42) 100%)",
              }}
            />
            <div className="absolute left-1/2 top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-1 ring-amber-200/40" style={{ backgroundColor: starColor }} />
            {selectedSystem3d.slice(0, 8).map((planet, idx) => {
              const maxRadius = Math.max(1, selectedSystemOrbitExtent);
              const norm = planet.orbitRadius / maxRadius;
              const ringSizePct = 18 + norm * 74;
              const angle = (planet.orbitPhaseStart ?? phaseSeedFromPlanet(planet.id, idx)) - minimapTime * planet.orbitSpeed;
              const ringRadiusPct = ringSizePct / 2;
              const dotX = 50 + Math.cos(angle) * ringRadiusPct;
              const dotY = 50 + Math.sin(angle) * ringRadiusPct;
              return (
                <div key={`minimap-${planet.id}`}>
                  <svg className="absolute inset-0 pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <circle cx="50" cy="50" r={ringRadiusPct} fill="none" stroke="rgba(156,163,175,0.22)" strokeWidth="0.6" />
                    <path
                      d={orbitArcPath(50, 50, ringRadiusPct, angle, angle + 0.6)}
                      fill="none"
                      stroke="rgba(156,163,175,0.65)"
                      strokeOpacity="0.6"
                      strokeWidth="0.9"
                      strokeLinecap="round"
                    />
                  </svg>
                  <div
                    className={`absolute h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full ${!focusOnStar && selectedPlanetIdx === idx ? "ring-2 ring-gray-100" : ""}`}
                    style={{ left: `${dotX}%`, top: `${dotY}%`, backgroundColor: planet.color }}
                    title={planet.planetName ?? planet.name}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </motion.div>

      <div className="absolute top-4 right-4 z-20 flex items-center gap-2">
        <div className="hidden md:flex items-center gap-1 rounded-sm border border-neutral-400/26 bg-neutral-900/82 px-1 py-1 mr-1">
          {[
            { id: "cinematic", label: "Cine" },
            { id: "tactical", label: "Tac" },
            { id: "close", label: "Close" },
            { id: "free", label: "Free" },
          ].map((preset) => (
            <Button
              key={preset.id}
              size="sm"
              variant="ghost"
              onClick={() => {
                setCameraPreset(preset.id as CameraPreset);
                if (preset.id !== "free") {
                  setFocusKey((k) => k + 1);
                }
              }}
              className={`h-6 px-2 font-data text-[10px] ${cameraPreset === preset.id ? "bg-neutral-500/22 text-neutral-50" : "text-neutral-200/75 hover:bg-neutral-500/12"}`}
            >
              {preset.label}
            </Button>
          ))}
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            const next = !trackPlanet;
            setTrackPlanet(next);
            if (next && !focusOnStar) {
              setFocusKey((k) => k + 1);
            }
          }}
          className="h-8 border-neutral-400/26 bg-neutral-900/82 text-neutral-50 hover:bg-neutral-500/12"
        >
          {trackPlanet ? "Track On" : "Track Off"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            if (visibleSystemIndices.length === 0) return;
            const currentVisibleIdx = selectedVisibleIdx === -1 ? 0 : selectedVisibleIdx;
            const prevVisibleIdx = Math.max(0, currentVisibleIdx - 1);
            setSelectedStarIdx(visibleSystemIndices[prevVisibleIdx]);
            setSelectedPlanetIdx(0);
            setFocusOnStar(true);
            selectedPlanetFocusRef.current.set(0, 0, 0);
            setFocusKey((k) => k + 1);
          }}
          disabled={visibleSystemIndices.length === 0 || selectedVisibleIdx <= 0}
          className="h-8 border-neutral-400/26 bg-neutral-900/82 text-neutral-50 hover:bg-neutral-500/12"
        >
          Prev System
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            if (visibleSystemIndices.length === 0) return;
            const currentVisibleIdx = selectedVisibleIdx === -1 ? 0 : selectedVisibleIdx;
            const nextVisibleIdx = Math.min(visibleSystemIndices.length - 1, currentVisibleIdx + 1);
            setSelectedStarIdx(visibleSystemIndices[nextVisibleIdx]);
            setSelectedPlanetIdx(0);
            setFocusOnStar(true);
            selectedPlanetFocusRef.current.set(0, 0, 0);
            setFocusKey((k) => k + 1);
          }}
          disabled={
            visibleSystemIndices.length === 0 ||
            selectedVisibleIdx === -1 ||
            selectedVisibleIdx >= visibleSystemIndices.length - 1
          }
          className="h-8 border-neutral-400/26 bg-neutral-900/82 text-neutral-50 hover:bg-neutral-500/12"
        >
          Next System
        </Button>
      </div>

      <div className="absolute bottom-4 right-4 z-20 rounded-md border border-neutral-400/20 bg-neutral-950/74 px-3 py-2 backdrop-blur-sm">
        <p className="font-data text-[10px] tracking-wider text-gray-100/75 uppercase">Controls</p>
        <p className="font-data text-[10px] text-gray-50/85">LMB select • drag orbit • wheel zoom • click star to center • track planet toggle</p>
      </div>

      <div
        className="pointer-events-none absolute inset-0 z-10 opacity-[0.16] mix-blend-screen"
        style={{
          backgroundImage:
            "repeating-linear-gradient(to bottom, rgba(196,203,214,0.06) 0px, rgba(196,203,214,0.06) 1px, transparent 1px, transparent 3px)",
        }}
      />
    </div>
  );
};

export default Visualizer;


