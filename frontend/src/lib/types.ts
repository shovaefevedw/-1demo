export type ModelName = "small" | "base" | "large";
export type ColorMode = "rgb" | "depth";
export type Projection = "perspective" | "orthographic";

export type Settings = {
  near: number;
  far: number;
  depthStrength: number;
  invertDepth: boolean;
  smoothing: number;
  pointSize: number;
  density: number;
  maxPoints: number;
  opacity: number;
  colorMode: ColorMode;
  background: string;
  fov: number;
  projection: Projection;
};

export const defaultSettings: Settings = {
  near: 0.4,
  far: 2.5,
  depthStrength: 1,
  invertDepth: false,
  smoothing: 0,
  pointSize: 2,
  density: 1,
  maxPoints: 500000,
  opacity: 1,
  colorMode: "rgb",
  background: "#101114",
  fov: 55,
  projection: "perspective",
};

export type DepthJob = {
  jobId: string;
  width: number;
  height: number;
  model: ModelName;
  modelId: string;
  device: string;
  predictionMin: number;
  predictionMax: number;
  elapsedMs: number;
};

export type CloudBuffers = {
  positions: Float32Array;
  colors: Float32Array;
  count: number;
  elapsedMs: number | null;
};

export type Health = {
  status: string;
  device: string;
  mps_available: boolean;
  torch: string;
};
