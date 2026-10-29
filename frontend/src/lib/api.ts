import type { CloudBuffers, DepthJob, Health, ModelName, Settings } from "./types";

export function settingsPayload(jobId: string, settings: Settings) {
  return {
    job_id: jobId,
    near: settings.near,
    far: settings.far,
    depth_strength: settings.depthStrength,
    invert_depth: settings.invertDepth,
    smoothing: settings.smoothing,
    density: settings.density,
    max_points: settings.maxPoints,
    color_mode: settings.colorMode,
    fov_deg: settings.fov,
    point_size: settings.pointSize,
    point_opacity: settings.opacity,
    background: settings.background,
    projection: settings.projection,
  };
}

async function readError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((item: { msg?: string }) => item.msg ?? "Invalid value").join("; ");
    }
  } catch {
    // The body was not JSON.
  }
  return response.statusText || "Request failed";
}

export async function fetchHealth(): Promise<Health> {
  const response = await fetch("/health");
  if (!response.ok) throw new Error("The local backend is not running.");
  return response.json();
}

export async function estimateDepth(file: File, model: ModelName): Promise<DepthJob> {
  const body = new FormData();
  body.append("file", file);
  body.append("model", model);
  const response = await fetch("/api/depth", { method: "POST", body });
  if (!response.ok) throw new Error(await readError(response));
  const data = await response.json();
  return {
    jobId: data.job_id,
    width: data.width,
    height: data.height,
    model: data.model_variant,
    modelId: data.model_id,
    device: data.device,
    predictionMin: data.prediction_min,
    predictionMax: data.prediction_max,
    elapsedMs: data.elapsed_ms,
  };
}

export function parsePreview(buffer: ArrayBuffer): { positions: Float32Array; colors: Float32Array; count: number } {
  if (buffer.byteLength < 4) throw new Error("The point cloud response was empty.");
  const view = new DataView(buffer);
  const count = view.getUint32(0, true);
  const positionBytes = count * 12;
  const colorBytes = count * 3;
  if (buffer.byteLength < 4 + positionBytes + colorBytes) {
    throw new Error("The point cloud response was incomplete.");
  }
  const positions = new Float32Array(buffer.slice(4, 4 + positionBytes));
  const rawColors = new Uint8Array(buffer, 4 + positionBytes, colorBytes);
  const colors = new Float32Array(colorBytes);
  for (let index = 0; index < colorBytes; index += 1) {
    colors[index] = rawColors[index] / 255;
  }
  return { positions, colors, count };
}

export async function fetchPreview(
  jobId: string,
  settings: Settings,
  signal: AbortSignal,
): Promise<CloudBuffers> {
  const response = await fetch("/api/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settingsPayload(jobId, settings)),
    signal,
  });
  if (!response.ok) throw new Error(await readError(response));
  const parsed = parsePreview(await response.arrayBuffer());
  const elapsed = response.headers.get("X-Elapsed-Ms");
  return {
    ...parsed,
    elapsedMs: elapsed ? Number(elapsed) : null,
  };
}

export async function writePointCloud(jobId: string, settings: Settings): Promise<void> {
  const response = await fetch("/api/pointcloud", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settingsPayload(jobId, settings)),
  });
  if (!response.ok) throw new Error(await readError(response));
}

export function downloadUrl(jobId: string, filename: string): string {
  return `/api/export/${jobId}/${filename}`;
}

export function triggerDownload(url: string): void {
  const link = document.createElement("a");
  link.href = url;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
}
