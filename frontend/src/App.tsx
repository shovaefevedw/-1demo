import { useEffect, useRef, useState } from "react";
import LeftPanel from "./components/LeftPanel";
import RightPanel from "./components/RightPanel";
import Viewer from "./components/Viewer";
import {
  downloadUrl,
  estimateDepth,
  fetchHealth,
  fetchPreview,
  triggerDownload,
  writePointCloud,
} from "./lib/api";
import { defaultSettings, type DepthJob, type Health, type ModelName, type Settings } from "./lib/types";

const GEOMETRY_KEYS: (keyof Settings)[] = [
  "near",
  "far",
  "depthStrength",
  "invertDepth",
  "smoothing",
  "density",
  "maxPoints",
  "colorMode",
  "pointSize",
  "opacity",
  "dotContrast",
  "dotColor",
  "background",
];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
  const [model, setModel] = useState<ModelName>("small");
  const [job, setJob] = useState<DepthJob | null>(null);
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [pointCount, setPointCount] = useState(0);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const photoUrlRef = useRef<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [cloudReady, setCloudReady] = useState(false);
  const requestId = useRef(0);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (model !== "small" && model !== "base" && model !== "large") setModel("large");
  }, [model]);

  useEffect(() => {
    const prevent = (event: DragEvent) => event.preventDefault();
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", prevent);
    return () => {
      window.removeEventListener("dragover", prevent);
      window.removeEventListener("drop", prevent);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!job || !cloudReady) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void refreshCloud(controller.signal);
    }, 120);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
    // Depth, density, dot size, and background rebuild the flat photograph.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.jobId, cloudReady, ...GEOMETRY_KEYS.map((key) => settings[key])]);

  function chooseFile(next: File) {
    const name = next.name.toLowerCase();
    const allowed = next.type === "image/jpeg" || next.type === "image/png" || /\.(jpe?g|png)$/.test(name);
    if (!allowed) {
      setError("Use a JPG, JPEG, or PNG photograph.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const url = URL.createObjectURL(next);
    setFile(next);
    setPreviewUrl(url);
    setJob(null);
    setCloudReady(false);
    replacePhoto(null);
    setPointCount(0);
    setError(null);
    const image = new Image();
    image.onload = () => setImageSize({ width: image.naturalWidth, height: image.naturalHeight });
    image.src = url;
  }

  async function generateDepth() {
    if (!file) return;
    setBusy(true);
    setError(null);
    setCloudReady(false);
    replacePhoto(null);
    setPointCount(0);
    try {
      const next = await estimateDepth(file, model);
      setJob(next);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Depth estimation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function refreshCloud(signal?: AbortSignal) {
    if (!job) return;
    const id = requestId.current + 1;
    requestId.current = id;
    setUpdating(true);
    setError(null);
    try {
      const cloud = await fetchPreview(job.jobId, settings, signal ?? new AbortController().signal);
      if (id !== requestId.current) {
        URL.revokeObjectURL(cloud.url);
        return;
      }
      replacePhoto(cloud.url);
      setPointCount(cloud.count);
      setElapsedMs(cloud.elapsedMs);
      setCloudReady(true);
    } catch (exc) {
      if (signal?.aborted) return;
      if (id !== requestId.current) return;
      setError(exc instanceof Error ? exc.message : "Could not build the point cloud.");
    } finally {
      if (id === requestId.current) setUpdating(false);
    }
  }

  async function saveAndDownload(filename: string) {
    if (!job) return;
    setExporting(true);
    setError(null);
    try {
      if (filename !== "depth_16bit.png" && filename !== "depth_preview.png") {
        await writePointCloud(job.jobId, settings);
      }
      triggerDownload(downloadUrl(job.jobId, filename));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Export failed.");
    } finally {
      setExporting(false);
    }
  }

  function replacePhoto(url: string | null) {
    if (photoUrlRef.current) URL.revokeObjectURL(photoUrlRef.current);
    photoUrlRef.current = url;
    setPhotoUrl(url);
  }

  function patchSettings(patch: Partial<Settings>) {
    setSettings((current) => {
      const next = { ...current, ...patch };
      if (next.far <= next.near) next.far = next.near + 0.1;
      return next;
    });
  }

  const modelStale = job != null && job.model !== model;

  return (
    <div className="flex h-full min-h-screen min-w-[1080px] flex-col bg-[#0c0d10] text-zinc-100">
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-white/10 px-4">
        <div className="flex items-baseline gap-3">
          <span className="text-sm font-medium tracking-tight">PhotoDepth 3D</span>
          <span className="text-[11px] text-zinc-500">Relative depth, not meters</span>
        </div>
        <span className="text-[11px] uppercase tracking-[0.14em] text-zinc-500">
          {health ? (health.device === "mps" ? "Apple Metal" : "CPU") : "Backend offline"}
          {modelStale ? " · regenerate depth" : ""}
        </span>
      </header>
      <div className="grid min-h-0 flex-1 grid-cols-[300px_minmax(0,1fr)_320px]">
        <LeftPanel
          file={file}
          previewUrl={previewUrl}
          imageSize={imageSize}
          model={model}
          job={job}
          busy={busy}
          error={error}
          onFile={chooseFile}
          onModel={setModel}
          onEstimate={() => void generateDepth()}
          onBuildCloud={() => void refreshCloud()}
        />
        <Viewer photoUrl={photoUrl} pointCount={pointCount} updating={updating} elapsedMs={elapsedMs} />
        <RightPanel
          settings={settings}
          canExport={job != null}
          exporting={exporting}
          onChange={patchSettings}
          onReset={() => setSettings(defaultSettings)}
          onDownloadDepth={() => void saveAndDownload("depth_16bit.png")}
          onDownloadPhoto={() => void saveAndDownload("pointcloud.png")}
          onDownloadPly={() => void saveAndDownload("pointcloud.ply")}
          onDownloadXyz={() => void saveAndDownload("pointcloud.xyz")}
          onDownloadAll={() => void saveAndDownload("results.zip")}
        />
      </div>
    </div>
  );
}
