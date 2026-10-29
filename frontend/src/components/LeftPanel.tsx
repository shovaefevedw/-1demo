import type { DepthJob, ModelName } from "../lib/types";
import { formatBytes } from "../lib/format";

type Props = {
  file: File | null;
  previewUrl: string | null;
  imageSize: { width: number; height: number } | null;
  model: ModelName;
  job: DepthJob | null;
  busy: boolean;
  error: string | null;
  onFile: (file: File) => void;
  onModel: (model: ModelName) => void;
  onEstimate: () => void;
  onBuildCloud: () => void;
};

const MODELS: { id: ModelName; label: string; note: string }[] = [
  { id: "small", label: "Small", note: "Recommended" },
  { id: "base", label: "Base", note: "More detail" },
  { id: "large", label: "Large", note: "Heavy" },
];

export default function LeftPanel({
  file,
  previewUrl,
  imageSize,
  model,
  job,
  busy,
  error,
  onFile,
  onModel,
  onEstimate,
  onBuildCloud,
}: Props) {
  return (
    <aside className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto border-r border-white/10 bg-[#121317] p-4">
      <div>
        <p className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Input</p>
        <h2 className="mt-1 text-sm font-medium text-zinc-100">Photograph</h2>
      </div>

      <label
        className="flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-white/15 bg-black/20 px-4 py-6 text-center transition hover:border-white/30"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          const next = event.dataTransfer.files?.[0];
          if (next) onFile(next);
        }}
      >
        <span className="text-sm text-zinc-200">Drop a JPG or PNG</span>
        <span className="mt-1 text-xs text-zinc-500">or choose a file</span>
        <input
          className="hidden"
          type="file"
          accept="image/jpeg,image/png,.jpg,.jpeg,.png"
          onChange={(event) => {
            const next = event.target.files?.[0];
            if (next) onFile(next);
          }}
        />
      </label>

      {previewUrl && (
        <img
          src={previewUrl}
          alt="Original photograph"
          className="max-h-52 w-full rounded-lg bg-black object-contain"
        />
      )}

      <dl className="space-y-1 text-xs text-zinc-400">
        <div className="flex justify-between gap-3">
          <dt>File</dt>
          <dd className="truncate text-zinc-200">{file ? file.name : "None"}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt>Size</dt>
          <dd className="text-zinc-200">{file ? formatBytes(file.size) : "—"}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt>Pixels</dt>
          <dd className="text-zinc-200">
            {imageSize ? `${imageSize.width} × ${imageSize.height}` : "—"}
          </dd>
        </div>
        {job && (
          <>
            <div className="flex justify-between gap-3">
              <dt>Depth range</dt>
              <dd className="text-zinc-200">
                {job.predictionMin.toFixed(2)} – {job.predictionMax.toFixed(2)}
              </dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt>Device</dt>
              <dd className="uppercase text-zinc-200">{job.device}</dd>
            </div>
          </>
        )}
      </dl>

      {job && (
        <img
          src={`/api/export/${job.jobId}/depth_preview.png`}
          alt="Visual depth preview. Brighter areas are closer."
          className="max-h-36 w-full rounded-lg bg-black object-contain"
        />
      )}

      <div>
        <p className="mb-2 text-[11px] uppercase tracking-[0.16em] text-zinc-500">Model</p>
        <div className="grid grid-cols-3 gap-1 rounded-lg bg-black/30 p-1">
          {MODELS.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onModel(item.id)}
              className={`rounded-md px-2 py-2 text-left ${
                model === item.id ? "bg-zinc-100 text-zinc-950" : "text-zinc-300 hover:bg-white/5"
              }`}
            >
              <span className="block text-xs font-medium">{item.label}</span>
              <span className={`block text-[10px] ${model === item.id ? "text-zinc-600" : "text-zinc-500"}`}>
                {item.note}
              </span>
            </button>
          ))}
        </div>
        <p className="mt-2 text-[11px] leading-relaxed text-zinc-500">
          Depth Anything V2 relative disparity. Small is the reliable default on Apple Silicon.
        </p>
      </div>

      {error && (
        <p className="rounded-lg border border-red-400/30 bg-red-400/10 px-3 py-2 text-xs leading-relaxed text-red-100">
          {error}
        </p>
      )}

      <div className="mt-auto space-y-2">
        <button
          type="button"
          disabled={!file || busy}
          onClick={onEstimate}
          className="w-full rounded-lg bg-[#e7e1d5] px-3 py-2.5 text-sm font-medium text-zinc-950 disabled:opacity-40"
        >
          {busy ? "Estimating depth…" : "Generate Depth"}
        </button>
        <button
          type="button"
          disabled={!job || busy}
          onClick={onBuildCloud}
          className="w-full rounded-lg border border-white/15 px-3 py-2.5 text-sm text-zinc-100 disabled:opacity-40"
        >
          Generate Point Cloud
        </button>
      </div>
    </aside>
  );
}
