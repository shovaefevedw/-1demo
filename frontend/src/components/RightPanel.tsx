import type { ColorMode, Projection, Settings } from "../lib/types";
import { formatCount } from "../lib/format";

type Props = {
  settings: Settings;
  canExport: boolean;
  exporting: boolean;
  onChange: (patch: Partial<Settings>) => void;
  onReset: () => void;
  onDownloadDepth: () => void;
  onDownloadPly: () => void;
  onDownloadXyz: () => void;
  onDownloadAll: () => void;
};

export default function RightPanel({
  settings,
  canExport,
  exporting,
  onChange,
  onReset,
  onDownloadDepth,
  onDownloadPly,
  onDownloadXyz,
  onDownloadAll,
}: Props) {
  return (
    <aside className="flex h-full min-h-0 flex-col gap-5 overflow-y-auto border-l border-white/10 bg-[#121317] p-4">
      <section className="space-y-3">
        <p className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Depth</p>
        <Slider
          label="Depth Strength"
          min={0.2}
          max={3}
          step={0.05}
          value={settings.depthStrength}
          display={settings.depthStrength.toFixed(2)}
          onChange={(depthStrength) => onChange({ depthStrength })}
        />
        <Slider
          label="Virtual Near"
          min={0.1}
          max={2}
          step={0.05}
          value={settings.near}
          display={settings.near.toFixed(2)}
          onChange={(near) => onChange({ near, far: Math.max(settings.far, near + 0.1) })}
        />
        <Slider
          label="Virtual Far"
          min={0.3}
          max={8}
          step={0.05}
          value={settings.far}
          display={settings.far.toFixed(2)}
          onChange={(far) => onChange({ far, near: Math.min(settings.near, far - 0.1) })}
        />
        <Slider
          label="Depth Smoothing"
          min={0}
          max={1}
          step={0.05}
          value={settings.smoothing}
          display={settings.smoothing === 0 ? "Off" : settings.smoothing.toFixed(2)}
          onChange={(smoothing) => onChange({ smoothing })}
        />
        <label className="flex items-center justify-between text-xs text-zinc-300">
          Invert Depth
          <input
            type="checkbox"
            checked={settings.invertDepth}
            onChange={(event) => onChange({ invertDepth: event.target.checked })}
          />
        </label>
        <p className="text-[11px] leading-relaxed text-zinc-500">
          Near and far are virtual scene units, not meters. Higher model disparity stays closer unless inverted.
        </p>
      </section>

      <section className="space-y-3">
        <p className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Point cloud</p>
        <Slider
          label="Point Size"
          min={0.5}
          max={8}
          step={0.1}
          value={settings.pointSize}
          display={settings.pointSize.toFixed(1)}
          onChange={(pointSize) => onChange({ pointSize })}
        />
        <Slider
          label="Point Density"
          min={0.1}
          max={1}
          step={0.05}
          value={settings.density}
          display={`${Math.round(settings.density * 100)}%`}
          onChange={(density) => onChange({ density })}
        />
        <Slider
          label="Maximum Points"
          min={50000}
          max={2000000}
          step={50000}
          value={settings.maxPoints}
          display={formatCount(settings.maxPoints)}
          onChange={(maxPoints) => onChange({ maxPoints })}
        />
        <Slider
          label="Point Opacity"
          min={0.15}
          max={1}
          step={0.05}
          value={settings.opacity}
          display={settings.opacity.toFixed(2)}
          onChange={(opacity) => onChange({ opacity })}
        />
        <div className="grid grid-cols-2 gap-1 rounded-lg bg-black/30 p-1">
          <Toggle
            active={settings.colorMode === "rgb"}
            label="RGB"
            onClick={() => onChange({ colorMode: "rgb" satisfies ColorMode })}
          />
          <Toggle
            active={settings.colorMode === "depth"}
            label="Depth"
            onClick={() => onChange({ colorMode: "depth" satisfies ColorMode })}
          />
        </div>
      </section>

      <section className="space-y-3">
        <p className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Display</p>
        <label className="flex items-center justify-between text-xs text-zinc-300">
          Background
          <input
            type="color"
            value={settings.background}
            onChange={(event) => onChange({ background: event.target.value })}
            className="h-7 w-10 cursor-pointer bg-transparent"
          />
        </label>
        <Slider
          label="Camera FOV"
          min={30}
          max={80}
          step={1}
          value={settings.fov}
          display={`${Math.round(settings.fov)}°`}
          onChange={(fov) => onChange({ fov })}
        />
        <div className="grid grid-cols-2 gap-1 rounded-lg bg-black/30 p-1">
          <Toggle
            active={settings.projection === "perspective"}
            label="Perspective"
            onClick={() => onChange({ projection: "perspective" satisfies Projection })}
          />
          <Toggle
            active={settings.projection === "orthographic"}
            label="Orthographic"
            onClick={() => onChange({ projection: "orthographic" satisfies Projection })}
          />
        </div>
        <button
          type="button"
          onClick={onReset}
          className="w-full rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-300 hover:bg-white/5"
        >
          Reset Settings
        </button>
      </section>

      <section className="mt-auto space-y-2">
        <p className="text-[11px] uppercase tracking-[0.16em] text-zinc-500">Export</p>
        <ExportButton disabled={!canExport || exporting} label="Download 16-bit Depth PNG" onClick={onDownloadDepth} />
        <ExportButton disabled={!canExport || exporting} label="Download PLY" onClick={onDownloadPly} />
        <ExportButton disabled={!canExport || exporting} label="Download XYZ" onClick={onDownloadXyz} />
        <ExportButton disabled={!canExport || exporting} label="Download All Results" onClick={onDownloadAll} />
      </section>
    </aside>
  );
}

function Slider({
  label,
  min,
  max,
  step,
  value,
  display,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  display: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="mb-1 flex items-center justify-between text-xs text-zinc-400">
        {label}
        <span className="tabular-nums text-zinc-200">{display}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function Toggle({ active, label, onClick }: { active: boolean; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-md px-2 py-1.5 text-xs ${active ? "bg-zinc-100 text-zinc-950" : "text-zinc-300 hover:bg-white/5"}`}
    >
      {label}
    </button>
  );
}

function ExportButton({
  label,
  disabled,
  onClick,
}: {
  label: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="w-full rounded-lg border border-white/10 px-3 py-2 text-left text-xs text-zinc-200 hover:bg-white/5 disabled:opacity-40"
    >
      {label}
    </button>
  );
}
