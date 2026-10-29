import { formatCount } from "../lib/format";

type Props = {
  photoUrl: string | null;
  pointCount: number;
  updating: boolean;
  elapsedMs: number | null;
};

export default function Viewer({ photoUrl, pointCount, updating, elapsedMs }: Props) {
  return (
    <section className="relative flex min-h-0 min-w-0 items-center justify-center overflow-hidden bg-[#0b0c0f]">
      {photoUrl ? (
        <img
          src={photoUrl}
          alt="Flat point cloud photograph"
          className="max-h-full max-w-full object-contain"
        />
      ) : (
        <p className="max-w-md px-8 text-center text-sm leading-relaxed text-zinc-400">
          Generate Depth to read front-to-back order. Generate Point Cloud then draws a flat
          photograph: larger, lighter dots are nearer, and smaller, darker dots are farther.
        </p>
      )}

      <div className="pointer-events-none absolute inset-x-0 top-0 p-3">
        <div className="inline-block rounded-lg bg-black/50 px-3 py-2 text-xs text-zinc-200 backdrop-blur-sm">
          <div className="tabular-nums">{pointCount > 0 ? `${formatCount(pointCount)} points` : "No point photograph"}</div>
          <div className="mt-0.5 text-[11px] text-zinc-400">
            {elapsedMs != null ? `${elapsedMs} ms` : "Flat photograph"}
            {updating ? " · updating" : ""}
          </div>
        </div>
      </div>
    </section>
  );
}
