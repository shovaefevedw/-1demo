export function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${Math.max(1, Math.round(value / 1024))} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}
