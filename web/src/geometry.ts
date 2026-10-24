// Normalised (0–1) image geometry shared by the canvas and the review page.
// A box is [x0, y0, x1, y1], a point [x, y]; stored values keep 4 decimals.

export const round4 = (v: number) => +v.toFixed(4);

export const inBox = (p: number[], b: number[]) => p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];

export const center = (b: number[]) => [round4((b[0] + b[2]) / 2), round4((b[1] + b[3]) / 2)];

export function iou(a: number[], b: number[]) {
  const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = w * h, u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
  return u > 0 ? i / u : 0;
}
