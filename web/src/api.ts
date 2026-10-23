// Thin client for the forge API (src/vlmforge/server/api.py).

export type Skill = "detect" | "point" | "query" | "caption";
export type Pred = { boxes?: number[][]; points?: number[][]; text?: string };

export interface ImageRow {
  id: string;
  path: string;
  width: number;
  height: number;
  source: string;
  added_at: string;
}

export interface Item {
  id: number;
  queue_id: number | null;
  image_id: string;
  skill: Skill;
  prompt: string;
  pred: Pred | null;
  pred_model: string | null;
  label: Pred | null;
  judge: JudgeVerdict | null;
  teacher: Pred | null;
  verdict: "accept" | "fix" | null;
  spot: number;
  status: "pending" | "auto" | "approved" | "corrected" | "discarded";
  reviewer: string | null;
  reviewed_at: string | null;
  path: string;
  width: number;
  height: number;
  source: string;
}

export interface JudgeVerdict {
  verdict: "accept" | "fix";
  note: string;
  model?: string;
  wrong?: number[];
  missed?: number;
  answer?: string;
  unsupported?: string[];
  raw?: string;
}

export interface Queue {
  id: number;
  name: string;
  skill: Skill;
  model: string;
  judge: string;
  mode: "manual" | "auto";
  config: string | null;
  note: string;
  created_at: string;
  counts: Record<string, number>;
  verdicts: Record<string, number>;
  spot: number;
  total: number;
  job?: { state: string; phase?: string; done: number; total: number; error: string | null; triage?: Record<string, number> } | null;
}

export interface Model {
  name: string;
  label: string;
  meta?: Record<string, unknown>;
}

export interface Run {
  name: string;
  adapters: string[];
  meta: Record<string, unknown> | null;
  last: Record<string, number> | null;
  curve: { step: number; loss?: number; val?: number }[];
  evals: Record<string, Record<string, number>>;
  running: boolean;
}

export interface Report {
  path: string;
  model: string;
  manifest: string;
  macro: Record<string, number>;
}

export interface EngineStatus {
  loaded: boolean;
  weights: string;
  weights_ok: boolean;
  judge: string;
  judge_ok: boolean;
  judge_loaded: boolean;
  current: string;
  pending: number;
  busy: boolean;
  gpu_mem_gb: number | null;
  last_error: string | null;
}

export interface Status {
  version: string;
  engine: EngineStatus;
  models: Model[];
}

async function j<T>(res: Promise<Response>): Promise<T> {
  const r = await res;
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const b = await r.json();
      if (b.detail) msg = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail);
    } catch {
      /* ignore */
    }
    throw new Error(msg);
  }
  return r.json();
}

const post = (url: string, body: unknown) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  status: () => j<Status>(fetch("/api/status")),
  models: () => j<Model[]>(fetch("/api/models")),
  images: (source?: string) => j<ImageRow[]>(fetch(`/api/images?limit=2000${source ? `&source=${encodeURIComponent(source)}` : ""}`)),
  sources: () => j<{ source: string; n: number }[]>(fetch("/api/sources")),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return j<ImageRow>(fetch("/api/images", { method: "POST", body: fd }));
  },
  scan: (dir: string, source: string) => j<{ added: number }>(post("/api/images/scan", { dir, source })),
  imageUrl: (id: string, w?: number) => `/api/images/${id}/file${w ? `?w=${w}` : ""}`,
  infer: (image_id: string, skill: Skill, prompt: string, model: string) =>
    j<{ pred: Pred; model: string; ms: number }>(post("/api/infer", { image_id, skill, prompt, model })),
  compare: (image_id: string, skill: Skill, prompt: string, models: string[]) =>
    j<Record<string, { pred: Pred; ms: number }>>(post("/api/compare", { image_id, skill, prompt, models })),
  saveItem: (body: Record<string, unknown>) => j<Item>(post("/api/items", body)),
  judge: (image_id: string, skill: Skill, prompt: string, pred: Pred) =>
    j<JudgeVerdict & { ms: number }>(post("/api/judge", { image_id, skill, prompt, pred })),
  judgeQueue: (id: number) => j<Queue>(post(`/api/queues/${id}/judge`, {})),
  queues: () => j<Queue[]>(fetch("/api/queues")),
  queue: (id: number) => j<Queue>(fetch(`/api/queues/${id}`)),
  createQueue: (body: Record<string, unknown>) => j<Queue>(post("/api/queues", body)),
  createAutoQueue: (body: Record<string, unknown>) => j<Queue>(post("/api/queues/auto", body)),
  queueImages: (id: number) => j<ImageSummary[]>(fetch(`/api/queues/${id}/images`)),
  nextImage: (id: number, reviewer: string, after?: string) =>
    j<ImageReview | Record<string, never>>(
      fetch(`/api/queues/${id}/next-image?reviewer=${encodeURIComponent(reviewer)}${after ? `&after=${after}` : ""}`),
    ),
  queueImage: (id: number, imageId: string) => j<ImageReview>(fetch(`/api/queues/${id}/image/${imageId}`)),
  reviewImage: (id: number, reviewer: string, items: { id: number; action: string; label?: Pred | null }[]) =>
    j<Item[]>(post(`/api/queues/${id}/review-image`, { reviewer, items })),
  queueItems: (id: number, status?: string) =>
    j<Item[]>(fetch(`/api/queues/${id}/items${status ? `?status=${status}` : ""}`)),
  next: (id: number, reviewer: string, after?: number) =>
    j<Item | Record<string, never>>(
      fetch(`/api/queues/${id}/next?reviewer=${encodeURIComponent(reviewer)}${after ? `&after=${after}` : ""}`),
    ),
  item: (id: number) => j<Item>(fetch(`/api/items/${id}`)),
  review: (id: number, action: string, reviewer: string, label?: Pred | null) =>
    j<Item>(post(`/api/items/${id}/review`, { action, reviewer, label })),
  reopen: (id: number) => j<Item>(post(`/api/items/${id}/reopen`, {})),
  runs: () => j<Run[]>(fetch("/api/runs")),
  reports: () => j<Report[]>(fetch("/api/heldout/reports")),
  heldoutDiff: (base: string, adapter: string, onlyChanged = true) =>
    j<{ manifest: string; n: number; items: HeldoutPair[] }>(
      fetch(
        `/api/heldout/diff?base=${encodeURIComponent(base)}&adapter=${encodeURIComponent(adapter)}&only_changed=${onlyChanged}`,
      ),
    ),
  export: (name: string, queue_ids?: number[], incoming = false) =>
    j<Record<string, unknown>>(post("/api/export", { name, queue_ids, incoming })),
};

export interface ImageSummary {
  image_id: string;
  n: number;
  pending: number;
  auto: number;
  reviewed: number;
  fix: number;
  spot: number;
  first_id: number;
}

export interface ImageReview {
  image: ImageRow;
  items: Item[];
}

export interface HeldoutPair {
  row: Record<string, unknown> & { skill: Skill; object?: string; question?: string; boxes?: number[][]; points?: number[][]; answer?: string };
  image_id: string;
  base: { pred: unknown; tp_fp_fn?: number[]; ok?: boolean };
  adapter: { pred: unknown; tp_fp_fn?: number[]; ok?: boolean };
}

export const reviewerStore = {
  get: () => localStorage.getItem("forge.reviewer") ?? "",
  set: (v: string) => localStorage.setItem("forge.reviewer", v),
};

export const skillLabel: Record<Skill, string> = {
  detect: "Detect",
  point: "Point",
  query: "Query",
  caption: "Caption",
};

export function promptLabel(skill: Skill): string {
  return skill === "query" ? "問題" : skill === "caption" ? "長度 (short / normal / long)" : "物件名稱";
}

export function defaultPrompt(skill: Skill): string {
  return skill === "query" ? "How many people are there?" : skill === "caption" ? "normal" : "person";
}

export function emptyLabel(skill: Skill): Pred {
  return skill === "detect" ? { boxes: [] } : skill === "point" ? { points: [] } : { text: "" };
}

export function summary(p: Pred | null | undefined): string {
  if (!p) return "—";
  if (p.boxes) return `${p.boxes.length} 框`;
  if (p.points) return `${p.points.length} 點`;
  return p.text ? `"${p.text.slice(0, 60)}${p.text.length > 60 ? "…" : ""}"` : "(空)";
}

export function judgeText(v: JudgeVerdict | null | undefined): string {
  if (!v) return "";
  const parts: string[] = [];
  if (v.wrong?.length) parts.push(`框/點 ${v.wrong.join(", ")} 不對`);
  if (v.missed) parts.push(`漏 ${v.missed} 個`);
  if (v.answer && v.verdict === "fix") parts.push(`judge 認為答案是「${v.answer}」`);
  if (v.unsupported?.length) parts.push(`圖裡沒有：${v.unsupported.join("、")}`);
  if (v.note) parts.push(v.note);
  return parts.join("；");
}

export const sourceLabel = (s: string) =>
  s === "upload" ? "上傳" : s === "heldout" ? "held-out（只能評估）" : s.startsWith("capture") ? `擷取 ${s.slice(8)}` : s;
