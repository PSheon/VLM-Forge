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

export interface JudgeVerdict {
  verdict: "accept" | "fix";
  note: string;
  model?: string;
  wrong?: number[];
  missed?: number;
  answer?: string;
  unsupported?: string[];
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

export interface Job {
  state: string;
  started?: number;
  phase?: string;
  done: number;
  total: number;
  error: string | null;
  student_error?: string;
  triage?: Record<string, number>;
}

export interface Batch {
  id: number;
  name: string;
  mode: "manual" | "auto";
  created_at: string;
  counts: Record<string, number>;
  verdicts: Record<string, number>;
  spot: number;
  total: number;
  job?: Job | null;
}

export interface Summary {
  images: number;
  items: number;
  images_pending: number;
  images_fix: number;
  pending: number;
  auto: number;
  reviewed: number;
  discarded: number;
  trainable: number;
  batches: Batch[];
  sources: { source: string; n: number }[];
}

export interface ImageSummary {
  image_id: string;
  n: number;
  pending: number;
  fix: number;
  spot: number;
  auto: number;
  queue_id: number;
}

export interface ImageReview {
  image: ImageRow;
  items: Item[];
}

export interface Run {
  name: string;
  state: "running" | "done" | "error";
  phase: string;
  started: string;
  finished?: string;
  steps: number;
  lr: number;
  rank: number;
  rows: number;
  skills: Record<string, number>;
  images: number;
  evals: Record<string, Record<string, number>>;
  gate: { passed: boolean; skills: { skill: string; base: number; adapter: number; delta: number; ok: boolean }[] } | null;
  error: string | null;
  curve: { step: number; loss?: number; val?: number }[];
  last: Record<string, number> | null;
  alive: boolean;
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

export interface Model {
  name: string;
  label: string;
}

export interface Status {
  version: string;
  engine: EngineStatus;
  models: Model[];
}

export interface HeldoutPair {
  row: Record<string, unknown> & { skill: Skill; object?: string; question?: string; boxes?: number[][]; points?: number[][]; answer?: string };
  image_id: string;
  base: { pred: unknown; tp_fp_fn?: number[]; ok?: boolean };
  adapter: { pred: unknown; tp_fp_fn?: number[]; ok?: boolean };
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
  // ① data
  scan: (dir: string, limit?: number) => j<{ source: string; added: number; found: number }>(post("/api/data/scan", { dir, limit })),
  autogen: (body: { source: string; n_images?: number; skills?: Skill[]; topics?: string[]; focus?: string }) => j<Batch>(post("/api/data/autogen", body)),
  summary: () => j<Summary>(fetch("/api/review/summary")),
  // ② review
  nextImage: (reviewer: string, after?: string) =>
    j<ImageReview | Record<string, never>>(fetch(`/api/review/next-image?reviewer=${encodeURIComponent(reviewer)}${after ? `&after=${after}` : ""}`)),
  reviewImages: (status: "pending" | "auto" | "reviewed") => j<ImageSummary[]>(fetch(`/api/review/images?status=${status}`)),
  reviewImage: (imageId: string) => j<ImageReview>(fetch(`/api/review/image/${imageId}`)),
  submitImage: (reviewer: string, items: { id: number; action: string; label?: Pred | null }[]) =>
    j<Item[]>(post("/api/review/image", { reviewer, items })),
  reopen: (id: number) => j<Item>(post(`/api/items/${id}/reopen`, {})),
  // ③ train
  train: (body: { name?: string; steps: number; lr: number; rank?: number }) => j<Run>(post("/api/train", body)),
  runs: () => j<{ runs: Run[]; trainable: number }>(fetch("/api/train")),
  // ④ test
  testSummary: () => j<{ suites: string[]; base: Record<string, Record<string, number>>; runs: Run[] }>(fetch("/api/test/summary")),
  heldoutDiff: (base: string, adapter: string, onlyChanged = true) =>
    j<{ manifest: string; n: number; items: HeldoutPair[] }>(
      fetch(`/api/heldout/diff?base=${encodeURIComponent(base)}&adapter=${encodeURIComponent(adapter)}&only_changed=${onlyChanged}`),
    ),
  images: (source?: string) => j<ImageRow[]>(fetch(`/api/images?limit=2000${source ? `&source=${encodeURIComponent(source)}` : ""}`)),
  sources: () => j<{ source: string; n: number }[]>(fetch("/api/sources")),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return j<ImageRow>(fetch("/api/images", { method: "POST", body: fd }));
  },
  imageUrl: (id: string, w?: number) => `/api/images/${id}/file${w ? `?w=${w}` : ""}`,
  infer: (image_id: string, skill: Skill, prompt: string, model: string) =>
    j<{ pred: Pred; model: string; ms: number }>(post("/api/infer", { image_id, skill, prompt, model })),
  compare: (image_id: string, skill: Skill, prompt: string, models: string[]) =>
    j<Record<string, { pred: Pred; ms: number }>>(post("/api/compare", { image_id, skill, prompt, models })),
};

export const reviewerStore = {
  get: () => localStorage.getItem("forge.reviewer") ?? "Paul",
  set: (v: string) => localStorage.setItem("forge.reviewer", v),
};

export const skillLabel: Record<Skill, string> = { detect: "Detect", point: "Point", query: "Query", caption: "Caption" };

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
  
  if (v.unsupported?.length) parts.push(`圖裡沒有：${v.unsupported.join("、")}`);
  if (v.note) parts.push(v.note);
  return parts.join("；");
}

export const suiteLabel: Record<string, string> = {
  "heldout-cctv": "工廠 CCTV",
  "heldout-crowdhuman-val": "CrowdHuman",
  "heldout-people-posture-gold": "機器狗 gold",
};

export const phaseLabel = (p?: string) =>
  p === "generating" ? "出題中" : p === "predicting" ? "模型作答中" : p === "judging" ? "比對中" : p?.startsWith("eval:") ? `評估 ${suiteLabel[p.slice(5)] ?? p.slice(5)}` : p === "training" ? "訓練中" : p ?? "";
