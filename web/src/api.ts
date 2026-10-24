// Thin client for the forge API (src/vlmforge/server/api.py). Every call maps to one
// endpoint; the UI spec (docs/ui-spec.md §10) lists which screen uses which.

export type Skill = "detect" | "point" | "query" | "caption";
export type Pred = { boxes?: number[][]; points?: number[][]; text?: string };
export type ItemStatus = "pending" | "auto" | "approved" | "corrected" | "discarded";

export interface ImageRow { id: string; path: string; width: number; height: number; source: string; added_at: string }

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
  status: ItemStatus;
  reviewer: string | null;
  reviewed_at: string | null;
  claimed_by: string | null;
  claimed_at: number | null;
  path: string;
  width: number;
  height: number;
  source: string;
}

export interface Job {
  state: "running" | "done" | "error" | "cancelled";
  phase?: string;
  done: number;
  total: number;
  error: string | null;
  started?: number;
  student_error?: string;
  triage?: Record<string, number>;
  resumed?: boolean;
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
  dataset_id?: number | null;
  job?: Job | null;
  unfinished?: { images: number; unpredicted: number; unjudged: number }; // autogen batches only
  images_pending?: number;
}

export interface DatasetConfig { mode?: "auto" | "both" | "fixed"; fixed?: string; topics?: string[]; focus?: string; skills?: Skill[]; probes?: string[] }
/** autogen.DEFAULT_PROBES: asked on every image in auto mode, absent = a 0-box detect question */
export const DEFAULT_PROBES = ["person", "forklift", "helmet"];

export interface Dataset {
  id: number;
  thumbs?: string[]; // first few image ids, for the list's thumbnail strip
  name: string;
  created_at: string;
  updated_at: string;
  config: DatasetConfig;
  note: string;
  images: number;
  val_images: number;
  items: number;
  pending: number;
  auto: number;
  reviewed: number;
  discarded: number;
  images_pending: number;
  detect: number;
  point: number;
  query: number;
  caption: number;
  trainable: number;
  trainable_images: number; // train-split images with at least one confirmed question
  trainable_by_skill?: Partial<Record<Skill, number>>;
  spot?: { total: number; pending: number; agree: number; changed: number }; // teacher-trust sample
  val_rows?: number; // confirmed questions in the val split
  batches?: Batch[];
  unlabelled?: number;
}

export interface GridCell { image_id: string; split: "train" | "val"; n: number; pending: number; fix: number; auto: number; kept: number }

export interface Summary {
  images: number; items: number; images_pending: number; images_fix: number;
  pending: number; auto: number; reviewed: number; discarded: number; trainable: number;
  batches: Batch[]; sources: { source: string; n: number }[];
}

export interface ImageSummary { image_id: string; n: number; pending: number; fix: number; spot: number; auto: number; queue_id: number; claimed_by: string | null }
export interface ImageReview { image: ImageRow; items: Item[] }

export interface GateRow { skill: string; base: number; adapter: number; delta: number; ok: boolean }
export interface Run {
  name: string;
  state: "running" | "done" | "error" | "cancelled";
  phase: string;
  started: string;
  finished?: string;
  steps: number; lr: number; rank: number; rows: number; images: number;
  skills: Record<string, number>;
  region_head?: boolean;
  datasets?: string[];
  evals: Record<string, Record<string, number>>;
  gate: { passed: boolean; skills: GateRow[] } | null;
  error: string | null;
  log_tail?: string[];
  curve: { step: number; loss?: number; val?: number }[];
  last: Record<string, number> | null;
  alive: boolean;
}

export interface EngineStatus {
  loaded: boolean; weights: string; weights_ok: boolean; judge: string; judge_ok: boolean; judge_loaded: boolean;
  judge_name?: string | null; judge_small_ok?: boolean; judge_need_gb?: number;
  current: string; pending: number; busy: boolean; gpu_mem_gb: number | null; gpu_free_gb?: number | null; last_error: string | null;
}
export interface Model { name: string; label: string }
export interface Status { version: string; engine: EngineStatus; models: Model[] }

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
    try { const b = await r.json(); if (b.detail) msg = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail); } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}
const post = (url: string, body: unknown) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const del = (url: string) => fetch(url, { method: "DELETE" });

export const api = {
  status: () => j<Status>(fetch("/api/status")),
  summary: () => j<Summary>(fetch("/api/review/summary")),
  // ① datasets
  datasets: () => j<Dataset[]>(fetch("/api/datasets")),
  dataset: (id: number) => j<Dataset>(fetch(`/api/datasets/${id}`)),
  createDataset: (name: string, config: DatasetConfig = {}, note = "") => j<Dataset>(post("/api/datasets", { name, config, note })),
  updateDataset: (id: number, body: { name?: string; config?: DatasetConfig }) => j<Dataset>(post(`/api/datasets/${id}`, body)),
  deleteDataset: (id: number) => j<{ ok: true }>(del(`/api/datasets/${id}`)),
  datasetGrid: (id: number) => j<GridCell[]>(fetch(`/api/datasets/${id}/grid`)),
  uploadToDataset: (id: number, file: File) => { const fd = new FormData(); fd.append("file", file); return j<{ added: number; image_ids: string[] }>(fetch(`/api/datasets/${id}/upload`, { method: "POST", body: fd })); },
  relabel: (id: number, image_ids: string[]) => j<Batch & { reset: { dropped: number; kept: number } }>(post(`/api/datasets/${id}/relabel`, { image_ids })),
  removeImages: (id: number, image_ids: string[]) => j<{ removed: number; items: number; files: number }>(post(`/api/datasets/${id}/images/remove`, { image_ids })),
  addFolder: (id: number, dir: string, limit?: number) => j<{ added: number }>(post(`/api/datasets/${id}/folder`, { dir, limit })),
  label: (id: number, n?: number) => j<Batch>(post(`/api/datasets/${id}/label${n ? `?n_images=${n}` : ""}`, {})),
  cancelBatch: (qid: number) => j<Job>(post(`/api/queues/${qid}/cancel`, {})),
  resumeBatch: (qid: number) => j<Batch>(post(`/api/queues/${qid}/resume`, {})),
  exportDataset: (id: number) => j<{ jsonl: string; rows: number; zip?: string; images?: number }>(post(`/api/datasets/${id}/export`, {})),
  downloadUrl: (id: number, kind: "jsonl" | "zip") => `/api/datasets/${id}/download/${kind}`,
  saveDatasetItem: (id: number, body: Record<string, unknown>) => j<Item>(post(`/api/datasets/${id}/items`, body)),
  // ② review
  nextImage: (reviewer: string, after?: string, datasetId?: number | null, spot?: boolean) =>
    j<ImageReview | Record<string, never>>(fetch(`/api/review/next-image?reviewer=${encodeURIComponent(reviewer)}${after ? `&after=${after}` : ""}${datasetId ? `&dataset_id=${datasetId}` : ""}${spot ? "&spot=true" : ""}`)),
  reviewImages: (status: string, datasetId?: number | null) => j<ImageSummary[]>(fetch(`/api/review/images?status=${status}${datasetId ? `&dataset_id=${datasetId}` : ""}`)),
  reviewImage: (imageId: string) => j<ImageReview>(fetch(`/api/review/image/${imageId}`)),
  review: (id: number, action: string, reviewer: string, label?: Pred | null) => j<Item>(post(`/api/items/${id}/review`, { action, reviewer, label })),
  submitImage: (reviewer: string, items: { id: number; action: string; label?: Pred | null }[]) => j<Item[]>(post("/api/review/image", { reviewer, items })),
  reopen: (id: number) => j<Item>(post(`/api/items/${id}/reopen`, {})),
  // ③ train
  train: (body: { name?: string; steps: number; lr: number; rank?: number; dataset_ids: number[]; region_head?: boolean }) => j<Run>(post("/api/train", body)),
  runs: () => j<{ runs: Run[]; trainable: number }>(fetch("/api/train")),
  cancelRun: (name: string) => j<{ ok: true }>(post(`/api/train/${name}/cancel`, {})),
  deleteRun: (name: string) => j<{ ok: true }>(del(`/api/train/${name}`)),
  orinUrl: (name: string) => `/api/train/${name}/orin.zip`,
  // ④ test
  testSummary: () => j<{ suites: string[]; base: Record<string, Record<string, number>>; runs: Run[] }>(fetch("/api/test/summary")),
  heldoutDiff: (base: string, adapter: string, onlyChanged = true) =>
    j<{ manifest: string; n: number; items: HeldoutPair[] }>(fetch(`/api/heldout/diff?base=${encodeURIComponent(base)}&adapter=${encodeURIComponent(adapter)}&only_changed=${onlyChanged}`)),
  images: (source?: string) => j<ImageRow[]>(fetch(`/api/images?limit=2000${source ? `&source=${encodeURIComponent(source)}` : ""}`)),
  sources: () => j<{ source: string; n: number }[]>(fetch("/api/sources")),
  upload: (file: File) => { const fd = new FormData(); fd.append("file", file); return j<ImageRow>(fetch("/api/images", { method: "POST", body: fd })); },
  imageUrl: (id: string, w?: number) => `/api/images/${id}/file${w ? `?w=${w}` : ""}`,
  compare: (image_id: string, skill: Skill, prompt: string, models: string[]) => j<Record<string, { pred: Pred; ms: number }>>(post("/api/compare", { image_id, skill, prompt, models })),
};

export const nameStore = {
  get: () => { try { return localStorage.getItem("forge.name") ?? ""; } catch { return ""; } },
  set: (v: string) => { try { localStorage.setItem("forge.name", v); } catch { /* ignore */ } },
};

export const skillLabel: Record<Skill, string> = { detect: "Detect", point: "Point", query: "Query", caption: "Caption" };
export const defaultPrompt = (skill: Skill) => (skill === "query" ? "How many people are there?" : skill === "caption" ? "normal" : "person");
export const emptyLabel = (skill: Skill): Pred => (skill === "detect" ? { boxes: [] } : skill === "point" ? { points: [] } : { text: "" });
export function summary(p: Pred | null | undefined): string {
  if (!p) return "—";
  if (p.boxes) return `${p.boxes.length} 框`;
  if (p.text !== undefined) { const t = answerText(p); return t ? `"${t.slice(0, 60)}${t.length > 60 ? "…" : ""}"` : "(空)"; }
  if (p.points) return `${p.points.length} 點`;
  return "(空)";
}
/** A text answer as shown: Moondream 3 sometimes answers by pointing ("There is " + a
 *  grounded point), so the points it grounded are appended as ⟨N 個點⟩. */
export function answerText(p: Pred | null | undefined): string {
  const t = (p?.text ?? "").trim();
  const n = p?.text !== undefined ? p?.points?.length ?? 0 : 0;
  return n ? `${t}⟨${n} 個點⟩` : t;
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
export const suiteLabel = (s: string) =>
  s === "heldout-cctv" ? "工廠 CCTV（真考卷）" : s === "heldout-crowdhuman-val" ? "CrowdHuman（參考）" : s === "heldout-people-posture-gold" ? "機器狗 gold（參考）" : s.startsWith("val-") ? `${s.slice(4)} 驗證集` : s;
export const phaseLabel = (p?: string) =>
  p === "generating" ? "出題中" : p === "predicting" ? "模型作答中" : p === "judging" ? "比對中" : p?.startsWith("eval:") ? `評估 ${suiteLabel(p.slice(5))}` : p === "training" ? "訓練中" : p === "cancelled" ? "已取消" : p ?? "";
export const itemTitle = (it: { skill: Skill; prompt: string }) => (it.skill === "query" ? it.prompt : it.skill === "caption" ? `寫一段描述（${it.prompt}）` : `${skillLabel[it.skill]}：${it.prompt}`);

/** Mirrors vlmforge.autogen.parse_fixed so the textarea can validate as you type. */
export function parseFixed(text: string): { ok: Record<Skill, string[]>; bad: number[] } {
  const ok: Record<Skill, string[]> = { detect: [], point: [], query: [], caption: [] };
  const bad: number[] = [];
  text.split("\n").forEach((raw, i) => {
    const line = raw.trim();
    if (!line) return;
    const m = line.match(/^(detect|point|query|caption)\s*:\s*(.*)$/i);
    if (!m) { bad.push(i + 1); return; }
    const skill = m[1].toLowerCase() as Skill, rest = m[2].trim();
    if (skill === "query") { if (rest) ok.query.push(rest); else bad.push(i + 1); }
    else if (skill === "caption") ok.caption.push(...(rest ? rest.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean) : ["normal"]));
    else { const xs = rest.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean); if (xs.length) ok[skill].push(...xs); else bad.push(i + 1); }
  });
  return { ok, bad };
}
export const fmtDelta = (d: number) => `${d >= 0 ? "+" : ""}${(100 * d).toFixed(1)}`;
export const fmt = (v?: number | null) => (v === undefined || v === null || Number.isNaN(v) ? "—" : v.toFixed(3));
