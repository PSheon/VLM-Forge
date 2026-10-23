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
  status: "pending" | "approved" | "corrected" | "discarded";
  reviewer: string | null;
  reviewed_at: string | null;
  path: string;
  width: number;
  height: number;
  source: string;
}

export interface Queue {
  id: number;
  name: string;
  skill: Skill;
  model: string;
  note: string;
  created_at: string;
  counts: Record<string, number>;
  total: number;
  job?: { state: string; done: number; total: number; error: string | null } | null;
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

async function j<T>(res: Promise<Response>): Promise<T> {
  const r = await res;
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      const b = await r.json();
      msg = b.detail ? JSON.stringify(b.detail) : msg;
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
  status: () => j<{ version: string; engine: Record<string, unknown>; models: Model[] }>(fetch("/api/status")),
  models: () => j<Model[]>(fetch("/api/models")),
  images: (source?: string) => j<ImageRow[]>(fetch(`/api/images${source ? `?source=${encodeURIComponent(source)}` : ""}`)),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return j<ImageRow>(fetch("/api/images", { method: "POST", body: fd }));
  },
  scan: (dir: string, source: string) => j<{ added: number }>(post("/api/images/scan", { dir, source })),
  imageUrl: (id: string) => `/api/images/${id}/file`,
  infer: (image_id: string, skill: Skill, prompt: string, model: string) =>
    j<{ pred: Pred; model: string; ms: number }>(post("/api/infer", { image_id, skill, prompt, model })),
  compare: (image_id: string, skill: Skill, prompt: string, models: string[]) =>
    j<Record<string, { pred: Pred; ms: number }>>(post("/api/compare", { image_id, skill, prompt, models })),
  saveItem: (body: Record<string, unknown>) => j<Item>(post("/api/items", body)),
  queues: () => j<Queue[]>(fetch("/api/queues")),
  queue: (id: number) => j<Queue>(fetch(`/api/queues/${id}`)),
  createQueue: (body: Record<string, unknown>) => j<Queue>(post("/api/queues", body)),
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
  detect: "Detect 框",
  point: "Point 點",
  query: "Query 問答",
  caption: "Caption 描述",
};

export function promptLabel(skill: Skill): string {
  return skill === "query" ? "問題" : skill === "caption" ? "長度 (short / normal / long)" : "物件名稱";
}
