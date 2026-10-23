// Runs / adapters (GOALS §6.4): every runs/<name>, its loss curve from log.jsonl, held-out
// macro scores from eval reports, and which adapter the GPU worker currently holds.

import { useEffect, useState } from "react";
import { api, type Report, type Run } from "../api";

export default function Runs() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [engine, setEngine] = useState<Record<string, unknown>>({});
  const [err, setErr] = useState("");
  const load = () => Promise.all([api.runs(), api.reports(), api.status()]).then(([r, p, s]) => { setRuns(r); setReports(p); setEngine(s.engine); }).catch((e) => setErr(String(e)));
  useEffect(() => { load(); const t = setInterval(load, 10000); return () => clearInterval(t); }, []);
  return (
    <div className="col">
      <div className="card row">
        <strong>GPU worker</strong>
        <span>模型 {engine.loaded ? "已載入" : "未載入"} · 目前 {String(engine.current ?? "—")} · 佇列 {String(engine.pending ?? 0)} · {String(engine.gpu_mem_gb ?? "?")} GB</span>
        {engine.last_error ? <span className="bad">{String(engine.last_error)}</span> : null}
        {err && <span className="bad">{err}</span>}
      </div>
      <div className="card">
        <strong>Held-out 報告</strong>
        <table><thead><tr><th>模型</th><th>suite</th><th>detect</th><th>point</th><th>query</th><th>caption</th></tr></thead>
          <tbody>{reports.map((r) => <tr key={r.path}><td>{r.model}</td><td>{r.manifest?.split("/").pop()}</td>{["detect", "point", "query", "caption"].map((k) => <td key={k}>{fmt(r.macro[`${k}/macro`])}</td>)}</tr>)}</tbody></table>
      </div>
      {runs.map((r) => (
        <div className="card" key={r.name}>
          <div className="row"><strong>{r.name}</strong>{r.running && <span className="ok">執行中</span>}<span className="muted">{meta(r)}</span></div>
          <div className="grid2">
            <Curve points={r.curve} />
            <div>
              <div className="muted">adapter：{r.adapters.length ? r.adapters.map((a) => a.split("/").pop()).join(", ") : "尚未存檔"}</div>
              {r.last && <div className="muted">最後：step {r.last.step}{r.last.loss !== undefined ? ` · loss ${r.last.loss.toFixed(4)}` : ""}{r.last.mem_gb !== undefined ? ` · ${r.last.mem_gb.toFixed(1)} GB` : ""}{r.last.samples_per_s !== undefined ? ` · ${r.last.samples_per_s.toFixed(1)} samples/s` : ""}</div>}
              {Object.keys(r.evals).length > 0 && (
                <table><tbody>{Object.entries(r.evals).map(([f, m]) => <tr key={f}><td>{f.replace(/^eval-/, "").replace(/\.json$/, "")}</td><td>{Object.entries(m).map(([k, v]) => `${k.split("/")[0]} ${fmt(v)}`).join(" · ")}</td></tr>)}</tbody></table>
              )}
            </div>
          </div>
        </div>
      ))}
      {!runs.length && <div className="card muted">還沒有訓練 run（runs/ 是空的）。</div>}
    </div>
  );
}

function fmt(v: number | undefined) { return v === undefined || Number.isNaN(v) ? "—" : v.toFixed(3); }
function meta(r: Run) {
  const m = r.meta ?? {};
  return ["rank", "alpha", "steps", "batch", "lr", "data_release"].filter((k) => m[k] !== undefined && m[k] !== null).map((k) => `${k} ${String(m[k])}`).join(" · ");
}

function Curve({ points }: { points: Run["curve"] }) {
  const tr = points.filter((p) => p.loss !== undefined), va = points.filter((p) => p.val !== undefined);
  if (!tr.length && !va.length) return <div className="muted">沒有 log</div>;
  const W = 420, H = 140, pad = 28;
  const xs = points.map((p) => p.step), maxX = Math.max(1, ...xs);
  const ys = [...tr.map((p) => p.loss!), ...va.map((p) => p.val!)];
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const X = (s: number) => pad + ((W - pad - 4) * s) / maxX;
  const Y = (v: number) => H - pad + ((pad - H + 6) * (v - minY)) / Math.max(1e-9, maxY - minY);
  const path = (ps: { step: number; v: number }[]) => ps.map((p, i) => `${i ? "L" : "M"}${X(p.step).toFixed(1)},${Y(p.v).toFixed(1)}`).join(" ");
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      <line x1={pad} y1={H - pad} x2={W} y2={H - pad} stroke="var(--line)" />
      <line x1={pad} y1={0} x2={pad} y2={H - pad} stroke="var(--line)" />
      <path d={path(tr.map((p) => ({ step: p.step, v: p.loss! })))} fill="none" stroke="var(--acc)" strokeWidth={1.5} />
      <path d={path(va.map((p) => ({ step: p.step, v: p.val! })))} fill="none" stroke="var(--box)" strokeWidth={1.5} />
      <text x={pad + 4} y={12} fontSize={11} fill="var(--muted)">loss {fmt(minY)}–{fmt(maxY)} · 藍 train · 黃 val</text>
      <text x={W - 40} y={H - 8} fontSize={11} fill="var(--muted)">{maxX}</text>
    </svg>
  );
}
