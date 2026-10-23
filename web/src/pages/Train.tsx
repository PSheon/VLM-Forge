// ③ 訓練: one button. Three knobs (name, steps, learning rate); everything else is the
// default. The run trains, evaluates on the held-out suites and gates itself.

import { useEffect, useState } from "react";
import { api, phaseLabel, suiteLabel, type Run, type Summary } from "../api";

export default function Train({ sum, go }: { sum: Summary | null; go: () => void }) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [name, setName] = useState("");
  const [steps, setSteps] = useState(500);
  const [lr, setLr] = useState("2e-4");
  const [err, setErr] = useState("");
  const load = () => api.runs().then((r) => setRuns(r.runs)).catch((e) => setErr(String(e)));
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, []);
  const running = runs.find((r) => r.state === "running");
  const start = async () => {
    setErr("");
    try { await api.train({ name: name || undefined, steps, lr: parseFloat(lr) }); setName(""); load(); } catch (e) { setErr(String(e)); }
  };
  const trainable = sum?.trainable ?? 0;
  return (
    <div className="col" style={{ maxWidth: 900 }}>
      <div className="card col">
        <div className="row">
          <strong>訓練集</strong>
          <span>{trainable} 筆{sum ? `（你確認 ${sum.reviewed}、自動通過 ${sum.auto}）` : ""}</span>
          {sum && sum.images_pending > 0 && <span className="muted hint">· 還有 {sum.images_pending} 張沒看，可以先訓練，沒看的不算進去</span>}
        </div>
        <div className="row">
          <label className="f">名稱（空＝自動）<input value={name} onChange={(e) => setName(e.target.value)} placeholder="ft-1025-1430" /></label>
          <label className="f">步數<input type="number" min={20} value={steps} onChange={(e) => setSteps(+e.target.value)} style={{ width: 100 }} /></label>
          <label className="f">學習率<input value={lr} onChange={(e) => setLr(e.target.value)} style={{ width: 100 }} /></label>
          <button className="primary" onClick={start} disabled={!!running || trainable === 0} style={{ alignSelf: "flex-end" }}>{running ? "訓練中…" : "開始訓練"}</button>
        </div>
        <div className="muted hint">LoRA r=16，region head 可訓，batch 8，每步約 1–2 秒；500 步約 15 分鐘，之後自動在 held-out 上評估。</div>
        {err && <div className="bad hint">{err}</div>}
      </div>
      {runs.map((r) => <RunCard key={r.name} r={r} go={go} />)}
      {runs.length === 0 && <div className="empty">還沒訓練過。</div>}
    </div>
  );
}

function RunCard({ r, go }: { r: Run; go: () => void }) {
  const total = r.steps, cur = r.last?.step ?? 0;
  return (
    <div className="card col">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span><strong>{r.name}</strong> <span className="muted hint">{r.rows} 筆 · {r.images} 張圖 · {r.steps} 步 · lr {r.lr}</span></span>
        <span className={r.state === "error" ? "bad" : r.state === "done" ? "ok" : ""}>{r.state === "running" ? `${phaseLabel(r.phase)}${r.phase === "training" ? ` ${cur}/${total}` : ""}` : r.state === "done" ? "完成" : "失敗"}</span>
      </div>
      {r.state === "running" && r.phase === "training" && <div className="bar"><i style={{ width: `${(100 * cur) / Math.max(1, total)}%` }} /></div>}
      {r.error && <div className="bad hint">{r.error}</div>}
      <div className="grid2">
        <Curve points={r.curve} />
        <div className="col hint">
          {r.last && <div className="muted">loss {r.last.loss?.toFixed(3)}{r.last["val/loss"] !== undefined ? ` · val ${r.last["val/loss"].toFixed(3)}` : ""}{r.last.mem_gb ? ` · ${r.last.mem_gb.toFixed(0)} GB` : ""}{r.last.samples_per_s ? ` · ${r.last.samples_per_s.toFixed(1)} 張/s` : ""}</div>}
          {Object.entries(r.evals).map(([s, m]) => <div key={s}>{suiteLabel[s] ?? s}：{Object.entries(m).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(" · ")}</div>)}
          {r.gate && <div className={r.gate.passed ? "ok" : "bad"}>{r.gate.passed ? "比 base 好（過關）" : "沒有比 base 好"}：{r.gate.skills.map((s) => `${s.skill} ${s.delta >= 0 ? "+" : ""}${(100 * s.delta).toFixed(1)}pt`).join(" · ")}</div>}
          {r.state === "done" && <button className="primary" onClick={go} style={{ alignSelf: "flex-start" }}>去測試看差異 →</button>}
        </div>
      </div>
    </div>
  );
}

export function Curve({ points }: { points: Run["curve"] }) {
  const tr = points.filter((p) => p.loss !== undefined), va = points.filter((p) => p.val !== undefined);
  if (!tr.length && !va.length) return <div className="muted hint">還沒有 log</div>;
  const W = 420, H = 140, pad = 28;
  const maxX = Math.max(1, ...points.map((p) => p.step));
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
      <text x={pad + 4} y={12} fontSize={11} fill="var(--muted)">loss {minY.toFixed(3)}–{maxY.toFixed(3)} · 藍 train · 黃 val</text>
    </svg>
  );
}
