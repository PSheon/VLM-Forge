// ③ 訓練 (docs/ui-spec §5): pick datasets, three knobs, one button; the run trains,
// evaluates on the held-out suites and each dataset's val split, and gates itself.

import { useCallback, useEffect, useState } from "react";
import { api, fmt, fmtDelta, phaseLabel, suiteLabel, type Dataset, type Run, type Status } from "../api";
import { Bar, ConfirmDialog, Pill, useToast } from "../components/ui";
import type { Step } from "../App";

export default function Train({ status, go, reviewer }: { status: Status | null; go: (s: Step) => void; reviewer: string }) {
  const { say } = useToast();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [picked, setPicked] = useState<number[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [name, setName] = useState("");
  const [steps, setSteps] = useState<number | "">("");
  const [lr, setLr] = useState("2e-4");
  const [adv, setAdv] = useState(false);
  const [rank, setRank] = useState(16);
  const [regionHead, setRegionHead] = useState(false);
  const [askDel, setAskDel] = useState<string | null>(null);
  const load = useCallback(() => { api.datasets().then((d) => { setDatasets(d.filter((x) => x.trainable > 0)); setPicked((p) => (p.length ? p : d.filter((x) => x.trainable > 0).map((x) => x.id))); }).catch(() => undefined); api.runs().then((r) => setRuns(r.runs)).catch(() => undefined); }, []);
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, [load]);
  const chosen = datasets.filter((d) => picked.includes(d.id));
  const rows = chosen.reduce((a, d) => a + d.trainable, 0);
  const images = chosen.reduce((a, d) => a + d.images - d.val_images, 0);
  const bySkill = { detect: chosen.reduce((a, d) => a + d.detect, 0), point: chosen.reduce((a, d) => a + d.point, 0), query: chosen.reduce((a, d) => a + d.query, 0), caption: chosen.reduce((a, d) => a + d.caption, 0) };
  const autoSteps = Math.max(100, Math.min(1000, images * 2));
  const running = runs.find((r) => r.state === "running");
  const e = status?.engine;
  const nameFinal = name.trim() || autoName();
  const blockers: string[] = [];
  if (rows === 0) blockers.push("沒有可訓練的題：先到 ② 確認一些");
  if (runs.some((r) => r.name === nameFinal)) blockers.push(`名稱「${nameFinal}」已經有了`);
  if (e && !e.weights_ok) blockers.push("模型權重缺");
  if (running) blockers.push(`正在訓練 ${running.name}，等它完成或取消`);
  const warns: string[] = [];
  for (const d of chosen) if (d.images_pending > 0) warns.push(`${d.name} 還有 ${d.images_pending} 張沒看：沒看的不算進去`);
  for (const [k, v] of Object.entries(bySkill)) if (v > 0 && v < 20) warns.push(`${k} 只有 ${v} 筆，效果有限`);
  if (regionHead) warns.push("開了 region head，這個 adapter Orin 的 Photon 載不了");

  const start = async () => {
    try { await api.train({ name: nameFinal, steps: steps === "" ? autoSteps : Number(steps), lr: parseFloat(lr), rank, dataset_ids: picked, region_head: regionHead }); say(`開始訓練 ${nameFinal}`); setName(""); load(); } catch (err) { say(String(err), "bad", { sticky: true }); }
  };
  return (
    <div className="col" style={{ maxWidth: 1100 }}>
      <div className="grid-2">
        <div className="panel col">
          <h3>訓練集</h3>
          {datasets.length === 0 && <div className="muted hint">還沒有確認過的題。</div>}
          {datasets.map((d) => (
            <label key={d.id} className="row"><input type="checkbox" checked={picked.includes(d.id)} onChange={(ev) => setPicked(ev.target.checked ? [...picked, d.id] : picked.filter((x) => x !== d.id))} /> <b>{d.name}</b><span className="muted small">{d.trainable} 筆 · 驗證集 {d.val_images} 張{d.images_pending ? ` · 還有 ${d.images_pending} 張沒看` : ""}</span></label>
          ))}
          <div className="small muted">合計 {rows} 筆 · {images} 張圖 · detect {bySkill.detect} · point {bySkill.point} · query {bySkill.query} · caption {bySkill.caption}</div>
          <div className="row">
            <label className="f">名稱<input value={name} placeholder={autoName()} onChange={(ev) => setName(ev.target.value)} onKeyDown={(ev) => ev.key === "Enter" && !blockers.length && start()} /></label>
            <label className="f">步數<input type="number" min={20} value={steps} placeholder={String(autoSteps)} onChange={(ev) => setSteps(ev.target.value === "" ? "" : +ev.target.value)} style={{ width: 110 }} /></label>
            <label className="f">學習率<input value={lr} onChange={(ev) => setLr(ev.target.value)} style={{ width: 100 }} /></label>
          </div>
          <button className="ghost" onClick={() => setAdv(!adv)} style={{ alignSelf: "flex-start" }}>{adv ? "收起進階" : "進階：rank、region head"}</button>
          {adv && <div className="row card"><label className="f">LoRA rank<select value={rank} onChange={(ev) => setRank(+ev.target.value)}>{[8, 16, 24, 32].map((r) => <option key={r} value={r}>{r}</option>)}</select></label><label className="row hint"><input type="checkbox" checked={regionHead} onChange={(ev) => setRegionHead(ev.target.checked)} /> 也訓 region head（座標頭；開了 Orin 載不了）</label></div>}
          <div className="note">LoRA 只掛文字 MLP（Orin 的 Photon 可直接載入）。預設步數＝圖數×2（100–1000）；{autoSteps} 步約 {Math.round(autoSteps / 36)} 分鐘，訓完自動在 CCTV held-out 與各資料集的驗證集評估。</div>
          {blockers.map((b) => <div key={b} className="bad hint">✗ {b}</div>)}
          {warns.map((w) => <div key={w} className="warn hint">⚠ {w}</div>)}
        </div>
        <div className="col">
          <button className="primary big" onClick={start} disabled={blockers.length > 0}>{running ? `訓練中 ${running.last?.step ?? 0}/${running.steps}` : "開始訓練"}</button>
          <div className="small muted">由 {reviewer || "—"} 啟動</div>
          {runs.map((r) => <RunRow key={r.name} r={r} onDelete={() => setAskDel(r.name)} onCancel={async () => { await api.cancelRun(r.name).catch((err) => say(String(err), "bad")); say("取消中…", "warn"); }} />)}
          {runs.length === 0 && <div className="empty">還沒訓練過。</div>}
        </div>
      </div>
      {runs.filter((r) => r.state !== "cancelled").slice(0, 3).map((r) => <RunCard key={r.name} r={r} go={go} />)}
      <ConfirmDialog open={!!askDel} title={`刪除 ${askDel}？`} body="adapter、log、評估報告都會刪。" confirmText="刪除" danger onCancel={() => setAskDel(null)} onConfirm={async () => { await api.deleteRun(askDel!).catch((err) => say(String(err), "bad")); setAskDel(null); load(); }} />
    </div>
  );
}

function autoName() { const d = new Date(); const p = (n: number) => String(n).padStart(2, "0"); return `ft-${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`; }

function RunRow({ r, onDelete, onCancel }: { r: Run; onDelete: () => void; onCancel: () => void }) {
  const pill = r.state === "running" ? <Pill kind="warn">{phaseLabel(r.phase)}{r.phase === "training" ? ` ${r.last?.step ?? 0}/${r.steps}` : ""}</Pill> : r.state === "done" ? (r.gate ? <Pill kind={r.gate.passed ? "ok" : "grey"}>{r.gate.passed ? "完成 · 比 base 好" : "完成 · 沒有比 base 好"}</Pill> : <Pill kind="ok">完成</Pill>) : r.state === "cancelled" ? <Pill kind="grey">取消</Pill> : <Pill kind="bad">失敗</Pill>;
  return (
    <div className="panel row" style={{ justifyContent: "space-between" }}>
      <span><b>{r.name}</b> <span className="muted small">{r.rows} 筆 · {r.steps} 步 · lr {r.lr}{r.datasets?.length ? ` · ${r.datasets.join("＋")}` : ""}</span></span>
      <span className="row">{pill}{r.state === "running" ? <button onClick={onCancel}>取消</button> : <button className="ghost" onClick={onDelete} title="刪除 run">✕</button>}</span>
    </div>
  );
}

export function RunCard({ r, go }: { r: Run; go: (s: Step) => void }) {
  const cur = r.last?.step ?? 0, total = r.steps;
  const left = r.state === "running" && r.phase === "training" && r.last?.samples_per_s ? Math.max(1, Math.round(((total - cur) * 8) / r.last.samples_per_s / 60)) : null;
  return (
    <div className="panel col">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span><b>{r.name}</b> <span className="muted small">{r.rows} 筆 · {r.images} 張 · {r.steps} 步 · lr {r.lr} · r{r.rank}{r.region_head ? " · region head" : ""}</span></span>
        <span className={r.state === "error" ? "bad" : r.state === "done" ? "ok" : ""}>{r.state === "running" ? `${phaseLabel(r.phase)}${r.phase === "training" ? ` ${cur}/${total}${left ? ` · 約還要 ${left} 分` : ""}` : ""}` : r.state === "done" ? "完成" : r.state === "cancelled" ? "已取消" : "失敗"}{r.last?.mem_gb ? ` · ${r.last.mem_gb.toFixed(0)} GB` : ""}</span>
      </div>
      {r.state === "running" && r.phase === "training" && <Bar done={cur} total={total} />}
      {r.error && <div className="bad hint">{r.error}</div>}
      {r.log_tail && r.log_tail.length > 0 && <pre className="code" style={{ maxHeight: 160 }}>{r.log_tail.join("\n")}</pre>}
      <div className="grid-2">
        <Curve r={r} />
        <div className="col hint">
          {r.last && <div className="muted">loss {r.last.loss?.toFixed(3)}{r.last["val/loss"] !== undefined ? ` · val ${r.last["val/loss"].toFixed(3)}` : ""}{r.last.samples_per_s ? ` · ${r.last.samples_per_s.toFixed(1)} 張/s` : ""}</div>}
          {r.gate && (
            <div className="check">
              <span className="h">指標</span><span className="h num">base</span><span className="h num">新</span><span className="h num">Δ</span><span className="h"></span>
              {r.gate.skills.map((s) => [<span key={s.skill + "a"}>{s.skill}</span>, <span key={s.skill + "b"} className="num">{fmt(s.base)}</span>, <span key={s.skill + "c"} className="num">{fmt(s.adapter)}</span>, <span key={s.skill + "d"} className={`num ${s.delta >= 0 ? "ok" : "bad"}`}>{fmtDelta(s.delta)}</span>, <Pill key={s.skill + "e"} kind={s.ok ? "ok" : "bad"}>{s.ok ? "PASS" : "FAIL"}</Pill>])}
            </div>
          )}
          {Object.entries(r.evals).filter(([k]) => k.startsWith("val-")).map(([k, m]) => <div key={k} className="muted">{suiteLabel(k)}：{Object.entries(m).map(([s, v]) => `${s} ${fmt(v)}`).join(" · ")}</div>)}
          {r.gate && <div className={r.gate.passed ? "ok" : "bad"}>{r.gate.passed ? "比 base 好（過關）" : "沒有比 base 好"}。規則：目標 skill ≥ base，其他 ≥ base − 1.0 點</div>}
          {r.state === "done" && <button className="primary" onClick={() => go("test")} style={{ alignSelf: "flex-start" }}>去測試看差異 →</button>}
        </div>
      </div>
    </div>
  );
}

function Curve({ r }: { r: Run }) {
  const tr = r.curve.filter((p) => p.loss !== undefined), va = r.curve.filter((p) => p.val !== undefined);
  if (!tr.length && !va.length) return <div className="muted hint">還沒有 log</div>;
  const W = 460, H = 150, pad = 30;
  const maxX = Math.max(1, r.steps, ...r.curve.map((p) => p.step));
  const ys = [...tr.map((p) => p.loss!), ...va.map((p) => p.val!)];
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const X = (s: number) => pad + ((W - pad - 6) * s) / maxX;
  const Y = (v: number) => H - pad + ((pad - H + 8) * (v - minY)) / Math.max(1e-9, maxY - minY);
  const path = (ps: { step: number; v: number }[]) => ps.map((p, i) => `${i ? "L" : "M"}${X(p.step).toFixed(1)},${Y(p.v).toFixed(1)}`).join(" ");
  const held = r.evals["heldout-cctv"]?.detect;
  return (
    <svg className="curve" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="loss 曲線">
      <line x1={pad} y1={H - pad} x2={W} y2={H - pad} stroke="var(--line)" /><line x1={pad} y1={4} x2={pad} y2={H - pad} stroke="var(--line)" />
      <text x={4} y={12} fontSize={10} fill="var(--muted)">{maxY.toFixed(2)}</text><text x={4} y={H - pad} fontSize={10} fill="var(--muted)">{minY.toFixed(2)}</text>
      <text x={W - 36} y={H - 10} fontSize={10} fill="var(--muted)">{maxX}</text>
      <path d={path(tr.map((p) => ({ step: p.step, v: p.loss! })))} fill="none" stroke="var(--acc)" strokeWidth={1.6} />
      <path d={path(va.map((p) => ({ step: p.step, v: p.val! })))} fill="none" stroke="var(--box2)" strokeWidth={1.6} strokeDasharray="4 3" />
      {va.map((p) => <circle key={p.step} cx={X(p.step)} cy={Y(p.val!)} r={3} fill="var(--box2)" />)}
      {held !== undefined && <text x={W - 150} y={16} fontSize={10} fill="var(--ok)">● held-out detect {held.toFixed(3)}</text>}
      <text x={pad + 6} y={12} fontSize={10} fill="var(--muted)">— train　- - val（每次評估）</text>
    </svg>
  );
}
