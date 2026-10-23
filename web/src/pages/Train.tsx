// ② 訓練 (docs/ui-spec §5): ① pick datasets → ② settings → run; history underneath. The action
// column holds the button and a summary of exactly what this run will train on.

import { useCallback, useEffect, useState } from "react";
import { api, fmt, fmtDelta, phaseLabel, suiteLabel, type Dataset, type Run, type Status } from "../api";
import { ActionsHead, Bar, ConfirmDialog, PageHead, Pill, useToast } from "../components/ui";
import { Fade, Item, Pop, Stagger } from "../components/motion";
import type { Step } from "../App";

const SKILL_KEYS = ["detect", "point", "query", "caption"] as const;
type SkillKey = (typeof SKILL_KEYS)[number];

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
  const [openRun, setOpenRun] = useState<string | null>(null);
  // every dataset is listed; only those with confirmed questions can be picked (all of them
  // are, by default), so a new or still-labelling dataset is visible with the reason it is out
  const load = useCallback(() => { api.datasets().then((d) => { setDatasets(d); setPicked((p) => (p.length ? p.filter((id) => d.some((x) => x.id === id && x.trainable > 0)) : d.filter((x) => x.trainable > 0).map((x) => x.id))); }).catch(() => undefined); api.runs().then((r) => setRuns(r.runs)).catch(() => undefined); }, []);
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, [load]);
  const chosen = datasets.filter((d) => picked.includes(d.id));
  const rows = chosen.reduce((a, d) => a + d.trainable, 0);
  const images = chosen.reduce((a, d) => a + (d.trainable_images ?? 0), 0);
  const val = chosen.reduce((a, d) => a + d.val_images, 0);
  const sk = (d: Dataset | Dataset[], k: SkillKey) => ([] as Dataset[]).concat(d).reduce((a, x) => a + (x.trainable_by_skill?.[k] ?? 0), 0);
  const bySkill = Object.fromEntries(SKILL_KEYS.map((k) => [k, sk(chosen, k)])) as Record<SkillKey, number>; // confirmed, train split only
  const autoSteps = Math.max(100, Math.min(1000, images * 2));
  const nSteps = steps === "" ? autoSteps : Number(steps);
  const running = runs.find((r) => r.state === "running");
  const e = status?.engine;
  const nameFinal = name.trim() || autoName();
  const blockers: string[] = [];
  if (rows === 0) blockers.push("還沒選資料集，或選的資料集沒有確認過的題");
  if (runs.some((r) => r.name === nameFinal)) blockers.push(`名稱「${nameFinal}」已經有了`);
  if (e && !e.weights_ok) blockers.push("模型權重缺");
  if (running) blockers.push(`正在訓練 ${running.name}，等它完成或取消`);
  const warns: string[] = [];
  for (const d of chosen) if (d.images_pending > 0) warns.push(`${d.name} 還有 ${d.images_pending} 張沒看：沒看的不算進去`);
  if (rows > 0 && images < 10) warns.push(`只有 ${images} 張圖有確認的題：模型會把這幾張背起來，多看幾張再訓比較值`);
  for (const k of SKILL_KEYS) if (bySkill[k] > 0 && bySkill[k] < 20) warns.push(`${k} 只有 ${bySkill[k]} 筆，效果有限`);
  if (regionHead) warns.push("開了 region head，這個 adapter Orin 的 Photon 載不了");
  const toggle = (d: Dataset) => { if (d.trainable === 0) return; setPicked((p) => (p.includes(d.id) ? p.filter((x) => x !== d.id) : [...p, d.id])); };
  const shown = running ?? runs.find((r) => r.name === openRun) ?? runs.find((r) => r.state !== "cancelled");

  const start = async () => {
    try { await api.train({ name: nameFinal, steps: nSteps, lr: parseFloat(lr), rank, dataset_ids: picked, region_head: regionHead }); say(`開始訓練 ${nameFinal}`); setName(""); setOpenRun(nameFinal); load(); } catch (err) { say(String(err), "bad", { sticky: true }); }
  };
  return (
    <div className="page">
      <div className="col">
        <PageHead title="訓練">{running ? <Pill kind="warn">{running.name} {running.phase === "training" ? `訓練中 ${running.last?.step ?? 0}/${running.steps}` : phaseLabel(running.phase)}</Pill> : <span className="muted hint">選資料集 → 確認設定 → 開始；訓完自動評分</span>}</PageHead>

        <div className="panel col">
          <div className="step-h"><span className="n">1</span><h3>選資料集</h3><span className="muted hint">點卡片勾選；只有確認過的題會進訓練，每個資料集留 10% 驗證集評分</span></div>
          {datasets.length === 0 && <div className="empty">還沒有資料集。<button className="ghost" onClick={() => go("datasets")}>去建立 →</button></div>}
          <Stagger className="ds-pick">
            {datasets.map((d) => {
              const on = picked.includes(d.id), off = d.trainable === 0;
              const why = d.batches?.[0]?.job?.state === "running" ? "標註中，還沒有確認的題" : d.items ? `還沒有確認的題（${d.images_pending} 張要看）` : "還沒標註";
              return (
                <Item key={d.id} className={`card${on ? " on" : ""}${off ? " off" : ""}`} onClick={() => toggle(d)} title={off ? why : on ? "點一下取消" : "點一下加入"}>
                  <div className="row" style={{ justifyContent: "space-between" }}><span className="row"><input type="checkbox" checked={on} disabled={off} readOnly /><span className="name">{d.name}</span></span>{d.images_pending > 0 && !off && <Pill kind="warn">{d.images_pending} 張沒看</Pill>}</div>
                  {off ? <div className="small muted">{why}</div> : (
                    <>
                      <div className="stat"><span>筆 <b>{d.trainable}</b></span><span>圖 <b>{d.trainable_images ?? 0}</b></span><span>驗證 <b>{d.val_images}</b></span></div>
                      <SkillMix counts={Object.fromEntries(SKILL_KEYS.map((k) => [k, sk(d, k)])) as Record<SkillKey, number>} />
                    </>
                  )}
                </Item>
              );
            })}
          </Stagger>
        </div>

        <div className="panel col">
          <div className="step-h"><span className="n">2</span><h3>設定</h3><span className="muted hint">預設值通常就好</span></div>
          <div className="row">
            <label className="f">名稱<input value={name} placeholder={autoName()} onChange={(ev) => setName(ev.target.value)} onKeyDown={(ev) => ev.key === "Enter" && !blockers.length && start()} /></label>
            <label className="f">步數<input type="number" min={20} value={steps} placeholder={String(autoSteps)} onChange={(ev) => setSteps(ev.target.value === "" ? "" : +ev.target.value)} style={{ width: 110 }} /></label>
            <label className="f">學習率<input value={lr} onChange={(ev) => setLr(ev.target.value)} style={{ width: 100 }} /></label>
            <button className="ghost" onClick={() => setAdv(!adv)} style={{ alignSelf: "flex-end" }}>{adv ? "收起進階" : "進階 ▾"}</button>
          </div>
          <div className="small muted">步數預設 {autoSteps} ＝ {images} 張 × 2（100–1000）。LoRA rank {rank}、只掛文字 MLP，Orin 的 Photon 可直接載入。</div>
          <Pop show={adv} className="row card"><label className="f">LoRA rank<select value={rank} onChange={(ev) => setRank(+ev.target.value)}>{[8, 16, 24, 32].map((r) => <option key={r} value={r}>{r}</option>)}</select></label><label className="row hint"><input type="checkbox" checked={regionHead} onChange={(ev) => setRegionHead(ev.target.checked)} /> 也訓 region head（座標頭；開了 Orin 載不了）</label></Pop>
        </div>

        <div className="panel col">
          <div className="step-h"><h3>訓練紀錄</h3><span className="muted hint">{runs.length ? `${runs.length} 次；點一列看曲線與分數` : "還沒訓練過"}</span></div>
          {shown && <Fade k={shown.name}><RunCard r={shown} go={go} /></Fade>}
          {runs.length > 0 && <div className="history">{runs.map((r) => <RunRow key={r.name} r={r} cur={r.name === shown?.name} onOpen={() => setOpenRun(r.name)} onDelete={() => setAskDel(r.name)} onCancel={async () => { await api.cancelRun(r.name).catch((err) => say(String(err), "bad")); say("取消中…", "warn"); }} />)}</div>}
        </div>
      </div>

      <div className="actions">

        <ActionsHead />
        <button className="primary big" onClick={start} disabled={blockers.length > 0}>{running ? `訓練中 ${running.last?.step ?? 0}/${running.steps}` : blockers.length ? "還不能訓練" : `開始訓練 ${nameFinal}`}</button>
        {blockers.map((b) => <Pop key={b} className="bad hint">✗ {b}</Pop>)}
        <div className="panel col summary">
          <b>這次會訓練</b>
          <dl>
            <dt>資料集</dt><dd>{chosen.length ? chosen.map((d) => d.name).join("＋") : "—"}</dd>
            <dt>題目</dt><dd>{rows} 筆 · {images} 張圖</dd>
            <dt>步數</dt><dd>{nSteps} 步 · 約 {Math.max(1, Math.round(nSteps / 36))} 分鐘</dd>
            <dt>評分</dt><dd>工廠 CCTV{val ? ` ＋ ${chosen.filter((d) => d.val_images).length} 個驗證集（${val} 張）` : ""}</dd>
          </dl>
          <SkillMix counts={bySkill} />
        </div>
        {warns.map((w) => <Pop key={w} className="warn hint">⚠ {w}</Pop>)}
        {!blockers.length && <div className="small muted">會記為 {reviewer || "—"} 啟動；過關＝工廠 CCTV 每個 skill ≥ base − 1 點且至少一個進步。</div>}
      </div>
      <ConfirmDialog open={!!askDel} title={`刪除 ${askDel}？`} body="adapter、log、評估報告都會刪。" confirmText="刪除" danger onCancel={() => setAskDel(null)} onConfirm={async () => { await api.deleteRun(askDel!).catch((err) => say(String(err), "bad")); setAskDel(null); load(); }} />
    </div>
  );
}

/** Confirmed questions per skill as bars; under 20 is flagged (too few to learn from). */
function SkillMix({ counts }: { counts: Record<SkillKey, number> }) {
  const max = Math.max(1, ...SKILL_KEYS.map((k) => counts[k]));
  return <div className="mix">{SKILL_KEYS.map((k) => [<span key={k + "a"} className={counts[k] > 0 && counts[k] < 20 ? "warn-text" : ""}>{k}</span>, <span key={k + "b"} className="b"><i style={{ width: `${(100 * counts[k]) / max}%` }} /></span>, <span key={k + "c"} className="v">{counts[k]}</span>])}</div>;
}

function autoName() { const d = new Date(); const p = (n: number) => String(n).padStart(2, "0"); return `ft-${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`; }

function RunRow({ r, cur, onOpen, onDelete, onCancel }: { r: Run; cur?: boolean; onOpen?: () => void; onDelete: () => void; onCancel: () => void }) {
  const pill = r.state === "running" ? <Pill kind="warn">{phaseLabel(r.phase)}{r.phase === "training" ? ` ${r.last?.step ?? 0}/${r.steps}` : ""}</Pill> : r.state === "done" ? (r.gate ? <Pill kind={r.gate.passed ? "ok" : "grey"}>{r.gate.passed ? "完成 · 比 base 好" : "完成 · 沒有比 base 好"}</Pill> : <Pill kind="ok">完成</Pill>) : r.state === "cancelled" ? <Pill kind="grey">取消</Pill> : <Pill kind="bad">失敗</Pill>;
  return (
    <div className={`it${cur ? " cur" : ""}`} onClick={onOpen}>
      <span><b>{r.name}</b> <span className="muted small">{r.rows} 筆 · {r.steps} 步 · lr {r.lr}{r.datasets?.length ? ` · ${r.datasets.join("＋")}` : ""}{r.started ? ` · ${r.started.slice(5, 16).replace("T", " ")}` : ""}</span></span>
      <span className="row" onClick={(e) => e.stopPropagation()}>{pill}{r.state === "running" ? <button onClick={onCancel}>取消</button> : <button className="ghost" onClick={onDelete} title="刪除 run" aria-label="刪除">✕</button>}</span>
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
          {r.gate && <div className={r.gate.passed ? "ok" : "bad"}>{r.gate.passed ? "比 base 好（過關）" : "沒有比 base 好"}。規則（工廠 CCTV 考卷）：每個 skill ≥ base − 1.0 點，且至少一個進步</div>}
          {r.state === "done" && <button className="primary" onClick={() => go("test")} style={{ alignSelf: "flex-start" }}>去測試看差異 →</button>}
        </div>
      </div>
    </div>
  );
}

function Curve({ r }: { r: Run }) {
  const tr = r.curve.filter((p) => p.loss != null), va = r.curve.filter((p) => p.val != null);
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
