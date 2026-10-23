// ① 資料: a folder on the GB300 becomes a source; one button runs
// generate -> answer -> compare on it. Advanced options stay folded.

import { useState } from "react";
import { api, phaseLabel, type Batch, type Skill, type Status, type Summary } from "../api";

export default function Data({ sum, refresh, status, go }: { sum: Summary | null; refresh: () => void; status: Status | null; go: () => void }) {
  const [dir, setDir] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ text: string; bad?: boolean } | null>(null);
  const [adv, setAdv] = useState(false);
  const [n, setN] = useState<number | "">("");
  const [skills, setSkills] = useState<Skill[]>(["detect", "point", "query", "caption"]);
  const [topics, setTopics] = useState("");
  const [focus, setFocus] = useState("");
  const sources = (sum?.sources ?? []).filter((s) => !s.source.startsWith("heldout"));
  const batches = sum?.batches ?? [];
  const running = batches.find((b) => b.job?.state === "running");

  const scan = async () => {
    if (!dir) return;
    setBusy(true); setMsg(null);
    try { const r = await api.scan(dir); setMsg({ text: `「${r.source}」加入 ${r.added} 張（共找到 ${r.found}）` }); setDir(""); refresh(); } catch (e) { setMsg({ text: String(e), bad: true }); } finally { setBusy(false); }
  };
  const run = async (source: string) => {
    setBusy(true); setMsg(null);
    try {
      await api.autogen({ source, n_images: n === "" ? undefined : Number(n), skills, topics: topics.split("\n").map((s) => s.trim()).filter(Boolean), focus });
      refresh();
    } catch (e) { setMsg({ text: String(e), bad: true }); } finally { setBusy(false); }
  };

  return (
    <div className="col" style={{ maxWidth: 900 }}>
      <div className="card col">
        <strong>資料夾</strong>
        <div className="row">
          <input placeholder="GB300 上的圖片資料夾，例如 /home/paul/VLM-Forge/workspace/captures/1025" value={dir} onChange={(e) => setDir(e.target.value)} style={{ flex: 1 }} onKeyDown={(e) => e.key === "Enter" && scan()} />
          <button className="primary" onClick={scan} disabled={busy || !dir}>加入</button>
        </div>
        {sources.length === 0 && <div className="muted hint">還沒有資料夾。貼一個路徑按「加入」；子目錄裡的 jpg/png 都會算進去。</div>}
        {sources.map((s) => (
          <div key={s.source} className="row" style={{ justifyContent: "space-between", borderTop: "1px solid var(--line)", paddingTop: 6 }}>
            <span><strong>{s.source}</strong> <span className="muted">{s.n} 張</span></span>
            <button className="primary" disabled={busy || !!running || !(status?.engine.judge_ok ?? true)} onClick={() => run(s.source)} title="模型出題、作答、比對；完成後到 ② 檢查">自動出題（{n === "" ? s.n : Math.min(Number(n), s.n)} 張）</button>
          </div>
        ))}
        <button onClick={() => setAdv(!adv)} style={{ alignSelf: "flex-start" }}>{adv ? "收起進階" : "進階…"}</button>
        {adv && (
          <div className="col card">
            <label className="f">只抽幾張（空＝全部）<input type="number" min={1} value={n} onChange={(e) => setN(e.target.value === "" ? "" : +e.target.value)} /></label>
            <div className="f"><span>出哪些題</span><div className="row">{(["detect", "point", "query", "caption"] as Skill[]).map((s) => <label key={s} className="row hint"><input type="checkbox" checked={skills.includes(s)} onChange={(e) => setSkills(e.target.checked ? [...skills, s] : skills.filter((x) => x !== s))} />{s}</label>)}</div></div>
            <label className="f">額外巡檢主題（每行一個；預設已含值班人員、安全帽／背心／口罩／手套、出口阻擋、門、地面、堆高機、燈、管制區、無人看管物品）<textarea value={topics} onChange={(e) => setTopics(e.target.value)} style={{ minHeight: 44 }} /></label>
            <label className="f">側重（給出題模型的一句話）<input value={focus} onChange={(e) => setFocus(e.target.value)} placeholder="例如 night-shift factory floor" /></label>
          </div>
        )}
        {msg && <div className={msg.bad ? "bad hint" : "ok hint"}>{msg.text}</div>}
        {!(status?.engine.judge_ok ?? true) && <div className="warn hint">出題模型不在（{status?.engine.judge}）。</div>}
      </div>

      {batches.length > 0 && (
        <div className="card col">
          <strong>批次</strong>
          {batches.map((b) => <BatchRow key={b.id} b={b} />)}
          {sum && sum.images_pending > 0 && !running && <button className="primary" onClick={go} style={{ alignSelf: "flex-start" }}>去檢查（還有 {sum.images_pending} 張要看）→</button>}
        </div>
      )}
    </div>
  );
}

function BatchRow({ b }: { b: Batch }) {
  const j = b.job;
  const pending = b.counts.pending ?? 0, auto = b.counts.auto ?? 0, done = (b.counts.approved ?? 0) + (b.counts.corrected ?? 0) + (b.counts.discarded ?? 0);
  return (
    <div style={{ borderTop: "1px solid var(--line)", paddingTop: 6 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span><strong>{b.name}</strong> <span className="muted">{b.total} 題</span></span>
        <span className="muted hint">{j?.state === "running" ? `${phaseLabel(j.phase)} ${j.done}/${j.total}` : j?.state === "error" ? <span className="bad">失敗：{j.error}</span> : `待看 ${pending} · 自動通過 ${auto} · 已看 ${done}`}</span>
      </div>
      {j?.state === "running" && <div className="bar"><i style={{ width: `${j.total ? (100 * j.done) / j.total : 0}%` }} /></div>}
      {j?.student_error && <div className="warn hint">模型沒作答（{j.student_error}）；這批全部交給你看。</div>}
    </div>
  );
}
