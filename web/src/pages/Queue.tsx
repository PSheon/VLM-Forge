// Review queue (GOALS §6.2): batch predictions over a set of images, reviewed one by one.
// Keys: Enter = approve (or save the correction), D = discard, N = skip to the next.

import { useCallback, useEffect, useState } from "react";
import { api, promptLabel, reviewerStore, skillLabel, type Item, type Model, type Pred, type Queue as Q, type Skill } from "../api";
import Canvas from "../components/Canvas";
import { emptyLabel, summary } from "./Workbench";

export default function Queue({ models }: { models: Model[] }) {
  const [queues, setQueues] = useState<Q[]>([]);
  const [cur, setCur] = useState<Q | null>(null);
  const [item, setItem] = useState<Item | null>(null);
  const [label, setLabel] = useState<Pred | null>(null);
  const [reviewer, setReviewer] = useState(reviewerStore.get());
  const [msg, setMsg] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [done, setDone] = useState<Item[]>([]);

  const refresh = useCallback(() => api.queues().then(setQueues).catch((e) => setMsg(String(e))), []);
  useEffect(() => { refresh(); const t = setInterval(refresh, 4000); return () => clearInterval(t); }, [refresh]);
  useEffect(() => { reviewerStore.set(reviewer); }, [reviewer]);

  const next = useCallback(async (after?: number) => {
    if (!cur || !reviewer) { setMsg("選一個佇列並填 reviewer"); return; }
    const it = await api.next(cur.id, reviewer, after);
    if (!("id" in it)) { setItem(null); setMsg("這個佇列沒有待審的項目了"); return; }
    setItem(it as Item); setLabel(structuredClone((it as Item).pred) ?? emptyLabel((it as Item).skill)); setMsg("");
  }, [cur, reviewer]);

  const act = useCallback(async (action: "approve" | "correct" | "discard") => {
    if (!item) return;
    const changed = JSON.stringify(label) !== JSON.stringify(item.pred);
    const a = action === "approve" && changed ? "correct" : action;
    const saved = await api.review(item.id, a, reviewer, a === "discard" ? null : label);
    setDone((d) => [saved, ...d].slice(0, 30));
    refresh();
    next();
  }, [item, label, reviewer, next, refresh]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (e.key === "Enter") { e.preventDefault(); act("approve"); }
      else if (e.key === "d" || e.key === "D") act("discard");
      else if (e.key === "n" || e.key === "N") next(item?.id);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [act, next, item]);

  const openQueue = (q: Q) => { setCur(q); setItem(null); setDone([]); };
  const spatial = item && (item.skill === "detect" || item.skill === "point");

  return (
    <div className="grid3">
      <aside className="card col">
        <div className="row"><strong>佇列</strong><button onClick={() => setShowNew(!showNew)}>{showNew ? "關閉" : "＋ 新佇列"}</button></div>
        {showNew && <NewQueue models={models} onCreated={(q) => { setShowNew(false); refresh(); openQueue(q); }} />}
        <div className="list">
          {queues.map((q) => (
            <div key={q.id} className={`it${cur?.id === q.id ? " cur" : ""}`} onClick={() => openQueue(q)}>
              <div><strong>{q.name}</strong> <span className="muted">{skillLabel[q.skill]} · {q.model === "base" ? "base" : q.model.split("/").pop()}</span></div>
              <div className="muted">待審 {q.counts.pending} · 通過 {q.counts.approved} · 修正 {q.counts.corrected} · 丟棄 {q.counts.discarded}</div>
              {q.job && q.job.state === "running" && <div className="bar"><i style={{ width: `${q.job.total ? (100 * q.job.done) / q.job.total : 0}%` }} /></div>}
              {q.job?.state === "error" && <div className="bad">{q.job.error}</div>}
            </div>
          ))}
        </div>
        <label>Reviewer<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="你的名字" /></label>
        <button className="primary" onClick={() => next()} disabled={!cur}>開始／下一題</button>
        {cur && <ExportBox q={cur} />}
      </aside>
      <section>
        {item ? (
          <>
            <div className="row" style={{ marginBottom: 6 }}>
              <strong style={{ fontSize: 18 }}>{item.skill === "query" ? item.prompt : item.skill === "caption" ? `caption / ${item.prompt}` : item.prompt}</strong>
              <span className="muted">#{item.id} · {item.path.split("/").slice(-2).join("/")} · 預測 {summary(item.pred)} · 目前 {summary(label)}</span>
            </div>
            <Canvas src={api.imageUrl(item.image_id)} skill={item.skill} value={label} onChange={setLabel} />
            {!spatial && <textarea style={{ marginTop: 8 }} value={label?.text ?? ""} onChange={(e) => setLabel({ text: e.target.value })} />}
          </>
        ) : (
          <div className="card muted">{msg || "左側選佇列，按「開始」。"}</div>
        )}
      </section>
      <aside className="card col">
        <div className="row">
          <button className="primary" onClick={() => act("approve")} disabled={!item}>確認 <kbd>Enter</kbd></button>
          <button onClick={() => act("discard")} disabled={!item}>丟棄 <kbd>D</kbd></button>
          <button onClick={() => next(item?.id)} disabled={!item}>跳過 <kbd>N</kbd></button>
          <button onClick={() => item && setLabel(structuredClone(item.pred) ?? emptyLabel(item.skill))} disabled={!item}>重設</button>
        </div>
        <div className="muted">確認＝接受畫面上的結果（改過就記為修正）。框：拖曳新增、拖角縮放、Delete 刪除。</div>
        {msg && item && <div className="bad">{msg}</div>}
        <div><strong>剛審過</strong></div>
        <div className="list" style={{ maxHeight: "40vh" }}>
          {done.map((d) => (
            <div key={d.id} className="it row" style={{ justifyContent: "space-between" }}>
              <span>#{d.id} {d.prompt} · <span className={d.status === "discarded" ? "bad" : "ok"}>{d.status}</span></span>
              <button onClick={async () => { const it = await api.reopen(d.id); setItem(it); setLabel(structuredClone(it.pred) ?? emptyLabel(it.skill)); setDone((x) => x.filter((y) => y.id !== d.id)); refresh(); }}>重審</button>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}

function NewQueue({ models, onCreated }: { models: Model[]; onCreated: (q: Q) => void }) {
  const [name, setName] = useState("");
  const [skill, setSkill] = useState<Skill>("detect");
  const [prompts, setPrompts] = useState("person");
  const [model, setModel] = useState("base");
  const [source, setSource] = useState("capture");
  const [err, setErr] = useState("");
  const create = async () => {
    try {
      const q = await api.createQueue({ name, skill, prompts: prompts.split(/[,\n]/).map((s) => s.trim()).filter(Boolean), model, source });
      onCreated(q);
    } catch (e) { setErr(String(e)); }
  };
  return (
    <div className="col card">
      <input placeholder="名稱，例如 cctv-detect-1025" value={name} onChange={(e) => setName(e.target.value)} />
      <select value={skill} onChange={(e) => setSkill(e.target.value as Skill)}>{(["detect", "point", "query", "caption"] as Skill[]).map((s) => <option key={s} value={s}>{skillLabel[s]}</option>)}</select>
      <textarea placeholder={`${promptLabel(skill)}，逗號或換行分隔`} value={prompts} onChange={(e) => setPrompts(e.target.value)} style={{ minHeight: 50 }} />
      <select value={model} onChange={(e) => setModel(e.target.value)}>{models.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}</select>
      <input placeholder="影像來源前綴 (upload / capture / heldout)" value={source} onChange={(e) => setSource(e.target.value)} />
      {err && <div className="bad">{err}</div>}
      <button className="primary" onClick={create} disabled={!name || !prompts}>建立並預測</button>
    </div>
  );
}

function ExportBox({ q }: { q: Q }) {
  const [res, setRes] = useState<string>("");
  return (
    <div className="col">
      <button onClick={async () => { try { const r = await api.export(q.name, [q.id], true); setRes(JSON.stringify(r, null, 1)); } catch (e) { setRes(String(e)); } }}>匯出此佇列（訓練列 + Data Core 包）</button>
      {res && <pre style={{ fontSize: 11, whiteSpace: "pre-wrap" }}>{res}</pre>}
    </div>
  );
}
