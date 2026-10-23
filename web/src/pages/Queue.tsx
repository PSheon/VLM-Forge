// Review queue (GOALS §6.2): batch predictions over a set of images, reviewed one by one.
// Keys: Enter = approve (or save the correction), D = discard, N = skip.

import { useCallback, useEffect, useState } from "react";
import { api, defaultPrompt, emptyLabel, judgeText, promptLabel, reviewerStore, skillLabel, sourceLabel, summary, type Item, type Model, type Pred, type Queue as Q, type Skill, type Status } from "../api";
import Canvas from "../components/Canvas";

export default function Queue({ models, status }: { models: Model[]; status: Status | null }) {
  const [queues, setQueues] = useState<Q[]>([]);
  const [cur, setCur] = useState<Q | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [item, setItem] = useState<Item | null>(null);
  const [label, setLabel] = useState<Pred | null>(null);
  const [reviewer, setReviewer] = useState(reviewerStore.get());
  const [msg, setMsg] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [filter, setFilter] = useState("");

  const refresh = useCallback(() => api.queues().then(setQueues).catch((e) => setMsg(String(e))), []);
  const loadItems = useCallback((q: Q | null) => { if (q) api.queueItems(q.id).then(setItems).catch(() => undefined); else setItems([]); }, []);
  useEffect(() => { refresh(); const t = setInterval(refresh, 4000); return () => clearInterval(t); }, [refresh]);
  useEffect(() => { loadItems(cur); }, [cur, loadItems]);
  useEffect(() => { // keep item grid fresh while predictions run
    if (cur?.job?.state === "running") { const t = setInterval(() => loadItems(cur), 3000); return () => clearInterval(t); }
  }, [cur, loadItems]);
  useEffect(() => { reviewerStore.set(reviewer); }, [reviewer]);
  useEffect(() => { // the selected queue's counts update with the poll
    if (cur) { const q = queues.find((x) => x.id === cur.id); if (q && JSON.stringify(q) !== JSON.stringify(cur)) setCur(q); }
  }, [queues]); // eslint-disable-line react-hooks/exhaustive-deps

  const open = (it: Item) => { setItem(it); setLabel(structuredClone(it.label ?? it.pred) ?? emptyLabel(it.skill)); setMsg(""); };

  const next = useCallback(async (after?: number) => {
    if (!cur) { setMsg("先選一個佇列"); return; }
    if (!reviewer) { setMsg("先填 reviewer 名字"); return; }
    const it = await api.next(cur.id, reviewer, after);
    if (!("id" in it)) { setItem(null); setMsg("這個佇列沒有待審的項目了"); return; }
    open(it as Item);
  }, [cur, reviewer]);

  const act = useCallback(async (action: "approve" | "correct" | "discard") => {
    if (!item) return;
    if (!reviewer) { setMsg("先填 reviewer 名字"); return; }
    const changed = JSON.stringify(label) !== JSON.stringify(item.pred);
    const a = action === "approve" && changed ? "correct" : action;
    try {
      await api.review(item.id, a, reviewer, a === "discard" ? null : label);
      refresh(); loadItems(cur);
      next(item.id);
    } catch (e) { setMsg(String(e)); }
  }, [item, label, reviewer, next, refresh, loadItems, cur]);

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

  const spatial = item && (item.skill === "detect" || item.skill === "point");
  const shown = filter === "fix" ? items.filter((i) => i.verdict === "fix" && i.status === "pending") : filter === "spot" ? items.filter((i) => i.spot) : filter ? items.filter((i) => i.status === filter) : items;

  return (
    <div className="grid3">
      <aside className="card col">
        <div className="row"><strong>佇列</strong><button onClick={() => setShowNew(!showNew)}>{showNew ? "關閉" : "＋ 新佇列"}</button></div>
        {showNew && <NewQueue models={models} ready={status?.engine.weights_ok ?? true} onCreated={(q) => { setShowNew(false); refresh(); setCur(q); setItem(null); }} />}
        {queues.length === 0 && !showNew && <div className="empty">還沒有佇列。「＋ 新佇列」會對一批圖跑模型預測，然後逐張審核。</div>}
        <div className="list">
          {queues.map((q) => (
            <div key={q.id} className={`it${cur?.id === q.id ? " cur" : ""}`} onClick={() => { setCur(q); setItem(null); }}>
              <div><strong>{q.name}</strong> <span className="muted">{skillLabel[q.skill]} · {q.model === "base" ? "base" : q.model.split("/").pop()}</span></div>
              <div className="muted">待人審 {q.counts.pending} · judge 通過 {q.counts.auto + (q.spot ?? 0)}（抽查 {q.spot ?? 0}）· 人審 {q.counts.approved + q.counts.corrected + q.counts.discarded}</div>
              {q.job && q.job.state === "running" && <div className="bar" title={`${q.job.phase === "judging" ? "judge 審核中" : "預測中"} ${q.job.done}/${q.job.total}`}><i style={{ width: `${q.job.total ? (100 * q.job.done) / q.job.total : 0}%`, background: q.job.phase === "judging" ? "var(--box)" : undefined }} /></div>}
              {q.job && q.job.state === "running" && <div className="muted hint">{q.job.phase === "judging" ? "judge 審核中" : "預測中"} {q.job.done}/{q.job.total}</div>}
              {q.job?.state === "error" && <div className="bad hint">預測失敗：{q.job.error}</div>}
            </div>
          ))}
        </div>
        <label className="f">Reviewer<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="你的名字" /></label>
        <button className="primary" onClick={() => next()} disabled={!cur}>下一題待審</button>
        {cur && <ExportBox q={cur} />}
      </aside>
      <section style={{ minWidth: 0 }}>
        {item ? (
          <>
            <div className="row" style={{ marginBottom: 6 }}>
              <strong style={{ fontSize: 17 }}>{item.skill === "query" ? item.prompt : item.skill === "caption" ? `caption / ${item.prompt}` : `${skillLabel[item.skill]}：${item.prompt}`}</strong>
              <span className="muted hint">#{item.id} · {item.path.split("/").slice(-2).join("/")} · 模型 {summary(item.pred)} · 目前 {summary(label)} · {item.status}{item.reviewer ? ` by ${item.reviewer}` : ""}</span>
            </div>
            {item.judge && (
              <div className={item.verdict === "fix" ? "warn hint" : "hint ok"} style={{ marginBottom: 6 }}>
                Judge {item.verdict === "fix" ? "標紅" : "通過"}{item.spot ? "（抽查樣本）" : ""}：{judgeText(item.judge) || "—"}
              </div>
            )}
            <Canvas src={api.imageUrl(item.image_id, 2048)} skill={item.skill} value={label} onChange={setLabel} height="calc(100vh - 200px)" />
            {!spatial && <textarea style={{ marginTop: 8 }} value={label?.text ?? ""} onChange={(e) => setLabel({ text: e.target.value })} />}
          </>
        ) : (
          <div className="empty">{msg || (cur ? "按「下一題待審」，或在右側點任一張。" : "左側選一個佇列。")}</div>
        )}
      </section>
      <aside className="card col">
        <div className="row">
          <button className="primary" onClick={() => act("approve")} disabled={!item}>確認 <kbd>Enter</kbd></button>
          <button onClick={() => act("discard")} disabled={!item}>丟棄 <kbd>D</kbd></button>
          <button onClick={() => next(item?.id)} disabled={!item}>跳過 <kbd>N</kbd></button>
          <button onClick={() => item && setLabel(structuredClone(item.pred) ?? emptyLabel(item.skill))} disabled={!item}>還原</button>
        </div>
        <div className="hint muted">確認＝接受畫面上的結果（改過就記為修正）。</div>
        {msg && item && <div className="bad hint">{msg}</div>}
        {cur && (
          <>
            <div className="row"><strong>項目</strong>
              <select value={filter} onChange={(e) => setFilter(e.target.value)}>
                <option value="">全部（{items.length}）</option>
                <option value="fix">judge 標紅（{cur.verdicts?.fix ?? 0}）</option>
                <option value="spot">抽查樣本（{cur.spot ?? 0}）</option>
                {(["pending", "auto", "approved", "corrected", "discarded"] as const).map((s) => <option key={s} value={s}>{statusLabel[s]}（{cur.counts[s] ?? 0}）</option>)}
              </select>
              {cur.judge && cur.verdicts?.none ? <button onClick={() => api.judgeQueue(cur.id).then(refresh)} title="對還沒 judge 的題再跑一次">再跑 judge</button> : null}
            </div>
            <div className="thumbs" style={{ maxHeight: "calc(100vh - 330px)" }}>
              {shown.map((it) => (
                <div key={it.id} className="th" onClick={() => open(it)} title={`#${it.id} ${it.prompt} · ${statusLabel[it.status]}${it.verdict ? ` · judge ${it.verdict}` : ""}`}>
                  <img src={api.imageUrl(it.image_id, 200)} className={item?.id === it.id ? "cur" : ""} loading="lazy" />
                  <b className={it.status === "pending" && it.verdict === "fix" ? "fix" : it.status}>{it.status === "pending" ? (it.verdict === "fix" ? "紅" : it.spot ? "抽" : it.pred ? summary(it.pred) : "…") : it.status === "auto" ? "J" : it.status[0]}</b>
                </div>
              ))}
            </div>
          </>
        )}
      </aside>
    </div>
  );
}

const statusLabel: Record<Item["status"], string> = { pending: "待人審", auto: "judge 通過", approved: "人確認", corrected: "人修正", discarded: "丟棄" };

function NewQueue({ models, ready, onCreated }: { models: Model[]; ready: boolean; onCreated: (q: Q) => void }) {
  const [name, setName] = useState("");
  const [skill, setSkill] = useState<Skill>("detect");
  const [prompts, setPrompts] = useState("person");
  const [model, setModel] = useState("base");
  const [judge, setJudge] = useState(true);
  const [sources, setSources] = useState<{ source: string; n: number }[]>([]);
  const [source, setSource] = useState("");
  const [err, setErr] = useState("");
  useEffect(() => { api.sources().then((s) => { setSources(s); if (!source && s.length) setSource(s[0].source); }); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const create = async () => {
    try {
      const q = await api.createQueue({ name, skill, prompts: prompts.split(/[,\n]/).map((s) => s.trim()).filter(Boolean), model, source, judge });
      onCreated(q);
    } catch (e) { setErr(String(e)); }
  };
  const n = sources.find((s) => s.source === source)?.n ?? 0;
  const k = prompts.split(/[,\n]/).filter((s) => s.trim()).length;
  return (
    <div className="col card">
      {!ready && <div className="warn hint">權重未就緒：佇列會建立，但預測會失敗。</div>}
      <label className="f">名稱<input placeholder="例如 cctv-detect-1025" value={name} onChange={(e) => setName(e.target.value)} /></label>
      <label className="f">圖片來源<select value={source} onChange={(e) => setSource(e.target.value)}>{sources.map((s) => <option key={s.source} value={s.source}>{sourceLabel(s.source)}（{s.n} 張）</option>)}</select></label>
      <label className="f">Skill<select value={skill} onChange={(e) => { setSkill(e.target.value as Skill); setPrompts(defaultPrompt(e.target.value as Skill)); }}>{(["detect", "point", "query", "caption"] as Skill[]).map((s) => <option key={s} value={s}>{skillLabel[s]}</option>)}</select></label>
      <label className="f">{promptLabel(skill)}（逗號或換行分隔，每張圖各一題）<textarea value={prompts} onChange={(e) => setPrompts(e.target.value)} style={{ minHeight: 48 }} /></label>
      <label className="f">模型<select value={model} onChange={(e) => setModel(e.target.value)}>{models.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}</select></label>
      <label className="row hint"><input type="checkbox" checked={judge} onChange={(e) => setJudge(e.target.checked)} /> 預測後先由 judge（Qwen3-VL-32B）審，人只看標紅的和 10% 抽查</label>
      {err && <div className="bad hint">{err}</div>}
      <button className="primary" onClick={create} disabled={!name || !k || !n}>建立並預測（{n * k} 題）</button>
      {source === "heldout" && <div className="warn hint">held-out 只能用來做 gold 或評估，匯出的列不能進訓練。</div>}
    </div>
  );
}

function ExportBox({ q }: { q: Q }) {
  const [res, setRes] = useState<string>("");
  const n = (q.counts.approved ?? 0) + (q.counts.corrected ?? 0);
  return (
    <div className="col">
      <button onClick={async () => { try { const r = await api.export(q.name, [q.id], true) as { train: { path: string; rows: number }; incoming?: { path: string } }; setRes(`${r.train.rows} 列 → ${r.train.path}${r.incoming ? `\nData Core 包 → ${r.incoming.path}` : ""}`); } catch (e) { setRes(String(e)); } }} disabled={!n}>匯出 {n} 筆已審核</button>
      {res && <pre className="hint" style={{ whiteSpace: "pre-wrap", margin: 0 }}>{res}</pre>}
    </div>
  );
}
