// Review queue (GOALS §6.2). Two kinds of queue:
//   auto   the tool picks images, the teacher writes questions + answers, the student
//          answers, agreement / judge decide, a human checks image by image
//   manual a batch of images x prompts, predicted by the student, judged, reviewed per item
// Review unit is the image: every item of the image is listed, flagged ones first.
// Keys: Enter = confirm the image (every item as shown), D = discard the selected item,
// N = next image, ↑/↓ = select item.

import { useCallback, useEffect, useMemo, useState } from "react";
import { api, defaultPrompt, emptyLabel, judgeText, promptLabel, reviewerStore, skillLabel, sourceLabel, summary, type ImageReview, type ImageSummary, type Item, type Model, type Pred, type Queue as Q, type Skill, type Status } from "../api";
import Canvas from "../components/Canvas";

type Decision = { action: "approve" | "discard"; label: Pred };

export default function Queue({ models, status }: { models: Model[]; status: Status | null }) {
  const [queues, setQueues] = useState<Q[]>([]);
  const [cur, setCur] = useState<Q | null>(null);
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [review, setReview] = useState<ImageReview | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [dec, setDec] = useState<Record<number, Decision>>({});
  const [reviewer, setReviewer] = useState(reviewerStore.get());
  const [msg, setMsg] = useState("");
  const [showNew, setShowNew] = useState<"" | "auto" | "manual">("");
  const [filter, setFilter] = useState("");

  const refresh = useCallback(() => api.queues().then(setQueues).catch((e) => setMsg(String(e))), []);
  const loadImages = useCallback((q: Q | null) => { if (q) api.queueImages(q.id).then(setImages).catch(() => undefined); else setImages([]); }, []);
  useEffect(() => { refresh(); const t = setInterval(refresh, 4000); return () => clearInterval(t); }, [refresh]);
  useEffect(() => { loadImages(cur); }, [cur?.id, loadImages]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (cur?.job?.state === "running") { const t = setInterval(() => loadImages(cur), 3000); return () => clearInterval(t); }
  }, [cur, loadImages]);
  useEffect(() => { reviewerStore.set(reviewer); }, [reviewer]);
  useEffect(() => {
    if (cur) { const q = queues.find((x) => x.id === cur.id); if (q && JSON.stringify(q) !== JSON.stringify(cur)) setCur(q); }
  }, [queues]); // eslint-disable-line react-hooks/exhaustive-deps

  const openReview = (r: ImageReview) => {
    const d: Record<number, Decision> = {};
    for (const it of r.items) d[it.id] = { action: it.status === "discarded" ? "discard" : "approve", label: structuredClone(it.label ?? it.pred ?? it.teacher) ?? emptyLabel(it.skill) };
    const order = sortItems(r.items);
    setReview({ ...r, items: order }); setDec(d); setSel(order[0]?.id ?? null); setMsg("");
  };
  const nextImage = useCallback(async (after?: string) => {
    if (!cur) { setMsg("先選一個佇列"); return; }
    if (!reviewer) { setMsg("先填 reviewer 名字"); return; }
    const r = await api.nextImage(cur.id, reviewer, after);
    if (!("image" in r)) { setReview(null); setMsg("這個佇列沒有待人審的圖了"); return; }
    openReview(r as ImageReview);
  }, [cur, reviewer]);
  const openImage = async (imageId: string) => { if (cur) openReview(await api.queueImage(cur.id, imageId)); };

  const confirmImage = useCallback(async () => {
    if (!review) return;
    if (!reviewer) { setMsg("先填 reviewer 名字"); return; }
    const items = review.items.filter((it) => it.status === "pending" || it.status === "auto").map((it) => {
      const d = dec[it.id];
      if (d.action === "discard") return { id: it.id, action: "discard" };
      const changed = JSON.stringify(d.label) !== JSON.stringify(it.pred);
      return { id: it.id, action: changed ? "correct" : "approve", label: d.label };
    });
    try {
      await api.reviewImage(cur!.id, reviewer, items);
      refresh(); loadImages(cur);
      nextImage(review.image.id);
    } catch (e) { setMsg(String(e)); }
  }, [review, dec, reviewer, cur, refresh, loadImages, nextImage]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (!review) return;
      if (e.key === "Enter") { e.preventDefault(); confirmImage(); }
      else if (e.key === "d" || e.key === "D") { if (sel !== null) setDec((d) => ({ ...d, [sel]: { ...d[sel], action: d[sel].action === "discard" ? "approve" : "discard" } })); }
      else if (e.key === "n" || e.key === "N") nextImage(review.image.id);
      else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const ids = review.items.map((i) => i.id);
        const k = sel === null ? 0 : ids.indexOf(sel);
        setSel(ids[Math.max(0, Math.min(ids.length - 1, k + (e.key === "ArrowDown" ? 1 : -1)))]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [review, sel, confirmImage, nextImage]);

  const selItem = review?.items.find((i) => i.id === sel) ?? null;
  const selDec = sel !== null ? dec[sel] : null;
  const setLabel = (label: Pred) => { if (sel !== null) setDec((d) => ({ ...d, [sel]: { ...d[sel], label } })); };
  const spatial = selItem && (selItem.skill === "detect" || selItem.skill === "point");
  const shownImages = useMemo(() => filter === "fix" ? images.filter((i) => i.fix > 0) : filter === "pending" ? images.filter((i) => i.pending > 0) : filter === "spot" ? images.filter((i) => i.spot) : filter === "auto" ? images.filter((i) => i.auto === i.n) : images, [images, filter]);

  return (
    <div className="grid3">
      <aside className="card col">
        <div className="row"><strong>佇列</strong>
          <button className={showNew === "auto" ? "cur" : ""} onClick={() => setShowNew(showNew === "auto" ? "" : "auto")}>＋ 自動出題</button>
          <button className={showNew === "manual" ? "cur" : ""} onClick={() => setShowNew(showNew === "manual" ? "" : "manual")}>＋ 指定題目</button>
        </div>
        {showNew === "auto" && <NewAuto models={models} ready={status?.engine.weights_ok ?? true} onCreated={(q) => { setShowNew(""); refresh(); setCur(q); setReview(null); }} />}
        {showNew === "manual" && <NewManual models={models} ready={status?.engine.weights_ok ?? true} onCreated={(q) => { setShowNew(""); refresh(); setCur(q); setReview(null); }} />}
        {queues.length === 0 && !showNew && <div className="empty">還沒有佇列。「自動出題」：工具自己選圖、出題、出答案，學生答，judge 對，你逐張圖檢查。</div>}
        <div className="list">
          {queues.map((q) => (
            <div key={q.id} className={`it${cur?.id === q.id ? " cur" : ""}`} onClick={() => { setCur(q); setReview(null); }}>
              <div><strong>{q.name}</strong> <span className="muted">{q.mode === "auto" ? "自動出題" : skillLabel[q.skill]} · {q.model === "base" ? "base" : q.model.split("/").pop()}</span></div>
              <div className="muted hint">待人審 {q.counts.pending} · judge 通過 {q.counts.auto}（抽查 {q.spot ?? 0}）· 人審 {q.counts.approved + q.counts.corrected + q.counts.discarded}</div>
              {q.job && q.job.state === "running" && <><div className="bar"><i style={{ width: `${q.job.total ? (100 * q.job.done) / q.job.total : 0}%`, background: q.job.phase === "generating" ? "var(--box2)" : q.job.phase === "judging" ? "var(--box)" : undefined }} /></div><div className="muted hint">{phaseLabel(q.job.phase)} {q.job.done}/{q.job.total}</div></>}
              {q.job?.state === "error" && <div className="bad hint">失敗：{q.job.error}</div>}
            </div>
          ))}
        </div>
        <label className="f">Reviewer<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="你的名字" /></label>
        <button className="primary" onClick={() => nextImage()} disabled={!cur}>下一張待審的圖</button>
        {cur && <ExportBox q={cur} />}
      </aside>

      <section style={{ minWidth: 0 }}>
        {review && selItem ? (
          <>
            <div className="row" style={{ marginBottom: 6 }}>
              <strong style={{ fontSize: 17 }}>{itemTitle(selItem)}</strong>
              <span className="muted hint">{review.image.path.split("/").slice(-2).join("/")} · 學生 {summary(selItem.pred)}{selItem.teacher ? ` · teacher ${summary(selItem.teacher)}` : ""} · 目前 {summary(selDec?.label)}</span>
            </div>
            {selItem.judge && <div className={selItem.verdict === "fix" ? "warn hint" : "hint ok"} style={{ marginBottom: 6 }}>{selItem.verdict === "fix" ? "不一致／標紅" : "一致"}{selItem.spot ? "（抽查）" : ""}：{judgeText(selItem.judge) || "—"}</div>}
            <Canvas
              src={api.imageUrl(review.image.id, 2048)}
              skill={selItem.skill}
              value={selDec?.label ?? null}
              onChange={setLabel}
              alt={spatial ? selItem.teacher : null}
              labels={spatial && selItem.teacher ? { main: "目前（可改）", alt: "teacher" } : undefined}
              height="calc(100vh - 210px)"
            />
            {!spatial && (
              <div className="grid2" style={{ marginTop: 8 }}>
                <label className="f">目前答案（可改）<textarea value={selDec?.label.text ?? ""} onChange={(e) => setLabel({ text: e.target.value })} /></label>
                <div className="col hint">
                  {selItem.teacher && <div>teacher：{selItem.teacher.text} <button onClick={() => setLabel({ text: selItem.teacher!.text ?? "" })}>用這個</button></div>}
                  {selItem.pred && <div>學生：{selItem.pred.text} <button onClick={() => setLabel({ text: selItem.pred!.text ?? "" })}>用這個</button></div>}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="empty">{msg || (cur ? "按「下一張待審的圖」，或在右側點一張。" : "左側選一個佇列。")}</div>
        )}
      </section>

      <aside className="card col">
        {review ? (
          <>
            <div className="row">
              <button className="primary" onClick={confirmImage}>整張確認 <kbd>Enter</kbd></button>
              <button onClick={() => nextImage(review.image.id)}>跳過 <kbd>N</kbd></button>
            </div>
            <div className="hint muted">這張圖的 {review.items.length} 題；確認＝每一題照目前畫面存（改過的記為修正，打叉的丟棄）。</div>
            <div className="list" style={{ maxHeight: "40vh" }}>
              {review.items.map((it) => {
                const d = dec[it.id];
                const changed = d && JSON.stringify(d.label) !== JSON.stringify(it.pred);
                return (
                  <div key={it.id} className={`it${sel === it.id ? " cur" : ""}`} onClick={() => setSel(it.id)} style={{ textDecoration: d?.action === "discard" ? "line-through" : undefined }}>
                    <span className={it.verdict === "fix" && it.status === "pending" ? "bad" : it.status === "auto" ? "muted" : "ok"}>{it.verdict === "fix" && it.status === "pending" ? "●" : "○"}</span> {itemTitle(it)}
                    <span className="muted"> — {summary(d?.label)}{changed ? " ✎" : ""}{it.status !== "pending" ? ` · ${statusLabel[it.status]}` : ""}</span>
                    <button style={{ float: "right", padding: "0 6px" }} title="丟棄這題 (D)" onClick={(e) => { e.stopPropagation(); setDec((x) => ({ ...x, [it.id]: { ...x[it.id], action: x[it.id].action === "discard" ? "approve" : "discard" } })); }}>{d?.action === "discard" ? "↺" : "✕"}</button>
                  </div>
                );
              })}
            </div>
            {msg && <div className="bad hint">{msg}</div>}
          </>
        ) : null}
        {cur && (
          <>
            <div className="row"><strong>圖</strong>
              <select value={filter} onChange={(e) => setFilter(e.target.value)}>
                <option value="">全部（{images.length}）</option>
                <option value="fix">有標紅（{images.filter((i) => i.fix > 0).length}）</option>
                <option value="pending">待人審（{images.filter((i) => i.pending > 0).length}）</option>
                <option value="spot">抽查（{images.filter((i) => i.spot).length}）</option>
                <option value="auto">全自動通過（{images.filter((i) => i.auto === i.n).length}）</option>
              </select>
            </div>
            <div className="thumbs" style={{ maxHeight: review ? "calc(100vh - 560px)" : "calc(100vh - 300px)" }}>
              {shownImages.map((im) => (
                <div key={im.image_id} className="th" onClick={() => openImage(im.image_id)} title={`${im.n} 題 · 標紅 ${im.fix} · 待審 ${im.pending}`}>
                  <img src={api.imageUrl(im.image_id, 200)} className={review?.image.id === im.image_id ? "cur" : ""} loading="lazy" />
                  <b className={im.fix > 0 ? "fix" : im.pending > 0 ? "" : im.reviewed === im.n ? "approved" : "auto"}>{im.fix > 0 ? `紅 ${im.fix}` : im.pending > 0 ? (im.spot ? "抽" : `${im.n}`) : im.reviewed === im.n ? "✓" : "J"}</b>
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
function phaseLabel(p?: string) { return p === "generating" ? "teacher 出題中" : p === "judging" ? "judge 對答案中" : "學生作答中"; }
function itemTitle(it: Item) { return it.skill === "query" ? it.prompt : it.skill === "caption" ? "Caption" : `${skillLabel[it.skill]}：${it.prompt}`; }
function sortItems(items: Item[]) {
  const rank = (it: Item) => (it.status === "pending" && it.verdict === "fix" ? 0 : it.status === "pending" ? 1 : 2);
  return [...items].sort((a, b) => rank(a) - rank(b) || a.id - b.id);
}

function useSources() {
  const [sources, setSources] = useState<{ source: string; n: number }[]>([]);
  useEffect(() => { api.sources().then(setSources); }, []);
  return sources;
}

function NewAuto({ models, ready, onCreated }: { models: Model[]; ready: boolean; onCreated: (q: Q) => void }) {
  const sources = useSources().filter((s) => !s.source.startsWith("heldout"));
  const [name, setName] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const [n, setN] = useState(50);
  const [skills, setSkills] = useState<Skill[]>(["detect", "point", "query", "caption"]);
  const [model, setModel] = useState("base");
  const [topics, setTopics] = useState("");
  const [focus, setFocus] = useState("");
  const [err, setErr] = useState("");
  useEffect(() => { if (!picked.length && sources.length) setPicked([sources[0].source]); }, [sources]); // eslint-disable-line react-hooks/exhaustive-deps
  const pool = sources.filter((s) => picked.includes(s.source)).reduce((a, s) => a + s.n, 0);
  const create = async () => {
    try { onCreated(await api.createAutoQueue({ name, sources: picked, n_images: n, skills, model, focus, topics: topics.split("\n").map((s) => s.trim()).filter(Boolean) })); } catch (e) { setErr(String(e)); }
  };
  return (
    <div className="col card">
      {!ready && <div className="warn hint">權重未就緒：佇列會建立，但學生作答會失敗。</div>}
      <label className="f">名稱<input placeholder="例如 auto-cctv-1025" value={name} onChange={(e) => setName(e.target.value)} /></label>
      <div className="f"><span>圖池（held-out 不在內）</span>
        {sources.length === 0 && <span className="muted hint">沒有可用的圖池：先在工作台「掃描目錄」登記擷取或 Data Core release 的圖。</span>}
        {sources.map((s) => <label key={s.source} className="row hint"><input type="checkbox" checked={picked.includes(s.source)} onChange={(e) => setPicked(e.target.checked ? [...picked, s.source] : picked.filter((x) => x !== s.source))} />{sourceLabel(s.source)}（{s.n}）</label>)}
      </div>
      <label className="f">抽幾張圖（圖池 {pool}）<input type="number" min={1} value={n} onChange={(e) => setN(+e.target.value)} /></label>
      <div className="f"><span>出哪些題</span><div className="row">{(["detect", "point", "query", "caption"] as Skill[]).map((s) => <label key={s} className="row hint"><input type="checkbox" checked={skills.includes(s)} onChange={(e) => setSkills(e.target.checked ? [...skills, s] : skills.filter((x) => x !== s))} />{skillLabel[s]}</label>)}</div></div>
      <label className="f">額外巡檢主題（每行一個；預設已含值班人員、安全帽／背心／口罩／手套、出口阻擋、門、地面、堆高機、燈、管制區、無人看管物品）<textarea value={topics} onChange={(e) => setTopics(e.target.value)} style={{ minHeight: 44 }} placeholder="例如：whether the loading dock door is open" /></label>
      <label className="f">側重（選填，給 teacher 的一句話）<input value={focus} onChange={(e) => setFocus(e.target.value)} placeholder="例如 night-shift factory floor" /></label>
      <label className="f">學生模型<select value={model} onChange={(e) => setModel(e.target.value)}>{models.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}</select></label>
      {err && <div className="bad hint">{err}</div>}
      <button className="primary" onClick={create} disabled={!name || !picked.length || !skills.length || !pool}>開始：選圖 → 出題 → 作答 → 對答案</button>
    </div>
  );
}

function NewManual({ models, ready, onCreated }: { models: Model[]; ready: boolean; onCreated: (q: Q) => void }) {
  const sources = useSources();
  const [name, setName] = useState("");
  const [skill, setSkill] = useState<Skill>("detect");
  const [prompts, setPrompts] = useState("person");
  const [model, setModel] = useState("base");
  const [judge, setJudge] = useState(true);
  const [source, setSource] = useState("");
  const [err, setErr] = useState("");
  useEffect(() => { if (!source && sources.length) setSource(sources[0].source); }, [sources]); // eslint-disable-line react-hooks/exhaustive-deps
  const n = sources.find((s) => s.source === source)?.n ?? 0;
  const k = prompts.split(/[,\n]/).filter((s) => s.trim()).length;
  const create = async () => {
    try { onCreated(await api.createQueue({ name, skill, prompts: prompts.split(/[,\n]/).map((s) => s.trim()).filter(Boolean), model, source, judge })); } catch (e) { setErr(String(e)); }
  };
  return (
    <div className="col card">
      {!ready && <div className="warn hint">權重未就緒：佇列會建立，但預測會失敗。</div>}
      <label className="f">名稱<input placeholder="例如 cctv-detect-1025" value={name} onChange={(e) => setName(e.target.value)} /></label>
      <label className="f">圖片來源<select value={source} onChange={(e) => setSource(e.target.value)}>{sources.map((s) => <option key={s.source} value={s.source}>{sourceLabel(s.source)}（{s.n} 張）</option>)}</select></label>
      <label className="f">Skill<select value={skill} onChange={(e) => { setSkill(e.target.value as Skill); setPrompts(defaultPrompt(e.target.value as Skill)); }}>{(["detect", "point", "query", "caption"] as Skill[]).map((s) => <option key={s} value={s}>{skillLabel[s]}</option>)}</select></label>
      <label className="f">{promptLabel(skill)}（逗號或換行分隔，每張圖各一題）<textarea value={prompts} onChange={(e) => setPrompts(e.target.value)} style={{ minHeight: 48 }} /></label>
      <label className="f">模型<select value={model} onChange={(e) => setModel(e.target.value)}>{models.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}</select></label>
      <label className="row hint"><input type="checkbox" checked={judge} onChange={(e) => setJudge(e.target.checked)} /> 預測後先由 judge 審，人只看標紅的和 10% 抽查</label>
      {err && <div className="bad hint">{err}</div>}
      <button className="primary" onClick={create} disabled={!name || !k || !n}>建立並預測（{n * k} 題）</button>
      {source === "heldout" && <div className="warn hint">held-out 只能用來做 gold 或評估，匯出的列不能進訓練。</div>}
    </div>
  );
}

function ExportBox({ q }: { q: Q }) {
  const [res, setRes] = useState<string>("");
  const n = (q.counts.approved ?? 0) + (q.counts.corrected ?? 0) + (q.counts.auto ?? 0);
  return (
    <div className="col">
      <button onClick={async () => { try { const r = await api.export(q.name, [q.id], true) as { train: { path: string; rows: number }; incoming?: { path: string } }; setRes(`${r.train.rows} 列 → ${r.train.path}${r.incoming ? `\nData Core 包 → ${r.incoming.path}` : ""}`); } catch (e) { setRes(String(e)); } }} disabled={!n}>匯出 {n} 筆（人審 + judge 通過）</button>
      {res && <pre className="hint" style={{ whiteSpace: "pre-wrap", margin: 0 }}>{res}</pre>}
    </div>
  );
}
