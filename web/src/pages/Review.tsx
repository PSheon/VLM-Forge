// ② 檢查 (docs/ui-spec §4): one image, one question at a time; every key saves and moves on.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, emptyLabel, itemTitle, judgeText, summary, type Dataset, type ImageReview, type ImageSummary, type Item, type Pred } from "../api";
import Canvas from "../components/Canvas";
import { Dialog, Pill, useToast } from "../components/ui";
import { Fade, Pop } from "../components/motion";
import type { Step } from "../App";

export default function Review({ reviewer, datasetId, openImage, onOpened, refresh, go, setGuard }: { reviewer: string; datasetId: number; openImage?: string | null; onOpened?: () => void; refresh: () => void; go: (s: Step) => void; setGuard: (g: (() => boolean) | null) => void }) {
  const { say } = useToast();
  const dsId = datasetId;
  const [sum, setSum] = useState<Dataset | null>(null);
  const [review, setReview] = useState<ImageReview | null>(null);
  const [idx, setIdx] = useState(0);
  const [label, setLabel] = useState<Pred | null>(null);
  const [dirty, setDirty] = useState(false);
  const [grid, setGrid] = useState(false);
  const [gridFilter, setGridFilter] = useState("pending");
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [msg, setMsg] = useState("");
  const [pendingJump, setPendingJump] = useState<(() => void) | null>(null);
  const started = useRef(false);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const dirtyRef = useRef(false);
  dirtyRef.current = dirty;

  const loadSum = useCallback(() => api.dataset(dsId).then(setSum).catch(() => undefined), [dsId]);
  useEffect(() => { loadSum(); const t = setInterval(loadSum, 5000); return () => clearInterval(t); }, [loadSum]);
  useEffect(() => {
    setGuard(() => { if (!dirtyRef.current) return true; say("有改動還沒存：Enter 存，或 Z 還原", "warn"); return false; });
    return () => setGuard(null);
  }, [setGuard, say]);

  const items = review?.items ?? [];
  const it = items[idx] ?? null;
  const initial = (x: Item): Pred => structuredClone(x.label ?? x.pred ?? x.teacher) ?? emptyLabel(x.skill);
  const spatial = !!it && (it.skill === "detect" || it.skill === "point");
  const pendingLeft = items.filter((x) => x.status === "pending").length;

  const open = useCallback((r: ImageReview) => {
    const order = [...r.items].sort((a, b) => rank(a) - rank(b) || a.id - b.id);
    const first = Math.max(0, order.findIndex((x) => x.status === "pending"));
    setReview({ ...r, items: order }); setIdx(first); setLabel(initial(order[first] ?? order[0])); setDirty(false); setMsg("");
  }, []);
  const nextImage = useCallback(async (after?: string) => {
    started.current = true;
    try {
      const r = await api.nextImage(reviewer, after, dsId);
      if (!("image" in r)) { setReview(null); setMsg("都看完了"); refresh(); return; }
      open(r as ImageReview);
    } catch (e) { setReview(null); setMsg(`取圖失敗：${String(e)}`); say(String(e), "bad"); }
  }, [reviewer, dsId, refresh, open, say]);
  useEffect(() => {
    if (openImage) { started.current = true; api.reviewImage(openImage).then((r) => { open(r); onOpened?.(); }).catch((e) => say(String(e), "bad")); return; }
    if (!started.current && sum && sum.images_pending > 0) nextImage();
  }, [sum?.images_pending, openImage]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (grid) api.reviewImages(gridFilter, dsId).then(setImages).catch(() => undefined); }, [grid, gridFilter, dsId, sum?.pending]);

  const guarded = (fn: () => void) => { if (dirty) setPendingJump(() => fn); else fn(); };
  const goTo = (i: number) => guarded(() => { if (!items.length) return; const k = Math.max(0, Math.min(items.length - 1, i)); setIdx(k); setLabel(initial(items[k])); setDirty(false); });
  const advance = (saved: Item[]) => {
    const nextItems = items.map((x) => saved.find((s) => s.id === x.id) ?? x);
    setReview((r) => (r ? { ...r, items: nextItems } : r));
    const k = nextItems.findIndex((x, i) => i > idx && x.status === "pending");
    const k2 = k >= 0 ? k : nextItems.findIndex((x) => x.status === "pending");
    if (k2 >= 0) { setIdx(k2); setLabel(initial(nextItems[k2])); setDirty(false); }
    else { say("這張看完了 ✓", "ok", { ms: 1200 }); refresh(); setTimeout(() => nextImage(review!.image.id), 300); }
  };

  const acceptWith = useCallback(async (p?: Pred) => {
    if (!it || !review) return;
    const final = p ?? label ?? emptyLabel(it.skill);
    const changed = JSON.stringify(final) !== JSON.stringify(it.pred);
    try { const saved = await api.review(it.id, changed ? "correct" : "approve", reviewer, final); say(changed ? "已存（你的修正）✓" : "已存 ✓"); advance([{ ...it, ...saved }]); } catch (e) { say(String(e), "bad"); }
  }, [it, review, label, reviewer]); // eslint-disable-line react-hooks/exhaustive-deps
  const accept = useCallback(() => acceptWith(), [acceptWith]);
  // question type decides the answer widget: yes/no → two keys, count → a number, else text
  const qt = it?.skill === "query" ? (/how many|count|number of/i.test(it.prompt) ? "count" : /^(is|are|does|do|can|has|have|was|were)\b/i.test(it.prompt) ? "yesno" : "text") : null;
  const setCount = (v: number) => { const n = Math.max(0, Math.round(v)); setLabel({ text: String(n) }); setDirty(true); };
  const [pressed, setPressed] = useState("");
  const focus = useMemo(() => {
    if (!it || (it.skill !== "detect" && it.skill !== "point")) return null;
    const pb = (p: number[]) => [p[0] - 0.02, p[1] - 0.02, p[0] + 0.02, p[1] + 0.02];
    const b = [...(it.label?.boxes ?? it.pred?.boxes ?? []), ...(it.teacher?.boxes ?? []), ...(it.label?.points ?? it.pred?.points ?? []).map(pb), ...(it.teacher?.points ?? []).map(pb)];
    return b.length ? b : null;
  }, [it?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const discard = useCallback(async () => {
    if (!it) return;
    try { const saved = await api.review(it.id, "discard", reviewer, null); say("已丟棄", "warn"); advance([{ ...it, ...saved }]); } catch (e) { say(String(e), "bad"); }
  }, [it, reviewer]); // eslint-disable-line react-hooks/exhaustive-deps
  const confirmRest = useCallback(async () => {
    if (!review) return;
    const rest = items.filter((x) => x.status === "pending").map((x) => (x.id === it?.id ? { id: x.id, action: JSON.stringify(label) !== JSON.stringify(x.pred) ? "correct" : "approve", label: label ?? emptyLabel(x.skill) } : { id: x.id, action: "approve" }));
    const ids = rest.map((r) => r.id);
    try {
      await api.submitImage(reviewer, rest);
      say(`這張 ${rest.length} 題全部存了 ✓`, "ok", { action: { label: "復原", run: async () => { await Promise.all(ids.map((id) => api.reopen(id))); say("已復原，這張回到待看", "warn"); refresh(); open(await api.reviewImage(review.image.id)); } } });
      refresh(); nextImage(review.image.id);
    } catch (e) { say(String(e), "bad"); }
  }, [review, items, it, label, reviewer, refresh, nextImage, open, say]);
  const edit = (p: Pred, what: string) => { setLabel(p); setDirty(true); say(`${what}（還沒存，Enter 存）`, "warn"); };
  const removePhrase = (ph: string) => { const t = (label?.text ?? "").replace(ph, "").replace(/\s{2,}/g, " ").replace(/\s+([,.;])/g, "$1").trim(); edit({ text: t }, `刪掉「${ph}」`); };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
      if (typing) { if (e.key === "Escape") t.blur(); if (e.key === "Enter" && !e.shiftKey && (t.tagName === "TEXTAREA" || t.classList.contains("count"))) { e.preventDefault(); t.blur(); accept(); } return; }
      if (!review || pendingJump) return;
      if (!e.metaKey && !e.ctrlKey) { setPressed(e.key.length === 1 ? e.key.toUpperCase() : e.key); window.setTimeout(() => setPressed(""), 260); }
      if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); confirmRest(); }
      else if (e.key === "Enter") { e.preventDefault(); accept(); }  // Space is the canvas pan key
      else if (qt === "yesno" && (e.key === "y" || e.key === "Y")) acceptWith({ text: "yes" });
      else if (qt === "yesno" && (e.key === "n" || e.key === "N")) acceptWith({ text: "no" });
      else if (qt === "count" && /^[0-9]$/.test(e.key)) { const cur = label?.text ?? ""; setCount(Number((/^\d+$/.test(cur) && pressed ? cur : "") + e.key)); }
      else if (qt === "count" && (e.key === "ArrowUp" || e.key === "ArrowDown")) { e.preventDefault(); setCount(Number(label?.text ?? 0) + (e.key === "ArrowUp" ? 1 : -1)); }
      else if (e.key === "d" || e.key === "D") discard();
      else if (e.key === "1" && it?.teacher) edit(structuredClone(it.teacher), "改用預標");
      else if (e.key === "2" && it?.pred) edit(structuredClone(it.pred), "改用模型答案");
      else if ((e.key === "e" || e.key === "E") && !spatial) { e.preventDefault(); textRef.current?.focus(); }
      else if ((e.key === "z" || e.key === "Z") && it) { setLabel(initial(it)); setDirty(false); say("已還原"); }
      else if (e.key === "ArrowRight" && !e.shiftKey && !spatialSelected()) goTo(idx + 1);
      else if (e.key === "ArrowLeft" && !e.shiftKey && !spatialSelected()) goTo(idx - 1);
      else if (e.key === "s" || e.key === "S") guarded(() => nextImage(review.image.id));
      else if (e.key === "g" || e.key === "G") setGrid((g) => !g);
      else if (e.key === "Escape") setGrid(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  const spatialSelected = () => !!document.querySelector(".cv-stage .sel"); // arrows nudge a selected box instead

  if (sum && sum.items === 0) return <div className="empty">這個資料集還沒有題。到「圖片」分頁加照片、按「標註」。</div>;
  if (!review || !it) {
    return <div className="empty">{msg === "都看完了" ? <>這個資料集都看完了（已確認 {sum?.trainable ?? 0} 題）。<button className="primary" onClick={() => go("train")} style={{ marginLeft: 8 }}>去訓練 →</button></> : msg ? <span className="row" style={{ justifyContent: "center" }}><span className="bad">{msg}</span><button onClick={() => nextImage()}>重試</button></span> : sum?.images_pending ? <span className="row" style={{ justifyContent: "center" }}><i className="spin" />載入中…</span> : "沒有待看的圖。到「圖片」分頁點任一張可以重看。"}</div>;
  }
  const changed = JSON.stringify(label) !== JSON.stringify(it.pred);
  const unsupported = it.skill === "caption" ? (it.judge?.unsupported ?? []).filter((p) => (label?.text ?? "").includes(p)) : [];
  const negative = it.skill === "detect" && (label?.boxes?.length ?? 0) === 0 && ((it.pred?.boxes?.length ?? 0) > 0 || (it.teacher?.boxes?.length ?? 0) > 0);
  const claimedOther = items.find((x) => x.claimed_by && x.claimed_by !== reviewer)?.claimed_by;
  const fmtHint = it.skill === "query" ? (/how many|count/i.test(it.prompt) ? "數量題：用數字作答" : /^(is|are|does|do|can|has|have)\b/i.test(it.prompt) ? "是非題：yes / no" : "") : "";
  return (
    <div className="rv">
      <div className="rv-top row">
        <span className="muted hint">{review.image.path.split("/").slice(-1)[0]}</span>
        <Pill kind={pendingLeft ? "warn" : "ok"}>這張還有 {pendingLeft} 題</Pill>
        <span className="muted hint">待看 {sum?.images_pending ?? 0} 張</span>
        <span style={{ marginLeft: "auto" }} className="row"><button onClick={() => setGrid(!grid)} title="G">所有圖</button></span>
      </div>
      <div className="page">
        <div className="col" style={{ minWidth: 0 }}>
          <Fade k={it.id} y={4}>
            <div className={`q ${it.verdict === "fix" ? "q-fix" : it.verdict === "accept" ? "q-acc" : ""}`}>
              <div className="row" style={{ gap: 12 }}>
                <span className="q-title">{itemTitle(it)}</span>
                <span className="muted hint">{it.verdict === "fix" ? "● 不一致，請你看" : it.verdict === "accept" ? "○ 一致" : "– 未比對（模型沒作答）"}{it.spot ? " · 抽查" : ""}{it.status !== "pending" ? ` · ${statusLabel[it.status]}${it.reviewer ? `（${it.reviewer}）` : ""}` : ""}</span>
                {it.judge && judgeText(it.judge) && <span className="hint muted">{judgeText(it.judge)}</span>}
                {fmtHint && <span className="hint muted">{fmtHint}</span>}
              </div>
            </div>
          </Fade>
          <Canvas src={api.imageUrl(review.image.id, 2048)} skill={it.skill} value={label} onChange={(p, what) => edit(p, what)} alt={spatial ? it.teacher : null} labels={spatial ? { main: "目前（可改）", alt: "預標" } : undefined} fitAspect={{ maxOffset: 270 }} focus={focus} focusKey={it.id} compact />
        </div>
        <Fade k={it.id} className="actions" y={10}>
          <div className="row">
            <button className="primary big" onClick={accept}>{changed ? "存修正" : "沒問題"} <kbd>Enter</kbd></button>
            <button onClick={discard}>丟棄 <kbd>D</kbd></button>
          </div>
          {spatial ? (
            <>
              <div className="hint muted">目前 {summary(label)} · 預標 {summary(it.teacher)} · 模型 {summary(it.pred)}{changed ? " · ✎ 已改" : ""}</div>
              {negative && <div className="warn hint">確認這張沒有 {it.prompt}？（預標或模型有框，你全刪了）</div>}
              <div className="row">{it.teacher && <button onClick={() => edit(structuredClone(it.teacher!), "改用預標")}><kbd>1</kbd> 用預標的{it.skill === "detect" ? "框" : "點"}</button>}{it.pred && <button onClick={() => edit(structuredClone(it.pred!), "改用模型答案")}><kbd>2</kbd> 用模型的{it.skill === "detect" ? "框" : "點"}</button>}</div>
            </>
          ) : (
            <>
              {qt === "yesno" && (
                <div className="yn">
                  <button className={`big${(label?.text ?? "").toLowerCase().startsWith("yes") ? " cur" : ""}`} onClick={() => acceptWith({ text: "yes" })}><kbd>Y</kbd> yes 是</button>
                  <button className={`big${(label?.text ?? "").toLowerCase().startsWith("no") ? " cur" : ""}`} onClick={() => acceptWith({ text: "no" })}><kbd>N</kbd> no 否</button>
                </div>
              )}
              {qt === "count" && (
                <div className="row"><button onClick={() => setCount(Number(label?.text ?? 0) - 1)} title="↓">−</button><input className="count" type="number" min={0} value={/^\d+$/.test(label?.text ?? "") ? label!.text : ""} placeholder="?" title={label?.text} onChange={(e) => setCount(Number(e.target.value))} /><button onClick={() => setCount(Number(label?.text ?? 0) + 1)} title="↑">＋</button><span className="muted hint">按數字、↑↓、Enter 存</span></div>
              )}
              {qt !== "yesno" && qt !== "count" && <textarea ref={textRef} value={label?.text ?? ""} onChange={(e) => { setLabel({ text: e.target.value }); setDirty(true); }} style={{ minHeight: it.skill === "caption" ? 110 : 64, fontSize: it.skill === "caption" ? 14 : 16 }} placeholder="答案" />}
              {it.skill === "caption" && unsupported.length > 0 && <div className="row hint"><span className="muted">圖裡沒有：</span>{unsupported.map((p) => <span key={p} className="chip bad" title="點 ✕ 從文字刪掉">{p}<button onClick={() => removePhrase(p)} aria-label="刪掉">✕</button></span>)}</div>}
              <div className="row">
                {it.teacher && <button onClick={() => edit(structuredClone(it.teacher!), "改用預標")} title={it.teacher.text}><kbd>1</kbd> 預標：{short(it.teacher.text)}</button>}
                {it.pred && <button onClick={() => edit(structuredClone(it.pred!), "改用模型答案")} title={it.pred.text}><kbd>2</kbd> 模型：{short(it.pred.text)}</button>}
              </div>
            </>
          )}
          <Pop show={dirty} className="warn hint">✎ 改過了，還沒存。<kbd>Enter</kbd> 存，<kbd>Z</kbd> 還原</Pop>
          <div className="row hint muted">
            <button onClick={confirmRest} title="Shift+Enter">剩下 {pendingLeft} 題全照畫面存</button>
            <button onClick={() => guarded(() => nextImage(review.image.id))} title="S">跳過這張</button>
          </div>
          <div className="qlist">
            {items.map((x, i) => <div key={x.id} className={`it${i === idx ? " cur" : ""}${x.status !== "pending" ? " done" : ""}`} onClick={() => goTo(i)} title={itemTitle(x)}><i className={`dot ${x.status !== "pending" ? "done" : x.verdict === "fix" ? "fix" : x.verdict === "accept" ? "acc" : "none"}`} /><span className="t">{itemTitle(x)}</span><span className="muted small">{x.status !== "pending" ? statusLabel[x.status] : x.verdict === "fix" ? "不一致" : x.verdict === "accept" ? (x.spot ? "抽查" : "一致") : "未比對"}</span></div>)}
          </div>
          <div className="hint muted">{reviewer} 認領中{claimedOther ? `（${claimedOther} 也在看）` : ""}</div>
        </Fade>
      </div>
      <KeyBar pressed={pressed} keys={[["Enter", changed ? "存修正" : "沒問題"], ...(qt === "yesno" ? [["Y", "yes"], ["N", "no"]] as [string, string][] : qt === "count" ? [["0–9", "數字"], ["↑↓", "加減"]] as [string, string][] : []), ["D", "丟棄"], ...(it.teacher ? [["1", "用預標"]] as [string, string][] : []), ...(it.pred ? [["2", "用模型"]] as [string, string][] : []), ...(spatial ? [["Tab", "選框"], ["Del", "刪框"], ["F", "全圖"]] as [string, string][] : qt === "text" || it.skill === "caption" ? [["E", "改文字"]] as [string, string][] : []), ["←→", "換題"], ["Shift+Enter", `剩 ${pendingLeft} 題全存`], ["S", "跳過"], ["G", "所有圖"]]} />
      <Dialog open={!!pendingJump} onClose={() => setPendingJump(null)}>
        <h2>有改動還沒存</h2>
        <div className="hint">要丟掉這題的改動嗎？</div>
        <div className="row" style={{ justifyContent: "flex-end" }}><button onClick={() => setPendingJump(null)}>留下</button><button className="danger" onClick={() => { const f = pendingJump!; setPendingJump(null); setDirty(false); f(); }}>丟掉</button></div>
      </Dialog>
      {grid && (
        <div className="overlay" onClick={() => setGrid(false)}>
          <div className="modal" style={{ width: "min(1100px, 92vw)" }} onClick={(e) => e.stopPropagation()}>
            <div className="row"><strong>圖（{images.length}）</strong>
              <select value={gridFilter} onChange={(e) => setGridFilter(e.target.value)}><option value="pending">待看</option><option value="auto">自動通過</option><option value="reviewed">已看過</option><option value="all">全部</option></select>
              <span className="muted hint">數字＝不一致的題數；灰＝別人認領中；點一張開始看</span></div>
            <div className="thumbs" style={{ maxHeight: "70vh", gridTemplateColumns: "repeat(auto-fill, minmax(110px, 1fr))" }}>
              {images.map((im) => (
                <div key={im.image_id} className={`th${im.claimed_by && im.claimed_by !== reviewer ? " taken" : ""}`} onClick={async () => { open(await api.reviewImage(im.image_id)); setGrid(false); }} title={im.claimed_by && im.claimed_by !== reviewer ? `${im.claimed_by} 認領中` : `${im.n} 題 · 不一致 ${im.fix}`}>
                  <img src={api.imageUrl(im.image_id, 200)} className={review.image.id === im.image_id ? "cur" : ""} loading="lazy" />
                  <b className={im.fix > 0 ? "fix" : im.pending > 0 ? "" : "approved"}>{im.fix > 0 ? im.fix : im.pending > 0 ? (im.spot ? "抽" : im.n) : "✓"}</b>
                  {im.claimed_by && im.claimed_by !== reviewer && <span className="who-tag">{im.claimed_by}</span>}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/** The keys that work right now, one line at the bottom; the one just pressed lights up. */
function KeyBar({ keys, pressed }: { keys: [string, string][]; pressed: string }) {
  const on = (k: string) => pressed && (k === pressed || k.split("–").includes(pressed) || (k === "←→" && (pressed === "ArrowLeft" || pressed === "ArrowRight")) || (k === "↑↓" && (pressed === "ArrowUp" || pressed === "ArrowDown")) || (k === "0–9" && /^[0-9]$/.test(pressed)) || (k === "Del" && (pressed === "Delete" || pressed === "Backspace")));
  return <div className="keybar">{keys.map(([k, l]) => <span key={k}><kbd className={on(k) ? "on" : ""}>{k}</kbd> {l}</span>)}</div>;
}

const statusLabel: Record<Item["status"], string> = { pending: "待看", auto: "自動通過", approved: "已確認", corrected: "已修正", discarded: "丟棄" };
const rank = (it: Item) => (it.status === "pending" && it.verdict === "fix" ? 0 : it.status === "pending" ? 1 : 2);
const short = (t?: string) => ((t ?? "").length > 22 ? (t ?? "").slice(0, 22) + "…" : t ?? "");
