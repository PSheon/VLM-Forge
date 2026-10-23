// ② 檢查: one image, one question at a time. Every key press saves and moves on.
//
//   Enter / Space  存這題（照畫面上的答案）並到下一題        D  丟棄這題
//   1 用預標   2 用模型   E 改答案   Z 還原                   ←/→  上一題／下一題
//   Shift+Enter  這張圖剩下的題全部照畫面存，到下一張        N  跳過這張
//   G 顯示／隱藏所有圖                                       ?  快捷鍵說明
//
// Items the model disagreed on come first (●); agreed ones (○) only need Enter.

import { useCallback, useEffect, useRef, useState } from "react";
import { api, emptyLabel, judgeText, skillLabel, summary, type ImageReview, type ImageSummary, type Item, type Pred, type Summary } from "../api";
import Canvas from "../components/Canvas";

const KEYS = [
  ["Enter / Space", "存這題，到下一題"],
  ["Shift+Enter", "這張剩下的題全部存，下一張"],
  ["D", "丟棄這題"],
  ["1 / 2", "用預標 / 用模型的答案"],
  ["E", "改答案（文字題）"],
  ["Z", "還原這題"],
  ["← / →", "上一題 / 下一題"],
  ["N", "跳過這張圖"],
  ["滾輪 · 空白鍵拖曳", "縮放 · 移動"],
  ["拖曳 · 拖角 · Delete", "畫框 · 縮放框 · 刪框"],
  ["G", "顯示所有圖"],
  ["?", "這個說明"],
];

export default function Review({ reviewer, sum, refresh, go }: { reviewer: string; sum: Summary | null; refresh: () => void; go: () => void }) {
  const [review, setReview] = useState<ImageReview | null>(null);
  const [idx, setIdx] = useState(0);
  const [label, setLabel] = useState<Pred | null>(null);
  const [dirty, setDirty] = useState(false);
  const [toast, setToast] = useState<{ text: string; kind?: "ok" | "warn" } | null>(null);
  const [grid, setGrid] = useState(false);
  const [help, setHelp] = useState(false);
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [msg, setMsg] = useState("");
  const started = useRef(false);
  const textRef = useRef<HTMLTextAreaElement>(null);
  const toastTimer = useRef<number | null>(null);

  const say = (text: string, kind: "ok" | "warn" = "ok") => {
    setToast({ text, kind });
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 1800);
  };

  const items = review?.items ?? [];
  const it = items[idx] ?? null;
  const initial = (x: Item): Pred => structuredClone(x.label ?? x.pred ?? x.teacher) ?? emptyLabel(x.skill);
  const spatial = !!it && (it.skill === "detect" || it.skill === "point");
  const pendingLeft = items.filter((x) => x.status === "pending").length;

  const open = useCallback((r: ImageReview, at = 0) => {
    const order = [...r.items].sort((a, b) => rank(a) - rank(b) || a.id - b.id);
    const first = Math.max(0, order.findIndex((x, i) => i >= at && x.status === "pending"));
    setReview({ ...r, items: order }); setIdx(first); setLabel(initial(order[first] ?? order[0])); setDirty(false); setMsg("");
  }, []);
  const nextImage = useCallback(async (after?: string) => {
    started.current = true;
    const r = await api.nextImage(reviewer, after);
    if (!("image" in r)) { setReview(null); setMsg("都看完了"); refresh(); return; }
    open(r as ImageReview);
  }, [reviewer, refresh, open]);
  useEffect(() => { if (!started.current && sum && sum.images_pending > 0) nextImage(); }, [sum?.images_pending]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (grid) api.reviewImages("pending").then(setImages).catch(() => undefined); }, [grid, sum?.pending]);

  const goTo = (i: number) => { if (!items.length) return; const k = Math.max(0, Math.min(items.length - 1, i)); setIdx(k); setLabel(initial(items[k])); setDirty(false); };
  const advance = (saved: Item[]) => {
    const nextItems = items.map((x) => saved.find((s) => s.id === x.id) ?? x);
    setReview((r) => (r ? { ...r, items: nextItems } : r));
    const k = nextItems.findIndex((x, i) => i > idx && x.status === "pending");
    const k2 = k >= 0 ? k : nextItems.findIndex((x) => x.status === "pending");
    if (k2 >= 0) { setIdx(k2); setLabel(initial(nextItems[k2])); setDirty(false); }
    else { say("這張看完了 ✓"); refresh(); nextImage(review!.image.id); }
  };

  const accept = useCallback(async () => {
    if (!it || !review) return;
    const final = label ?? emptyLabel(it.skill);
    const changed = JSON.stringify(final) !== JSON.stringify(it.pred);
    try {
      const saved = await api.review(it.id, changed ? "correct" : "approve", reviewer, final) as unknown as Item;
      say(changed ? "已存（你的修正）✓" : "已存 ✓");
      advance([{ ...it, ...saved }]);
    } catch (e) { say(String(e), "warn"); }
  }, [it, review, label, reviewer]); // eslint-disable-line react-hooks/exhaustive-deps

  const discard = useCallback(async () => {
    if (!it) return;
    try { const saved = await api.review(it.id, "discard", reviewer, null) as unknown as Item; say("已丟棄", "warn"); advance([{ ...it, ...saved }]); } catch (e) { say(String(e), "warn"); }
  }, [it, reviewer]); // eslint-disable-line react-hooks/exhaustive-deps

  const confirmRest = useCallback(async () => {
    if (!review) return;
    const rest = items.filter((x) => x.status === "pending").map((x) => x.id === it?.id ? { id: x.id, action: JSON.stringify(label) !== JSON.stringify(x.pred) ? "correct" : "approve", label: label ?? emptyLabel(x.skill) } : { id: x.id, action: "approve" });
    try { await api.submitImage(reviewer, rest); say(`這張 ${rest.length} 題全部存了 ✓`); refresh(); nextImage(review.image.id); } catch (e) { say(String(e), "warn"); }
  }, [review, items, it, label, reviewer, refresh, nextImage]);

  const edit = (p: Pred, what: string) => { setLabel(p); setDirty(true); say(`${what}（還沒存，Enter 存）`, "warn"); };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT");
      if (typing) { if (e.key === "Escape") (t as HTMLElement).blur(); if (e.key === "Enter" && !e.shiftKey && t.tagName === "TEXTAREA") { e.preventDefault(); (t as HTMLElement).blur(); accept(); } return; }
      if (!review) return;
      if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); confirmRest(); }
      else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); accept(); }
      else if (e.key === "d" || e.key === "D") discard();
      else if (e.key === "1" && it?.teacher) edit(structuredClone(it.teacher), "改用預標");
      else if (e.key === "2" && it?.pred) edit(structuredClone(it.pred), "改用模型答案");
      else if ((e.key === "e" || e.key === "E") && !spatial) { e.preventDefault(); textRef.current?.focus(); }
      else if ((e.key === "z" || e.key === "Z") && it) { setLabel(initial(it)); setDirty(false); say("已還原"); }
      else if (e.key === "ArrowRight" || e.key === "j") goTo(idx + 1);
      else if (e.key === "ArrowLeft" || e.key === "k") goTo(idx - 1);
      else if (e.key === "n" || e.key === "N") nextImage(review.image.id);
      else if (e.key === "g" || e.key === "G") setGrid((g) => !g);
      else if (e.key === "?") setHelp((h) => !h);
      else if (e.key === "Escape") { setHelp(false); setGrid(false); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }); // closures over the current item

  if (sum && sum.items === 0) return <div className="empty">還沒有資料。先到 ① 資料 加資料夾、自動出題。</div>;
  if (!review || !it) return <div className="empty">{msg === "都看完了" ? <>都看完了。<button className="primary" onClick={go} style={{ marginLeft: 8 }}>去訓練 →</button></> : msg || (sum?.images_pending ? "載入中…" : "沒有待看的圖。")}</div>;

  const changed = JSON.stringify(label) !== JSON.stringify(it.pred);
  return (
    <div className="rv">
      <div className="rv-top row">
        <span className="muted hint">{review.image.path.split("/").slice(-1)[0]}</span>
        <span className="strip">{items.map((x, i) => <i key={x.id} className={`${i === idx ? "cur " : ""}${x.status !== "pending" ? "done" : x.verdict === "fix" ? "fix" : x.verdict === "accept" ? "acc" : "none"}`} title={title(x)} onClick={() => goTo(i)} />)}</span>
        <span className="muted hint">題 {idx + 1}/{items.length} · 還有 {pendingLeft} 題 · 待看 {sum?.images_pending ?? 0} 張</span>
        <span style={{ marginLeft: "auto" }} className="row">
          <button onClick={() => setGrid(!grid)} title="G">所有圖</button>
          <button onClick={() => setHelp(!help)} title="?">快捷鍵</button>
        </span>
      </div>
      <div className="rv-main">
        <Canvas src={api.imageUrl(review.image.id, 2048)} skill={it.skill} value={label} onChange={(p) => edit(p, it.skill === "detect" ? "框已改" : "點已改")} alt={spatial ? it.teacher : null} labels={spatial && it.teacher ? { main: "目前", alt: "預標" } : undefined} height="calc(100vh - 150px)" compact />
        <aside className="rv-side">
          <div className={`q ${it.verdict === "fix" ? "q-fix" : it.verdict === "accept" ? "q-acc" : ""}`}>
            <div className="muted hint">{it.verdict === "fix" ? "● 不一致，請你看" : it.verdict === "accept" ? "○ 一致" : "– 未比對"}{it.spot ? " · 抽查" : ""}</div>
            <div className="q-title">{title(it)}</div>
            {it.judge && judgeText(it.judge) && <div className="hint muted">{judgeText(it.judge)}</div>}
          </div>
          {spatial ? (
            <div className="hint muted">目前 {summary(label)} · 預標 {summary(it.teacher)} · 模型 {summary(it.pred)}{changed ? " · ✎ 已改" : ""}</div>
          ) : (
            <>
              <textarea ref={textRef} value={label?.text ?? ""} onChange={(e) => { setLabel({ text: e.target.value }); setDirty(true); }} style={{ minHeight: 64 }} placeholder="答案" />
              <div className="row">
                {it.teacher && <button onClick={() => edit(structuredClone(it.teacher!), "改用預標")} title={it.teacher.text}><kbd>1</kbd> 預標：{short(it.teacher.text)}</button>}
                {it.pred && <button onClick={() => edit(structuredClone(it.pred!), "改用模型答案")} title={it.pred.text}><kbd>2</kbd> 模型：{short(it.pred.text)}</button>}
              </div>
            </>
          )}
          {dirty && <div className="warn hint">✎ 改過了，還沒存。<kbd>Enter</kbd> 存，<kbd>Z</kbd> 還原</div>}
          <div className="row">
            <button className="primary big" onClick={accept}>{changed ? "存修正" : "沒問題"} <kbd>Enter</kbd></button>
            <button onClick={discard}>丟棄 <kbd>D</kbd></button>
          </div>
          <div className="row hint muted">
            <button onClick={confirmRest} title="Shift+Enter">剩下 {pendingLeft} 題全照畫面存</button>
            <button onClick={() => nextImage(review.image.id)} title="N">跳過這張</button>
          </div>
        </aside>
      </div>
      {toast && <div className={`toast ${toast.kind ?? "ok"}`}>{toast.text}</div>}
      {help && (
        <div className="overlay" onClick={() => setHelp(false)}>
          <div className="card col" onClick={(e) => e.stopPropagation()}>
            <strong>快捷鍵</strong>
            <table><tbody>{KEYS.map(([k, v]) => <tr key={k}><td><kbd>{k}</kbd></td><td>{v}</td></tr>)}</tbody></table>
          </div>
        </div>
      )}
      {grid && (
        <div className="overlay" onClick={() => setGrid(false)}>
          <div className="card col" style={{ width: "min(1100px, 92vw)" }} onClick={(e) => e.stopPropagation()}>
            <div className="row"><strong>待看的圖（{images.length}）</strong><span className="muted hint">數字＝不一致的題數；點一張開始看</span></div>
            <div className="thumbs" style={{ maxHeight: "70vh", gridTemplateColumns: "repeat(auto-fill, minmax(110px, 1fr))" }}>
              {images.map((im) => (
                <div key={im.image_id} className="th" onClick={async () => { open(await api.reviewImage(im.image_id)); setGrid(false); }}>
                  <img src={api.imageUrl(im.image_id, 200)} className={review.image.id === im.image_id ? "cur" : ""} loading="lazy" />
                  <b className={im.fix > 0 ? "fix" : ""}>{im.fix > 0 ? im.fix : im.spot ? "抽" : im.n}</b>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const rank = (it: Item) => (it.status === "pending" && it.verdict === "fix" ? 0 : it.status === "pending" ? 1 : 2);
const title = (it: Item) => (it.skill === "query" ? it.prompt : it.skill === "caption" ? "寫一段描述（caption）" : `${skillLabel[it.skill]}：${it.prompt}`);
const short = (t?: string) => ((t ?? "").length > 22 ? (t ?? "").slice(0, 22) + "…" : t ?? "");
