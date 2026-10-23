// ② 檢查: one image at a time with all its items; items the model disagreed on come
// first and are marked. Enter = confirm the whole image, D = discard the selected item,
// N = next image, ↑/↓ = select item.

import { useCallback, useEffect, useState } from "react";
import { api, emptyLabel, judgeText, skillLabel, summary, type ImageReview, type ImageSummary, type Item, type Pred, type Summary } from "../api";
import Canvas from "../components/Canvas";

type Decision = { action: "approve" | "discard"; label: Pred };

export default function Review({ reviewer, sum, refresh, go }: { reviewer: string; sum: Summary | null; refresh: () => void; go: () => void }) {
  const [review, setReview] = useState<ImageReview | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [dec, setDec] = useState<Record<number, Decision>>({});
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [filter, setFilter] = useState<"pending" | "auto" | "reviewed">("pending");
  const [msg, setMsg] = useState("");
  const [started, setStarted] = useState(false);

  const loadImages = useCallback(() => api.reviewImages(filter).then(setImages).catch(() => undefined), [filter]);
  useEffect(() => { loadImages(); }, [loadImages, sum?.pending]);

  const open = (r: ImageReview) => {
    const d: Record<number, Decision> = {};
    for (const it of r.items) d[it.id] = { action: it.status === "discarded" ? "discard" : "approve", label: structuredClone(it.label ?? it.pred ?? it.teacher) ?? emptyLabel(it.skill) };
    const order = [...r.items].sort((a, b) => rank(a) - rank(b) || a.id - b.id);
    setReview({ ...r, items: order }); setDec(d); setSel(order[0]?.id ?? null); setMsg("");
  };
  const next = useCallback(async (after?: string) => {
    setStarted(true);
    const r = await api.nextImage(reviewer, after);
    if (!("image" in r)) { setReview(null); setMsg("都看完了"); refresh(); return; }
    open(r as ImageReview);
  }, [reviewer, refresh]);
  useEffect(() => { if (!started && sum && sum.images_pending > 0) next(); }, [sum?.images_pending]); // eslint-disable-line react-hooks/exhaustive-deps

  const confirm = useCallback(async () => {
    if (!review) return;
    const items = review.items.filter((it) => it.status === "pending" || it.status === "auto").map((it) => {
      const d = dec[it.id];
      if (d.action === "discard") return { id: it.id, action: "discard" };
      return { id: it.id, action: JSON.stringify(d.label) !== JSON.stringify(it.pred) ? "correct" : "approve", label: d.label };
    });
    try { await api.submitImage(reviewer, items); refresh(); loadImages(); next(review.image.id); } catch (e) { setMsg(String(e)); }
  }, [review, dec, reviewer, refresh, loadImages, next]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (!review) return;
      if (e.key === "Enter") { e.preventDefault(); confirm(); }
      else if (e.key === "d" || e.key === "D") { if (sel !== null) setDec((d) => ({ ...d, [sel]: { ...d[sel], action: d[sel].action === "discard" ? "approve" : "discard" } })); }
      else if (e.key === "n" || e.key === "N") next(review.image.id);
      else if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const ids = review.items.map((i) => i.id);
        const k = sel === null ? 0 : ids.indexOf(sel);
        setSel(ids[Math.max(0, Math.min(ids.length - 1, k + (e.key === "ArrowDown" ? 1 : -1)))]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [review, sel, confirm, next]);

  const it = review?.items.find((i) => i.id === sel) ?? null;
  const d = sel !== null ? dec[sel] : null;
  const setLabel = (label: Pred) => { if (sel !== null) setDec((x) => ({ ...x, [sel]: { ...x[sel], label } })); };
  const spatial = it && (it.skill === "detect" || it.skill === "point");

  if (sum && sum.items === 0) return <div className="empty">還沒有資料。先到 ① 資料 加資料夾、自動出題。</div>;

  return (
    <div className="grid-review">
      <section style={{ minWidth: 0 }}>
        {review && it ? (
          <>
            <div className="row" style={{ marginBottom: 6 }}>
              <strong style={{ fontSize: 17 }}>{title(it)}</strong>
              <span className="muted hint">{review.image.path.split("/").slice(-2).join("/")}{it.teacher ? ` · 預標 ${summary(it.teacher)}` : ""}{it.pred ? ` · 模型 ${summary(it.pred)}` : ""} · 目前 {summary(d?.label)}</span>
            </div>
            {it.judge && <div className={it.verdict === "fix" ? "warn hint" : "hint ok"} style={{ marginBottom: 6 }}>{it.verdict === "fix" ? "不一致，請你看" : "一致"}{it.spot ? "（抽查）" : ""}：{judgeText(it.judge) || "—"}</div>}
            <Canvas src={api.imageUrl(review.image.id, 2048)} skill={it.skill} value={d?.label ?? null} onChange={setLabel} alt={spatial ? it.teacher : null} labels={spatial && it.teacher ? { main: "目前（可改）", alt: "預標" } : undefined} height="calc(100vh - 230px)" />
            {!spatial && (
              <div className="grid2" style={{ marginTop: 8 }}>
                <label className="f">目前答案（可改）<textarea value={d?.label.text ?? ""} onChange={(e) => setLabel({ text: e.target.value })} /></label>
                <div className="col hint">
                  {it.teacher && <div>預標：{it.teacher.text} <button onClick={() => setLabel({ text: it.teacher!.text ?? "" })}>用這個</button></div>}
                  {it.pred && <div>模型：{it.pred.text} <button onClick={() => setLabel({ text: it.pred!.text ?? "" })}>用這個</button></div>}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="empty">{msg === "都看完了" ? <>都看完了。<button className="primary" onClick={go} style={{ marginLeft: 8 }}>去訓練 →</button></> : msg || (sum?.images_pending ? "載入中…" : "沒有待看的圖。")}</div>
        )}
      </section>
      <aside className="card col">
        {review && (
          <>
            <div className="row">
              <button className="primary" onClick={confirm}>整張確認 <kbd>Enter</kbd></button>
              <button onClick={() => next(review.image.id)}>跳過 <kbd>N</kbd></button>
            </div>
            <div className="hint muted">這張圖 {review.items.length} 題。確認＝每題照目前畫面存；✕ 的丟棄。</div>
            <div className="list" style={{ maxHeight: "36vh" }}>
              {review.items.map((x) => {
                const dd = dec[x.id];
                const changed = dd && JSON.stringify(dd.label) !== JSON.stringify(x.pred);
                return (
                  <div key={x.id} className={`it${sel === x.id ? " cur" : ""}`} onClick={() => setSel(x.id)} style={{ textDecoration: dd?.action === "discard" ? "line-through" : undefined }}>
                    <span className={x.verdict === "fix" && x.status === "pending" ? "bad" : x.status === "auto" ? "muted" : "ok"}>{x.verdict === "fix" && x.status === "pending" ? "●" : "○"}</span> {title(x)}
                    <span className="muted"> — {summary(dd?.label)}{changed ? " ✎" : ""}{x.status !== "pending" ? ` · ${statusLabel[x.status]}` : ""}</span>
                    <button style={{ float: "right", padding: "0 6px" }} title="丟棄這題 (D)" onClick={(e) => { e.stopPropagation(); setDec((y) => ({ ...y, [x.id]: { ...y[x.id], action: y[x.id].action === "discard" ? "approve" : "discard" } })); }}>{dd?.action === "discard" ? "↺" : "✕"}</button>
                  </div>
                );
              })}
            </div>
            {msg && msg !== "都看完了" && <div className="bad hint">{msg}</div>}
          </>
        )}
        <div className="row"><strong>圖</strong>
          <select value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)}>
            <option value="pending">待看（{sum?.images_pending ?? 0}，其中不一致 {sum?.images_fix ?? 0}）</option>
            <option value="auto">自動通過</option>
            <option value="reviewed">已看過</option>
          </select>
        </div>
        <div className="thumbs" style={{ maxHeight: review ? "calc(100vh - 520px)" : "calc(100vh - 220px)" }}>
          {images.map((im) => (
            <div key={im.image_id} className="th" onClick={async () => open(await api.reviewImage(im.image_id))} title={`${im.n} 題 · 不一致 ${im.fix}`}>
              <img src={api.imageUrl(im.image_id, 200)} className={review?.image.id === im.image_id ? "cur" : ""} loading="lazy" />
              <b className={im.fix > 0 ? "fix" : im.pending > 0 ? "" : "approved"}>{im.fix > 0 ? `${im.fix}` : im.pending > 0 ? (im.spot ? "抽" : `${im.n}`) : "✓"}</b>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}

const statusLabel: Record<Item["status"], string> = { pending: "待看", auto: "自動通過", approved: "已確認", corrected: "已修正", discarded: "丟棄" };
const rank = (it: Item) => (it.status === "pending" && it.verdict === "fix" ? 0 : it.status === "pending" ? 1 : 2);
const title = (it: Item) => (it.skill === "query" ? it.prompt : it.skill === "caption" ? "Caption" : `${skillLabel[it.skill]}：${it.prompt}`);
