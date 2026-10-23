// Image viewer / editor with zoom and pan; boxes and points in normalised 0-1 coordinates
// (docs/ui-spec §4.3–4.4).
//
//   wheel                zoom around the cursor (1x–12x)      Space + drag / middle drag   pan
//   detect: drag on empty = new box, drag a box = move, drag a corner = resize,
//           click = select, Tab = next box, arrows nudge 1px (Shift: 10px), Delete = remove
//   point:  click = add, drag = move, Tab = next point, arrows nudge, Delete = remove
//   H (held) hides every overlay to see the raw image.  F = fit the whole image again.
// `focus` (boxes, 0-1) zooms to their union when the image or the question changes, so a
// small box in a 2560 px frame is looked at, not hunted for. `fitAspect` sizes the viewer to
// the image's aspect (no letterbox) up to the viewport minus `maxOffset` px.
// `alt` draws a second, read-only set (dashed blue) — the pre-label — toggled by `showAlt`.

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { Pop } from "./motion";
import type { Pred, Skill } from "../api";

type Drag =
  | { kind: "new"; x0: number; y0: number; x1: number; y1: number }
  | { kind: "move"; i: number; dx: number; dy: number; orig: number[] }
  | { kind: "resize"; i: number; corner: number }
  | { kind: "point"; i: number }
  | { kind: "pan"; sx: number; sy: number; tx: number; ty: number }
  | null;

export interface CanvasProps {
  src: string;
  skill: Skill;
  value: Pred | null;
  onChange?: (v: Pred, what: string) => void;
  alt?: Pred | null;
  readOnly?: boolean;
  labels?: { main?: string; alt?: string };
  height?: string;
  compact?: boolean;
  tags?: string[]; // per-box badges (e.g. TP / FP) for read-only comparisons
  focus?: number[][] | null; // boxes to zoom to (keyed by `focusKey`)
  focusKey?: string | number;
  fitAspect?: { maxOffset: number; min?: number };
}

const MIN_Z = 1, MAX_Z = 12;

export default function Canvas({ src, skill, value, onChange, alt, readOnly, labels, height, compact, tags, focus, focusKey, fitAspect }: CanvasProps) {
  const view = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const img = useRef<HTMLImageElement>(null);
  // natural size tagged with the URL it belongs to: a cached image can fire `load` before
  // the reset effect runs, so a plain "set null on src change" left 載入中 on screen forever
  const [natFor, setNatFor] = useState<{ w: number; h: number; url: string } | null>(null);
  const [tries, setTries] = useState(0);
  const [failed, setFailed] = useState<string | null>(null);
  const [slow, setSlow] = useState(false);
  const url = tries ? `${src}${src.includes("?") ? "&" : "?"}r=${tries}` : src;
  const nat = natFor && natFor.url === url ? natFor : null;
  const ready = !!nat;
  const [fit, setFit] = useState({ w: 0, h: 0, ox: 0, oy: 0 });
  const [viewH, setViewH] = useState<number | null>(null); // fitAspect: the viewer's own height
  const [z, setZ] = useState(1);
  const [t, setT] = useState({ x: 0, y: 0 });
  const [sel, setSel] = useState<number | null>(null);
  const [drag, setDrag] = useState<Drag>(null);
  const [pan, setPan] = useState(false);
  const [space, setSpace] = useState(false);
  const [hide, setHide] = useState(false);
  const [showAlt, setShowAlt] = useState(true);
  const spatial = skill === "detect" || skill === "point";
  const boxes = value?.boxes ?? [];
  const points = value?.points ?? [];
  const editable = !!onChange && !readOnly && spatial;
  const panning = pan || space;
  const n = skill === "detect" ? boxes.length : points.length;

  const layout = useCallback(() => {
    if (!view.current || !nat) return;
    const vw = view.current.clientWidth, vh = view.current.clientHeight;
    const s = Math.min(vw / nat.w, vh / nat.h);
    const w = nat.w * s, h = nat.h * s;
    setFit({ w, h, ox: (vw - w) / 2, oy: (vh - h) / 2 });
  }, [nat]);
  useLayoutEffect(layout, [layout]);
  useEffect(() => { if (!view.current) return; const ro = new ResizeObserver(layout); ro.observe(view.current); return () => ro.disconnect(); }, [layout]);
  useEffect(() => {
    if (!fitAspect) return;
    const size = () => { if (!view.current || !nat) return; const vw = view.current.clientWidth; const maxH = Math.max(fitAspect.min ?? 320, window.innerHeight - fitAspect.maxOffset); setViewH(Math.round(Math.min(maxH, (vw * nat.h) / nat.w))); };
    size();
    window.addEventListener("resize", size);
    return () => window.removeEventListener("resize", size);
  }, [fitAspect, nat]);
  const fk = `${src}|${focusKey ?? ""}`;
  useEffect(() => {
    // zoom to the union of the focus boxes once the layout for this image is known
    if (!fit.w || !view.current) return;
    if (!focus || !focus.length) { setZ(1); setT({ x: 0, y: 0 }); return; } // text question: whole image
    const u = [Math.min(...focus.map((b) => b[0])), Math.min(...focus.map((b) => b[1])), Math.max(...focus.map((b) => b[2])), Math.max(...focus.map((b) => b[3]))];
    const bw = Math.max(0.04, u[2] - u[0]), bh = Math.max(0.04, u[3] - u[1]);
    const vw = view.current.clientWidth, vh = view.current.clientHeight;
    const z1 = clamp(Math.min((0.7 * vw) / (bw * fit.w), (0.7 * vh) / (bh * fit.h)), MIN_Z, 6);
    if (z1 < 1.15) { setZ(1); setT({ x: 0, y: 0 }); return; }
    const cx = ((u[0] + u[2]) / 2) * fit.w, cy = ((u[1] + u[3]) / 2) * fit.h;
    setZ(z1);
    setT(limit({ x: vw / 2 - fit.ox - cx * z1, y: vh / 2 - fit.oy - cy * z1 }, z1));
  }, [fk, fit.w, fit.h]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setSel(null); setZ(1); setT({ x: 0, y: 0 }); setTries(0); setFailed(null); }, [src]);
  const measured = useCallback((el: HTMLImageElement | null) => { if (el && el.complete && el.naturalWidth) setNatFor({ w: el.naturalWidth, h: el.naturalHeight, url: el.getAttribute("src") ?? url }); }, [url]);
  useLayoutEffect(() => { measured(img.current); }, [measured]); // already in cache → no load event to wait for
  useEffect(() => { setSlow(false); if (ready) return; const t = window.setTimeout(() => setSlow(true), 6000); return () => window.clearTimeout(t); }, [url, ready]);
  const retry = () => { setFailed(null); setTries((n) => n + 1); };
  useEffect(() => setSel(null), [skill]);

  useEffect(() => {
    const el = view.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      setZ((z0) => { const z1 = clamp(z0 * Math.exp(-e.deltaY * 0.0015), MIN_Z, MAX_Z); setT((t0) => limit({ x: px - (px - t0.x) * (z1 / z0), y: py - (py - t0.y) * (z1 / z0) }, z1)); return z1; });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [fit]); // eslint-disable-line react-hooks/exhaustive-deps

  const limit = (tt: { x: number; y: number }, zz: number) => {
    if (!view.current) return tt;
    const vw = view.current.clientWidth, vh = view.current.clientHeight;
    const w = fit.w * zz, h = fit.h * zz;
    const x = clamp(tt.x, Math.min(vw - w - fit.ox * zz, 0), Math.max(0, vw - w - fit.ox * zz));
    const y = clamp(tt.y, Math.min(vh - h - fit.oy * zz, 0), Math.max(0, vh - h - fit.oy * zz));
    return { x, y };
  };
  const reset = () => { setZ(1); setT({ x: 0, y: 0 }); };
  const zoomTo = (z1: number) => { if (!view.current) return; const cx = view.current.clientWidth / 2, cy = view.current.clientHeight / 2; setT((t0) => limit({ x: cx - (cx - t0.x) * (z1 / z), y: cy - (cy - t0.y) * (z1 / z) }, z1)); setZ(z1); };
  const removeSel = () => {
    if (sel === null || !onChange) return;
    if (skill === "detect") onChange({ boxes: boxes.filter((_, i) => i !== sel) }, "刪了一框");
    else onChange({ points: points.filter((_, i) => i !== sel) }, "刪了一點");
    setSel(null);
  };
  const nudge = (dx: number, dy: number) => {
    if (sel === null || !onChange || !nat) return;
    const ux = dx / (fit.w * z), uy = dy / (fit.h * z);
    if (skill === "detect") onChange({ boxes: boxes.map((b, i) => (i === sel ? fix([b[0] + ux, b[1] + uy, b[2] + ux, b[3] + uy]) : b)) }, "框已改");
    else onChange({ points: points.map((p, i) => (i === sel ? [clamp(p[0] + ux), clamp(p[1] + uy)] : p)) }, "點已改");
  };

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const tg = e.target as HTMLElement;
      if (tg && (tg.tagName === "INPUT" || tg.tagName === "TEXTAREA" || tg.tagName === "SELECT")) return;
      if (e.code === "Space") { e.preventDefault(); setSpace(true); }
      if (e.key === "h" || e.key === "H") setHide(true);
      if (e.key === "f" || e.key === "F") reset();
      if (!editable) return;
      if ((e.key === "Delete" || e.key === "Backspace") && sel !== null) { e.preventDefault(); removeSel(); }
      if (e.key === "Tab" && n > 0) { e.preventDefault(); setSel(sel === null ? 0 : (sel + 1) % n); }
      if (e.key.startsWith("Arrow") && sel !== null) {
        e.preventDefault();
        const d = e.shiftKey ? 10 : 1;
        nudge(e.key === "ArrowLeft" ? -d : e.key === "ArrowRight" ? d : 0, e.key === "ArrowUp" ? -d : e.key === "ArrowDown" ? d : 0);
      }
    };
    const up = (e: KeyboardEvent) => { if (e.code === "Space") setSpace(false); if (e.key === "h" || e.key === "H") setHide(false); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  });

  const norm = (e: React.PointerEvent) => { const r = stage.current!.getBoundingClientRect(); return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)]; };
  const down = (e: React.PointerEvent) => {
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    if (panning || e.button === 1 || e.button === 2 || !editable) { if (e.button <= 2) setDrag({ kind: "pan", sx: e.clientX, sy: e.clientY, tx: t.x, ty: t.y }); return; }
    if (e.button !== 0) return;
    const [x, y] = norm(e);
    const tg = e.target as HTMLElement;
    const i = tg.dataset.i !== undefined ? +tg.dataset.i : null;
    if (tg.dataset.corner !== undefined && i !== null) { setSel(i); setDrag({ kind: "resize", i, corner: +tg.dataset.corner }); }
    else if (tg.classList.contains("bx") && i !== null) { setSel(i); setDrag({ kind: "move", i, dx: x - boxes[i][0], dy: y - boxes[i][1], orig: boxes[i] }); }
    else if (tg.classList.contains("pt") && i !== null) { setSel(i); setDrag({ kind: "point", i }); }
    else if (skill === "detect") { setSel(null); setDrag({ kind: "new", x0: x, y0: y, x1: x, y1: y }); }
    else { onChange!({ points: [...points, [+x.toFixed(4), +y.toFixed(4)]] }, "加了一點"); setSel(points.length); }
  };
  const move = (e: React.PointerEvent) => {
    if (!drag) return;
    if (drag.kind === "pan") { setT(limit({ x: drag.tx + e.clientX - drag.sx, y: drag.ty + e.clientY - drag.sy }, z)); return; }
    const [x, y] = norm(e);
    if (drag.kind === "new") setDrag({ ...drag, x1: x, y1: y });
    else if (drag.kind === "move") { const [bx0, by0, bx1, by1] = drag.orig; const w = bx1 - bx0, h = by1 - by0; const nx = clamp(x - drag.dx, 0, 1 - w), ny = clamp(y - drag.dy, 0, 1 - h); onChange!({ boxes: boxes.map((b, i) => (i === drag.i ? [nx, ny, nx + w, ny + h] : b)) }, "框已改"); }
    else if (drag.kind === "resize") { const b = boxes[drag.i].slice(); if (drag.corner === 0 || drag.corner === 2) b[0] = x; else b[2] = x; if (drag.corner === 0 || drag.corner === 1) b[1] = y; else b[3] = y; onChange!({ boxes: boxes.map((o, i) => (i === drag.i ? b : o)) }, "框已改"); }
    else if (drag.kind === "point") onChange!({ points: points.map((p, i) => (i === drag.i ? [+x.toFixed(4), +y.toFixed(4)] : p)) }, "點已改");
  };
  const up = () => {
    if (drag?.kind === "new") { const b = fix([drag.x0, drag.y0, drag.x1, drag.y1]); if (b[2] - b[0] > 0.003 && b[3] - b[1] > 0.003) { onChange!({ boxes: [...boxes, b] }, "加了一框"); setSel(boxes.length); } }
    else if (drag?.kind === "resize") onChange!({ boxes: boxes.map((b, i) => (i === drag.i ? fix(b) : b)) }, "框已改");
    setDrag(null);
  };

  const pct = (v: number) => `${v * 100}%`;
  const boxStyle = (b: number[]) => ({ left: pct(b[0]), top: pct(b[1]), width: pct(b[2] - b[0]), height: pct(b[3] - b[1]) });
  const cursor = panning ? (drag?.kind === "pan" ? "grabbing" : "grab") : editable ? "crosshair" : "default";
  const overlays = !hide;

  return (
    <div className="cv-wrap">
      <div className="cv-bar row">
        <button onClick={() => zoomTo(clamp(z / 1.5, MIN_Z, MAX_Z))} title="縮小">−</button>
        <span className="muted num" style={{ minWidth: 44, textAlign: "center" }}>{Math.round(z * 100)}%</span>
        <button onClick={() => zoomTo(clamp(z * 1.5, MIN_Z, MAX_Z))} title="放大">＋</button>
        <button onClick={reset} title="F">適應 <kbd>F</kbd></button>
        {editable && <button className={panning ? "cur" : ""} onClick={() => setPan(!pan)} title="按住 Space 也可以拖動">移動</button>}
        {editable && sel !== null && <button onClick={removeSel}>刪除選取 <kbd>Del</kbd></button>}
        {!compact && <span className="muted">滾輪縮放 · {editable ? (skill === "detect" ? "拖曳畫框 · 拖角縮放 · 點框選取" : "點一下加點 · 拖動移動") : "拖曳移動"}</span>}
        {labels && (
          <span className="legend" style={{ marginLeft: "auto" }}>
            {labels.main && <><i style={{ background: "var(--box)" }} />{labels.main} </>}
            {labels.alt && alt && <button className={`lg${showAlt ? " on" : ""}`} onClick={() => setShowAlt(!showAlt)} title="顯示／隱藏預標"><i style={{ borderTop: "3px dashed var(--box2)", background: "none", height: 0 }} />{labels.alt}</button>}
            <span className="muted hint" title="按住 H 看原圖">H 原圖</span>
          </span>
        )}
      </div>
      <div className="cv-view" ref={view} style={{ height: fitAspect && viewH ? viewH : height ?? "calc(100vh - 170px)", cursor }} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} onContextMenu={(e) => e.preventDefault()}>
        <div className="cv-stage" ref={stage} style={{ width: fit.w, height: fit.h, left: fit.ox, top: fit.oy, transform: `translate(${t.x}px, ${t.y}px) scale(${z})`, ["--s" as string]: z }}>
          <img ref={img} src={url} className={ready ? "on" : ""} draggable={false} alt="" onLoad={(e) => measured(e.currentTarget)} onError={() => setFailed(url)} />
          {overlays && spatial && showAlt && alt?.boxes?.map((b, i) => <div key={`a${i}`} className="bx alt ro" style={boxStyle(b)}><span className="tag alt">{labels?.alt ?? "預標"} {i + 1}</span></div>)}
          {overlays && spatial && showAlt && alt?.points?.map((p, i) => <div key={`ap${i}`} className="pt alt ro" style={{ left: pct(p[0]), top: pct(p[1]) }} />)}
          {overlays && skill === "detect" && boxes.map((b, i) => (
            <div key={i} data-i={i} className={`bx${sel === i ? " sel" : ""}${editable ? "" : " ro"}`} style={boxStyle(b)}>
              <span className="tag">{tags?.[i] ?? `${i + 1}`}</span>
              {editable && sel === i && [0, 1, 2, 3].map((c) => <div key={c} data-i={i} data-corner={c} className={`h h${c}`} />)}
            </div>
          ))}
          {overlays && skill === "point" && points.map((p, i) => <div key={i} data-i={i} className={`pt${sel === i ? " sel" : ""}${editable ? "" : " ro"}`} style={{ left: pct(p[0]), top: pct(p[1]) }} title={`${i + 1}`} />)}
          {drag?.kind === "new" && <div className="draft" style={boxStyle(fix([drag.x0, drag.y0, drag.x1, drag.y1]))} />}
        </div>
        <Pop show={!ready} className="cv-loading">
          {failed === url ? <span className="row bad">圖片載不出來 <button onClick={retry}>重試</button></span> : <span className="row muted"><i className="spin" />載入中…{slow && <button onClick={retry}>重試</button>}</span>}
        </Pop>
      </div>
    </div>
  );
}

function clamp(v: number, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, v)); }
function fix(b: number[]) { return [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])].map((v) => +clamp(v).toFixed(4)); }
