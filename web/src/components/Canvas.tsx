// Image viewer / editor with zoom and pan, boxes and points in normalised 0-1 coordinates.
//
//   wheel            zoom around the cursor (1x–12x)
//   Space + drag, middle / right drag, or the 移動 mode   pan
//   detect: drag on empty space = new box, drag a box = move, drag a corner = resize,
//           click = select, Delete / Backspace = remove
//   point:  click = add, drag = move, Delete = remove
//
// `alt` draws a second, read-only set (dashed blue) for comparisons. The stage (image +
// overlays) is scaled with a CSS transform, so overlay geometry stays in image space and
// handles / borders are counter-scaled to a constant screen size.

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
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
  onChange?: (v: Pred) => void;
  alt?: Pred | null;
  readOnly?: boolean;
  labels?: { main?: string; alt?: string };
  height?: string; // CSS height of the viewport
}

const MIN_Z = 1, MAX_Z = 12;

export default function Canvas({ src, skill, value, onChange, alt, readOnly, labels, height }: CanvasProps) {
  const view = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const [fit, setFit] = useState({ w: 0, h: 0, ox: 0, oy: 0 });
  const [z, setZ] = useState(1);
  const [t, setT] = useState({ x: 0, y: 0 });
  const [sel, setSel] = useState<number | null>(null);
  const [drag, setDrag] = useState<Drag>(null);
  const [pan, setPan] = useState(false);
  const [space, setSpace] = useState(false);
  const spatial = skill === "detect" || skill === "point";
  const boxes = value?.boxes ?? [];
  const points = value?.points ?? [];
  const editable = !!onChange && !readOnly && spatial;
  const panning = pan || space;

  // fit the image into the viewport (contain); recomputed on resize
  const layout = useCallback(() => {
    if (!view.current || !nat) return;
    const vw = view.current.clientWidth, vh = view.current.clientHeight;
    const s = Math.min(vw / nat.w, vh / nat.h);
    const w = nat.w * s, h = nat.h * s;
    setFit({ w, h, ox: (vw - w) / 2, oy: (vh - h) / 2 });
  }, [nat]);
  useLayoutEffect(layout, [layout]);
  useEffect(() => {
    if (!view.current) return;
    const ro = new ResizeObserver(layout);
    ro.observe(view.current);
    return () => ro.disconnect();
  }, [layout]);
  useEffect(() => { setSel(null); setZ(1); setT({ x: 0, y: 0 }); setNat(null); }, [src]);
  useEffect(() => setSel(null), [skill]);

  // wheel zoom (native listener: React's is passive)
  useEffect(() => {
    const el = view.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      setZ((z0) => {
        const z1 = clamp(z0 * Math.exp(-e.deltaY * 0.0015), MIN_Z, MAX_Z);
        setT((t0) => limit({ x: px - (px - t0.x) * (z1 / z0), y: py - (py - t0.y) * (z1 / z0) }, z1));
        return z1;
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [fit]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const tg = e.target as HTMLElement;
      if (tg && (tg.tagName === "INPUT" || tg.tagName === "TEXTAREA" || tg.tagName === "SELECT")) return;
      if (e.code === "Space") { e.preventDefault(); setSpace(true); }
      if (editable && (e.key === "Delete" || e.key === "Backspace") && sel !== null) {
        e.preventDefault();
        removeSel();
      }
      if (e.key === "0" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); reset(); }
    };
    const up = (e: KeyboardEvent) => { if (e.code === "Space") setSpace(false); };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }); // every render: closures over current state

  const limit = (tt: { x: number; y: number }, zz: number) => {
    // keep the scaled image overlapping the viewport
    if (!view.current) return tt;
    const vw = view.current.clientWidth, vh = view.current.clientHeight;
    const w = fit.w * zz, h = fit.h * zz;
    const minX = Math.min(0, vw - w - fit.ox * zz) - fit.ox * zz + Math.min(0, 0);
    const x = clamp(tt.x, Math.min(minX, 0), Math.max(0, vw - w - fit.ox * zz));
    const minY = Math.min(0, vh - h - fit.oy * zz) - fit.oy * zz;
    const y = clamp(tt.y, Math.min(minY, 0), Math.max(0, vh - h - fit.oy * zz));
    return { x, y };
  };
  const reset = () => { setZ(1); setT({ x: 0, y: 0 }); };
  const zoomTo = (z1: number) => {
    if (!view.current) return;
    const cx = view.current.clientWidth / 2, cy = view.current.clientHeight / 2;
    setT((t0) => limit({ x: cx - (cx - t0.x) * (z1 / z), y: cy - (cy - t0.y) * (z1 / z) }, z1));
    setZ(z1);
  };
  const removeSel = () => {
    if (sel === null || !onChange) return;
    if (skill === "detect") onChange({ boxes: boxes.filter((_, i) => i !== sel) });
    else onChange({ points: points.filter((_, i) => i !== sel) });
    setSel(null);
  };

  const norm = (e: React.PointerEvent) => {
    const r = stage.current!.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)];
  };

  const down = (e: React.PointerEvent) => {
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
    if (panning || e.button === 1 || e.button === 2 || !editable) {
      if (e.button === 0 || e.button === 1 || e.button === 2) setDrag({ kind: "pan", sx: e.clientX, sy: e.clientY, tx: t.x, ty: t.y });
      return;
    }
    if (e.button !== 0) return;
    const [x, y] = norm(e);
    const tg = e.target as HTMLElement;
    const i = tg.dataset.i !== undefined ? +tg.dataset.i : null;
    if (tg.dataset.corner !== undefined && i !== null) {
      setSel(i);
      setDrag({ kind: "resize", i, corner: +tg.dataset.corner });
    } else if (tg.classList.contains("bx") && i !== null) {
      setSel(i);
      setDrag({ kind: "move", i, dx: x - boxes[i][0], dy: y - boxes[i][1], orig: boxes[i] });
    } else if (tg.classList.contains("pt") && i !== null) {
      setSel(i);
      setDrag({ kind: "point", i });
    } else if (skill === "detect") {
      setSel(null);
      setDrag({ kind: "new", x0: x, y0: y, x1: x, y1: y });
    } else {
      onChange!({ points: [...points, [x, y]] });
      setSel(points.length);
    }
  };

  const move = (e: React.PointerEvent) => {
    if (!drag) return;
    if (drag.kind === "pan") {
      setT(limit({ x: drag.tx + e.clientX - drag.sx, y: drag.ty + e.clientY - drag.sy }, z));
      return;
    }
    const [x, y] = norm(e);
    if (drag.kind === "new") setDrag({ ...drag, x1: x, y1: y });
    else if (drag.kind === "move") {
      const [bx0, by0, bx1, by1] = drag.orig;
      const w = bx1 - bx0, h = by1 - by0;
      const nx = clamp(x - drag.dx, 0, 1 - w), ny = clamp(y - drag.dy, 0, 1 - h);
      onChange!({ boxes: boxes.map((b, i) => (i === drag.i ? [nx, ny, nx + w, ny + h] : b)) });
    } else if (drag.kind === "resize") {
      const b = boxes[drag.i].slice();
      if (drag.corner === 0 || drag.corner === 2) b[0] = x; else b[2] = x;
      if (drag.corner === 0 || drag.corner === 1) b[1] = y; else b[3] = y;
      onChange!({ boxes: boxes.map((o, i) => (i === drag.i ? b : o)) });
    } else if (drag.kind === "point") {
      onChange!({ points: points.map((p, i) => (i === drag.i ? [x, y] : p)) });
    }
  };

  const up = () => {
    if (drag?.kind === "new") {
      const b = fix([drag.x0, drag.y0, drag.x1, drag.y1]);
      if (b[2] - b[0] > 0.003 && b[3] - b[1] > 0.003) {
        onChange!({ boxes: [...boxes, b] });
        setSel(boxes.length);
      }
    } else if (drag?.kind === "resize") {
      onChange!({ boxes: boxes.map((b, i) => (i === drag.i ? fix(b) : b)) });
    }
    setDrag(null);
  };

  const pct = (v: number) => `${v * 100}%`;
  const boxStyle = (b: number[]) => ({ left: pct(b[0]), top: pct(b[1]), width: pct(b[2] - b[0]), height: pct(b[3] - b[1]) });
  const cursor = panning ? (drag?.kind === "pan" ? "grabbing" : "grab") : editable ? "crosshair" : "default";

  return (
    <div className="cv-wrap">
      <div className="cv-bar row">
        <button onClick={() => zoomTo(clamp(z / 1.5, MIN_Z, MAX_Z))} title="縮小">−</button>
        <span className="muted" style={{ minWidth: 44, textAlign: "center" }}>{Math.round(z * 100)}%</span>
        <button onClick={() => zoomTo(clamp(z * 1.5, MIN_Z, MAX_Z))} title="放大">＋</button>
        <button onClick={reset} title="Ctrl+0">適應</button>
        {editable && <button className={panning ? "cur" : ""} onClick={() => setPan(!pan)} title="按住 Space 也可以拖動">移動</button>}
        {editable && sel !== null && <button onClick={removeSel}>刪除選取 <kbd>Del</kbd></button>}
        <span className="muted">滾輪縮放 · {editable ? (skill === "detect" ? "拖曳畫框 · 拖角縮放 · 點框選取" : "點一下加點 · 拖動移動") : "拖曳移動"}</span>
        {labels && (
          <span className="legend" style={{ marginLeft: "auto" }}>
            {labels.main && <><i style={{ background: "var(--box)" }} />{labels.main} </>}
            {labels.alt && <><i style={{ background: "var(--box2)" }} />{labels.alt}</>}
          </span>
        )}
      </div>
      <div
        className="cv-view"
        ref={view}
        style={{ height: height ?? "calc(100vh - 170px)", cursor }}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={up}
        onContextMenu={(e) => e.preventDefault()}
      >
        <div
          className="cv-stage"
          ref={stage}
          style={{ width: fit.w, height: fit.h, left: fit.ox, top: fit.oy, transform: `translate(${t.x}px, ${t.y}px) scale(${z})`, ["--s" as string]: z }}
        >
          <img src={src} draggable={false} alt="" onLoad={(e) => setNat({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
          {spatial && alt?.boxes?.map((b, i) => <div key={`a${i}`} className="bx alt ro" style={boxStyle(b)} />)}
          {spatial && alt?.points?.map((p, i) => <div key={`ap${i}`} className="pt alt ro" style={{ left: pct(p[0]), top: pct(p[1]) }} />)}
          {skill === "detect" &&
            boxes.map((b, i) => (
              <div key={i} data-i={i} className={`bx${sel === i ? " sel" : ""}${editable ? "" : " ro"}`} style={boxStyle(b)}>
                {editable && sel === i &&
                  [0, 1, 2, 3].map((c) => (
                    <div key={c} data-i={i} data-corner={c} className={`h h${c}`} />
                  ))}
              </div>
            ))}
          {skill === "point" &&
            points.map((p, i) => <div key={i} data-i={i} className={`pt${sel === i ? " sel" : ""}${editable ? "" : " ro"}`} style={{ left: pct(p[0]), top: pct(p[1]) }} />)}
          {drag?.kind === "new" && <div className="draft" style={boxStyle(fix([drag.x0, drag.y0, drag.x1, drag.y1]))} />}
        </div>
        {!nat && <div className="cv-loading muted">載入中…</div>}
      </div>
    </div>
  );
}

function clamp(v: number, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, v));
}
function fix(b: number[]) {
  return [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])].map((v) => +clamp(v).toFixed(4));
}
