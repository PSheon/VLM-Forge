// Image with editable boxes / points in normalised 0-1 coordinates.
// detect: drag on empty space = new box, drag a box = move, drag a corner = resize,
// click = select, Delete / Backspace = remove. point: click = add, drag = move.
// `alt` draws a second, read-only set (dashed blue) for comparisons.

import { useEffect, useRef, useState } from "react";
import type { Pred, Skill } from "../api";

type Drag =
  | { kind: "new"; x0: number; y0: number; x1: number; y1: number }
  | { kind: "move"; i: number; dx: number; dy: number; orig: number[] }
  | { kind: "resize"; i: number; corner: number }
  | { kind: "point"; i: number }
  | null;

export interface CanvasProps {
  src: string;
  skill: Skill;
  value: Pred | null;
  onChange?: (v: Pred) => void;
  alt?: Pred | null;
  readOnly?: boolean;
  labels?: { main?: string; alt?: string };
}

export default function Canvas({ src, skill, value, onChange, alt, readOnly, labels }: CanvasProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [drag, setDrag] = useState<Drag>(null);
  const spatial = skill === "detect" || skill === "point";
  const boxes = value?.boxes ?? [];
  const points = value?.points ?? [];
  const editable = !!onChange && !readOnly && spatial;

  useEffect(() => setSel(null), [src, skill]);

  useEffect(() => {
    if (!editable) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === "Delete" || e.key === "Backspace") && sel !== null) {
        const t = e.target as HTMLElement;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
        e.preventDefault();
        if (skill === "detect") onChange!({ boxes: boxes.filter((_, i) => i !== sel) });
        else onChange!({ points: points.filter((_, i) => i !== sel) });
        setSel(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editable, sel, boxes, points, skill, onChange]);

  const norm = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)];
  };

  const down = (e: React.PointerEvent) => {
    if (!editable) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const [x, y] = norm(e);
    const t = e.target as HTMLElement;
    const i = t.dataset.i !== undefined ? +t.dataset.i : null;
    if (t.dataset.corner !== undefined && i !== null) {
      setSel(i);
      setDrag({ kind: "resize", i, corner: +t.dataset.corner });
    } else if (t.classList.contains("bx") && i !== null) {
      setSel(i);
      setDrag({ kind: "move", i, dx: x - boxes[i][0], dy: y - boxes[i][1], orig: boxes[i] });
    } else if (t.classList.contains("pt") && i !== null) {
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
      if (b[2] - b[0] > 0.005 && b[3] - b[1] > 0.005) {
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

  return (
    <div className="cv" ref={ref} onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
      <img src={src} draggable={false} alt="" />
      {spatial && alt?.boxes?.map((b, i) => <div key={`a${i}`} className="bx alt ro" style={boxStyle(b)} />)}
      {spatial && alt?.points?.map((p, i) => <div key={`ap${i}`} className="pt alt" style={{ left: pct(p[0]), top: pct(p[1]), pointerEvents: "none" }} />)}
      {skill === "detect" &&
        boxes.map((b, i) => (
          <div key={i} data-i={i} className={`bx${sel === i ? " sel" : ""}${editable ? "" : " ro"}`} style={boxStyle(b)}>
            {editable && sel === i &&
              [0, 1, 2, 3].map((c) => (
                <div
                  key={c}
                  data-i={i}
                  data-corner={c}
                  className="h"
                  style={{ left: c % 2 ? undefined : -6, right: c % 2 ? -6 : undefined, top: c < 2 ? -6 : undefined, bottom: c < 2 ? undefined : -6, cursor: c === 0 || c === 3 ? "nwse-resize" : "nesw-resize" }}
                />
              ))}
          </div>
        ))}
      {skill === "point" &&
        points.map((p, i) => <div key={i} data-i={i} className={`pt${sel === i ? " sel" : ""}`} style={{ left: pct(p[0]), top: pct(p[1]) }} />)}
      {drag?.kind === "new" && <div className="draft" style={boxStyle(fix([drag.x0, drag.y0, drag.x1, drag.y1]))} />}
      {labels && (
        <div className="legend" style={{ position: "absolute", left: 6, bottom: 6, fontSize: 12, lineHeight: 1.4, background: "rgba(0,0,0,.55)", color: "#fff", padding: "2px 6px", borderRadius: 4 }}>
          {labels.main && <div><span style={{ background: "var(--box)" }} />{labels.main}</div>}
          {labels.alt && <div><span style={{ background: "var(--box2)" }} />{labels.alt}</div>}
        </div>
      )}
    </div>
  );
}

function clamp(v: number, lo = 0, hi = 1) {
  return Math.max(lo, Math.min(hi, v));
}
function fix(b: number[]) {
  return [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])].map((v) => +clamp(v).toFixed(4));
}
