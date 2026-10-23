// Shared UI pieces: toasts (docs/ui-spec §7), dialogs, pills, the name dialog, key help.

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { AnimatePresence, Pop, motion } from "./motion";

type ToastKind = "ok" | "warn" | "bad";
interface Toast { id: number; text: string; kind: ToastKind; action?: { label: string; run: () => void }; sticky?: boolean }
interface ToastApi { say: (text: string, kind?: ToastKind, opts?: { action?: Toast["action"]; sticky?: boolean; ms?: number }) => void; dismiss: () => void }
const ToastCtx = createContext<ToastApi>({ say: () => undefined, dismiss: () => undefined });
export const useToast = () => useContext(ToastCtx);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<Toast | null>(null);
  const timer = useRef<number | null>(null);
  const dismiss = useCallback(() => { setToast(null); if (timer.current) window.clearTimeout(timer.current); }, []);
  const say = useCallback<ToastApi["say"]>((text, kind = "ok", opts = {}) => {
    if (timer.current) window.clearTimeout(timer.current);
    setToast({ id: Date.now(), text, kind, action: opts.action, sticky: opts.sticky });
    if (!opts.sticky) timer.current = window.setTimeout(() => setToast(null), opts.ms ?? (opts.action ? 3000 : kind === "ok" ? 1800 : 2500));
  }, []);
  return (
    <ToastCtx.Provider value={{ say, dismiss }}>
      {children}
      <AnimatePresence>
        {toast && (
          <motion.div key={toast.id} className={`toast ${toast.kind}`} role="status" initial={{ opacity: 0, y: 18, x: "-50%", scale: 0.98 }} animate={{ opacity: 1, y: 0, x: "-50%", scale: 1 }} exit={{ opacity: 0, y: 10, x: "-50%", transition: { duration: 0.14 } }}>
            <span>{toast.text}</span>
            {toast.action && <button onClick={() => { toast.action!.run(); dismiss(); }}>{toast.action.label}</button>}
            {toast.sticky && <button className="x" onClick={dismiss} aria-label="關閉">✕</button>}
          </motion.div>
        )}
      </AnimatePresence>
    </ToastCtx.Provider>
  );
}

export function Dialog({ open, onClose, children, width = 420 }: { open: boolean; onClose?: () => void; children: ReactNode; width?: number }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose?.(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="overlay" onClick={onClose} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, transition: { duration: 0.14 } }}>
          <motion.div className="modal" style={{ width: `min(${width}px, 94vw)` }} onClick={(e) => e.stopPropagation()} role="dialog" initial={{ opacity: 0, scale: 0.96, y: 10 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.98, y: 6, transition: { duration: 0.14 } }}>
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/** The title row every page starts with (docs/ui-spec §1): back link on sub-views, title,
 *  then status pills / counts. It sits in the left column so the action column lines up. */
export function PageHead({ back, title, children }: { back?: { label: string; onClick: () => void }; title: ReactNode; children?: ReactNode }) {
  return <div className="row page-head">{back && <button className="ghost back" onClick={back.onClick} title="Esc">{back.label}</button>}<h2>{title}</h2>{children}</div>;
}

/** The first row of every action column, level with the page's title row: page tools on the
 *  left (e.g. 所有圖), the shortcut help on the right. Keeps the primary button aligned with the
 *  first line of content under the title. */
export function ActionsHead({ children }: { children?: ReactNode }) {
  return (
    <div className="actions-head">
      <span className="row">{children}</span>
      <button className="ghost help" onClick={() => window.dispatchEvent(new Event("forge:help"))} title="快捷鍵（?）" aria-label="快捷鍵"><kbd>?</kbd> 快捷鍵</button>
    </div>
  );
}

/** A collapsible block in the action column (settings, export): one line with a summary. */
export function Section({ title, summary, open, onToggle, dirty, children }: { title: string; summary: string; open: boolean; onToggle: () => void; dirty?: boolean; children: ReactNode }) {
  return (
    <div className={`section${open ? " open" : ""}`}>
      <button className="section-h" onClick={onToggle} aria-expanded={open}><b>{title}</b><span className={`small ${dirty ? "warn-text" : "muted"}`}>{summary}</span><span className="chev">{open ? "▴" : "▾"}</span></button>
      <Pop show={open} className="section-b col">{children}</Pop>
    </div>
  );
}

/** Esc leaves a sub-view (review, diff, playground) unless the user is typing. */
export function useEscBack(active: boolean, back: () => void) {
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => { const t = e.target as HTMLElement; if (e.key !== "Escape" || (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT"))) return; if (document.querySelector(".overlay")) return; back(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, back]);
}

/** Ask once, from a click (browsers require a gesture), whether long jobs may notify. */
export function askNotify() {
  try { if ("Notification" in window && Notification.permission === "default") void Notification.requestPermission(); } catch { /* unsupported */ }
}
/** A desktop notification when the tab is in the background and permission was given. */
export function notify(title: string, body: string) {
  try { if ("Notification" in window && Notification.permission === "granted" && document.hidden) new Notification(title, { body }); } catch { /* unsupported */ }
}

/** Grey placeholder blocks shown until the first response arrives (no layout jump). */
export function Skeleton({ h = 16, w = "100%", r = 6, n = 1, gap = 8 }: { h?: number; w?: number | string; r?: number; n?: number; gap?: number }) {
  return <div className="col" style={{ gap }} aria-busy="true" aria-label="載入中">{Array.from({ length: n }, (_, i) => <div key={i} className="skel" style={{ height: h, width: w, borderRadius: r }} />)}</div>;
}

export function Pill({ kind, children, title, onClick }: { kind?: "ok" | "bad" | "warn" | "grey"; children: ReactNode; title?: string; onClick?: () => void }) {
  return <span className={`pill ${kind ?? "grey"}${onClick ? " click" : ""}`} title={title} onClick={onClick}>{children}</span>;
}

export function NameDialog({ open, initial, onDone }: { open: boolean; initial: string; onDone: (name: string) => void }) {
  const [v, setV] = useState(initial);
  useEffect(() => setV(initial), [initial, open]);
  return (
    <Dialog open={open} onClose={initial ? () => onDone(initial) : undefined}>
      <h2>你是誰？</h2>
      <div className="hint muted">檢查時每一筆都會記這個名字，和別人一起審也分得清楚。</div>
      <input id="name-input" autoFocus value={v} placeholder="名字" onChange={(e) => setV(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && v.trim()) onDone(v.trim()); }} />
      <button className="primary big" disabled={!v.trim()} onClick={() => onDone(v.trim())}>{initial ? "改好了" : "開始"}</button>
    </Dialog>
  );
}

export function ConfirmDialog({ open, title, body, confirmText, typed, onCancel, onConfirm, danger }: { open: boolean; title: string; body?: ReactNode; confirmText: string; typed?: string; onCancel: () => void; onConfirm: () => void; danger?: boolean }) {
  const [v, setV] = useState("");
  useEffect(() => setV(""), [open]);
  return (
    <Dialog open={open} onClose={onCancel}>
      <h2>{title}</h2>
      {body && <div className="hint">{body}</div>}
      {typed && <input autoFocus value={v} placeholder={`輸入「${typed}」確認`} onChange={(e) => setV(e.target.value)} />}
      <div className="row" style={{ justifyContent: "flex-end" }}>
        <button onClick={onCancel}>取消</button>
        <button className={danger ? "danger" : "primary"} disabled={!!typed && v !== typed} onClick={onConfirm}>{confirmText}</button>
      </div>
    </Dialog>
  );
}

export function HelpOverlay({ open, onClose, keys, title = "快捷鍵" }: { open: boolean; onClose: () => void; keys: [string, string][]; title?: string }) {
  return (
    <Dialog open={open} onClose={onClose} width={520}>
      <h2>{title}</h2>
      <div className="keys">{keys.map(([k, v]) => [<kbd key={k + "k"}>{k}</kbd>, <span key={k + "v"}>{v}</span>])}</div>
      <div className="hint muted">按 <kbd>?</kbd> 或 <kbd>Esc</kbd> 關閉</div>
    </Dialog>
  );
}

/** ETA in minutes from a job's start, done and total; null before the first unit. */
export function eta(started: number | undefined, done: number, total: number): number | null {
  if (!started || done <= 0 || total <= done) return null;
  return Math.max(1, Math.round((((Date.now() / 1000 - started) / done) * (total - done)) / 60));
}

export function Bar({ done, total, color }: { done: number; total: number; color?: string }) {
  return <div className="bar"><motion.i initial={false} animate={{ width: `${total ? Math.min(100, (100 * done) / total) : 0}%` }} transition={{ duration: 0.5, ease: "easeOut" }} style={{ background: color }} /></div>;
}
