// Shared UI pieces: toasts (docs/ui-spec §7), dialogs, pills, the name dialog, key help.

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

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
      {toast && (
        <div className={`toast ${toast.kind}`} role="status">
          <span>{toast.text}</span>
          {toast.action && <button onClick={() => { toast.action!.run(); dismiss(); }}>{toast.action.label}</button>}
          {toast.sticky && <button className="x" onClick={dismiss} aria-label="關閉">✕</button>}
        </div>
      )}
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
  if (!open) return null;
  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" style={{ width: `min(${width}px, 94vw)` }} onClick={(e) => e.stopPropagation()} role="dialog">{children}</div>
    </div>
  );
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
  return <div className="bar"><i style={{ width: `${total ? Math.min(100, (100 * done) / total) : 0}%`, background: color }} /></div>;
}
