// Motion primitives (docs/ui-spec §0): short, consistent, and off when the OS asks for
// reduced motion. Everything animated in the app goes through these so timing stays uniform.
//
//   <Fade k={key}>        content that swaps (a step, a tab, a question): fade + 6px rise
//   <Stagger>/<Item>      lists of cards appearing one after another
//   <Pop>                 small things that appear (pills, hints, warnings)
//   presence              AnimatePresence re-exported for toasts / dialogs

import { AnimatePresence, MotionConfig, motion, type HTMLMotionProps } from "motion/react";
import type { ReactNode } from "react";

export const EASE = [0.2, 0.8, 0.2, 1] as const;
export const FAST = 0.16, BASE = 0.24;

export function MotionRoot({ children }: { children: ReactNode }) {
  return <MotionConfig reducedMotion="user" transition={{ duration: BASE, ease: EASE }}>{children}</MotionConfig>;
}

/** A block whose content is replaced: remounts on `k` with a fade + rise. */
export function Fade({ k, children, className, style, y = 6 }: { k: string | number; children: ReactNode; className?: string; style?: React.CSSProperties; y?: number }) {
  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div key={k} className={className} style={style} initial={{ opacity: 0, y }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -y / 2, transition: { duration: FAST } }}>
        {children}
      </motion.div>
    </AnimatePresence>
  );
}

/** A list container whose <Item> children appear one after another. */
export function Stagger({ children, className, style, gap = 0.035 }: { children: ReactNode; className?: string; style?: React.CSSProperties; gap?: number }) {
  return (
    <motion.div className={className} style={style} initial="out" animate="in" variants={{ in: { transition: { staggerChildren: gap } }, out: {} }}>
      {children}
    </motion.div>
  );
}
export function Item({ children, className, style, onClick, title, layoutId }: { children: ReactNode; className?: string; style?: React.CSSProperties; onClick?: () => void; title?: string; layoutId?: string }) {
  return (
    <motion.div layout layoutId={layoutId} className={className} style={style} onClick={onClick} title={title} variants={{ in: { opacity: 1, y: 0 }, out: { opacity: 0, y: 8 } }} exit={{ opacity: 0, scale: 0.98, transition: { duration: FAST } }}>
      {children}
    </motion.div>
  );
}

/** Something small that appears or disappears in place. */
export function Pop({ children, className, style, show = true, ...rest }: { children: ReactNode; className?: string; style?: React.CSSProperties; show?: boolean } & Omit<HTMLMotionProps<"div">, "children" | "className" | "style">) {
  return (
    <AnimatePresence initial={false}>
      {show && (
        <motion.div className={className} style={style} initial={{ opacity: 0, scale: 0.96, y: 4 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.98, transition: { duration: FAST } }} {...rest}>
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export { AnimatePresence, motion };
