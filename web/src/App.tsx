// The four steps (GOALS §6): ① 資料 → ② 檢查 → ③ 訓練 → ④ 測試. One screen, one main
// button each; the header shows where the data stands.

import { useCallback, useEffect, useState } from "react";
import { api, reviewerStore, type Status, type Summary } from "./api";
import Data from "./pages/Data";
import Review from "./pages/Review";
import Train from "./pages/Train";
import Test from "./pages/Test";

const STEPS = [
  ["data", "① 資料", "選資料夾，自動出題"],
  ["review", "② 檢查", "逐張看圖"],
  ["train", "③ 訓練", "一個按鈕"],
  ["test", "④ 測試", "base vs 新模型"],
] as const;
type Step = (typeof STEPS)[number][0];

export default function App() {
  const [step, setStep] = useState<Step>((location.hash.slice(1) as Step) || "data");
  const [status, setStatus] = useState<Status | null>(null);
  const [sum, setSum] = useState<Summary | null>(null);
  const [down, setDown] = useState(false);
  const [reviewer, setReviewer] = useState(reviewerStore.get());
  const refresh = useCallback(() => {
    api.status().then((s) => { setStatus(s); setDown(false); }).catch(() => setDown(true));
    api.summary().then(setSum).catch(() => undefined);
  }, []);
  useEffect(() => { refresh(); const t = setInterval(refresh, 5000); return () => clearInterval(t); }, [refresh]);
  useEffect(() => { location.hash = step; }, [step]);
  useEffect(() => { reviewerStore.set(reviewer); }, [reviewer]);
  const e = status?.engine;
  const running = sum?.batches.some((b) => b.job?.state === "running");
  const counts: Record<Step, string> = {
    data: sum ? `${sum.images} 張圖 · ${sum.items} 題${running ? " · 進行中" : ""}` : "",
    review: sum ? (sum.images_pending ? `還有 ${sum.images_pending} 張要看` : sum.items ? "都看完了" : "") : "",
    train: sum ? `${sum.trainable} 筆可訓練` : "",
    test: "",
  };
  return (
    <>
      <header className="top">
        <strong>VLM-Forge</strong>
        <nav className="steps">
          {STEPS.map(([k, l, d]) => (
            <a key={k} href={`#${k}`} className={k === step ? "cur" : ""} title={d} onClick={(ev) => { ev.preventDefault(); setStep(k); }}>
              <span>{l}</span><small className="muted">{counts[k]}</small>
            </a>
          ))}
        </nav>
        <div className="right hint">
          <input value={reviewer} onChange={(ev) => setReviewer(ev.target.value)} title="審核時記錄的名字" style={{ width: 64, padding: "1px 6px", border: "1px dashed var(--line)", background: "transparent" }} />
          {down ? <span className="bad">連不到 server</span> : e ? (
            <>
              {!e.weights_ok ? <span className="warn">模型權重缺</span> : null}
              {!e.judge_ok ? <span className="warn">出題模型缺</span> : null}
              {e.busy ? <span className="ok">GPU 工作中</span> : <span className="muted">GPU 閒置</span>}
              {e.last_error && <span className="bad" title={e.last_error}>上次錯誤</span>}
            </>
          ) : "連線中…"}
        </div>
      </header>
      <main>
        {step === "data" && <Data sum={sum} refresh={refresh} status={status} go={() => setStep("review")} />}
        {step === "review" && <Review reviewer={reviewer} sum={sum} refresh={refresh} go={() => setStep("train")} />}
        {step === "train" && <Train sum={sum} go={() => setStep("test")} />}
        {step === "test" && <Test models={status?.models ?? []} />}
      </main>
    </>
  );
}
