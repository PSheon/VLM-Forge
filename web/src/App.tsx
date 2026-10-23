// Shell (docs/ui-spec §1–2): three steps with progress, who you are, GPU status, what is
// missing, and the per-page key help. Data (images, labelling, review, export) lives inside
// a dataset; training and testing are the next two steps.

import { useCallback, useEffect, useState } from "react";
import { api, nameStore, type Batch, type Status, type Summary } from "./api";
import { HelpOverlay, NameDialog, Pill, ToastProvider, useToast } from "./components/ui";
import Datasets from "./pages/Datasets";
import Train from "./pages/Train";
import Test from "./pages/Test";

const STEPS = [
  ["datasets", "① 資料集", "丟照片、AI 導師先標、你檢查、匯出"],
  ["train", "② 訓練", "選資料集，一個按鈕"],
  ["test", "③ 測試", "base vs 新模型、匯出到 Orin"],
] as const;
export type Step = (typeof STEPS)[number][0];

const HELP: Record<Step, [string, string][]> = {
  datasets: [["拖曳照片到虛線區", "加入資料集"], ["點縮圖", "進檢查看那張"], ["Enter / Space", "檢查：存這題，到下一題"], ["Shift+Enter", "這張剩下的全照畫面存"], ["D", "丟棄這題"], ["1 / 2", "用預標 / 用模型的答案"], ["E", "改答案（文字題）"], ["Z", "還原這題"], ["← / →", "上一題 / 下一題"], ["N", "跳過這張"], ["G", "所有圖"], ["H（按住）", "看原圖"], ["Tab", "選下一個框/點"], ["方向鍵（選取時）", "微移 1px，Shift 10px"], ["Delete", "刪框/點"], ["滾輪 · 空白鍵拖曳", "縮放 · 移動"]],
  train: [["Enter（欄位內）", "開始訓練"], ["?", "說明"]],
  test: [["← / →（看差異）", "上一題 / 下一題"], ["?", "說明"]],
};

export default function App() {
  return <ToastProvider><Shell /></ToastProvider>;
}

function Shell() {
  const [step, setStep] = useState<Step>(() => { const h = location.hash.slice(1).split("?")[0] as Step; return STEPS.some(([k]) => k === h) ? h : "datasets"; });
  const [status, setStatus] = useState<Status | null>(null);
  const [sum, setSum] = useState<Summary | null>(null);
  const [down, setDown] = useState(false);
  const [name, setName] = useState(nameStore.get());
  const [askName, setAskName] = useState(!nameStore.get());
  const [help, setHelp] = useState(false);
  const [guard, setGuardState] = useState<(() => boolean) | null>(null); // review's unsaved-changes check
  const setGuard = useCallback((g: (() => boolean) | null) => setGuardState(() => g), []);
  const { say } = useToast();
  const refresh = useCallback(() => {
    api.status().then((s) => { setStatus(s); setDown(false); }).catch(() => setDown(true));
    api.summary().then(setSum).catch(() => undefined);
  }, []);
  useEffect(() => { refresh(); const t = setInterval(refresh, 5000); return () => clearInterval(t); }, [refresh]);
  useEffect(() => { if (location.hash.slice(1).split("?")[0] !== step) location.hash = step; }, [step]);
  useEffect(() => {
    const onHash = () => { const h = location.hash.slice(1).split("?")[0] as Step; if (STEPS.some(([k]) => k === h)) setStep(h); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { const t = e.target as HTMLElement; if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return; if (e.key === "?") setHelp((h) => !h); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const go = (s: Step) => { if (guard && !guard()) return; setStep(s); };
  const e = status?.engine;
  const running: Batch | undefined = sum?.batches.find((b) => b.job?.state === "running");
  const counts: Record<Step, string> = {
    datasets: sum ? (sum.images ? `${sum.images} 張 · ${sum.items} 題${sum.images_pending ? ` · 還有 ${sum.images_pending} 張要看` : ""}` : "還沒有資料") : "",
    train: sum ? `${sum.trainable} 筆可訓練` : "",
    test: "",
  };
  const gpuText = e?.busy ? (running ? `工作中：標註 ${running.name} ${running.job?.done ?? 0}/${running.job?.total ?? 0}` : "工作中") : "閒置";

  return (
    <>
      <header className="top">
        <strong className="brand">VLM-Forge</strong>
        <nav className="steps">
          {STEPS.map(([k, l, d]) => (
            <a key={k} href={`#${k}`} className={k === step ? "cur" : ""} title={d} onClick={(ev) => { ev.preventDefault(); go(k); }}>
              <span className="t">{l}</span><small className="muted">{counts[k]}</small>
            </a>
          ))}
        </nav>
        <div className="right hint">
          <span className="who" title="點一下改名字" onClick={() => setAskName(true)}>{name || "（沒填名字）"}</span>
          {down ? <Pill kind="bad">連不到 server，重試中…</Pill> : e ? (
            <>
              {!e.weights_ok && <Pill kind="warn" title={`模型權重不在 ${e.weights}`} onClick={() => say(`模型權重不在 ${e.weights}。在 GB300 上：hf download moondream/moondream3.1-9B-A2B model.safetensors --local-dir model-cache/moondream31/weights`, "warn", { sticky: true })}>模型權重缺</Pill>}
              {!e.judge_ok && <Pill kind="warn" title={`出題模型不在 ${e.judge}`} onClick={() => say(`出題模型不在 ${e.judge}（model-cache/judge 要指到 Qwen3-VL-32B-Instruct）`, "warn", { sticky: true })}>出題模型缺</Pill>}
              {e.busy ? <Pill kind="ok">GPU {gpuText}</Pill> : e.gpu_free_gb !== undefined && e.gpu_free_gb !== null && e.gpu_free_gb < 70 && !e.judge_loaded ? <Pill kind="warn" title="出題模型要約 70 GB、Moondream 約 24 GB；別的訓練正在用 GPU，等它結束">GPU 被別的訓練佔用，剩 {e.gpu_free_gb.toFixed(0)} GB</Pill> : <Pill kind="grey">GPU 閒置{e.gpu_free_gb ? `，剩 ${e.gpu_free_gb.toFixed(0)} GB` : ""}</Pill>}
              {e.last_error && <Pill kind="bad" title={e.last_error} onClick={() => say(e.last_error!, "bad", { sticky: true })}>上次錯誤</Pill>}
            </>
          ) : <span className="muted">連線中…</span>}
          <button className="ghost" onClick={() => setHelp(true)} title="?">?</button>
        </div>
      </header>
      <main>
        {step === "datasets" && <Datasets status={status} refresh={refresh} go={go} reviewer={name} setGuard={setGuard} />}
        {step === "train" && <Train status={status} go={go} reviewer={name} />}
        {step === "test" && <Test models={status?.models ?? []} reviewer={name} />}
      </main>
      <NameDialog open={askName} initial={name} onDone={(v) => { nameStore.set(v); setName(v); setAskName(false); say(name ? `之後記為：${v}` : `你好，${v}`); }} />
      <HelpOverlay open={help} onClose={() => setHelp(false)} keys={HELP[step]} />
    </>
  );
}
