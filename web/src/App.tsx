// Shell (docs/ui-spec §1–2): three steps with progress, who you are, GPU status, what is
// missing, and the per-page key help. Data (images, labelling, review, export) lives inside
// a dataset; training and testing are the next two steps.

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { api, nameStore, type Batch, type Run, type Status, type Summary } from "./api";
import { HelpOverlay, NameDialog, Pill, ToastProvider, notify, useToast } from "./components/ui";
import { Fade, MotionRoot } from "./components/motion";
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
  datasets: [["拖曳照片到虛線區", "加入資料集"], ["點縮圖", "進檢查看那張"], ["縮圖左上角 ☐ · Shift+點", "多選圖片 · 連選"], ["Delete（選了圖）", "從資料集移除"], ["A（檢查）", "這張圖加一題"], ["Esc（選了圖）", "取消選取"], ["Enter", "檢查：存這題，到下一題"], ["Shift+Enter", "這張剩下的全照畫面存"], ["D", "丟棄這題"], ["Y / N", "是非題：yes / no（直接存）"], ["0–9 · ↑↓", "數量題：填數字 / 加減"], ["1 / 2", "用預標 / 用模型的答案"], ["3", "框／點題：補上預標有、目前沒有的"], ["P", "detect 題：存的時候同時存同物件的 point（框中心）開／關"], ["Backspace", "刪最後一個框／點"], ["E", "改答案（文字題）"], ["Z", "還原這題（還沒存的改動）"], ["Ctrl+Z", "重開上一題（已經存了也可以）"], ["← / →", "上一題 / 下一題"], ["S", "跳過這張"], ["G", "所有圖"], ["F", "回到全圖"], ["H（按住）", "看原圖"], ["拖曳 / 拖角 / 點框", "detect：畫框、縮放、選取（預標框被蓋到會變綠 ✓）"], ["點一下 / 拖點", "point：加點、移動；點在預標框裡會吸到框中心，按住 Alt 放在原位（紅點＝不在任何預標框裡）"], ["Tab", "選下一個框/點"], ["方向鍵（選取時）", "微移 1px，Shift 10px"], ["Delete", "刪框/點"], ["滾輪 · Space／中鍵／右鍵拖曳", "縮放 · 移動畫面"]],
  train: [["Enter（欄位內）", "開始訓練"], ["?", "說明"]],
  test: [["← / →（看差異）", "上一題 / 下一題"], ["?", "說明"]],
};

export default function App() {
  return <MotionRoot><ToastProvider><Shell /></ToastProvider></MotionRoot>;
}

function Shell() {
  const [step, setStep] = useState<Step>(() => { const h = location.hash.slice(1).split("?")[0] as Step; return STEPS.some(([k]) => k === h) ? h : "datasets"; });
  const [status, setStatus] = useState<Status | null>(null);
  const [sum, setSum] = useState<Summary | null>(null);
  const [down, setDown] = useState(false);
  const [name, setName] = useState(nameStore.get());
  const [askName, setAskName] = useState(!nameStore.get());
  const [help, setHelp] = useState(false);
  const [sys, setSys] = useState(false);
  useEffect(() => {
    // the system popover closes on any click outside it, and on Esc
    if (!sys) return;
    const close = (e: Event) => { if (e instanceof KeyboardEvent ? e.key === "Escape" : !(e.target as HTMLElement).closest(".sys")) setSys(false); };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", close);
    return () => { document.removeEventListener("pointerdown", close); document.removeEventListener("keydown", close); };
  }, [sys]);
  const [guard, setGuardState] = useState<(() => boolean) | null>(null); // review's unsaved-changes check
  const setGuard = useCallback((g: (() => boolean) | null) => setGuardState(() => g), []);
  const { say } = useToast();
  const [runs, setRuns] = useState<Run[]>([]);
  const refresh = useCallback(() => {
    api.status().then((s) => { setStatus(s); setDown(false); }).catch(() => setDown(true));
    api.summary().then(setSum).catch(() => undefined);
    api.runs().then((r) => setRuns(r.runs)).catch(() => undefined);
  }, []);
  // long jobs: progress in the tab title, a notification (and a toast) when one finishes
  const liveBatch = sum?.batches.find((b) => b.job?.state === "running");
  const liveRun = runs.find((r) => r.state === "running");
  useEffect(() => {
    const t = liveBatch ? `標註 ${liveBatch.job?.done ?? 0}/${liveBatch.job?.total ?? 0}` : liveRun ? (liveRun.phase === "training" ? `訓練 ${liveRun.last?.step ?? 0}/${liveRun.steps}` : "評估中") : "";
    document.title = t ? `(${t}) VLM-Forge` : "VLM-Forge";
  }, [liveBatch, liveRun]);
  const seen = useRef<Record<string, string>>({});
  useEffect(() => {
    const now: Record<string, string> = {};
    for (const b of sum?.batches ?? []) if (b.job) now[`b${b.id}`] = b.job.state;
    for (const r of runs) now[`r${r.name}`] = r.state;
    for (const [k, v] of Object.entries(now)) {
      if (seen.current[k] !== "running" || v === "running") continue;
      const what = k.startsWith("b") ? `標註 ${sum?.batches.find((b) => `b${b.id}` === k)?.name ?? ""}` : `訓練 ${k.slice(1)}`;
      const text = v === "done" ? `${what} 完成` : v === "error" ? `${what} 失敗` : `${what} 停了`;
      say(text, v === "done" ? "ok" : "bad", { sticky: v !== "done" });
      notify("VLM-Forge", text);
    }
    seen.current = now;
  }, [sum, runs]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { refresh(); const t = setInterval(refresh, 5000); return () => clearInterval(t); }, [refresh]);
  useEffect(() => { if (location.hash.slice(1).split("?")[0] !== step) location.hash = step; }, [step]);
  useEffect(() => {
    const onHash = () => { const h = location.hash.slice(1).split("?")[0] as Step; if (STEPS.some(([k]) => k === h)) setStep(h); };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { const t = e.target as HTMLElement; if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return; if (e.key === "?") setHelp((h) => !h); };
    const onHelp = () => setHelp(true); // the ? 快捷鍵 button in every action column
    window.addEventListener("keydown", onKey);
    window.addEventListener("forge:help", onHelp);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("forge:help", onHelp); };
  }, []);

  const go = (s: Step) => { if (guard && !guard()) return; setStep(s); };
  const e = status?.engine;
  const running: Batch | undefined = sum?.batches.find((b) => b.job?.state === "running");
  // one short state per step (the numbers live on each page's title row): a dot when there is
  // something to do there, ✓ when it is done
  const nModels = Math.max(0, (status?.models.length ?? 1) - 1);
  const badge: Record<Step, ReactNode> = {
    datasets: !sum ? "" : sum.images_pending ? <><i className="dot d-warn" />{sum.images_pending} 張待看</> : sum.items ? "✓ 都看完了" : "還沒有資料",
    train: !sum ? "" : sum.trainable ? `${sum.trainable} 筆可訓` : "還沒有可訓的題",
    test: nModels ? `${nModels} 個模型可比` : "還沒有模型",
  };
  const gpuText = e?.busy ? (running ? `標註 ${running.name} ${running.job?.done ?? 0}/${running.job?.total ?? 0}` : "GPU 工作中") : "閒置";
  const lowGpu = !!e && !e.busy && e.gpu_free_gb != null && e.gpu_free_gb < (e.judge_need_gb ?? 70) && !e.judge_loaded;
  const sysLevel = down || e?.last_error ? "bad" : e && (!e.weights_ok || !e.judge_ok) ? "bad" : lowGpu || e?.judge_name === "Qwen3-VL-8B" ? "warn" : e?.busy ? "busy" : "ok";
  const sysText = down ? "連不到 server" : !e ? "連線中…" : !e.weights_ok ? "模型權重缺" : !e.judge_ok ? "出題模型缺" : e.last_error ? "上次有錯誤" : e.busy ? gpuText : lowGpu ? `GPU 剩 ${e.gpu_free_gb!.toFixed(0)} GB` : e.judge_name === "Qwen3-VL-8B" ? "用 8B 出題" : "系統正常";

  return (
    <>
      <header className="top">
        <strong className="brand">VLM-Forge</strong>
        <nav className="steps">
          {STEPS.map(([k, l, d]) => (
            <a key={k} href={`#${k}`} className={k === step ? "cur" : ""} title={d} onClick={(ev) => { ev.preventDefault(); go(k); }}>
              <span className="t">{l}</span><small className="muted">{badge[k]}</small>
            </a>
          ))}
        </nav>
        <div className="right hint">
          <span className="who" title="點一下改名字" onClick={() => setAskName(true)}>{name || "（沒填名字）"}</span>
          <span className="sys">
            <Pill kind={sysLevel === "bad" ? "bad" : sysLevel === "warn" ? "warn" : "grey"} onClick={() => setSys(!sys)} title="系統狀態"><i className={`dot d-${sysLevel}`} />{sysText}</Pill>
            {sys && (
              <div className="sys-pop">
                {down && <div className="it"><b>server</b><span className="bad">連不到，每 5 秒重試，畫面不會清空</span></div>}
                {e && <>
                  <div className="it"><b>GPU</b><span>{e.busy ? `工作中：${gpuText}` : "閒置"}{e.gpu_free_gb != null ? ` · 剩 ${e.gpu_free_gb.toFixed(0)} GB` : ""}{lowGpu ? <span className="warn-text">（出題要 {e.judge_need_gb ?? 70} GB，別的訓練佔著；等它結束）</span> : ""}</span></div>
                  <div className="it"><b>模型權重</b><span className={e.weights_ok ? "" : "bad"}>{e.weights_ok ? "✓ Moondream 3.1" : `缺：hf download moondream/moondream3.1-9B-A2B model.safetensors --local-dir model-cache/moondream31/weights`}</span></div>
                  <div className="it"><b>出題模型</b><span className={e.judge_ok ? "" : "bad"}>{!e.judge_ok ? "缺：model-cache/judge（Qwen3-VL-32B）或 model-cache/judge-8b" : e.judge_name ? `${e.judge_name}${e.judge_name === "Qwen3-VL-8B" ? "（GPU 放不下 32B；CCTV 考卷上不比 32B 差；GPU 空了重啟回 32B）" : ""}` : "還沒載入（第一次標註時載入）"}</span></div>
                  {e.last_error && <div className="it"><b>上次錯誤</b><span className="bad" style={{ overflowWrap: "anywhere" }}>{e.last_error} <button className="ghost" onClick={() => fetch("/api/status/clear-error", { method: "POST" }).then(refresh)}>知道了</button></span></div>}
                </>}
              </div>
            )}
          </span>
        </div>
      </header>
      <main>
        <Fade k={step}>
          {step === "datasets" && <Datasets status={status} refresh={refresh} go={go} reviewer={name} setGuard={setGuard} />}
          {step === "train" && <Train status={status} go={go} reviewer={name} />}
          {step === "test" && <Test models={status?.models ?? []} reviewer={name} go={go} />}
        </Fade>
      </main>
      <NameDialog open={askName} initial={name} onDone={(v) => { nameStore.set(v); setName(v); setAskName(false); say(name ? `之後記為：${v}` : `你好，${v}`); }} />
      <HelpOverlay open={help} onClose={() => setHelp(false)} keys={HELP[step]} />
    </>
  );
}
