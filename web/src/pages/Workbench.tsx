// Workbench (GOALS §6.1): pick an image, run a skill with base or an adapter, correct the
// output on the canvas / in the text box, save it as a reviewed sample. Ctrl+Enter runs.

import { useEffect, useState } from "react";
import { api, defaultPrompt, emptyLabel, judgeText, promptLabel, reviewerStore, skillLabel, summary, type ImageRow, type JudgeVerdict, type Model, type Pred, type Skill, type Status } from "../api";
import Canvas from "../components/Canvas";
import ImagePicker from "../components/ImagePicker";

const SKILLS: Skill[] = ["detect", "point", "query", "caption"];

export default function Workbench({ models, status }: { models: Model[]; status: Status | null }) {
  const [image, setImage] = useState<ImageRow | null>(null);
  const [skill, setSkill] = useState<Skill>("detect");
  const [prompt, setPrompt] = useState("person");
  const [model, setModel] = useState("base");
  const [pred, setPred] = useState<Pred | null>(null);
  const [label, setLabel] = useState<Pred | null>(null);
  const [ms, setMs] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ text: string; bad?: boolean } | null>(null);
  const [reviewer, setReviewer] = useState(reviewerStore.get());
  const ready = status?.engine.weights_ok ?? true;

  useEffect(() => { reviewerStore.set(reviewer); }, [reviewer]);
  useEffect(() => { setPred(null); setLabel(null); setMs(null); setPrompt(defaultPrompt(skill)); }, [skill]);
  useEffect(() => { setPred(null); setLabel(null); setMs(null); setMsg(null); }, [image?.id]);

  const run = async () => {
    if (!image || busy) return;
    setBusy(true); setMsg(null);
    try {
      const r = await api.infer(image.id, skill, prompt, model);
      setPred(r.pred); setLabel(structuredClone(r.pred)); setMs(r.ms);
    } catch (e) { setMsg({ text: String(e), bad: true }); } finally { setBusy(false); }
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); run(); } };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const [verdict, setVerdict] = useState<(JudgeVerdict & { ms: number }) | null>(null);
  useEffect(() => setVerdict(null), [label]);
  const askJudge = async () => {
    if (!image) return;
    setBusy(true); setMsg(null);
    try { setVerdict(await api.judge(image.id, skill, prompt, label ?? emptyLabel(skill))); } catch (e) { setMsg({ text: String(e), bad: true }); } finally { setBusy(false); }
  };

  const save = async (action: "approve" | "discard") => {
    if (!image) return;
    if (!reviewer) { setMsg({ text: "先填 reviewer 名字", bad: true }); return; }
    setBusy(true); setMsg(null);
    try {
      const final = label ?? emptyLabel(skill);
      const changed = JSON.stringify(final) !== JSON.stringify(pred);
      const a = action === "discard" ? "discard" : changed ? "correct" : "approve";
      const it = await api.saveItem({ image_id: image.id, skill, prompt, pred, pred_model: pred ? model : null, action: a, label: a === "discard" ? null : final, reviewer });
      setMsg({ text: `已存 #${it.id}（${it.status === "approved" ? "模型答對，照存" : it.status === "corrected" ? "人工修正" : "丟棄"}）` });
    } catch (e) { setMsg({ text: String(e), bad: true }); } finally { setBusy(false); }
  };

  const spatial = skill === "detect" || skill === "point";
  const changed = pred !== null && JSON.stringify(label) !== JSON.stringify(pred);

  return (
    <div className="grid3">
      <aside className="card"><ImagePicker value={image?.id ?? null} onPick={setImage} /></aside>
      <section style={{ minWidth: 0 }}>
        {image ? (
          <Canvas src={api.imageUrl(image.id, 2048)} skill={skill} value={label ?? emptyLabel(skill)} onChange={setLabel} />
        ) : (
          <div className="empty">左側選一張圖或上傳，右側選 skill 後按「執行」；模型的輸出會畫在圖上，直接修正再存。</div>
        )}
        {image && <div className="muted hint" style={{ marginTop: 4 }}>{image.path.split("/").slice(-3).join("/")} · {image.width}×{image.height}</div>}
      </section>
      <aside className="card col">
        {!ready && <div className="warn hint">模型權重還沒放到 GB300（{status?.engine.weights}）；畫框、存 gold 可以用，「執行」會失敗。</div>}
        <div className="tabs">{SKILLS.map((s) => <button key={s} className={s === skill ? "cur" : ""} onClick={() => setSkill(s)}>{skillLabel[s]}</button>)}</div>
        <label className="f">{promptLabel(skill)}<input value={prompt} onChange={(e) => setPrompt(e.target.value)} /></label>
        <label className="f">模型<select value={model} onChange={(e) => setModel(e.target.value)}>{models.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}</select></label>
        <button className="primary" onClick={run} disabled={!image || busy}>{busy ? "執行中…" : "執行"} <kbd>Ctrl+Enter</kbd></button>
        {ms !== null && <div className="hint muted">{ms} ms · 模型輸出 {summary(pred)}{changed ? " · 已修改" : ""}</div>}
        {pred && (
          <div className="row">
            <button onClick={askJudge} disabled={busy} title="讓 Qwen3-VL-32B 檢查目前畫面上的結果">請 judge 檢查</button>
            {verdict && <span className={verdict.verdict === "fix" ? "warn hint" : "ok hint"}>{verdict.verdict === "fix" ? "標紅" : "通過"}：{judgeText(verdict) || "—"}（{verdict.ms} ms）</span>}
          </div>
        )}
        {!spatial && <label className="f">答案（可改）<textarea value={label?.text ?? ""} onChange={(e) => setLabel({ text: e.target.value })} /></label>}
        {spatial && <div className="hint muted">目前 {summary(label ?? emptyLabel(skill))}</div>}
        <label className="f">Reviewer<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="你的名字" /></label>
        <div className="row">
          <button className="primary" onClick={() => save("approve")} disabled={busy || !image} title="把圖上現在的結果存成人工確認的樣本">存為確認樣本</button>
          <button onClick={() => save("discard")} disabled={busy || !image} title="記錄這題不可用">丟棄</button>
          <button onClick={() => setLabel(structuredClone(pred))} disabled={!pred || !changed}>還原模型輸出</button>
        </div>
        {msg && <div className={msg.bad ? "bad hint" : "ok hint"}>{msg.text}</div>}
        <div className="hint muted">存下的樣本在「審核佇列」頁匯出成訓練列。</div>
      </aside>
    </div>
  );
}
