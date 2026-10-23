// Workbench (GOALS §6.1): pick an image, run a skill with base or an adapter, correct the
// output on the canvas / in the text box, save it as a reviewed sample.

import { useEffect, useState } from "react";
import { api, promptLabel, reviewerStore, skillLabel, type ImageRow, type Model, type Pred, type Skill } from "../api";
import Canvas from "../components/Canvas";
import ImagePicker from "../components/ImagePicker";

const SKILLS: Skill[] = ["detect", "point", "query", "caption"];

export default function Workbench({ models }: { models: Model[] }) {
  const [image, setImage] = useState<ImageRow | null>(null);
  const [skill, setSkill] = useState<Skill>("detect");
  const [prompt, setPrompt] = useState("person");
  const [model, setModel] = useState("base");
  const [pred, setPred] = useState<Pred | null>(null);
  const [label, setLabel] = useState<Pred | null>(null);
  const [ms, setMs] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [reviewer, setReviewer] = useState(reviewerStore.get());

  useEffect(() => { reviewerStore.set(reviewer); }, [reviewer]);
  useEffect(() => {
    setPred(null); setLabel(null); setMs(null);
    if (skill === "caption" && !["short", "normal", "long"].includes(prompt)) setPrompt("normal");
    if (skill === "query" && ["short", "normal", "long", "person"].includes(prompt)) setPrompt("How many people are there?");
    if ((skill === "detect" || skill === "point") && !prompt) setPrompt("person");
  }, [skill]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = async () => {
    if (!image) return;
    setBusy(true); setMsg("");
    try {
      const r = await api.infer(image.id, skill, prompt, model);
      setPred(r.pred); setLabel(structuredClone(r.pred)); setMs(r.ms);
    } catch (e) { setMsg(String(e)); } finally { setBusy(false); }
  };

  const save = async (action: "approve" | "correct" | "discard") => {
    if (!image || !reviewer) { setMsg("先填 reviewer 名字"); return; }
    setBusy(true); setMsg("");
    try {
      const changed = JSON.stringify(label) !== JSON.stringify(pred);
      const a = action === "approve" && changed ? "correct" : action;
      const it = await api.saveItem({ image_id: image.id, skill, prompt, pred, pred_model: pred ? model : null, action: a, label: a === "discard" ? null : label ?? emptyLabel(skill), reviewer });
      setMsg(`已存為 ${it.status} (#${it.id})`);
    } catch (e) { setMsg(String(e)); } finally { setBusy(false); }
  };

  const text = label?.text ?? "";
  const spatial = skill === "detect" || skill === "point";

  return (
    <div className="grid3">
      <aside className="card"><ImagePicker value={image?.id ?? null} onPick={setImage} /></aside>
      <section>
        {image ? (
          <Canvas src={api.imageUrl(image.id)} skill={skill} value={label ?? emptyLabel(skill)} onChange={setLabel} />
        ) : (
          <div className="card muted">左側選一張圖，或上傳。</div>
        )}
        {image && <div className="muted" style={{ marginTop: 6 }}>{image.path} · {image.width}×{image.height}</div>}
      </section>
      <aside className="card col">
        <div className="tabs">{SKILLS.map((s) => <button key={s} className={s === skill ? "cur" : ""} onClick={() => setSkill(s)}>{skillLabel[s]}</button>)}</div>
        <label>{promptLabel(skill)}<input value={prompt} onChange={(e) => setPrompt(e.target.value)} style={{ width: "100%" }} /></label>
        <label>模型<select value={model} onChange={(e) => setModel(e.target.value)} style={{ width: "100%" }}>{models.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}</select></label>
        <button className="primary" onClick={run} disabled={!image || busy}>{busy ? "…" : "執行"}</button>
        {ms !== null && <div className="muted">{ms} ms · 模型輸出：{summary(pred)}</div>}
        {!spatial && <label>答案（可改）<textarea value={text} onChange={(e) => setLabel({ text: e.target.value })} /></label>}
        {spatial && <div className="muted">{skill === "detect" ? "拖曳=新增框、拖框=移動、拖角=縮放、Delete=刪除" : "點=新增、拖=移動、Delete=刪除"}；目前 {summary(label ?? emptyLabel(skill))}</div>}
        <label>Reviewer<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="你的名字" /></label>
        <div className="row">
          <button className="primary" onClick={() => save("approve")} disabled={busy || !image}>存為 gold</button>
          <button onClick={() => save("discard")} disabled={busy || !image}>丟棄</button>
          <button onClick={() => { setLabel(structuredClone(pred)); }} disabled={!pred}>重設為模型輸出</button>
        </div>
        {msg && <div>{msg}</div>}
      </aside>
    </div>
  );
}

export function emptyLabel(skill: Skill): Pred {
  return skill === "detect" ? { boxes: [] } : skill === "point" ? { points: [] } : { text: "" };
}

export function summary(p: Pred | null): string {
  if (!p) return "—";
  if (p.boxes) return `${p.boxes.length} 框`;
  if (p.points) return `${p.points.length} 點`;
  return p.text ? `"${p.text.slice(0, 60)}${p.text.length > 60 ? "…" : ""}"` : "(空)";
}
