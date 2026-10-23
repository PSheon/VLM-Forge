// Compare (GOALS §6.3): the same image through two models side by side, or two held-out
// eval reports item by item (only the items whose score changed).

import { useEffect, useState } from "react";
import { api, promptLabel, skillLabel, type HeldoutPair, type ImageRow, type Model, type Pred, type Report, type Skill } from "../api";
import Canvas from "../components/Canvas";
import ImagePicker from "../components/ImagePicker";
import TextDiff from "../components/TextDiff";
import { summary } from "./Workbench";

export default function Compare({ models }: { models: Model[] }) {
  const [mode, setMode] = useState<"live" | "heldout">("live");
  return (
    <div className="col">
      <div className="tabs"><button className={mode === "live" ? "cur" : ""} onClick={() => setMode("live")}>單張對照</button><button className={mode === "heldout" ? "cur" : ""} onClick={() => setMode("heldout")}>Held-out 報告對照</button></div>
      {mode === "live" ? <Live models={models} /> : <Heldout />}
    </div>
  );
}

function Live({ models }: { models: Model[] }) {
  const [image, setImage] = useState<ImageRow | null>(null);
  const [skill, setSkill] = useState<Skill>("detect");
  const [prompt, setPrompt] = useState("person");
  const [a, setA] = useState("base");
  const [b, setB] = useState(models[1]?.name ?? "base");
  const [res, setRes] = useState<Record<string, { pred: Pred; ms: number }> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const run = async () => {
    if (!image) return;
    setBusy(true); setErr("");
    try { setRes(await api.compare(image.id, skill, prompt, [a, b])); } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };
  const spatial = skill === "detect" || skill === "point";
  const pa = res?.[a]?.pred ?? null, pb = res?.[b]?.pred ?? null;
  return (
    <div className="grid3">
      <aside className="card"><ImagePicker value={image?.id ?? null} onPick={setImage} /></aside>
      <section className="col">
        {image && res && spatial && (
          <div className="grid2">
            <div><div className="muted">{lbl(a)} · {summary(pa)} · {res[a].ms} ms</div><Canvas src={api.imageUrl(image.id)} skill={skill} value={pa} alt={pb} readOnly labels={{ main: lbl(a), alt: lbl(b) }} /></div>
            <div><div className="muted">{lbl(b)} · {summary(pb)} · {res[b].ms} ms</div><Canvas src={api.imageUrl(image.id)} skill={skill} value={pb} alt={pa} readOnly labels={{ main: lbl(b), alt: lbl(a) }} /></div>
          </div>
        )}
        {image && res && !spatial && (
          <div className="grid2">
            <div className="card"><div className="muted">{lbl(a)} · {res[a].ms} ms</div><p>{pa?.text}</p></div>
            <div className="card"><div className="muted">{lbl(b)} · {res[b].ms} ms（相對 {lbl(a)} 的差異）</div><TextDiff a={pa?.text ?? ""} b={pb?.text ?? ""} /></div>
          </div>
        )}
        {image && !res && <Canvas src={api.imageUrl(image.id)} skill={skill} value={null} readOnly />}
        {!image && <div className="card muted">左側選一張圖。</div>}
      </section>
      <aside className="card col">
        <div className="tabs">{(["detect", "point", "query", "caption"] as Skill[]).map((s) => <button key={s} className={s === skill ? "cur" : ""} onClick={() => { setSkill(s); setRes(null); setPrompt(s === "query" ? "How many people are there?" : s === "caption" ? "normal" : "person"); }}>{skillLabel[s]}</button>)}</div>
        <label>{promptLabel(skill)}<input value={prompt} onChange={(e) => setPrompt(e.target.value)} style={{ width: "100%" }} /></label>
        <label>模型 A<select value={a} onChange={(e) => setA(e.target.value)} style={{ width: "100%" }}>{models.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}</select></label>
        <label>模型 B<select value={b} onChange={(e) => setB(e.target.value)} style={{ width: "100%" }}>{models.map((m) => <option key={m.name} value={m.name}>{m.label}</option>)}</select></label>
        <button className="primary" onClick={run} disabled={!image || busy}>{busy ? "…" : "兩個都跑"}</button>
        {err && <div className="bad">{err}</div>}
      </aside>
    </div>
  );
  function lbl(n: string) { return models.find((m) => m.name === n)?.label ?? n; }
}

function Heldout() {
  const [reports, setReports] = useState<Report[]>([]);
  const [base, setBase] = useState("");
  const [adapter, setAdapter] = useState("");
  const [onlyChanged, setOnlyChanged] = useState(true);
  const [pairs, setPairs] = useState<HeldoutPair[] | null>(null);
  const [n, setN] = useState(0);
  const [cur, setCur] = useState(0);
  const [err, setErr] = useState("");
  useEffect(() => { api.reports().then(setReports).catch((e) => setErr(String(e))); }, []);
  const load = async () => {
    try { const r = await api.heldoutDiff(base, adapter, onlyChanged); setPairs(r.items); setN(r.n); setCur(0); setErr(""); } catch (e) { setErr(String(e)); }
  };
  const p = pairs?.[cur];
  const gt = (row: HeldoutPair["row"]): Pred | null => row.skill === "detect" ? { boxes: row.boxes ?? [] } : row.skill === "point" ? { points: row.points ?? [] } : null;
  const asPred = (skill: Skill, v: unknown): Pred => skill === "detect" ? { boxes: v as number[][] } : skill === "point" ? { points: v as number[][] } : { text: String(v ?? "") };
  const rb = reports.find((r) => r.path === base), ra = reports.find((r) => r.path === adapter);
  return (
    <div className="col">
      <div className="card row">
        <label>Base 報告 <select value={base} onChange={(e) => setBase(e.target.value)}><option value="">—</option>{reports.map((r) => <option key={r.path} value={r.path}>{short(r)}</option>)}</select></label>
        <label>Adapter 報告 <select value={adapter} onChange={(e) => setAdapter(e.target.value)}><option value="">—</option>{reports.filter((r) => !base || r.manifest === rb?.manifest).map((r) => <option key={r.path} value={r.path}>{short(r)}</option>)}</select></label>
        <label><input type="checkbox" checked={onlyChanged} onChange={(e) => setOnlyChanged(e.target.checked)} /> 只看分數變動的題</label>
        <button className="primary" onClick={load} disabled={!base || !adapter}>載入</button>
        {rb && ra && <span className="muted">macro：{Object.keys(rb.macro).map((k) => `${k} ${rb.macro[k].toFixed(3)} → ${(ra.macro[k] ?? NaN).toFixed(3)}`).join(" · ")}</span>}
        {err && <span className="bad">{err}</span>}
      </div>
      {pairs && (
        <div className="grid3">
          <aside className="card"><div className="muted">{pairs.length} / {n} 題</div>
            <div className="list">{pairs.map((q, i) => <div key={i} className={`it${i === cur ? " cur" : ""}`} onClick={() => setCur(i)}>{i + 1}. {q.row.skill} {q.row.object ?? q.row.question} <span className={better(q) ? "ok" : "bad"}>{score(q.base)} → {score(q.adapter)}</span></div>)}</div>
          </aside>
          <section>
            {p && (p.row.skill === "detect" || p.row.skill === "point") && (
              <div className="grid2">
                <div><div className="muted">base（實線）vs GT（虛線）· {score(p.base)}</div><Canvas src={api.imageUrl(p.image_id)} skill={p.row.skill} value={asPred(p.row.skill, p.base.pred)} alt={gt(p.row)} readOnly labels={{ main: "base", alt: "GT" }} /></div>
                <div><div className="muted">adapter（實線）vs GT（虛線）· {score(p.adapter)}</div><Canvas src={api.imageUrl(p.image_id)} skill={p.row.skill} value={asPred(p.row.skill, p.adapter.pred)} alt={gt(p.row)} readOnly labels={{ main: "adapter", alt: "GT" }} /></div>
              </div>
            )}
            {p && p.row.skill !== "detect" && p.row.skill !== "point" && (
              <div className="grid2">
                <div><Canvas src={api.imageUrl(p.image_id)} skill="query" value={null} readOnly /></div>
                <div className="card col"><div><strong>{p.row.question}</strong> · GT：{p.row.answer}</div><div>base：{String(p.base.pred)} {p.base.ok ? "✓" : "✗"}</div><div>adapter：{String(p.adapter.pred)} {p.adapter.ok ? "✓" : "✗"}</div></div>
              </div>
            )}
          </section>
          <aside className="card muted">tp/fp/fn（框、點）或 ✓/✗（問答）。綠＝adapter 比 base 好。</aside>
        </div>
      )}
    </div>
  );
  function short(r: Report) { return `${r.model} · ${r.manifest.split("/").pop()}`; }
  function score(x: HeldoutPair["base"]) { return x.tp_fp_fn ? x.tp_fp_fn.join("/") : x.ok ? "✓" : "✗"; }
  function better(q: HeldoutPair) {
    if (q.base.tp_fp_fn && q.adapter.tp_fp_fn) { const f = (t: number[]) => (2 * t[0]) / Math.max(1, 2 * t[0] + t[1] + t[2]); return f(q.adapter.tp_fp_fn) >= f(q.base.tp_fp_fn); }
    return !!q.adapter.ok;
  }
}
