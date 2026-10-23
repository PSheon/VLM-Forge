// ④ 測試: base vs every trained adapter on the held-out suites; "看差異" opens the items
// whose score changed; "單張試玩" runs any image through base and an adapter.

import { useEffect, useState } from "react";
import { api, defaultPrompt, skillLabel, suiteLabel, summary, type HeldoutPair, type ImageRow, type Model, type Pred, type Run, type Skill } from "../api";
import Canvas from "../components/Canvas";
import ImagePicker from "../components/ImagePicker";
import TextDiff from "../components/TextDiff";

export default function Test({ models }: { models: Model[] }) {
  const [tab, setTab] = useState<"scores" | "diff" | "play">("scores");
  const [data, setData] = useState<{ suites: string[]; base: Record<string, Record<string, number>>; runs: Run[] } | null>(null);
  const [diff, setDiff] = useState<{ run: Run; suite: string } | null>(null);
  useEffect(() => { api.testSummary().then(setData).catch(() => undefined); }, []);
  const done = data?.runs.filter((r) => r.state === "done" && Object.keys(r.evals).length) ?? [];
  const skills = ["detect", "point", "query", "caption"];
  return (
    <div className="col">
      <div className="tabs"><button className={tab === "scores" ? "cur" : ""} onClick={() => setTab("scores")}>分數</button><button className={tab === "diff" ? "cur" : ""} onClick={() => setTab("diff")} disabled={!done.length}>看差異</button><button className={tab === "play" ? "cur" : ""} onClick={() => setTab("play")}>單張試玩</button></div>
      {tab === "scores" && data && (
        <div className="card">
          <table>
            <thead><tr><th>held-out</th><th>skill</th><th>base</th>{done.map((r) => <th key={r.name}>{r.name}</th>)}</tr></thead>
            <tbody>
              {data.suites.map((s) => skills.filter((k) => data.base[s]?.[k] !== undefined || done.some((r) => r.evals[s]?.[k] !== undefined)).map((k) => (
                <tr key={s + k}><td>{suiteLabel[s] ?? s}</td><td>{k}</td><td>{fmt(data.base[s]?.[k])}</td>
                  {done.map((r) => { const v = r.evals[s]?.[k]; const b = data.base[s]?.[k]; const d = v !== undefined && b !== undefined ? v - b : undefined; return <td key={r.name} className={d === undefined ? "" : d >= 0 ? "ok" : "bad"}>{fmt(v)}{d !== undefined ? ` (${d >= 0 ? "+" : ""}${(100 * d).toFixed(1)})` : ""} {v !== undefined && <button style={{ padding: "0 6px" }} onClick={() => { setDiff({ run: r, suite: s }); setTab("diff"); }}>差異</button>}</td>; })}
                </tr>
              )))}
            </tbody>
          </table>
          {!done.length && <div className="muted hint" style={{ marginTop: 8 }}>還沒有訓練好的模型；base 的分數是起點。</div>}
          <div className="muted hint" style={{ marginTop: 8 }}>數字是 macro（逐來源×物件平均 F1 / exact）。held-out 的圖不進訓練。</div>
        </div>
      )}
      {tab === "diff" && <Diff runs={done} initial={diff} />}
      {tab === "play" && <Play models={models} />}
    </div>
  );
}

const fmt = (v?: number) => (v === undefined ? "—" : v.toFixed(3));

function Diff({ runs, initial }: { runs: Run[]; initial: { run: Run; suite: string } | null }) {
  const [run, setRun] = useState(initial?.run.name ?? runs[0]?.name ?? "");
  const [suite, setSuite] = useState(initial?.suite ?? "heldout-cctv");
  const [pairs, setPairs] = useState<HeldoutPair[] | null>(null);
  const [n, setN] = useState(0);
  const [cur, setCur] = useState(0);
  const [err, setErr] = useState("");
  const load = async () => {
    try { const r = await api.heldoutDiff(`runs/baselines/base-${suite}.json`, `runs/${run}/eval-${run}-${suite}.json`); setPairs(r.items); setN(r.n); setCur(0); setErr(""); } catch (e) { setErr(String(e)); }
  };
  useEffect(() => { if (run && suite) load(); }, [run, suite]); // eslint-disable-line react-hooks/exhaustive-deps
  const p = pairs?.[cur];
  const gt = (row: HeldoutPair["row"]): Pred | null => row.skill === "detect" ? { boxes: row.boxes ?? [] } : row.skill === "point" ? { points: row.points ?? [] } : null;
  const asPred = (skill: Skill, v: unknown): Pred => skill === "detect" ? { boxes: v as number[][] } : skill === "point" ? { points: v as number[][] } : { text: String(v ?? "") };
  const H = "calc(100vh - 300px)";
  return (
    <div className="col">
      <div className="card row">
        <label className="row">模型 <select value={run} onChange={(e) => setRun(e.target.value)}>{runs.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}</select></label>
        <label className="row">held-out <select value={suite} onChange={(e) => setSuite(e.target.value)}>{Object.entries(suiteLabel).map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select></label>
        {pairs && <span className="muted hint">{pairs.length} / {n} 題分數有變</span>}
        {err && <span className="bad hint">{err}</span>}
      </div>
      {pairs && (
        <div className="grid-review">
          <section style={{ minWidth: 0 }}>
            {p && (p.row.skill === "detect" || p.row.skill === "point") && (
              <div className="grid2">
                <div><div className="muted hint">base（實線）vs 正解（虛線）· {score(p.base)}</div><Canvas src={api.imageUrl(p.image_id, 2048)} skill={p.row.skill} value={asPred(p.row.skill, p.base.pred)} alt={gt(p.row)} readOnly labels={{ main: "base", alt: "正解" }} height={H} /></div>
                <div><div className="muted hint">新模型（實線）vs 正解（虛線）· {score(p.adapter)}</div><Canvas src={api.imageUrl(p.image_id, 2048)} skill={p.row.skill} value={asPred(p.row.skill, p.adapter.pred)} alt={gt(p.row)} readOnly labels={{ main: "新模型", alt: "正解" }} height={H} /></div>
              </div>
            )}
            {p && p.row.skill !== "detect" && p.row.skill !== "point" && (
              <div className="grid2">
                <Canvas src={api.imageUrl(p.image_id, 2048)} skill="query" value={null} readOnly height={H} />
                <div className="card col"><div><strong>{p.row.question}</strong> · 正解：{p.row.answer}</div><div>base：{String(p.base.pred)} {p.base.ok ? "✓" : "✗"}</div><div>新模型：{String(p.adapter.pred)} {p.adapter.ok ? "✓" : "✗"}</div></div>
              </div>
            )}
          </section>
          <aside className="card"><div className="list" style={{ maxHeight: "calc(100vh - 260px)" }}>{pairs.map((q, i) => <div key={i} className={`it${i === cur ? " cur" : ""}`} onClick={() => setCur(i)}>{i + 1}. {q.row.skill} {q.row.object ?? q.row.question} <span className={better(q) ? "ok" : "bad"}>{score(q.base)} → {score(q.adapter)}</span></div>)}</div><div className="muted hint">tp/fp/fn 或 ✓/✗；綠＝新模型比 base 好。</div></aside>
        </div>
      )}
    </div>
  );
  function score(x: HeldoutPair["base"]) { return x.tp_fp_fn ? x.tp_fp_fn.join("/") : x.ok ? "✓" : "✗"; }
  function better(q: HeldoutPair) {
    if (q.base.tp_fp_fn && q.adapter.tp_fp_fn) { const f = (t: number[]) => (2 * t[0]) / Math.max(1, 2 * t[0] + t[1] + t[2]); return f(q.adapter.tp_fp_fn) >= f(q.base.tp_fp_fn); }
    return !!q.adapter.ok;
  }
}

function Play({ models }: { models: Model[] }) {
  const [image, setImage] = useState<ImageRow | null>(null);
  const [skill, setSkill] = useState<Skill>("detect");
  const [prompt, setPrompt] = useState("person");
  const [b, setB] = useState(models[1]?.name ?? "base");
  const [res, setRes] = useState<Record<string, { pred: Pred; ms: number }> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  useEffect(() => setRes(null), [image?.id]);
  const run = async () => {
    if (!image) return;
    setBusy(true); setErr("");
    try { setRes(await api.compare(image.id, skill, prompt, b === "base" ? ["base"] : ["base", b])); } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };
  const spatial = skill === "detect" || skill === "point";
  const pa = res?.base?.pred ?? null, pb = res?.[b]?.pred ?? null;
  const H = "calc(100vh - 260px)";
  const src = image ? api.imageUrl(image.id, 2048) : "";
  const lbl = (n: string) => models.find((m) => m.name === n)?.label ?? n;
  return (
    <div className="grid3">
      <aside className="card"><ImagePicker value={image?.id ?? null} onPick={setImage} /></aside>
      <section className="col" style={{ minWidth: 0 }}>
        {image && res && spatial && (b === "base" ? <Canvas src={src} skill={skill} value={pa} readOnly height={H} /> : (
          <div className="grid2">
            <div><div className="muted hint">base · {summary(pa)} · {res.base.ms} ms</div><Canvas src={src} skill={skill} value={pa} alt={pb} readOnly labels={{ main: "base", alt: lbl(b) }} height={H} /></div>
            <div><div className="muted hint">{lbl(b)} · {summary(pb)} · {res[b].ms} ms</div><Canvas src={src} skill={skill} value={pb} alt={pa} readOnly labels={{ main: lbl(b), alt: "base" }} height={H} /></div>
          </div>
        ))}
        {image && res && !spatial && (
          <div className="grid2">
            <div className="card"><div className="muted hint">base · {res.base.ms} ms</div><p>{pa?.text}</p></div>
            {b !== "base" && <div className="card"><div className="muted hint">{lbl(b)} · {res[b].ms} ms（相對 base 的差異）</div><TextDiff a={pa?.text ?? ""} b={pb?.text ?? ""} /></div>}
          </div>
        )}
        {image && !res && <Canvas src={src} skill={skill} value={null} readOnly height={H} />}
        {!image && <div className="empty">左側選一張圖（任何來源都行，held-out 也可以）。</div>}
      </section>
      <aside className="card col">
        <div className="tabs">{(["detect", "point", "query", "caption"] as Skill[]).map((s) => <button key={s} className={s === skill ? "cur" : ""} onClick={() => { setSkill(s); setRes(null); setPrompt(defaultPrompt(s)); }}>{skillLabel[s]}</button>)}</div>
        <label className="f">{skill === "query" ? "問題" : skill === "caption" ? "長度" : "物件"}<input value={prompt} onChange={(e) => setPrompt(e.target.value)} /></label>
        <label className="f">和 base 比的模型<select value={b} onChange={(e) => setB(e.target.value)}>{models.map((m) => <option key={m.name} value={m.name}>{m.name === "base" ? "只跑 base" : m.label}</option>)}</select></label>
        <button className="primary" onClick={run} disabled={!image || busy}>{busy ? "執行中…" : "執行"}</button>
        {err && <div className="bad hint">{err}</div>}
      </aside>
    </div>
  );
}
