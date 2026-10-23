// ④ 測試 (docs/ui-spec §6): scores (base vs every run, held-out + val splits), per-item
// diff, a single-image playground that can save into a dataset, and the Orin bundle.

import { useEffect, useState } from "react";
import { api, defaultPrompt, emptyLabel, fmt, fmtDelta, skillLabel, suiteLabel, summary, type Dataset, type HeldoutPair, type ImageRow, type Model, type Pred, type Run, type Skill } from "../api";
import Canvas from "../components/Canvas";
import ImagePicker from "../components/ImagePicker";
import TextDiff from "../components/TextDiff";
import { Pill, useToast } from "../components/ui";
import { Fade } from "../components/motion";

type Tab = "scores" | "diff" | "play" | "orin";

export default function Test({ models, reviewer }: { models: Model[]; reviewer: string }) {
  const [tab, setTab] = useState<Tab>("scores");
  const [data, setData] = useState<{ suites: string[]; base: Record<string, Record<string, number>>; runs: Run[] } | null>(null);
  const [diff, setDiff] = useState<{ run: string; suite: string } | null>(null);
  const load = () => api.testSummary().then(setData).catch(() => undefined);
  useEffect(() => { load(); const t = setInterval(load, 10000); return () => clearInterval(t); }, []);
  const done = data?.runs.filter((r) => r.state === "done" && Object.keys(r.evals).length) ?? [];
  const suites = Array.from(new Set([...(data?.suites ?? []), ...done.flatMap((r) => Object.keys(r.evals))]));
  const skills = ["detect", "point", "query", "caption"];
  return (
    <div className="col">
      <div className="tabs">{([["scores", "分數"], ["diff", "看差異"], ["play", "單張試玩"], ["orin", "匯出到 Orin"]] as [Tab, string][]).map(([k, l]) => <button key={k} className={tab === k ? "cur" : ""} onClick={() => setTab(k)} disabled={(k === "diff" || k === "orin") && !done.length}>{l}</button>)}</div>
      <Fade k={tab}>
      {tab === "scores" && data && (
        <div className="panel">
          <div className="tw"><table>
            <thead><tr><th>考卷</th><th>skill</th><th className="num">base</th>{done.map((r) => <th key={r.name} className="num">{r.name}</th>)}</tr></thead>
            <tbody>
              {suites.map((s) => skills.filter((k) => data.base[s]?.[k] !== undefined || done.some((r) => r.evals[s]?.[k] !== undefined)).map((k, i) => (
                <tr key={s + k} className={s.includes("crowdhuman") || s.includes("posture") ? "muted" : ""}>
                  <td>{i === 0 ? suiteLabel(s) : ""}</td><td>{k}</td><td className="num">{fmt(data.base[s]?.[k])}</td>
                  {done.map((r) => { const v = r.evals[s]?.[k]; const b = data.base[s]?.[k]; const d = v !== undefined && b !== undefined ? v - b : undefined; return <td key={r.name} className={`num ${d === undefined ? "" : d >= 0 ? "ok" : "bad"}`}>{fmt(v)}{d !== undefined ? ` (${fmtDelta(d)})` : ""} {v !== undefined && <button className="ghost small" onClick={() => { setDiff({ run: r.name, suite: s }); setTab("diff"); }}>差異</button>}</td>; })}
                </tr>
              )))}
            </tbody>
          </table></div>
          {!done.length && <div className="muted hint" style={{ marginTop: 8 }}>還沒有訓練好的模型；base 的分數是起點。</div>}
          <div className="muted hint" style={{ marginTop: 8 }}>數字＝macro（逐來源×物件平均 F1；問答 exact）；括號＝與 base 的差（百分點）。held-out 的圖從不進訓練；灰色列只當參考。</div>
        </div>
      )}
      {tab === "diff" && <Diff runs={done} suites={suites} initial={diff} />}
      {tab === "play" && <Play models={models} reviewer={reviewer} />}
      {tab === "orin" && <Orin runs={done} />}
      </Fade>
    </div>
  );
}

function Diff({ runs, suites, initial }: { runs: Run[]; suites: string[]; initial: { run: string; suite: string } | null }) {
  const [run, setRun] = useState(initial?.run ?? runs[0]?.name ?? "");
  const [suite, setSuite] = useState(initial?.suite ?? "heldout-cctv");
  const [filter, setFilter] = useState<"changed" | "better" | "worse" | "all">("changed");
  const [pairs, setPairs] = useState<HeldoutPair[] | null>(null);
  const [n, setN] = useState(0);
  const [cur, setCur] = useState(0);
  const [err, setErr] = useState("");
  const manifestStem = (s: string) => (s.startsWith("val-") ? `val-${s.slice(4)}` : s);
  useEffect(() => {
    if (!run || !suite) return;
    api.heldoutDiff(`runs/baselines/base-${manifestStem(suite)}.json`, `runs/${run}/eval-${run}-${manifestStem(suite)}.json`, filter !== "all").then((r) => { setPairs(r.items); setN(r.n); setCur(0); setErr(""); }).catch((e) => setErr(String(e)));
  }, [run, suite, filter]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (!pairs) return; if (e.key === "ArrowRight") setCur((c) => Math.min(shown.length - 1, c + 1)); if (e.key === "ArrowLeft") setCur((c) => Math.max(0, c - 1)); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  const shown = (pairs ?? []).filter((q) => filter === "all" || filter === "changed" || (filter === "better" ? better(q) : !better(q)));
  const p = shown[cur];
  const gt = (row: HeldoutPair["row"]): Pred | null => (row.skill === "detect" ? { boxes: row.boxes ?? [] } : row.skill === "point" ? { points: row.points ?? [] } : null);
  const asPred = (skill: Skill, v: unknown): Pred => (skill === "detect" ? { boxes: v as number[][] } : skill === "point" ? { points: v as number[][] } : { text: String(v ?? "") });
  const H = "calc(100vh - 300px)";
  return (
    <div className="col">
      <div className="panel row">
        <label className="row">模型 <select value={run} onChange={(e) => setRun(e.target.value)}>{runs.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}</select></label>
        <label className="row">考卷 <select value={suite} onChange={(e) => setSuite(e.target.value)}>{suites.map((s) => <option key={s} value={s}>{suiteLabel(s)}</option>)}</select></label>
        <div className="tabs">{([["changed", "只看變化"], ["better", "變好"], ["worse", "變差"], ["all", "全部"]] as const).map(([k, l]) => <button key={k} className={filter === k ? "cur" : ""} onClick={() => setFilter(k)}>{l}</button>)}</div>
        {pairs && <span className="muted hint">{shown.length} / {n} 題</span>}
        {err && <span className="bad hint">{err}</span>}
      </div>
      {pairs && (
        <div className="rv-main">
          <Fade k={`${run}/${suite}/${cur}`} style={{ minWidth: 0 }}>
            {p && (p.row.skill === "detect" || p.row.skill === "point") && (
              <div className="grid-2">
                <div><div className="muted hint">base（實線）vs 正解（虛線）· {score(p.base)}</div><Canvas src={api.imageUrl(p.image_id, 2048)} skill={p.row.skill} value={asPred(p.row.skill, p.base.pred)} alt={gt(p.row)} readOnly labels={{ main: "base", alt: "正解" }} height={H} compact /></div>
                <div><div className="muted hint">新模型（實線）vs 正解（虛線）· {score(p.adapter)}</div><Canvas src={api.imageUrl(p.image_id, 2048)} skill={p.row.skill} value={asPred(p.row.skill, p.adapter.pred)} alt={gt(p.row)} readOnly labels={{ main: "新模型", alt: "正解" }} height={H} compact /></div>
              </div>
            )}
            {p && p.row.skill !== "detect" && p.row.skill !== "point" && (
              <div className="grid-2">
                <Canvas src={api.imageUrl(p.image_id, 2048)} skill="query" value={null} readOnly height={H} compact />
                <div className="panel col"><div><strong>{p.row.question ?? "caption"}</strong></div><div>正解：{p.row.answer}</div><div>base：{String(p.base.pred)} {p.base.ok ? "✓" : "✗"}</div><div>新模型：{String(p.adapter.pred)} {p.adapter.ok ? "✓" : "✗"}</div>{p.row.skill === "caption" && <TextDiff a={String(p.base.pred ?? "")} b={String(p.adapter.pred ?? "")} />}</div>
              </div>
            )}
            {!p && <div className="empty">這個篩選沒有題。</div>}
          </Fade>
          <aside className="panel"><div className="list" style={{ maxHeight: "calc(100vh - 280px)" }}>{shown.map((q, i) => <div key={i} className={`it${i === cur ? " cur" : ""}`} onClick={() => setCur(i)}>{i + 1}. {q.row.skill} {q.row.object ?? q.row.question} <span className={better(q) ? "ok" : "bad"}>{score(q.base)} → {score(q.adapter)}</span></div>)}</div><div className="muted hint">tp/fp/fn 或 ✓/✗；綠＝新模型比 base 好。← / → 換題。</div></aside>
        </div>
      )}
    </div>
  );
  function score(x: HeldoutPair["base"]) { return x.tp_fp_fn ? x.tp_fp_fn.join("/") : x.ok ? "✓" : "✗"; }
  function better(q: HeldoutPair) { if (q.base.tp_fp_fn && q.adapter.tp_fp_fn) { const f = (t: number[]) => (2 * t[0]) / Math.max(1, 2 * t[0] + t[1] + t[2]); return f(q.adapter.tp_fp_fn) >= f(q.base.tp_fp_fn); } return !!q.adapter.ok; }
}

function Play({ models, reviewer }: { models: Model[]; reviewer: string }) {
  const { say } = useToast();
  const [image, setImage] = useState<ImageRow | null>(null);
  const [skill, setSkill] = useState<Skill>("detect");
  const [prompt, setPrompt] = useState("person");
  const [b, setB] = useState(models[1]?.name ?? "base");
  const [res, setRes] = useState<Record<string, { pred: Pred; ms: number }> | null>(null);
  const [busy, setBusy] = useState(false);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [saveTo, setSaveTo] = useState<number | "">("");
  const [label, setLabel] = useState<Pred | null>(null);
  useEffect(() => { api.datasets().then(setDatasets).catch(() => undefined); }, []);
  useEffect(() => { setRes(null); setLabel(null); }, [image?.id, skill]);
  const run = async () => {
    if (!image) return;
    setBusy(true);
    try { const r = await api.compare(image.id, skill, prompt, b === "base" ? ["base"] : ["base", b]); setRes(r); setLabel(structuredClone(r[b === "base" ? "base" : b].pred)); } catch (e) { say(String(e), "bad", { sticky: true }); } finally { setBusy(false); }
  };
  const save = async () => {
    if (!image || saveTo === "" || !label) return;
    try { await api.saveDatasetItem(Number(saveTo), { image_id: image.id, skill, prompt, pred: res?.base?.pred ?? null, pred_model: "base", action: "correct", label, reviewer }); say("存成資料集的題 ✓"); } catch (e) { say(String(e), "bad"); }
  };
  const spatial = skill === "detect" || skill === "point";
  const pa = res?.base?.pred ?? null, pb = res?.[b]?.pred ?? null;
  const H = "calc(100vh - 260px)";
  const src = image ? api.imageUrl(image.id, 2048) : "";
  const lbl = (n: string) => models.find((m) => m.name === n)?.label ?? n;
  return (
    <div className="grid-3">
      <aside className="panel"><ImagePicker value={image?.id ?? null} onPick={setImage} /></aside>
      <section className="col" style={{ minWidth: 0 }}>
        {image && res && spatial && (b === "base" ? <Canvas src={src} skill={skill} value={label ?? pa} onChange={(p) => setLabel(p)} labels={{ main: "base（可改後存）" }} height={H} compact /> : (
          <div className="grid-2">
            <div><div className="muted hint">base · {summary(pa)} · {res.base.ms} ms</div><Canvas src={src} skill={skill} value={pa} alt={pb} readOnly labels={{ main: "base", alt: lbl(b) }} height={H} compact /></div>
            <div><div className="muted hint">{lbl(b)} · {summary(pb)} · {res[b].ms} ms</div><Canvas src={src} skill={skill} value={pb} alt={pa} readOnly labels={{ main: lbl(b), alt: "base" }} height={H} compact /></div>
          </div>
        ))}
        {image && res && !spatial && (
          <div className="grid-2">
            <div className="panel col"><div className="muted hint">base · {res.base.ms} ms</div><textarea value={label?.text ?? pa?.text ?? ""} onChange={(e) => setLabel({ text: e.target.value })} /></div>
            {b !== "base" && <div className="panel"><div className="muted hint">{lbl(b)} · {res[b].ms} ms（相對 base 的差異）</div><TextDiff a={pa?.text ?? ""} b={pb?.text ?? ""} /></div>}
          </div>
        )}
        {image && !res && <Canvas src={src} skill={skill} value={null} readOnly height={H} compact />}
        {!image && <div className="empty">左側選一張圖（資料集的圖、上傳的圖、held-out 都可以），右側選 skill 和題目，按「執行」看 base（和新模型）怎麼答。</div>}
      </section>
      <aside className="panel col">
        <div className="tabs">{(["detect", "point", "query", "caption"] as Skill[]).map((s) => <button key={s} className={s === skill ? "cur" : ""} onClick={() => { setSkill(s); setPrompt(defaultPrompt(s)); }}>{skillLabel[s]}</button>)}</div>
        <label className="f">{skill === "query" ? "問題" : skill === "caption" ? "長度" : "物件"}<input value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()} /></label>
        <label className="f">和 base 比的模型<select value={b} onChange={(e) => setB(e.target.value)}>{models.map((m) => <option key={m.name} value={m.name}>{m.name === "base" ? "只跑 base" : m.label}</option>)}</select></label>
        <button className="primary" onClick={run} disabled={!image || busy}>{busy ? "執行中…" : "執行"}</button>
        {res && (
          <div className="col card">
            <div className="hint muted">把目前畫面上的答案存成資料集的題（已確認）：</div>
            <div className="row"><select value={saveTo} onChange={(e) => setSaveTo(e.target.value === "" ? "" : +e.target.value)}><option value="">選資料集</option>{datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select><button onClick={save} disabled={saveTo === "" || !(label ?? pa)}>存成題</button></div>
            {!label && <div className="hint muted">{summary(pa ?? emptyLabel(skill))}</div>}
          </div>
        )}
      </aside>
    </div>
  );
}

function Orin({ runs }: { runs: Run[] }) {
  const [sel, setSel] = useState(runs[0]?.name ?? "");
  const r = runs.find((x) => x.name === sel);
  return (
    <div className="grid-2">
      <div className="panel col">
        <h3>匯出到 Orin AGX</h3>
        <label className="f">模型<select value={sel} onChange={(e) => setSel(e.target.value)}>{runs.map((x) => <option key={x.name} value={x.name}>{x.name}</option>)}</select></label>
        {r && <div className="row hint"><Pill kind={r.region_head ? "warn" : "ok"}>{r.region_head ? "有 region head：Photon 載不了，只能在 Forge 跑" : "Photon 相容（只掛文字 MLP）"}</Pill><span className="muted">r{r.rank} · {r.rows} 筆 · {r.steps} 步</span></div>}
        <a className="btn primary" href={r ? api.orinUrl(r.name) : "#"} aria-disabled={!r}>下載 adapter 包（adapter.pt、local_adapter.py、README、bench_orin.py、scores.json）</a>
        <pre className="code">{`pip install moondream          # 帶 kestrel（Photon）
from PIL import Image
from local_adapter import Tuned
m = Tuned("adapter.pt")
img = Image.open("frame.jpg")
m.detect(img, "person"); m.query(img, "Is anyone without a helmet?")
m.base.detect(img, "person")   # 同一個 engine 跑 base 對照`}</pre>
        <div className="muted hint">官方 client 的 adapter 只認雲端 id，所以包裡附 <code>local_adapter.py</code>（kestrel 的 AdapterProvider，從本機檔載）。Orin 上跑 <code>python bench_orin.py --images ./frames --adapter adapter.pt</code> 量延遲（base vs adapter，每題毫秒，寫到 bench.json）；分數不在 Orin 上算。</div>
      </div>
      <div className="panel col"><h3>分數（這個模型）</h3>{r && Object.entries(r.evals).map(([s, m]) => <div key={s} className="hint">{suiteLabel(s)}：{Object.entries(m).map(([k, v]) => `${k} ${fmt(v)}`).join(" · ")}</div>)}{r?.gate && <div className={r.gate.passed ? "ok" : "bad"}>{r.gate.passed ? "比 base 好" : "沒有比 base 好"}</div>}</div>
    </div>
  );
}
