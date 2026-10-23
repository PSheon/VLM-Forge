// ③ 測試 (docs/ui-spec §6): the scores are the page; 看差異 (per-item diff) and 試一張 (a
// playground that can save into a dataset) are sub-views; the Orin bundle is a section in the
// action column. Same pattern as the dataset page: no tab bar, ← / Esc goes back.

import { useCallback, useEffect, useState, type ReactNode } from "react";
import { api, defaultPrompt, emptyLabel, fmt, fmtDelta, phaseLabel, skillLabel, suiteLabel, summary, type Dataset, type HeldoutPair, type ImageRow, type Model, type Pred, type Run, type Skill } from "../api";
import Canvas, { type View } from "../components/Canvas";
import ImagePicker from "../components/ImagePicker";
import TextDiff from "../components/TextDiff";
import { PageHead, Pill, Section, useEscBack, useToast } from "../components/ui";
import { Fade, Pop } from "../components/motion";
import type { Step } from "../App";

type Sub = "scores" | "diff" | "play";
const SKILLS: Skill[] = ["detect", "point", "query", "caption"];

export default function Test({ models, reviewer, go }: { models: Model[]; reviewer: string; go: (s: Step) => void }) {
  const [view, setView] = useState<Sub>("scores");
  const [data, setData] = useState<{ suites: string[]; base: Record<string, Record<string, number>>; runs: Run[] } | null>(null);
  const [diff, setDiff] = useState<{ run: string; suite: string } | null>(null);
  const load = () => api.testSummary().then(setData).catch(() => undefined);
  useEffect(() => { load(); const t = setInterval(load, 10000); return () => clearInterval(t); }, []);
  const back = useCallback(() => setView("scores"), []);
  useEscBack(view !== "scores", back);
  const done = data?.runs.filter((r) => r.state === "done" && Object.keys(r.evals).length) ?? [];
  const running = data?.runs.find((r) => r.state === "running");
  const suites = Array.from(new Set([...(data?.suites ?? []), ...done.flatMap((r) => Object.keys(r.evals))]));
  const status = running ? <Pill kind="warn">{running.name} {running.phase === "training" ? `訓練中 ${running.last?.step ?? 0}/${running.steps}` : phaseLabel(running.phase)}</Pill> : done.length ? <Pill kind="ok">{done.length} 個模型可比</Pill> : <Pill kind="grey">還沒有訓練好的模型</Pill>;
  const head = (title: string, hint: string) => <PageHead back={view === "scores" ? undefined : { label: "← 分數", onClick: back }} title={title}>{status}<span className="muted hint">{hint}</span></PageHead>;
  return (
    <Fade k={view}>
      {view === "scores" && data && <Scores header={head("測試", "base＝原本的 Moondream；每個訓練好的模型都和它比")} data={data} done={done} running={running} suites={suites} go={go} onDiff={(run, suite) => { setDiff({ run, suite }); setView("diff"); }} onPlay={() => setView("play")} />}
      {view === "diff" && <Diff header={head("看差異", "逐題比 base 和新模型")} runs={done} suites={suites} initial={diff} />}
      {view === "play" && <Play header={head("試一張", "任何一張圖，問 base 和新模型")} models={models} reviewer={reviewer} />}
      {view === "scores" && !data && <div className="page"><div className="col">{head("測試", "")}<div className="empty"><i className="spin" />載入中…</div></div><div className="actions" /></div>}
    </Fade>
  );
}

const suiteKind = (s: string): ["target" | "val" | "ref", string] => (s === "heldout-cctv" ? ["target", "v1 目標考卷"] : s.startsWith("val-") ? ["val", "你的資料集留下的 10%"] : ["ref", "參考，不當目標"]);

function Scores({ header, data, done, running, suites, go, onDiff, onPlay }: { header: ReactNode; data: { base: Record<string, Record<string, number>> }; done: Run[]; running?: Run; suites: string[]; go: (s: Step) => void; onDiff: (run: string, suite: string) => void; onPlay: () => void }) {
  const best = done[0];
  const [orin, setOrin] = useState(false);
  return (
    <div className="page">
    <div className="col">
      {header}
      <div className="suites">
        {suites.map((s) => {
          const [kind, why] = suiteKind(s);
          const skills = SKILLS.filter((k) => data.base[s]?.[k] !== undefined || done.some((r) => r.evals[s]?.[k] !== undefined));
          if (!skills.length) return null;
          return (
            <div key={s} className={`panel col suite ${kind}`}>
              <div className="row"><h3>{suiteLabel(s)}</h3><Pill kind={kind === "target" ? "warn" : "grey"}>{why}</Pill></div>
              <div className="tw"><table className="score">
                <thead><tr><th>skill</th><th className="num">base</th>{done.map((r) => <th key={r.name} className="num"><div className="col" style={{ gap: 2, alignItems: "flex-end" }}><span>{r.name}</span>{r.gate && kind === "target" && <Pill kind={r.gate.passed ? "ok" : "bad"}>{r.gate.passed ? "過關" : "沒過"}</Pill>}</div></th>)}</tr></thead>
                <tbody>
                  {skills.map((k) => (
                    <tr key={k}>
                      <td>{skillLabel[k]}</td>
                      <td className="num">{fmt(data.base[s]?.[k])}</td>
                      {done.map((r) => { const v = r.evals[s]?.[k]; const b = data.base[s]?.[k]; const d = v !== undefined && b !== undefined ? v - b : undefined; return (
                        <td key={r.name} className="num">
                          {v === undefined ? <span className="muted">—</span> : <span className="row" style={{ justifyContent: "flex-end", gap: 6 }}><span>{fmt(v)}</span>{d !== undefined && <Pill kind={d > 0.0005 ? "ok" : d < -0.0005 ? "bad" : "grey"}>{fmtDelta(d)}</Pill>}<button className="ghost small" onClick={() => onDiff(r.name, s)} title="逐題看 base 和這個模型的差別">看差異</button></span>}
                        </td>
                      ); })}
                    </tr>
                  ))}
                </tbody>
              </table></div>
            </div>
          );
        })}
      </div>
      <div className="muted hint">分數 0–1，越高越好：Detect／Point＝找對的比例（F1），Query／Caption＝答對的比例；綠紅小字＝比 base 高／低幾個百分點。考卷的圖從不進訓練。v1 目標：工廠 CCTV 的 Detect 和 Query 都不低於 base。</div>
    </div>
    <div className="actions">
      {best ? <button className="primary big" onClick={() => onDiff(best.name, "heldout-cctv")}>看 {best.name} 的差異 →</button> : <button className="primary big" onClick={() => go("train")} disabled={!!running}>{running ? `${running.name} 訓練中…` : "去訓練 →"}</button>}
      <Pop show={!done.length} className="hint muted">{running ? `第 ${running.last?.step ?? 0}/${running.steps} 步；訓完自動在每張考卷評分，這裡會多一欄。` : "左邊是 base 的起點分數；訓一個模型後會多一欄。"}</Pop>
      {best?.gate && <div className={`hint ${best.gate.passed ? "ok" : "bad"}`}>{best.name}：{best.gate.passed ? "比 base 好（工廠 CCTV）" : "沒有比 base 好（工廠 CCTV）"}</div>}
      <div className="row hint"><button className="ghost" onClick={onPlay}>試一張圖 →</button>{best && <button className="ghost" onClick={() => onDiff(best.name, "heldout-cctv")}>看差異 →</button>}</div>
      {done.length > 0 && <Section title="匯出到 Orin" summary={`${done.length} 個模型可匯出`} open={orin} onToggle={() => setOrin(!orin)}><Orin runs={done} /></Section>}
    </div>
    </div>
  );
}

function Diff({ header, runs, suites, initial }: { header: ReactNode; runs: Run[]; suites: string[]; initial: { run: string; suite: string } | null }) {
  const [run, setRun] = useState(initial?.run ?? runs[0]?.name ?? "");
  const [suite, setSuite] = useState(initial?.suite ?? "heldout-cctv");
  const [filter, setFilter] = useState<"changed" | "better" | "worse" | "all">("changed");
  const [pairs, setPairs] = useState<HeldoutPair[] | null>(null);
  const [n, setN] = useState(0);
  const [cur, setCur] = useState(0);
  const [err, setErr] = useState("");
  const [view, setView] = useState<View | null>(null); // both canvases zoom and pan together
  useEffect(() => setView(null), [cur, run, suite]);
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
  const H = "calc(100vh - 330px)";
  const nBetter = shown.filter(better).length;
  return (
    <div className="page">
    <div className="col">
      {header}
      <div className="panel row">
        <label className="row">模型 <select value={run} onChange={(e) => setRun(e.target.value)}>{runs.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}</select></label>
        <label className="row">考卷 <select value={suite} onChange={(e) => setSuite(e.target.value)}>{suites.map((s) => <option key={s} value={s}>{suiteLabel(s)}</option>)}</select></label>
        <div className="tabs">{([["changed", "只看變化"], ["better", "變好"], ["worse", "變差"], ["all", "全部"]] as const).map(([k, l]) => <button key={k} className={filter === k ? "cur" : ""} onClick={() => setFilter(k)}>{l}</button>)}</div>
        {pairs && <span className="muted hint">{shown.length} / {n} 題{filter === "changed" && shown.length ? <>（變好 <span className="ok">{nBetter}</span>、變差 <span className="bad">{shown.length - nBetter}</span>）</> : ""}</span>}
        {err && <span className="bad hint">{err}</span>}
      </div>
      {pairs && (
          <Fade k={`${run}/${suite}/${cur}`} style={{ minWidth: 0 }}>
            {p && <div className="row hint" style={{ marginBottom: 6 }}><Pill kind={better(p) ? "ok" : "bad"}>{better(p) ? "新模型較好" : "新模型較差"}</Pill><span className="muted">{p.row.skill} · {p.row.object ?? p.row.question ?? "caption"} · 第 {cur + 1}/{shown.length} 題 · ← → 換題 · 兩張圖同步縮放</span></div>}
            {p && (p.row.skill === "detect" || p.row.skill === "point") && (
              <div className="grid-2">
                <div><div className="muted hint">base（實線）vs 正解（虛線）· {score(p.base)}</div><Canvas src={api.imageUrl(p.image_id, 2048)} skill={p.row.skill} value={asPred(p.row.skill, p.base.pred)} alt={gt(p.row)} readOnly labels={{ main: "base", alt: "正解" }} height={H} compact sync={view} onSync={setView} /></div>
                <div><div className="muted hint">{run}（實線）vs 正解（虛線）· {score(p.adapter)}</div><Canvas src={api.imageUrl(p.image_id, 2048)} skill={p.row.skill} value={asPred(p.row.skill, p.adapter.pred)} alt={gt(p.row)} readOnly labels={{ main: run, alt: "正解" }} height={H} compact sync={view} onSync={setView} /></div>
              </div>
            )}
            {p && p.row.skill !== "detect" && p.row.skill !== "point" && (
              <div className="grid-2">
                <Canvas src={api.imageUrl(p.image_id, 2048)} skill="query" value={null} readOnly height={H} compact />
                <div className="panel col"><div className="q-title">{p.row.question ?? "caption"}</div><div>正解：<b>{p.row.answer}</b></div><div className={p.base.ok ? "ok" : "bad"}>base：{String(p.base.pred)} {p.base.ok ? "✓" : "✗"}</div><div className={p.adapter.ok ? "ok" : "bad"}>{run}：{String(p.adapter.pred)} {p.adapter.ok ? "✓" : "✗"}</div>{p.row.skill === "caption" && <TextDiff a={String(p.base.pred ?? "")} b={String(p.adapter.pred ?? "")} />}</div>
              </div>
            )}
            {!p && <div className="empty">這個篩選沒有題。</div>}
          </Fade>
      )}
    </div>
    <div className="actions">
      <div className="row"><button onClick={() => setCur((c) => Math.max(0, c - 1))} disabled={cur <= 0} title="←">←</button><button className="primary big" onClick={() => setCur((c) => Math.min(shown.length - 1, c + 1))} disabled={!shown.length || cur >= shown.length - 1}>下一題 → <kbd>→</kbd></button></div>
      {pairs && <div className="list" style={{ maxHeight: "calc(100vh - 220px)" }}>{shown.map((q, i) => <div key={i} className={`it${i === cur ? " cur" : ""}`} onClick={() => setCur(i)}>{i + 1}. {q.row.skill} {q.row.object ?? q.row.question} <span className={better(q) ? "ok" : "bad"}>{score(q.base)} → {score(q.adapter)}</span></div>)}</div>}
      <div className="muted hint">框題＝對/多/漏 的數量，文字題＝✓/✗；綠＝新模型比 base 好。</div>
    </div>
    </div>
  );
  function score(x: HeldoutPair["base"]) { return x.tp_fp_fn ? x.tp_fp_fn.join("/") : x.ok ? "✓" : "✗"; }
  function better(q: HeldoutPair) { if (q.base.tp_fp_fn && q.adapter.tp_fp_fn) { const f = (t: number[]) => (2 * t[0]) / Math.max(1, 2 * t[0] + t[1] + t[2]); return f(q.adapter.tp_fp_fn) >= f(q.base.tp_fp_fn); } return !!q.adapter.ok; }
}

// quick questions for the factory / warehouse checks the tool is built for (GOALS §4)
const QUICK: [Skill, string][] = [
  ["query", "Is there a person on duty?"], ["query", "Is anyone without a helmet?"], ["query", "Is anyone without a high-visibility vest?"],
  ["query", "Is an exit or aisle blocked?"], ["query", "Is there debris or liquid on the floor?"], ["query", "How many forklifts are there?"],
  ["detect", "person"], ["detect", "forklift"], ["detect", "helmet"], ["point", "person"], ["caption", "normal"],
];

function Play({ header, models, reviewer }: { header: ReactNode; models: Model[]; reviewer: string }) {
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
  const [view, setView] = useState<View | null>(null);
  useEffect(() => { api.datasets().then(setDatasets).catch(() => undefined); }, []);
  useEffect(() => { if (!image) api.images().then((im) => { if (im.length) setImage(im[0]); }).catch(() => undefined); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setRes(null); setLabel(null); }, [image?.id, skill]);
  useEffect(() => { if (models[1] && b === "base") setB(models[1].name); }, [models]); // eslint-disable-line react-hooks/exhaustive-deps
  const run = async () => {
    if (!image || busy) return;
    setBusy(true);
    try { const r = await api.compare(image.id, skill, prompt, b === "base" ? ["base"] : ["base", b]); setRes(r); setLabel(structuredClone(r[b === "base" ? "base" : b].pred)); } catch (e) { say(String(e), "bad", { sticky: true }); } finally { setBusy(false); }
  };
  const save = async () => {
    if (!image || saveTo === "" || !label) return;
    try { await api.saveDatasetItem(Number(saveTo), { image_id: image.id, skill, prompt, pred: res?.base?.pred ?? null, pred_model: "base", action: "correct", label, reviewer }); say("存成資料集的題 ✓"); } catch (e) { say(String(e), "bad"); }
  };
  const spatial = skill === "detect" || skill === "point";
  const pa = res?.base?.pred ?? null, pb = res?.[b]?.pred ?? null;
  const H = "calc(100vh - 290px)";
  const src = image ? api.imageUrl(image.id, 2048) : "";
  const lbl = (n: string) => models.find((m) => m.name === n)?.label ?? n;
  const pick = (s: Skill, p: string) => { setSkill(s); setPrompt(p); };
  return (
    <div className="page">
    <div className="col">
    {header}
    <div className="play-left">
      <aside className="panel"><ImagePicker value={image?.id ?? null} onPick={setImage} /></aside>
      <section className="col" style={{ minWidth: 0 }}>
        {image && <div className="row hint"><span className="muted">{image.path.split("/").slice(-1)[0]}</span>{res && <><Pill kind="grey">base · {summary(pa)} · {res.base.ms} ms</Pill>{b !== "base" && res[b] && <Pill kind="ok">{lbl(b)} · {summary(pb)} · {res[b].ms} ms</Pill>}</>}{busy && <span className="muted"><i className="spin" />模型在答…</span>}</div>}
        {image && res && spatial && (b === "base" ? <Canvas src={src} skill={skill} value={label ?? pa} onChange={(p) => setLabel(p)} labels={{ main: "base（可改後存）" }} height={H} compact /> : (
          <div className="grid-2">
            <div><div className="muted hint">base</div><Canvas src={src} skill={skill} value={pa} alt={pb} readOnly labels={{ main: "base", alt: lbl(b) }} height={H} compact sync={view} onSync={setView} /></div>
            <div><div className="muted hint">{lbl(b)}</div><Canvas src={src} skill={skill} value={pb} alt={pa} readOnly labels={{ main: lbl(b), alt: "base" }} height={H} compact sync={view} onSync={setView} /></div>
          </div>
        ))}
        {image && res && !spatial && (
          <div className="grid-2">
            <Canvas src={src} skill="query" value={null} readOnly height={H} compact />
            <div className="col">
              <div className="panel col"><div className="muted hint">base · {res.base.ms} ms</div><textarea value={label?.text ?? pa?.text ?? ""} onChange={(e) => setLabel({ text: e.target.value })} /></div>
              {b !== "base" && res[b] && <div className="panel"><div className="muted hint">{lbl(b)} · {res[b].ms} ms（和 base 的差異）</div><TextDiff a={pa?.text ?? ""} b={pb?.text ?? ""} /></div>}
            </div>
          </div>
        )}
        {image && !res && <Canvas src={src} skill={skill} value={null} readOnly height={H} compact />}
        {!image && <div className="empty">左邊選一張圖。</div>}
      </section>
    </div>
    </div>
      <div className="actions">
        <button className="primary big" onClick={run} disabled={!image || busy}>{busy ? "執行中…" : !image ? "先選一張圖" : `執行：${skillLabel[skill]} ${short(prompt)} ⏎`}</button>
        <div className="tabs">{SKILLS.map((s) => <button key={s} className={s === skill ? "cur" : ""} onClick={() => pick(s, defaultPrompt(s))}>{skillLabel[s]}</button>)}</div>
        <label className="f">{skill === "query" ? "問題" : skill === "caption" ? "長度（short / normal / long）" : "找什麼"}<input value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()} /></label>
        <div className="chips">{QUICK.map(([s, p]) => <button key={s + p} className={`chip-btn${s === skill && p === prompt ? " on" : ""}`} onClick={() => pick(s, p)} title={skillLabel[s]}>{s === "query" ? p : `${skillLabel[s]}: ${p}`}</button>)}</div>
        <label className="f">和 base 比的模型<select value={b} onChange={(e) => setB(e.target.value)}>{models.map((m) => <option key={m.name} value={m.name}>{m.name === "base" ? "只跑 base" : m.label}</option>)}</select></label>
        <Pop show={!!res} className="col card">
          <div className="hint muted">把畫面上的答案存成資料集的題（視為已確認）：</div>
          <div className="row"><select value={saveTo} onChange={(e) => setSaveTo(e.target.value === "" ? "" : +e.target.value)}><option value="">選資料集</option>{datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}</select><button onClick={save} disabled={saveTo === "" || !(label ?? pa)}>存成題</button></div>
          {!label && <div className="hint muted">{summary(pa ?? emptyLabel(skill))}</div>}
        </Pop>
      </div>
    </div>
  );
}

const short = (t: string) => (t.length > 26 ? t.slice(0, 26) + "…" : t);

function Orin({ runs }: { runs: Run[] }) {
  const [sel, setSel] = useState(runs[0]?.name ?? "");
  const [code, setCode] = useState(false);
  const r = runs.find((x) => x.name === sel);
  return (
    <>
      <label className="f">模型<select value={sel} onChange={(e) => setSel(e.target.value)}>{runs.map((x) => <option key={x.name} value={x.name}>{x.name}</option>)}</select></label>
      {r && <div className="col hint"><Pill kind={r.region_head ? "warn" : "ok"}>{r.region_head ? "有 region head：Photon 載不了，只能在 Forge 跑" : "Photon 相容（只掛文字 MLP）"}</Pill><span className="muted">r{r.rank} · {r.rows} 筆 · {r.steps} 步</span></div>}
      <a className="btn" href={r ? api.orinUrl(r.name) : "#"} aria-disabled={!r} style={{ textAlign: "center" }}>下載 adapter 包（zip）</a>
      <div className="muted small">內含 adapter.pt、local_adapter.py、README.md、bench_orin.py、scores.json。官方 client 的 adapter 只認雲端 id，所以附 local_adapter.py 從本機檔載。</div>
      <button className="ghost" onClick={() => setCode(!code)} style={{ alignSelf: "flex-start" }}>{code ? "收起用法" : "Orin 上怎麼用 ▾"}</button>
      {code && <pre className="code" style={{ fontSize: 11 }}>{`pip install moondream   # 帶 kestrel
from PIL import Image
from local_adapter import Tuned
m = Tuned("adapter.pt")
img = Image.open("frame.jpg")
m.detect(img, "person")
m.base.detect(img, "person")  # base 對照
# 延遲：python bench_orin.py --images ./frames`}</pre>}
    </>
  );
}
