// ① 資料集 (docs/ui-spec §3): the list, and the dataset page with four tabs —
// 圖片（加圖、標註）／檢查（一次一題）／設定（AI 導師）／匯出. Everything about data
// happens here; training and testing are the next two steps.

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { DEFAULT_PROBES, api, parseFixed, phaseLabel, type Batch, type Dataset, type GridCell, type Skill, type Status } from "../api";
import { ActionsHead, Bar, ConfirmDialog, Dialog, PageHead, Pill, Section, Skeleton, askNotify, eta, useToast } from "../components/ui";
import { Fade, Item, Pop, Stagger } from "../components/motion";
import Review from "./Review";
import type { Step } from "../App";

type View = "images" | "review"; // the image grid, or the one-question-at-a-time review

export default function Datasets({ status, refresh, go, reviewer, setGuard }: { status: Status | null; refresh: () => void; go: (s: Step) => void; reviewer: string; setGuard: (g: (() => boolean) | null) => void }) {
  const [list, setList] = useState<Dataset[] | null>(null); // null until the first response
  const [cur, setCur] = useState<number | null>(() => { const m = location.hash.match(/ds=(\d+)/); return m ? +m[1] : null; });
  const [showNew, setShowNew] = useState(false);
  const load = useCallback(() => api.datasets().then(setList).catch(() => setList((l) => l ?? [])), []);
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, [load]);
  const detail = list?.find((d) => d.id === cur) ?? null;
  if (detail) return <Detail key={detail.id} d={detail} status={status} reload={() => { load(); refresh(); }} go={go} reviewer={reviewer} setGuard={setGuard} onBack={() => setCur(null)} onDeleted={() => { setCur(null); load(); }} />;
  const tot = (k: "images" | "items" | "trainable" | "images_pending") => (list ?? []).reduce((a, d) => a + (d[k] ?? 0), 0);
  return (
    <div className="page">
      <div className="col">
        <PageHead title="資料集"><span className="muted hint">一個資料集＝一批照片＋AI 導師出的題＋你確認的答案</span></PageHead>
        {list === null && <Skeleton h={88} r={10} n={3} />}
        {list?.length === 0 && <div className="empty">還沒有資料集。建一個，把照片丟進去，AI 導師先標一輪，你再檢查。</div>}
        <Stagger className="ds-list">
          {(list ?? []).map((d) => <Item key={d.id}><DatasetRow d={d} onClick={() => setCur(d.id)} /></Item>)}
        </Stagger>
      </div>
      <div className="actions">
        <ActionsHead />
        <button className="primary big" onClick={() => setShowNew(true)}>＋ 新資料集</button>
        {!!list?.length && <div className="stat" style={{ flexWrap: "wrap" }}><span>資料集 <b>{list?.length}</b></span><span>圖 <b>{tot("images")}</b></span><span>題 <b>{tot("items")}</b></span><span>已確認 <b>{tot("trainable")}</b></span><span>還要看 <b>{tot("images_pending")}</b> 張</span></div>}
        <div className="hint muted">可以陸續加圖、重複標註；每個資料集自動留 10% 當驗證集。</div>
      </div>
      <NewDataset open={showNew} existing={(list ?? []).map((d) => d.name)} onClose={() => setShowNew(false)} onCreated={(d) => { setShowNew(false); load(); setCur(d.id); }} />
    </div>
  );
}

/** An autogen batch whose job is gone (server restart) or stopped, with work left. */
function interrupted(b?: Batch): boolean {
  const u = b?.unfinished;
  return !!u && b?.job?.state !== "running" && u.images + u.unpredicted + u.unjudged > 0;
}

function statusPill(d: Dataset): [string, "ok" | "bad" | "warn" | "grey"] {
  const b = d.batches?.[0];
  if (b?.job?.state === "running") return [`標註中 ${b.job.done}/${b.job.total}`, "warn"];
  if (b?.job?.state === "error") return ["上次標註失敗", "bad"];
  if (interrupted(b)) return ["標註中斷，可接著跑", "bad"];
  if (d.images_pending > 0) return [`還有 ${d.images_pending} 張要看`, "warn"];
  if (d.items > 0) return ["全部確認", "ok"];
  if (d.images > 0) return [`${d.images} 張還沒標`, "warn"];
  return ["沒有圖", "grey"];
}

/** One dataset as a list row: thumbnails, name, how far review got, the numbers, its state. */
function DatasetRow({ d, onClick }: { d: Dataset; onClick: () => void }) {
  const [text, kind] = statusPill(d);
  const seen = d.images ? Math.round((100 * (d.images - d.images_pending)) / d.images) : 0;
  return (
    <div className="ds-row" onClick={onClick} role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onClick()}>
      <div className="strip-th">{(d.thumbs ?? []).slice(0, 4).map((id) => <img key={id} src={api.imageUrl(id, 200)} alt="" loading="lazy" />)}{!(d.thumbs ?? []).length && <span className="muted small">沒有圖</span>}</div>
      <div className="col" style={{ gap: 4, minWidth: 0 }}>
        <div className="row"><span className="name">{d.name}</span><Pill kind={kind}>{text}</Pill></div>
        <div className="row small muted"><span>{d.config.mode === "fixed" ? "只指定題目" : d.config.mode === "both" ? "自動出題＋指定題目" : "自動出題"}</span><span>· 更新 {d.updated_at.slice(5, 16).replace("T", " ")}</span></div>
        <div className="progress" title={`看過 ${seen}%`}><i style={{ width: `${seen}%` }} /></div>
      </div>
      <div className="stat"><span>圖 <b>{d.images}</b></span><span>題 <b>{d.items}</b></span><span>已確認 <b>{d.trainable}</b></span><span>驗證 <b>{d.val_images}</b></span></div>
      <span className="chev muted">›</span>
    </div>
  );
}

function NewDataset({ open, existing, onClose, onCreated }: { open: boolean; existing: string[]; onClose: () => void; onCreated: (d: Dataset) => void }) {
  const [name, setName] = useState("");
  const [note, setNote] = useState("");
  const [err, setErr] = useState("");
  const { say } = useToast();
  useEffect(() => { setName(""); setNote(""); setErr(""); }, [open]);
  const dup = existing.includes(name.trim());
  const create = async () => { try { const d = await api.createDataset(name.trim(), { mode: "auto" }, note); say(`建好了：${d.name}`); onCreated(d); } catch (e) { setErr(String(e)); } };
  return (
    <Dialog open={open} onClose={onClose}>
      <h2>新資料集</h2>
      <label className="f">名稱<input autoFocus value={name} placeholder="例如 工廠-1025" onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && name.trim() && !dup) create(); }} /></label>
      {dup && <div className="bad hint">已經有這個名字</div>}
      <label className="f">備註（選填）<input value={note} onChange={(e) => setNote(e.target.value)} /></label>
      {err && <div className="bad hint">{err}</div>}
      <div className="row" style={{ justifyContent: "flex-end" }}><button onClick={onClose}>取消</button><button className="primary" disabled={!name.trim() || dup} onClick={create}>建立</button></div>
    </Dialog>
  );
}

function Detail({ d, status, reload, go, reviewer, setGuard, onBack, onDeleted }: { d: Dataset; status: Status | null; reload: () => void; go: (s: Step) => void; reviewer: string; setGuard: (g: (() => boolean) | null) => void; onBack: () => void; onDeleted: () => void }) {
  const { say } = useToast();
  const [tab, setTab] = useState<View>("images");
  const [spotOnly, setSpotOnly] = useState(false);
  const [openSec, setOpenSec] = useState<"" | "teacher" | "export">("");
  const [openImage, setOpenImage] = useState<string | null>(null);
  const [full, setFull] = useState<Dataset | null>(null);
  const [grid, setGrid] = useState<GridCell[] | null>(null);
  const [dir, setDir] = useState("");
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const [mode, setMode] = useState(d.config.mode ?? "auto");
  const [fixed, setFixed] = useState(d.config.fixed ?? "");
  const [topics, setTopics] = useState((d.config.topics ?? []).join("\n"));
  const [focus, setFocus] = useState(d.config.focus ?? "");
  const [probes, setProbes] = useState((d.config.probes ?? DEFAULT_PROBES).join("\n"));
  const probeList = probes.split("\n").map((x) => x.trim()).filter(Boolean);
  const [skills, setSkills] = useState<Skill[]>(d.config.skills ?? ["detect", "point", "query", "caption"]);
  const [n, setN] = useState<number | "">("");
  const [askDelete, setAskDelete] = useState(false);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [askRemove, setAskRemove] = useState(false);
  const [askRelabel, setAskRelabel] = useState(false);
  const lastPick = useRef<string | null>(null);
  const [rename, setRename] = useState(false);
  const [newName, setNewName] = useState(d.name);
  const [exp, setExp] = useState<{ jsonl: string; rows: number; zip?: string; images?: number } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const load = useCallback(() => { api.dataset(d.id).then(setFull).catch(() => undefined); api.datasetGrid(d.id).then(setGrid).catch(() => undefined); }, [d.id]);
  useEffect(() => { load(); }, [load, d.updated_at, d.items, d.pending]);
  const batch: Batch | undefined = full?.batches?.[0];
  const running = batch?.job?.state === "running";
  useEffect(() => { if (running) { const t = setInterval(load, 2500); return () => clearInterval(t); } }, [running, load]);
  const parsed = parseFixed(fixed);
  const nFixed = parsed.ok.detect.length + parsed.ok.point.length + parsed.ok.query.length + parsed.ok.caption.length;
  const unlabelled = full?.unlabelled ?? 0;
  const judgeOk = status?.engine.judge_ok ?? true;
  const gpuBusy = !!status?.engine.busy && !running;
  const cfgDirty = mode !== (d.config.mode ?? "auto") || fixed !== (d.config.fixed ?? "") || topics !== (d.config.topics ?? []).join("\n") || focus !== (d.config.focus ?? "") || skills.join() !== (d.config.skills ?? ["detect", "point", "query", "caption"]).join() || probeList.join() !== (d.config.probes ?? DEFAULT_PROBES).join();

  const saveConfig = async () => {
    await api.updateDataset(d.id, { config: { mode, fixed, topics: topics.split("\n").map((s) => s.trim()).filter(Boolean), focus, skills, probes: probeList } });
    reload();
  };
  const addFiles = async (files: FileList | File[] | null) => {
    if (!files || !files.length) return;
    setBusy(true);
    try { let added = 0; for (const f of Array.from(files)) added += (await api.uploadToDataset(d.id, f)).added; say(`加入 ${added} 張`); reload(); load(); } catch (e) { say(String(e), "bad", { sticky: true }); } finally { setBusy(false); }
  };
  const addFolder = async () => {
    if (!dir.trim()) return;
    setBusy(true);
    try { const r = await api.addFolder(d.id, dir.trim()); say(`加入 ${r.added} 張`); setDir(""); reload(); load(); } catch (e) { say(String(e), "bad", { sticky: true }); } finally { setBusy(false); }
  };
  const label = async () => {
    askNotify();
    setBusy(true);
    try { if (cfgDirty) await saveConfig(); await api.label(d.id, n === "" ? undefined : Number(n)); say(`開始標註 ${n === "" ? unlabelled : Math.min(Number(n), unlabelled)} 張，先載入出題模型…`, "ok", { ms: 2500 }); reload(); load(); } catch (e) { say(String(e), "bad", { sticky: true }); } finally { setBusy(false); }
  };
  const cancel = async () => { if (batch) { await api.cancelBatch(batch.id).catch(() => undefined); say("取消中，這一批出完就停", "warn"); } };
  const resume = async () => { if (!batch) return; setBusy(true); try { await api.resumeBatch(batch.id); say("接著跑：先補模型作答與比對…", "ok", { ms: 2500 }); reload(); load(); } catch (e) { say(String(e), "bad", { sticky: true }); } finally { setBusy(false); } };
  const doExport = async () => { setBusy(true); try { const r = await api.exportDataset(d.id); setExp(r); say(`匯出 ${r.rows} 列`); } catch (e) { say(String(e), "bad"); } finally { setBusy(false); } };
  const doRename = async () => { try { await api.updateDataset(d.id, { name: newName.trim() }); setRename(false); say("改好了"); reload(); } catch (e) { say(String(e), "bad"); } };
  // multi-select on the grid: the checkbox (or any thumb once something is picked) toggles,
  // Shift extends from the last pick; the removal itself lives in the action column
  const togglePick = (id: string, shift: boolean) => {
    const ids = (grid ?? []).map((g) => g.image_id);
    const a = shift && lastPick.current ? ids.indexOf(lastPick.current) : -1, b = ids.indexOf(id); // read the ref now, not inside the updater
    setPicked((cur) => {
      const next = new Set(cur);
      if (shift && a >= 0) { const on = !cur.has(id); for (const x of ids.slice(Math.min(a, b), Math.max(a, b) + 1)) { if (on) next.add(x); else next.delete(x); } }
      else if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
    lastPick.current = id;
  };
  useEffect(() => { setPicked(new Set()); }, [d.id]);
  useEffect(() => { if (grid) setPicked((cur) => { const have = new Set(grid.map((g) => g.image_id)); const next = new Set([...cur].filter((x) => have.has(x))); return next.size === cur.size ? cur : next; }); }, [grid]);
  useEffect(() => {
    if (tab !== "images" || !picked.size) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if ((t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) || document.querySelector(".overlay")) return;
      if (e.key === "Escape") setPicked(new Set());
      else if ((e.key === "Delete" || e.key === "Backspace") && !running) { e.preventDefault(); setAskRemove(true); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });
  const pickedCells = (grid ?? []).filter((g) => picked.has(g.image_id));
  const pickedQ = pickedCells.reduce((s, g) => s + g.n, 0), pickedDone = pickedCells.reduce((s, g) => s + g.n - g.pending, 0);
  const pickedMachine = pickedCells.reduce((s, g) => s + g.n - (g.kept ?? 0), 0); // what 重新標註 replaces
  const doRelabel = async () => {
    setAskRelabel(false); setBusy(true); askNotify();
    try { if (cfgDirty) await saveConfig(); const r = await api.relabel(d.id, [...picked]); setPicked(new Set()); say(`重新標註 ${r.job?.total ?? picked.size} 張：換掉 ${r.reset.dropped} 題，保留你看過的 ${r.reset.kept} 題`, "ok", { ms: 3000 }); reload(); load(); } catch (e) { say(String(e), "bad", { sticky: true }); } finally { setBusy(false); }
  };
  const doRemove = async () => {
    setAskRemove(false); setBusy(true);
    try { const r = await api.removeImages(d.id, [...picked]); setPicked(new Set()); say(`移除了 ${r.removed} 張（${r.items} 題）`, "warn"); reload(); load(); } catch (e) { say(String(e), "bad", { sticky: true }); } finally { setBusy(false); }
  };
  const doDelete = async () => { await api.deleteDataset(d.id); say(`刪了 ${d.name}`, "warn"); setAskDelete(false); onDeleted(); };
  const freeGb = status?.engine.gpu_free_gb ?? null;
  const judgeLoaded = status?.engine.judge_loaded ?? false;
  const needGb = status?.engine.judge_need_gb ?? 70;
  const lowMem = freeGb !== null && !judgeLoaded && freeGb < needGb;
  const labelText = interrupted(batch) ? "先把上一輪接著跑完" : !judgeOk ? "出題模型缺，不能標" : gpuBusy ? "排隊中…（GPU 忙）" : lowMem ? `GPU 只剩 ${freeGb!.toFixed(0)} GB（出題要 ${needGb} GB），等別的訓練結束` : unlabelled === 0 ? "沒有新圖" : `標註新加的 ${n === "" ? unlabelled : Math.min(Number(n), unlabelled)} 張`;
  const total = (full ?? d).images;
  const sp = (full ?? d).spot;
  const pendingImgs = (full ?? d).images_pending, trainable = (full ?? d).trainable;
  const labelBlocked = !judgeOk || gpuBusy || lowMem || (mode !== "auto" && nFixed === 0);
  // one orange button: the next thing to do (docs/ui-spec §3)
  const cta: { text: string; run?: () => void; disabled?: boolean } | null = interrupted(batch) ? null
    : total === 0 ? { text: "選照片加進來", run: () => fileInput.current?.click(), disabled: busy }
    : running ? { text: `標註中 ${batch?.job?.done ?? 0}/${batch?.job?.total ?? 0}`, disabled: true }
    : unlabelled > 0 && !labelBlocked ? { text: labelText, run: label, disabled: busy }
    : pendingImgs > 0 ? { text: `去檢查（還有 ${pendingImgs} 張）→`, run: () => setTab("review") }
    : unlabelled > 0 ? { text: labelText, disabled: true }
    : trainable > 0 ? { text: `已確認 ${trainable} 題 → 去訓練`, run: () => go("train") }
    : null;
  const skippedNote = batch?.job?.student_error ? `模型沒作答（${batch.job.student_error}），這批全部交給你看` : null;
  const [pillText, pillKind] = statusPill(full ?? d);
  // title row inside the left column, so the action column (and its primary button) starts at
  // the same height as on every other page. No tab bar: the grid is the page, review opens from
  // a thumbnail or the primary button, settings and export live in the action column.
  const header = (extra?: ReactNode) => (
    <PageHead back={tab === "review" ? { label: `← 全部圖片（${total}）`, onClick: () => setTab("images") } : { label: "← 資料集", onClick: onBack }} title={rename ? <span className="row"><input autoFocus value={newName} onChange={(e) => setNewName(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") doRename(); if (e.key === "Escape") setRename(false); }} style={{ width: 220, fontSize: 15 }} /><button className="primary" onClick={doRename}>存</button><button onClick={() => setRename(false)}>取消</button></span> : <>{d.name}<button className="ghost" onClick={() => setRename(true)} title="改名" aria-label="改名">✎</button></>}>
      {extra ?? <>
        <Pill kind={pillKind} onClick={tab === "images" && pendingImgs > 0 ? () => setTab("review") : undefined}>{pillText}</Pill>
        <span className="hint muted">{(full ?? d).items ? `題 ${(full ?? d).items} · 已確認 ${(full ?? d).trainable} · ` : ""}驗證集 {(full ?? d).val_images} 張</span>
      </>}
    </PageHead>
  );
  const modeText = mode === "auto" ? "自動出題" : mode === "both" ? `自動出題＋指定 ${nFixed} 題` : `只指定 ${nFixed} 題`;
  return (
    <div className="col">
      <Fade k={tab}>
      {tab === "images" && (
        <div className="page">
          <div className="col">
            {header()}
            <div className={`drop${drag ? " on" : ""}`} onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={(e) => { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files); }}>
              把照片拖到這裡，或 <button onClick={() => fileInput.current?.click()} disabled={busy}>選檔案</button>（可多張、可 zip）
              <input ref={fileInput} type="file" accept="image/*,.zip" multiple hidden onChange={(e) => addFiles(e.target.files)} />
            </div>
            <div className="row">
              <input placeholder="GB300 上的資料夾路徑，例如 /home/paul/VLM-Forge/workspace/captures/1026" value={dir} onChange={(e) => setDir(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addFolder()} style={{ flex: 1 }} />
              <button onClick={addFolder} disabled={busy || !dir.trim()}>加入資料夾</button>
            </div>
            <div className="small muted">{picked.size ? `已選 ${picked.size} 張 · 點縮圖加選或取消 · Shift 連選 · Esc 取消選取` : grid?.length ? `${grid.length} 張（val ${grid.filter((g) => g.split === "val").length}）· 紅色數字＝這張還有幾題不一致 · 點一張進去檢查 · 左上角 ☐ 多選` : grid === null ? "載入中…" : "還沒有圖"}</div>
            {grid === null && (full ?? d).images > 0 && <div className="thumbs" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(84px, 1fr))" }}>{Array.from({ length: Math.min(30, (full ?? d).images) }, (_, i) => <div key={i} className="skel" style={{ aspectRatio: "4/3", borderRadius: 4 }} />)}</div>}
            <Grid cells={grid ?? []} picked={picked} onPick={togglePick} onOpen={(cell) => { if (cell.n) { setOpenImage(cell.image_id); setTab("review"); } }} />
          </div>
          <div className="actions">
            <ActionsHead />
            {cta && <button className="primary big" onClick={cta.run} disabled={cta.disabled}>{cta.text}</button>}
            <Pop show={total === 0} className="hint muted">拖曳、選檔、zip，或貼資料夾路徑；10% 自動留作驗證集。</Pop>
            <Pop show={total > 0 && unlabelled > 0 && !running && !interrupted(batch) && !labelBlocked} className="hint muted">{modeText}，約 20 秒/張；一致的自動通過，不一致的才要你看。</Pop>
            {total > 0 && mode !== "auto" && nFixed === 0 && <div className="bad hint">AI 導師設定裡還沒有有效的指定題目</div>}
            <div className="row hint">
              {cta?.run !== label && unlabelled > 0 && !running && !interrupted(batch) && <span className="muted">{labelBlocked ? labelText : ""}</span>}
              {cta && !cta.text.startsWith("去檢查") && pendingImgs > 0 && !running && <button className="ghost" onClick={() => setTab("review")}>去檢查（{pendingImgs} 張）→</button>}
            </div>
            <Pop show={picked.size > 0} className="panel col">
              <div className="row" style={{ justifyContent: "space-between" }}><b>已選 {picked.size} 張</b><span className="row"><button className="ghost" onClick={() => setPicked(new Set((grid ?? []).map((g) => g.image_id)))}>全選</button><button className="ghost" onClick={() => setPicked(new Set())}>取消選取</button></span></div>
              <div className="small muted">{pickedQ ? `${pickedQ} 題，其中 ${pickedDone} 題已確認或自動通過` : "還沒標註"}</div>
              {running ? <div className="hint warn-text">標註中，跑完或取消後才能重新標註或移除</div> : (
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <button onClick={() => setAskRelabel(true)} disabled={busy || labelBlocked} title={labelBlocked ? labelText : "用目前的 AI 導師設定重出這些圖的題；你看過的題保留"}>重新標註 {picked.size} 張…</button>
                  <button className="ghost bad" onClick={() => setAskRemove(true)} disabled={busy}>從資料集移除…</button>
                </div>
              )}
              {!running && labelBlocked && <div className="hint muted">{labelText}</div>}
            </Pop>
            {!!sp?.total && <SpotCard sp={sp} onOpen={() => { setSpotOnly(true); setTab("review"); }} />}
            {batch && <BatchCard b={batch} onCancel={cancel} onResume={resume} busy={busy || gpuBusy} skippedNote={skippedNote} />}
            {full && full.batches && full.batches.length > 1 && (
              <div className="col"><strong className="small">批次紀錄</strong>{full.batches.slice(1, 6).map((b) => <div key={b.id} className="small muted">{b.created_at.slice(5, 16).replace("T", " ")} · {b.total} 題 · {b.job?.state === "cancelled" ? "取消" : b.job?.state === "error" ? "失敗" : "完成"} · 一致 {b.verdicts?.accept ?? 0} · 不一致 {b.verdicts?.fix ?? 0}</div>)}</div>
            )}

            <Section title="AI 導師設定" summary={`${modeText}${cfgDirty ? " · 改了還沒存" : ""}`} open={openSec === "teacher"} onToggle={() => setOpenSec(openSec === "teacher" ? "" : "teacher")} dirty={cfgDirty}>
              <label className="row hint"><input type="radio" name="mode" checked={mode === "auto"} onChange={() => setMode("auto")} /> 自動出題（物件、問答、巡檢題、caption）</label>
              <label className="row hint"><input type="radio" name="mode" checked={mode === "both"} onChange={() => setMode("both")} /> 自動出題＋指定題目</label>
              <label className="row hint"><input type="radio" name="mode" checked={mode === "fixed"} onChange={() => setMode("fixed")} /> 只指定題目</label>
              {mode !== "auto" && (
                <>
                  <textarea value={fixed} onChange={(e) => setFixed(e.target.value)} placeholder={"每行一題，例如：\ndetect: person, helmet\npoint: person\nquery: Is anyone without a helmet?\ncaption: normal"} style={{ minHeight: 110, fontFamily: "var(--mono)", fontSize: 12 }} />
                  <div className="hint">{nFixed > 0 ? <span className="ok">✓ {nFixed} 題有效（detect {parsed.ok.detect.length}、point {parsed.ok.point.length}、query {parsed.ok.query.length}、caption {parsed.ok.caption.length}）</span> : <span className="muted">還沒有有效的題</span>}{parsed.bad.length > 0 && <span className="bad"> · ✗ 第 {parsed.bad.join("、")} 行看不懂（要像 <code>detect: person</code>）</span>}</div>
                </>
              )}
              <label className="f">額外巡檢主題（每行一個；已內建值班、安全帽／背心／口罩／手套、出口、門、地面、堆高機、燈、管制區、無人看管物品）<textarea value={topics} onChange={(e) => setTopics(e.target.value)} style={{ minHeight: 54 }} /></label>
              {mode !== "fixed" && <label className="f">每張圖都問的物件（每行一個；導師沒看到就當作「沒有」，教模型別亂畫框）<textarea value={probes} onChange={(e) => setProbes(e.target.value)} style={{ minHeight: 54 }} placeholder={DEFAULT_PROBES.join("\n")} /></label>}
              <label className="f">側重（給導師的一句話）<input value={focus} onChange={(e) => setFocus(e.target.value)} placeholder="例如 night-shift factory floor" /></label>
              <div className="f"><span>出哪些 skill</span><div className="row">{(["detect", "point", "query", "caption"] as Skill[]).map((s) => <label key={s} className="row hint"><input type="checkbox" checked={skills.includes(s)} onChange={(e) => setSkills(e.target.checked ? [...skills, s] : skills.filter((x) => x !== s))} />{s}</label>)}</div></div>
              <label className="f">一次只標幾張（空＝全部還沒標的）<input type="number" min={1} value={n} onChange={(e) => setN(e.target.value === "" ? "" : +e.target.value)} style={{ width: 120 }} /></label>
              <div className="row save-row"><button onClick={saveConfig} disabled={!cfgDirty}>{cfgDirty ? "儲存設定" : "已儲存"}</button><span className="muted small">按「標註」時也會自動存；已出的題不變。</span></div>
            </Section>

            <Section title="匯出" summary={exp ? `已匯出 ${exp.rows} 列` : `${trainable} 題可匯出`} open={openSec === "export"} onToggle={() => setOpenSec(openSec === "export" ? "" : "export")}>
              <div className="hint muted">已確認的題 → jsonl（座標 0–1，含 split 欄）＋ 圖 zip。訓練不需要先匯出，這是給別的工具用的。</div>
              <button onClick={doExport} disabled={busy || !trainable}>匯出 {trainable} 題</button>
              {exp && <div className="hint">寫到 <code>{exp.jsonl}</code> · <a href={api.downloadUrl(d.id, "jsonl")}>下載 jsonl</a>{exp.zip && <> · <a href={api.downloadUrl(d.id, "zip")}>下載圖 zip（{exp.images} 張）</a></>}</div>}
            </Section>

            <div className="danger-zone">
              <div className="small muted">建立 {d.created_at.slice(0, 16).replace("T", " ")}</div>
              <button className="ghost bad" onClick={() => setAskDelete(true)}>刪除資料集…</button>
            </div>
          </div>
        </div>
      )}

      {tab === "review" && <Review spotOnly={spotOnly} setSpotOnly={setSpotOnly} reviewer={reviewer} datasetId={d.id} openImage={openImage} onOpened={() => setOpenImage(null)} refresh={() => { reload(); load(); }} go={go} setGuard={setGuard} header={header} onExit={() => setTab("images")} />}
      </Fade>
      <ConfirmDialog open={askRelabel} title={`重新標註 ${picked.size} 張？`} body={<>用<b>目前的 AI 導師設定</b>（{modeText}{probeList.length && mode !== "fixed" ? `、每張圖都問 ${probeList.join("／")}` : ""}）重出這些圖的題。自動通過與待看的 {pickedMachine} 題換掉；你看過、改過、丟棄或自己加的題保留，也不會重複出。約 20 秒/張。</>} confirmText={`重新標註 ${picked.size} 張`} onCancel={() => setAskRelabel(false)} onConfirm={doRelabel} />
      <ConfirmDialog open={askRemove} title={`從 ${d.name} 移除 ${picked.size} 張？`} body={<>這些圖{pickedQ ? <>和它們的 <b>{pickedQ} 題</b>（已確認 {pickedDone} 題）</> : ""}會從資料集拿掉，之後訓練不會用到，不能復原。上傳的原檔沒有別的資料集在用就一起刪；GB300 資料夾裡的原檔不動。</>} confirmText={`移除 ${picked.size} 張`} danger onCancel={() => setAskRemove(false)} onConfirm={doRemove} />
      <ConfirmDialog open={askDelete} title={`刪除 ${d.name}？`} body="只刪資料集與歸屬；圖和題留在工作區。" confirmText="刪除" typed={d.name} danger onCancel={() => setAskDelete(false)} onConfirm={doDelete} />
    </div>
  );
}

/** How far the spot check got and what it says about the teacher: of the agreed questions you
 *  re-checked, the share you left exactly as they were. */
function SpotCard({ sp, onOpen }: { sp: NonNullable<Dataset["spot"]>; onOpen: () => void }) {
  const seen = sp.agree + sp.changed;
  const pct = seen ? Math.round((100 * sp.agree) / seen) : null;
  const verdict = pct === null ? "看幾題就有數字" : seen < 30 ? "樣本還少，先看到 30 題" : pct >= 90 ? "自動通過的題可信" : pct >= 75 ? "大致可信，建議多抽一些" : "不太可信：先別用自動通過的題訓練";
  return (
    <div className="panel col">
      <div className="row" style={{ justifyContent: "space-between" }}><b>AI 導師可信度</b>{pct !== null && <Pill kind={seen < 30 ? "grey" : pct >= 90 ? "ok" : pct >= 75 ? "warn" : "bad"}>{pct}% 直接同意</Pill>}</div>
      <div className="small muted">抽查 看了 {seen}/{sp.total} 題 · 同意 {sp.agree} · 改了或丟棄 {sp.changed}</div>
      <Bar done={seen} total={sp.total} />
      <div className="row" style={{ justifyContent: "space-between" }}><span className="hint">{verdict}</span>{sp.pending > 0 && <button className="ghost" onClick={onOpen}>只看抽查（{sp.pending}）→</button>}</div>
    </div>
  );
}

function BatchCard({ b, onCancel, onResume, busy, skippedNote }: { b: Batch; onCancel: () => void; onResume: () => void; busy: boolean; skippedNote: string | null }) {
  const j = b.job;
  const stuck = interrupted(b);
  const u = b.unfinished;
  const idx = stuck ? (u!.images > 0 ? 0 : u!.unpredicted > 0 ? 1 : 2) : j?.phase === "generating" ? 0 : j?.phase === "predicting" ? 1 : j?.phase === "judging" ? 2 : 3;
  const state = stuck ? "interrupted" : j?.state ?? "done";
  const left = eta(j?.started, j?.done ?? 0, j?.total ?? 0);
  return (
    <div className={`panel col ${state === "error" || state === "interrupted" ? "card-bad" : state === "running" ? "card-accent" : ""}`}>
      <div className="row" style={{ justifyContent: "space-between" }}><b>{state === "running" ? "這一輪" : "上一輪"}：{b.created_at.slice(5, 16).replace("T", " ")}{j?.resumed ? "（接著跑）" : ""}</b>{state === "running" && <button onClick={onCancel}>取消</button>}</div>
      <div className="phase">{["出題", "模型作答", "比對", "人看"].map((l, i) => <div key={l} className={i < idx || state === "done" ? "ok" : i === idx && state === "running" ? "on" : ""}>{l}</div>)}</div>
      {state === "interrupted" && (
        <>
          <div className="small bad">中斷了（server 重啟或取消）：{[u!.images > 0 ? `${u!.images} 張還沒出題` : "", u!.unpredicted > 0 ? `${u!.unpredicted} 題模型還沒答` : "", u!.unjudged > 0 ? `${u!.unjudged} 題還沒比對` : ""].filter(Boolean).join(" · ")}。出好的題都還在。</div>
          <button className="primary" onClick={onResume} disabled={busy}>接著跑</button>
        </>
      )}
      {state === "running" && <><Bar done={j!.done} total={j!.total} /><div className="small">{phaseLabel(j!.phase)} {j!.done}/{j!.total}{left !== null ? `（約還要 ${left} 分鐘）` : j!.done === 0 ? "（載入出題模型中…）" : ""}</div></>}
      {state === "done" && <div className="small">{b.total} 題 · 一致 {b.verdicts?.accept ?? 0} · 不一致 {b.verdicts?.fix ?? 0} · 自動通過 {b.counts.auto ?? 0} · 還要看 {b.counts.pending ?? 0} 題{b.images_pending !== undefined ? `（${b.images_pending} 張）` : ""}</div>}
      {state === "cancelled" && <div className="small warn-text">已取消：出好的 {b.total} 題保留</div>}
      {state === "error" && <div className="small bad">失敗（{phaseLabel(j?.phase)}）：{j?.error}</div>}
      {skippedNote && <div className="warn hint">{skippedNote}</div>}
    </div>
  );
}

function Grid({ cells, picked, onPick, onOpen }: { cells: GridCell[]; picked: Set<string>; onPick: (id: string, shift: boolean) => void; onOpen: (c: GridCell) => void }) {
  if (!cells.length) return null;
  const picking = picked.size > 0; // once something is picked, a click on any thumb picks
  return (
    <div className={`thumbs${picking ? " picking" : ""}`} style={{ maxHeight: "calc(100vh - 330px)", gridTemplateColumns: "repeat(auto-fill, minmax(84px, 1fr))" }}>
      {cells.map((c) => (
        <div key={c.image_id} className={`th${picked.has(c.image_id) ? " picked" : ""}`} onClick={(e) => (picking ? onPick(c.image_id, e.shiftKey) : onOpen(c))} title={`${c.n} 題 · 不一致 ${c.fix} · ${c.split === "val" ? "驗證集" : "訓練"}`}>
          <img src={api.imageUrl(c.image_id, 200)} loading="lazy" />
          <span className="pick" role="checkbox" aria-checked={picked.has(c.image_id)} aria-label="選取" onClick={(e) => { e.stopPropagation(); onPick(c.image_id, e.shiftKey); }}>{picked.has(c.image_id) ? "✓" : ""}</span>
          {c.split === "val" && <b className="val">val</b>}
          {c.n > 0 && <b className={c.fix > 0 ? "fix" : c.pending > 0 ? "" : "approved"}>{c.fix > 0 ? c.fix : c.pending > 0 ? c.pending : "✓"}</b>}
        </div>
      ))}
    </div>
  );
}
