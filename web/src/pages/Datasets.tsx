// ① 資料集 (docs/ui-spec §3): the list, and the dataset page with four tabs —
// 圖片（加圖、標註）／檢查（一次一題）／設定（AI 導師）／匯出. Everything about data
// happens here; training and testing are the next two steps.

import { useCallback, useEffect, useRef, useState } from "react";
import { api, parseFixed, phaseLabel, type Batch, type Dataset, type GridCell, type Skill, type Status } from "../api";
import { Bar, ConfirmDialog, Dialog, Pill, eta, useToast } from "../components/ui";
import Review from "./Review";
import type { Step } from "../App";

type Tab = "images" | "review" | "settings" | "export";

export default function Datasets({ status, refresh, go, reviewer, setGuard }: { status: Status | null; refresh: () => void; go: (s: Step) => void; reviewer: string; setGuard: (g: (() => boolean) | null) => void }) {
  const [list, setList] = useState<Dataset[]>([]);
  const [cur, setCur] = useState<number | null>(() => { const m = location.hash.match(/ds=(\d+)/); return m ? +m[1] : null; });
  const [showNew, setShowNew] = useState(false);
  const load = useCallback(() => api.datasets().then(setList).catch(() => undefined), []);
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, [load]);
  const detail = list.find((d) => d.id === cur) ?? null;
  if (detail) return <Detail key={detail.id} d={detail} status={status} reload={() => { load(); refresh(); }} go={go} reviewer={reviewer} setGuard={setGuard} onBack={() => setCur(null)} onDeleted={() => { setCur(null); load(); }} />;
  return (
    <div className="col">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>資料集</h2>
        <button className="primary" onClick={() => setShowNew(true)}>＋ 新資料集</button>
      </div>
      {list.length === 0 && <div className="empty">還沒有資料集。建一個，把照片丟進去，AI 導師先標一輪，你再檢查。</div>}
      <div className="cards">
        {list.map((d) => <DatasetCard key={d.id} d={d} onClick={() => setCur(d.id)} />)}
      </div>
      <NewDataset open={showNew} existing={list.map((d) => d.name)} onClose={() => setShowNew(false)} onCreated={(d) => { setShowNew(false); load(); setCur(d.id); }} />
    </div>
  );
}

function statusPill(d: Dataset): [string, "ok" | "bad" | "warn" | "grey"] {
  const b = d.batches?.[0];
  if (b?.job?.state === "running") return [`標註中 ${b.job.done}/${b.job.total}`, "warn"];
  if (b?.job?.state === "error") return ["上次標註失敗", "bad"];
  if (d.images_pending > 0) return [`還有 ${d.images_pending} 張要看`, "warn"];
  if (d.items > 0) return ["全部確認", "ok"];
  if (d.images > 0) return ["還沒標", "grey"];
  return ["空的", "grey"];
}

function DatasetCard({ d, onClick }: { d: Dataset; onClick: () => void }) {
  const [text, kind] = statusPill(d);
  return (
    <div className="card" onClick={onClick}>
      <div className="row" style={{ justifyContent: "space-between" }}><span className="name">{d.name}</span><Pill kind={kind}>{text}</Pill></div>
      <div className="stat"><span>圖 <b>{d.images}</b></span><span>題 <b>{d.items}</b></span><span>已確認 <b>{d.trainable}</b></span><span>驗證 <b>{d.val_images}</b></span></div>
      <div className="small muted">{d.config.mode === "fixed" ? "只指定題目" : d.config.mode === "both" ? "自動出題＋指定題目" : "自動出題"} · 更新 {d.updated_at.slice(5, 16).replace("T", " ")}</div>
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
  const [tab, setTab] = useState<Tab>("images");
  const [openImage, setOpenImage] = useState<string | null>(null);
  const [full, setFull] = useState<Dataset | null>(null);
  const [grid, setGrid] = useState<GridCell[]>([]);
  const [dir, setDir] = useState("");
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const [mode, setMode] = useState(d.config.mode ?? "auto");
  const [fixed, setFixed] = useState(d.config.fixed ?? "");
  const [topics, setTopics] = useState((d.config.topics ?? []).join("\n"));
  const [focus, setFocus] = useState(d.config.focus ?? "");
  const [skills, setSkills] = useState<Skill[]>(d.config.skills ?? ["detect", "point", "query", "caption"]);
  const [n, setN] = useState<number | "">("");
  const [askDelete, setAskDelete] = useState(false);
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
  const cfgDirty = mode !== (d.config.mode ?? "auto") || fixed !== (d.config.fixed ?? "") || topics !== (d.config.topics ?? []).join("\n") || focus !== (d.config.focus ?? "") || skills.join() !== (d.config.skills ?? ["detect", "point", "query", "caption"]).join();

  const saveConfig = async () => {
    await api.updateDataset(d.id, { config: { mode, fixed, topics: topics.split("\n").map((s) => s.trim()).filter(Boolean), focus, skills } });
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
    setBusy(true);
    try { if (cfgDirty) await saveConfig(); await api.label(d.id, n === "" ? undefined : Number(n)); say(`開始標註 ${n === "" ? unlabelled : Math.min(Number(n), unlabelled)} 張，先載入出題模型…`, "ok", { ms: 2500 }); reload(); load(); } catch (e) { say(String(e), "bad", { sticky: true }); } finally { setBusy(false); }
  };
  const cancel = async () => { if (batch) { await api.cancelBatch(batch.id).catch(() => undefined); say("取消中，這一批出完就停", "warn"); } };
  const doExport = async () => { setBusy(true); try { const r = await api.exportDataset(d.id); setExp(r); say(`匯出 ${r.rows} 列`); } catch (e) { say(String(e), "bad"); } finally { setBusy(false); } };
  const doRename = async () => { try { await api.updateDataset(d.id, { name: newName.trim() }); setRename(false); say("改好了"); reload(); } catch (e) { say(String(e), "bad"); } };
  const doDelete = async () => { await api.deleteDataset(d.id); say(`刪了 ${d.name}`, "warn"); setAskDelete(false); onDeleted(); };
  const labelText = !judgeOk ? "出題模型缺，不能標" : gpuBusy ? "排隊中…（GPU 忙）" : unlabelled === 0 ? "沒有新圖" : `標註新加的 ${n === "" ? unlabelled : Math.min(Number(n), unlabelled)} 張`;
  const skippedNote = batch?.job?.student_error ? `模型沒作答（${batch.job.student_error}），這批全部交給你看` : null;
  const [pillText, pillKind] = statusPill(full ?? d);
  const tabs: [Tab, string][] = [["images", `圖片 ${full?.images ?? d.images}`], ["review", `檢查${(full ?? d).images_pending ? ` ${(full ?? d).images_pending}` : ""}`], ["settings", "設定"], ["export", "匯出"]];

  return (
    <div className="col">
      <div className="row">
        <button className="ghost" onClick={onBack}>← 資料集</button>
        {rename ? <><input value={newName} onChange={(e) => setNewName(e.target.value)} style={{ width: 220 }} /><button className="primary" onClick={doRename}>存</button><button onClick={() => setRename(false)}>取消</button></> : <h2 onDoubleClick={() => setRename(true)} title="雙擊改名">{d.name}</h2>}
        <Pill kind={pillKind}>{pillText}</Pill>
        <span className="small muted">題 {(full ?? d).items} · 已確認 {(full ?? d).trainable} · 驗證集 {(full ?? d).val_images} 張</span>
        <div className="tabs" style={{ marginLeft: "auto" }}>{tabs.map(([k, l]) => <button key={k} className={tab === k ? "cur" : ""} onClick={() => setTab(k)}>{l}</button>)}</div>
      </div>

      {tab === "images" && (
        <div className="grid-2">
          <div className="col">
            <div className={`drop${drag ? " on" : ""}`} onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)} onDrop={(e) => { e.preventDefault(); setDrag(false); addFiles(e.dataTransfer.files); }}>
              把照片拖到這裡，或 <button onClick={() => fileInput.current?.click()} disabled={busy}>選檔案</button>（可多張、可 zip）
              <input ref={fileInput} type="file" accept="image/*,.zip" multiple hidden onChange={(e) => addFiles(e.target.files)} />
            </div>
            <div className="row">
              <input placeholder="GB300 上的資料夾路徑，例如 /home/paul/VLM-Forge/workspace/captures/1026" value={dir} onChange={(e) => setDir(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addFolder()} style={{ flex: 1 }} />
              <button onClick={addFolder} disabled={busy || !dir.trim()}>加入資料夾</button>
            </div>
            <div className="small muted">{grid.length ? `${grid.length} 張 · 驗證集 ${grid.filter((g) => g.split === "val").length} 張 · 角標：val＝驗證集、紅＝不一致題數、✓＝都看過。點一張進去檢查。` : "還沒有圖"}</div>
            <Grid cells={grid} onOpen={(cell) => { if (cell.n) { setOpenImage(cell.image_id); setTab("review"); } }} />
          </div>
          <div className="col">
            {batch && <BatchCard b={batch} onCancel={cancel} skippedNote={skippedNote} />}
            <button className="primary big" onClick={label} disabled={busy || running || !judgeOk || gpuBusy || unlabelled === 0 || (mode !== "auto" && nFixed === 0)}>{running ? `標註中 ${batch?.job?.done ?? 0}/${batch?.job?.total ?? 0}` : labelText}</button>
            <div className="note">用「設定」分頁的方式出題（{mode === "auto" ? "自動出題" : mode === "both" ? `自動出題＋指定 ${nFixed} 題` : `只指定 ${nFixed} 題`}）：先出題（Qwen3-VL-32B，約 20 秒/張，8 張一批）→ Moondream 作答 → 比對。一致的題自動通過，不一致的到「檢查」。</div>
            {(full ?? d).images_pending > 0 && !running && <button onClick={() => setTab("review")}>去檢查（還有 {(full ?? d).images_pending} 張）→</button>}
            {(full ?? d).trainable > 0 && <button onClick={() => go("train")}>已確認 {(full ?? d).trainable} 題 → 去訓練</button>}
            {full && full.batches && full.batches.length > 1 && (
              <div className="col"><strong className="small">批次紀錄</strong>{full.batches.slice(1, 6).map((b) => <div key={b.id} className="small muted">{b.created_at.slice(5, 16).replace("T", " ")} · {b.total} 題 · {b.job?.state === "cancelled" ? "取消" : b.job?.state === "error" ? "失敗" : "完成"} · 一致 {b.verdicts?.accept ?? 0} · 不一致 {b.verdicts?.fix ?? 0}</div>)}</div>
            )}
          </div>
        </div>
      )}

      {tab === "review" && <Review reviewer={reviewer} datasetId={d.id} openImage={openImage} onOpened={() => setOpenImage(null)} refresh={() => { reload(); load(); }} go={go} setGuard={setGuard} />}

      {tab === "settings" && (
        <div className="grid-2">
          <div className="col panel">
            <h3>AI 導師怎麼標</h3>
            <label className="row"><input type="radio" name="mode" checked={mode === "auto"} onChange={() => setMode("auto")} /> 自動出題（導師看圖決定物件、問答、巡檢題、caption）</label>
            <label className="row"><input type="radio" name="mode" checked={mode === "both"} onChange={() => setMode("both")} /> 自動出題＋指定題目</label>
            <label className="row"><input type="radio" name="mode" checked={mode === "fixed"} onChange={() => setMode("fixed")} /> 只指定題目</label>
            {mode !== "auto" && (
              <>
                <textarea value={fixed} onChange={(e) => setFixed(e.target.value)} placeholder={"每行一題，例如：\ndetect: person, helmet\npoint: person\nquery: Is anyone without a helmet?\ncaption: normal"} style={{ minHeight: 110, fontFamily: "var(--mono)", fontSize: 13 }} />
                <div className="hint">{nFixed > 0 ? <span className="ok">✓ {nFixed} 題有效（detect {parsed.ok.detect.length}、point {parsed.ok.point.length}、query {parsed.ok.query.length}、caption {parsed.ok.caption.length}）</span> : <span className="muted">還沒有有效的題</span>}{parsed.bad.length > 0 && <span className="bad"> · ✗ 第 {parsed.bad.join("、")} 行看不懂（要像 <code>detect: person</code>）</span>}</div>
              </>
            )}
            <div className="row"><button className="primary" onClick={saveConfig} disabled={!cfgDirty}>{cfgDirty ? "儲存設定" : "已儲存"}</button><span className="muted hint">下次按「標註」會用這組設定；已出的題不變。</span></div>
          </div>
          <div className="col panel">
            <h3>進階</h3>
            <label className="f">額外巡檢主題（每行一個；預設已含值班人員、安全帽／背心／口罩／手套、出口阻擋、門、地面、堆高機、燈、管制區、無人看管物品）<textarea value={topics} onChange={(e) => setTopics(e.target.value)} style={{ minHeight: 60 }} /></label>
            <label className="f">側重（給導師的一句話）<input value={focus} onChange={(e) => setFocus(e.target.value)} placeholder="例如 night-shift factory floor" /></label>
            <div className="f"><span>出哪些 skill</span><div className="row">{(["detect", "point", "query", "caption"] as Skill[]).map((s) => <label key={s} className="row hint"><input type="checkbox" checked={skills.includes(s)} onChange={(e) => setSkills(e.target.checked ? [...skills, s] : skills.filter((x) => x !== s))} />{s}</label>)}</div></div>
            <label className="f">一次只標幾張（空＝全部還沒標的）<input type="number" min={1} value={n} onChange={(e) => setN(e.target.value === "" ? "" : +e.target.value)} style={{ width: 120 }} /></label>
          </div>
        </div>
      )}

      {tab === "export" && (
        <div className="grid-2">
          <div className="panel col">
            <h3>匯出</h3>
            <div className="hint muted">已確認的題（人確認＋自動通過）寫成 jsonl（座標 0–1，含 train/val 的 split 欄）；圖另外打包 zip，jsonl 在別台機器也能用。</div>
            <button className="primary" onClick={doExport} disabled={busy || !(full ?? d).trainable}>匯出 {(full ?? d).trainable} 題</button>
            {exp && <div className="hint">寫到 <code>{exp.jsonl}</code>（{exp.rows} 列）· <a href={api.downloadUrl(d.id, "jsonl")}>下載 jsonl</a>{exp.zip && <> · <a href={api.downloadUrl(d.id, "zip")}>下載圖 zip（{exp.images} 張）</a></>}</div>}
          </div>
          <div className="panel col">
            <h3>資料集</h3>
            <div className="small muted">建立 {d.created_at.slice(0, 16).replace("T", " ")} · 更新 {d.updated_at.slice(0, 16).replace("T", " ")}</div>
            <div className="row"><button onClick={() => setRename(true)}>改名</button><button className="danger" onClick={() => setAskDelete(true)}>刪除資料集</button></div>
          </div>
        </div>
      )}
      <ConfirmDialog open={askDelete} title={`刪除 ${d.name}？`} body="只刪資料集與歸屬；圖和題留在工作區。" confirmText="刪除" typed={d.name} danger onCancel={() => setAskDelete(false)} onConfirm={doDelete} />
    </div>
  );
}

function BatchCard({ b, onCancel, skippedNote }: { b: Batch; onCancel: () => void; skippedNote: string | null }) {
  const j = b.job;
  const idx = j?.phase === "generating" ? 0 : j?.phase === "predicting" ? 1 : j?.phase === "judging" ? 2 : 3;
  const state = j?.state ?? "done";
  const left = eta(j?.started, j?.done ?? 0, j?.total ?? 0);
  return (
    <div className={`panel col ${state === "error" ? "card-bad" : state === "running" ? "card-accent" : ""}`}>
      <div className="row" style={{ justifyContent: "space-between" }}><b>{state === "running" ? "這一輪" : "上一輪"}：{b.created_at.slice(5, 16).replace("T", " ")}</b>{state === "running" && <button onClick={onCancel}>取消</button>}</div>
      <div className="phase">{["出題", "模型作答", "比對", "人看"].map((l, i) => <div key={l} className={i < idx || state === "done" ? "ok" : i === idx && state === "running" ? "on" : ""}>{l}</div>)}</div>
      {state === "running" && <><Bar done={j!.done} total={j!.total} /><div className="small">{phaseLabel(j!.phase)} {j!.done}/{j!.total}{left !== null ? `（約還要 ${left} 分鐘）` : j!.done === 0 ? "（載入出題模型中…）" : ""}</div></>}
      {state === "done" && <div className="small">{b.total} 題 · 一致 {b.verdicts?.accept ?? 0} · 不一致 {b.verdicts?.fix ?? 0} · 自動通過 {b.counts.auto ?? 0} · 還要看 {b.counts.pending ?? 0}</div>}
      {state === "cancelled" && <div className="small warn-text">已取消：出好的 {b.total} 題保留</div>}
      {state === "error" && <div className="small bad">失敗（{phaseLabel(j?.phase)}）：{j?.error}</div>}
      {skippedNote && <div className="warn hint">{skippedNote}</div>}
    </div>
  );
}

function Grid({ cells, onOpen }: { cells: GridCell[]; onOpen: (c: GridCell) => void }) {
  if (!cells.length) return null;
  return (
    <div className="thumbs" style={{ maxHeight: "calc(100vh - 330px)", gridTemplateColumns: "repeat(auto-fill, minmax(84px, 1fr))" }}>
      {cells.map((c) => (
        <div key={c.image_id} className="th" onClick={() => onOpen(c)} title={`${c.n} 題 · 不一致 ${c.fix} · ${c.split === "val" ? "驗證集" : "訓練"}`}>
          <img src={api.imageUrl(c.image_id, 200)} loading="lazy" />
          {c.split === "val" && <b className="val">val</b>}
          {c.n > 0 && <b className={c.fix > 0 ? "fix" : c.pending > 0 ? "" : "approved"}>{c.fix > 0 ? c.fix : c.pending > 0 ? c.pending : "✓"}</b>}
        </div>
      ))}
    </div>
  );
}
