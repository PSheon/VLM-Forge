// Upload or pick a workspace image. Thumbnails are server-side 200 px copies.

import { useEffect, useState } from "react";
import { api, sourceLabel, type ImageRow } from "../api";

export default function ImagePicker({ value, onPick }: { value: string | null; onPick: (im: ImageRow) => void }) {
  const [images, setImages] = useState<ImageRow[]>([]);
  const [sources, setSources] = useState<{ source: string; n: number }[]>([]);
  const [source, setSource] = useState("");
  const [scanDir, setScanDir] = useState("");
  const [showScan, setShowScan] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const load = () =>
    Promise.all([api.images(source || undefined), api.sources()])
      .then(([im, s]) => { setImages(im); setSources(s); })
      .catch((e) => setErr(String(e)));
  useEffect(() => { load(); }, [source]); // eslint-disable-line react-hooks/exhaustive-deps

  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    setBusy(true); setErr("");
    try {
      let last: ImageRow | null = null;
      for (const f of Array.from(files)) last = await api.upload(f);
      await load();
      if (last) onPick(last);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  const scan = async () => {
    if (!scanDir) return;
    setBusy(true); setErr("");
    try {
      const r = await api.scan(scanDir, `capture:${scanDir.split("/").filter(Boolean).pop()}`);
      setErr(`加入 ${r.added} 張`);
      await load();
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  return (
    <div className="col">
      <div className="row">
        <label className="row" style={{ gap: 4 }}>
          <input type="file" accept="image/*" multiple disabled={busy} onChange={(e) => upload(e.target.files)} style={{ display: "none" }} id="up" />
          <button onClick={() => document.getElementById("up")?.click()} disabled={busy}>上傳圖片</button>
        </label>
        <button onClick={() => setShowScan(!showScan)} title="把 GB300 上某個目錄的圖全部登記進來（例如擷取的 CCTV）">掃描目錄</button>
      </div>
      {showScan && (
        <div className="row">
          <input placeholder="/home/paul/VLM-Forge/workspace/captures/…" value={scanDir} onChange={(e) => setScanDir(e.target.value)} style={{ flex: 1 }} />
          <button onClick={scan} disabled={busy || !scanDir}>掃描</button>
        </div>
      )}
      <select value={source} onChange={(e) => setSource(e.target.value)}>
        <option value="">全部來源（{sources.reduce((a, s) => a + s.n, 0)}）</option>
        {sources.map((s) => <option key={s.source} value={s.source}>{sourceLabel(s.source)}（{s.n}）</option>)}
      </select>
      {err && <div className="hint">{err}</div>}
      {images.length === 0 && <div className="empty">還沒有圖片。上傳，或掃描一個目錄。</div>}
      <div className="thumbs">
        {images.map((im) => (
          <img key={im.id} src={api.imageUrl(im.id, 200)} className={im.id === value ? "cur" : ""} title={im.path.split("/").slice(-2).join("/")} onClick={() => onPick(im)} loading="lazy" />
        ))}
      </div>
    </div>
  );
}
