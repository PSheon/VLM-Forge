// Upload or pick a workspace image; the source filter narrows to uploads / captures / held-out.

import { useEffect, useState } from "react";
import { api, type ImageRow } from "../api";

export default function ImagePicker({ value, onPick }: { value: string | null; onPick: (im: ImageRow) => void }) {
  const [images, setImages] = useState<ImageRow[]>([]);
  const [source, setSource] = useState("");
  const [scanDir, setScanDir] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const load = () => api.images(source || undefined).then(setImages).catch((e) => setErr(String(e)));
  useEffect(() => { load(); }, [source]); // eslint-disable-line react-hooks/exhaustive-deps

  const upload = async (files: FileList | null) => {
    if (!files) return;
    setBusy(true);
    try {
      let last: ImageRow | null = null;
      for (const f of Array.from(files)) last = await api.upload(f);
      await load();
      if (last) onPick(last);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  const scan = async () => {
    if (!scanDir) return;
    setBusy(true);
    try { await api.scan(scanDir, `capture:${scanDir.split("/").filter(Boolean).pop()}`); await load(); } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  return (
    <div className="col">
      <div className="row">
        <label className="row">上傳 <input type="file" accept="image/*" multiple disabled={busy} onChange={(e) => upload(e.target.files)} /></label>
      </div>
      <div className="row">
        <input placeholder="掃描目錄 (GB300 路徑)" value={scanDir} onChange={(e) => setScanDir(e.target.value)} style={{ flex: 1 }} />
        <button onClick={scan} disabled={busy || !scanDir}>掃描</button>
      </div>
      <select value={source} onChange={(e) => setSource(e.target.value)}>
        <option value="">全部來源</option>
        <option value="upload">上傳</option>
        <option value="capture">擷取</option>
        <option value="heldout">held-out</option>
      </select>
      {err && <div className="bad">{err}</div>}
      <div className="muted">{images.length} 張</div>
      <div className="thumbs">
        {images.map((im) => (
          <img key={im.id} src={api.imageUrl(im.id)} className={im.id === value ? "cur" : ""} title={im.path} onClick={() => onPick(im)} loading="lazy" />
        ))}
      </div>
    </div>
  );
}
