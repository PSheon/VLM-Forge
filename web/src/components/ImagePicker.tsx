// Pick a workspace image (any source) or upload one. Thumbnails are server-side 200 px copies.

import { useEffect, useState } from "react";
import { api, type ImageRow } from "../api";

export default function ImagePicker({ value, onPick }: { value: string | null; onPick: (im: ImageRow) => void }) {
  const [images, setImages] = useState<ImageRow[]>([]);
  const [sources, setSources] = useState<{ source: string; n: number }[]>([]);
  const [source, setSource] = useState("");
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

  return (
    <div className="col">
      <div className="row">
        <input type="file" accept="image/*" multiple disabled={busy} onChange={(e) => upload(e.target.files)} style={{ display: "none" }} id="up" />
        <button onClick={() => document.getElementById("up")?.click()} disabled={busy}>上傳圖片</button>
        <select value={source} onChange={(e) => setSource(e.target.value)} style={{ flex: 1 }}>
          <option value="">全部（{sources.reduce((a, s) => a + s.n, 0)}）</option>
          {sources.map((s) => <option key={s.source} value={s.source}>{s.source}（{s.n}）</option>)}
        </select>
      </div>
      {err && <div className="bad hint">{err}</div>}
      {images.length === 0 && <div className="empty">沒有圖片。</div>}
      <div className="thumbs">
        {images.map((im) => (
          <img key={im.id} src={api.imageUrl(im.id, 200)} className={im.id === value ? "cur" : ""} title={im.path.split("/").slice(-2).join("/")} onClick={() => onPick(im)} loading="lazy" />
        ))}
      </div>
    </div>
  );
}
