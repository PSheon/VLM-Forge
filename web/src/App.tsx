import { useEffect, useState } from "react";
import { api, type Model } from "./api";
import Workbench from "./pages/Workbench";
import Queue from "./pages/Queue";
import Compare from "./pages/Compare";
import Runs from "./pages/Runs";

const PAGES = [
  ["workbench", "工作台"],
  ["queue", "審核佇列"],
  ["compare", "對照"],
  ["runs", "Runs"],
] as const;
type Page = (typeof PAGES)[number][0];

export default function App() {
  const [page, setPage] = useState<Page>((location.hash.slice(1) as Page) || "workbench");
  const [models, setModels] = useState<Model[]>([{ name: "base", label: "base" }]);
  const [status, setStatus] = useState<{ version: string; engine: Record<string, unknown> } | null>(null);
  useEffect(() => {
    const load = () => api.status().then((s) => { setStatus(s); setModels(s.models); }).catch(() => undefined);
    load();
    const t = setInterval(load, 15000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => { location.hash = page; }, [page]);
  return (
    <>
      <header className="top">
        <strong>VLM-Forge</strong>
        <nav className="row">{PAGES.map(([k, l]) => <a key={k} href={`#${k}`} className={k === page ? "cur" : ""} onClick={(e) => { e.preventDefault(); setPage(k); }}>{l}</a>)}</nav>
        <div className="right muted">
          {status ? <>v{status.version} · GPU {status.engine.loaded ? `已載入 (${String(status.engine.current)})` : "待命"} · 排隊 {String(status.engine.pending ?? 0)}</> : "連線中…"}
        </div>
      </header>
      <main>
        {page === "workbench" && <Workbench models={models} />}
        {page === "queue" && <Queue models={models} />}
        {page === "compare" && <Compare models={models} />}
        {page === "runs" && <Runs />}
      </main>
    </>
  );
}
