import { useEffect, useState } from "react";
import { api, type Model, type Status } from "./api";
import Workbench from "./pages/Workbench";
import Queue from "./pages/Queue";
import Compare from "./pages/Compare";
import Runs from "./pages/Runs";

const PAGES = [
  ["workbench", "工作台", "單張試模型、修正、存樣本"],
  ["queue", "審核佇列", "整批預標，逐張審"],
  ["compare", "對照", "base vs adapter"],
  ["runs", "Runs", "訓練與評估結果"],
] as const;
type Page = (typeof PAGES)[number][0];

export default function App() {
  const [page, setPage] = useState<Page>((location.hash.slice(1) as Page) || "workbench");
  const [models, setModels] = useState<Model[]>([{ name: "base", label: "base" }]);
  const [status, setStatus] = useState<Status | null>(null);
  const [down, setDown] = useState(false);
  useEffect(() => {
    const load = () => api.status().then((s) => { setStatus(s); setModels(s.models); setDown(false); }).catch(() => setDown(true));
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => { location.hash = page; }, [page]);
  const e = status?.engine;
  return (
    <>
      <header className="top">
        <strong>VLM-Forge</strong>
        <nav className="row">{PAGES.map(([k, l, d]) => <a key={k} href={`#${k}`} title={d} className={k === page ? "cur" : ""} onClick={(ev) => { ev.preventDefault(); setPage(k); }}>{l}</a>)}</nav>
        <div className="right hint">
          {down ? <span className="bad">連不到 server</span> : e ? (
            <>
              {!e.weights_ok ? <span className="warn">權重未就緒</span> : e.loaded ? <span className="ok">GPU 已載入 {e.current === "base" ? "base" : e.current.split("/").pop()}</span> : <span className="muted">GPU 待命（第一次執行會載入，約 1 分鐘）</span>}
              {e.pending > 0 && <span className="muted">排隊 {e.pending}</span>}
              {e.last_error && <span className="bad" title={e.last_error}>上次錯誤</span>}
              <span className="muted">v{status!.version}</span>
            </>
          ) : "連線中…"}
        </div>
      </header>
      <main>
        {page === "workbench" && <Workbench models={models} status={status} />}
        {page === "queue" && <Queue models={models} status={status} />}
        {page === "compare" && <Compare models={models} />}
        {page === "runs" && <Runs />}
      </main>
    </>
  );
}
