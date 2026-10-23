# VLM-Forge

Moondream 3.1 微調工具鏈：detect / point / query / caption 的 LoRA 訓練、評估、以及人工監督修正的 UI。決議全部在 [GOALS.md](GOALS.md)。

- 用途限內部 R&D；adapter 不出貨（GOALS §2）。
- 跨出本 repo 的名稱一律 `syncvlm-*`。

## 安裝

```bash
uv sync
```

## 指令

```bash
forge manifests                          # held-out manifests ← Data Core release（data/manifests/）
forge eval heldout-cctv [--adapter P]    # 評分；報告到 runs/baselines/ 或 adapter 旁
forge train configs/detect-cctv-v1.yaml  # LoRA 微調 → runs/<name>/<name>.pt
forge gate --base runs/baselines/base-*.json --adapter runs/x/eval-*.json --skill detect
forge capture --out workspace/captures/<date> --url 'rtsp://…{ip}…{ch}…' --nvr 10.12.31.6 …
forge serve --port 8796                  # 監督 UI（需先 cd web && npm run build）
forge export NAME [--queue ID…] [--incoming]   # 審核結果 → workspace/export/NAME.jsonl（+ Data Core 包）
```

GB300 上一律 `numactl --membind=0 uv run forge …`。

## 目錄

| 目錄 | 內容 |
| --- | --- |
| `src/vlmforge/model/` | Moondream 3.1 純 PyTorch 移植（vision / text / region / LoRA） |
| `src/vlmforge/{data,evaluate,train,gate,capture}.py` | manifests、評分、訓練、gate、RTSP 擷取 |
| `src/vlmforge/server/` | UI 後端：workspace（SQLite）、GPU worker、API、匯出 |
| `web/` | UI 前端（Vite + React） |
| `configs/` | 訓練設定 |
| `model-cache/`、`runs/`、`adapters/`、`workspace/`、`data/` | 不進 git |

## 分支

`feat/* → dev → stage → main`。PR 以 `dev` 為 base；進 `stage` 要過 GOALS §4 的 gate。
