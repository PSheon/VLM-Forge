# GOALS — 唯一決議入口

VLM-Forge 的唯一目標、範圍與規格。與程式、其他文件或其他 session 牴觸時以本文件為準；改方向先改本文件。只寫現行決議，過時內容直接刪除（歷史在 git）。
簽核：Paul（grill Q1–Q24，2026-10-23）。Repo 由 Claude 全權負責，細節自行決定，只在真正的阻礙時問 Paul。

## 1. 目標

VLM-Forge 是 **Moondream 3.1 微調的唯一家**：detect / point / query / caption 四種 skill 的 LoRA 微調、評估、以及讓人監督與修正模型輸出的 UI，整套工具鏈自給自足。

動機：①把 3.1 的輸出變成我們資料的飛輪（模型預標 → **judge 自動審** → 人只看被標紅的和抽查樣本 → 回 Data Core）；②量「3.1 底子 + 我們的資料」的上限；③給 SyncVision 更強的 teacher。

不在範圍：SyncVision（自訓模型，在 SyncAI-Lib-VisionTraining_Exp）；playground 的推論展示（SyncAI-App-VLMPlayground_Exp）。playground 的 `training/` 已搬到本 repo，playground 本身在 v1 不動、不加 adapter 切換。

## 2. 命名與授權

- 本 repo 是全公司命名規則的**特例**：程式、文件、commit 可以直接寫 Moondream 3.1。
- **跨出 repo 的東西一律用 `syncvlm-*`**：adapter 名稱、manifest 欄位、報告、任何會進 playground 或 Data Core 的字串。
- Repo 在 Paul 的個人帳號（PSheon），private。
- **用途限內部 R&D**。微調出的 adapter 是上游 Model License 1.0 的 Derivative Model：**不出貨給客戶**。要不要部署到 playground／內網服務，等授權條款報告（`docs/license-report.md`，Claude 寫）交付後由 Paul 另決。在此之前權重只留 GB300 與 `gs://syncai-vlm-forge`。
- VisionTraining `GOALS.md` §2 對應改為「3.1 微調移到 VLM-Forge，用途限內部 R&D」。

## 3. 範圍（v1）

| 項目 | 決議 |
| --- | --- |
| skill | detect、point、query、caption 四種 |
| 微調方式 | 只做 LoRA：單任務 adapter + 多任務合訓 adapter。Full fine-tune 等 LoRA 天花板有數據再開 |
| 順序 | detect（工廠 CCTV 人與物件）→ point → query → caption → 多任務 |
| 訓練底層 | 從 VisionTraining `src/syncvision/teacher/` vendor 的純 PyTorch 移植（含 `lora_train.py`），在 GB300 訓練與評估 |
| PRO 6000 | 只做 kestrel 推論 parity（微調後權重在正式 engine 裡結果一致）；adapter 經 GCS 傳過去 |

## 4. 驗收（KPI）與 gate

- Held-out 直接沿用 Data Core `syncvision-heldout-eval` v1.0.0，**與 SyncVision 共用同一套 KPI 定義**（macro：逐組算 F1／exact 再平均）：detect = CCTV 210 + CrowdHuman 500；query = people-posture 175。
- 缺的 gold 由本 repo 的 UI 補：**point gold** 從 CrowdHuman 框中心點推導再人工校正；**caption gold** 由人在 UI 審核通過約 200 張 held-out 圖。做完登錄進 Data Core 成為 held-out release v1.1。
- **Gate（進 stage 的條件）**：adapter 目標 skill 的 macro 分數 **≥ base**，且**其他三個 skill 退步 ≤ 1 個百分點**。不過就不進 stage。
- 正式分數只算 Data Core tagged release；直讀路徑的結果只算探索。held-out 來源一律不進訓練。

## 5. 微調規格

| 部分 | 決議 |
| --- | --- |
| LoRA 位置 | 語言模型 attention + MLP（含 MoE experts） |
| rank | 起手 r=16、alpha=32；只在 detect 線掃一次 r ∈ {8, 16, 32}，之後固定 |
| region head | 解碼側全參數可訓（`RegionHead`：layer norm + coordinate / size head，detect / point 的座標品質主要靠它）；輸入側的座標／尺寸 encoder 凍結 |
| 視覺編碼器、projector | 凍結 |
| replay | 自蒸餾 replay 必混（預設 0.5），否則 adapter 會塌到訓練答案 |
| adapter 檔 | `<name>.pt`：`lora_state_dict`（engine 的 TextLoRA 版面，alpha/r 已折進 B）、`rank`、`alpha`、`region_head`（上游張量名）。kestrel parity 時 LoRA 直接載入，region head 要寫進一份 patch 過的 model 檔 |

## 6. UI

- 技術：FastAPI + Vite/React（TypeScript），推論後端是 §3 的純 PyTorch 移植。
- 部署：GB300 內網 **:8796**，不做登入；每筆修正記 reviewer 名字與時間；多人同審，每張圖認領後鎖 10 分鐘。
- **Judge（人類監督的前一層）**：佇列預測完後由本機的 **Qwen3-VL-32B-Instruct**（Apache-2.0，`model-cache/judge`，與 VisionTraining caption 抽審同一個模型）逐題審：detect/point 看畫了編號框的圖，回報錯框與漏框；query 判答案對錯；caption 列出圖裡不存在的片語。`accept` 的題離開人工佇列（status `auto`），只留 **max(20, 10%) 的抽查樣本**給人；`fix` 的題排在人工佇列最前面。Judge 永遠不寫 label：`auto` 匯出為 `machine_reviewed`（Data Core §7 的正式抽審在登錄時做），人審過的為 `human`。
- v1 四塊：
  1. **工作台**：選圖或上傳 → 選 skill 與模型（base 或任一 adapter）→ 直接拖框、移點、改文字 → 存成 gold 或訓練樣本。
  2. **審核佇列**：對一個資料集批次跑預測，人逐張通過／修正／丟棄，鍵盤快捷鍵。
  3. **對照**：同一張圖 base vs adapter 並排，框／點標差異，文字 diff。
  4. **Runs／adapter 清單**：訓練 run、loss 曲線（UI 從 `runs/<name>/log.jsonl` 自繪）、held-out 分數、目前載入的 adapter。
- 訓練的啟動留在 CLI，UI 不做「按鈕開訓練」。

## 7. 資料

- 只讀 Data Core tagged release（GB300 本機 `/home/paul/SyncAI-Data-Core-vlm`）；本 repo 不存資料，`data/` 只是 cache。held-out 的圖依 Data Core 規則不離開資料主機。
- 人工修正先存本機工作區 `workspace/`（SQLite + 圖檔，gitignore）；`forge export` 匯成 Data Core `_incoming` 格式，由 Data Core 登錄、過 rules §7 後才成為 release。
- **實驗 run 可以直接讀本機修正**快速迭代；**要打 tag 進 main 的 adapter 只能用 Data Core tagged release 訓練**。
- 不訓練：任何 `machine_unreviewed`、`evaluation_only`、held-out 來源。
- **CCTV 訓練資料**：GB300 上現有的 105 張工廠 CCTV 全部是 held-out（`factory-cctv-20260910`），不能訓練。detect 線的訓練資料來自**新擷取**的工廠 CCTV 畫面（`forge capture`，NVR 10.12.31.{6,8,9,12}，不同日期），經 UI 預標＋人工修正後登錄成新的 source + labels layer；held-out 的 2026-09-10／09-11 畫面永遠排除。同一批攝影機、不同時間：held-out 量的是「這些攝影機的畫面」，不是未見過的場域，報告如實註明。

## 8. 交付

- 每個 adapter 打包成 `adapters/<skill|multi>/<name>@<version>/`：safetensors + `manifest.yaml`（base 權重 sha、LoRA 設定、訓練用的 release tag、held-out 分數、git commit）。
- 權重放 **`gs://syncai-vlm-forge`**（Paul 建）；GitHub release 只帶 manifest 與 sha256；git 裡不放權重。

## 9. 流程與基礎設施

- 分支：`feat/* → dev → stage → main`；PR 一律以 `dev` 為 base，不疊在 feat 分支上；進 `stage` 須通過 §4 gate 並附 eval report；`main` 只放打了 tag 的 adapter release。
- CI（GitHub Actions）只跑 lint + 無 GPU 的單元測試。
- 工具：`uv` + `pyproject.toml`；追蹤 `runs/<name>/`（log.jsonl、TensorBoard）。
- **GB300 排程（SyncVision 優先）**：UI 推論隨時可跑（單模型約 20 GB）；Forge 訓練限 ≤ 60 GB 顯存，開跑前先量 SyncVision 的 step 速度，跑中若 SyncVision 變慢 > 10% 或 OOM，Forge 訓練暫停；一律 `numactl --membind=0`。

## 10. 里程碑與現況

| 里程碑 | 驗收 | 狀態 |
| --- | --- | --- |
| M1 detect 線跑通 | CCTV 210 張 held-out 上 base 分數；一個 detect adapter 訓完並過 gate；對照 UI 看得到 base vs adapter；adapter 打 tag、上傳 GCS | 進行中 |
| M2 point | point gold（CrowdHuman 中心點 + 校正）→ adapter 過 gate | — |
| M3 query | adapter 過 gate | — |
| M4 caption | caption gold 200 張 → adapter 過 gate | — |
| M5 多任務 | 單一 adapter 四 skill 全部不低於各自單任務 adapter − 1 pt | — |

只在里程碑到點時通知 Paul。

待 Paul：建 bucket `gs://syncai-vlm-forge`（M1 前）。
