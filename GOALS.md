# GOALS — 唯一決議入口

VLM-Forge 的目標、範圍與規格。與程式、其他文件或其他 session 牴觸時以本文件為準；改方向先改本文件。只寫現行決議，過時內容直接刪除（歷史在 git）。
簽核：Paul（重新 grill Q1–Q16，2026-10-23）。Repo 由 Claude 全權負責，細節自行決定，只在 v1 完成點與真正卡住時通知 Paul。

## 1. 目標

Paul 個人的 **Moondream 3.1 微調工具**，跑在 GB300（資料在那裡）。一條線走完：

**上傳照片／指定資料夾 → AI 導師先標一輪 → 人確認 → 存成資料集 → 微調 → 測試 → 匯出到 Orin**

成功的樣子：在自己的工廠／倉儲影像上，微調後的 3.1 在 detect 與巡檢問答（有沒有人、有沒有戴安全帽／背心、出口有沒有被擋、門是否開、地上有無雜物、堆高機是否在動…）明顯贏過原版。內部使用（playground、機器人、之後當 SyncVision 的 teacher），不出貨。

**v1 完成點**：用 ≥200 張審核過的圖訓出一個 adapter，在 CCTV held-out 的 detect 與 query 都 ≥ base，而且 ①→④ 全程在 UI 完成、不碰終端機。

## 2. 不做

Data Core 匯出包、GCS、CI、`stage` 分支與 gate 儀式、PRO 6000 kestrel parity、RTSP 擷取（等有 NVR 連線資訊再談）、vLLM（列研究項目）、帳號／權限／指派、多 GPU、通用 benchmark、GGUF／量化匯出、RLHF、多模型抽象層。已寫的程式留著，不列目標。

## 3. 模型與部署

| 項目 | 決議 |
| --- | --- |
| 權重 | Hugging Face `moondream/moondream3.1-9B-A2B`（公開、不需登入），`model-cache/moondream31/`；tokenizer `moondream/starmie-v1` |
| 推論與訓練底層 | `src/vlmforge/model/`：純 PyTorch 移植（vision／text／region／LoRA），GB300 可跑；base 在三套 held-out 的分數與 SyncVision 記錄的 SyncVLM 一致（detect 0.368／0.370） |
| 部署目標 | **Orin AGX，用 Moondream 官方 Photon 跑 base + adapter**（kestrel 有 Jetson aarch64 kernel） |
| adapter 格式 | 官方 `lora_state_dict` 版面（`dense.{i}.up_a/up_b/down_a/down_b`、`moe.{j}.*`，alpha/r 已折進 B），**只掛文字 MLP**，rank 8–32；**region head 預設不訓**（訓了就只能在 Forge 跑，是進階選項） |
| 出題／比對模型 | Qwen3-VL-32B-Instruct（`model-cache/judge`，載入需 ~70 GB），8 張一批；GPU 被佔住放不下時退到 Qwen3-VL-8B-Instruct（`model-cache/judge-8b`，~22 GB），UI 標明「出題模型：8B（小）」，重啟後記憶體夠就回 32B |
| GPU | 與 SyncVision 共用 GB300：SyncVision 優先；Forge 訓練 ≤ 60 GB；一律 `numactl --membind=0` |

## 4. 資料集

- **資料集是一等公民**：有名字（例如「工廠-1025」）、建立／更新時間，沒有版本號。
- 圖可混來源、陸續加：上傳、拖曳、zip、GB300 資料夾路徑。held-out 的圖永遠不能進資料集。
- **AI 導師兩種模式**：①自動出題（預設）——導師看圖決定物件（含框）、3–4 題一般問答（數量、是非、屬性、空間、動作）、2–3 題巡檢問答、一段 caption；②指定題目——資料集設定裡寫死要問的物件與問題，每張圖都問。
- 學生（Moondream）答同一批題；比對：框用 IoU 配對、點看是否落在導師框內、文字比數字／是非／字串、caption 由 judge 查幻覺。一致→自動通過；不一致→人看。
- 標籤：文字用導師的答案、框／點用學生的（學它自己的框風格）；人改過的以人為準。匯出 = 資料集的 `jsonl`（座標 0–1）＋圖 zip。
- 每個資料集自動切 **10% 當驗證集**（不進訓練）。

## 5. 檢查（人）

- **一次一題、每按一次就存**，看完一張自動換下一張。只看不一致的題；一致的題 Enter 帶過；全部一致的圖只抽 **10%／至少 20 張**。
- 快捷鍵：`Enter`/`Space` 存並下一題、`Shift+Enter` 剩下全存、`D` 丟棄、`1` 用預標、`2` 用模型、`E` 改文字、`Z` 還原、`←/→` 換題、`N` 跳過這張、`G` 所有圖、`?` 說明。改過沒存要有明顯提示；存了要有提示。
- **多人**：進站 dialog 填名字（記在瀏覽器，右上可改），每筆記名字與時間，認領後鎖 10 分鐘；不做帳號、權限、指派。

## 6. 訓練

- UI 一顆按鈕；只露名稱、步數、學習率。預設 r=16、alpha=32、lr 2e-4、步數依資料量（≈ 每張圖 2 步，100–1000）、batch 8、replay 0.5、region head 關。預設值由 Claude 依實驗調整，不逐項問。
- 訓完自動在 held-out 與驗證集評估、和 base 比；一次只跑一個訓練。

## 7. 測試

- **分數在 GB300 算**（④ 測試頁）：CCTV held-out（105 張／210 題，真考卷）＋每個資料集的驗證集；CrowdHuman 與機器狗 gold 只當參考。指標：detect／point F1 macro、query exact（數字、是非）、count MAE；顯示 base vs 新模型與差值，逐題看差異，單張試玩。
- **Orin 只跑延遲腳本**：同一組圖，base vs adapter，每題毫秒；不重算分數。
- ④ 提供「匯出到 Orin」：adapter 檔＋載入範例。

## 8. 介面

- **從資料出發、三步**（Paul 2026-10-23）：**① 資料集 → ② 訓練 → ③ 測試**。資料集頁分四個分頁：圖片（加圖、標註與進度）、檢查（一次一題，固定在這個資料集；點縮圖格任一張直接進）、設定（AI 導師怎麼標）、匯出。和市面工具一致（Roboflow／Label Studio／Encord：標註與審核都在資料集／專案裡，訓練另起一段引用資料集）。每步一個畫面、一個主按鈕；步驟列顯示各步進度。名詞只用「預標」「模型」「不一致（請你看）」「自動通過」「抽查」。
- 桌機 1440 為主、平板可用、不做手機；跟系統深淺色；繁中。
- 改版先出 HTML 可點稿給 Paul 確認，再改程式。

## 9. 工程規範

- Python AI 專案最佳實踐：乾淨的分層（`model/` 純模型、`train.py`／`evaluate.py`／`autogen.py`／`judge.py` 各司其職、`server/` 只做 API 與工作區、`web/` 只做畫面）；每個模組有說明用途與資料格式的 docstring；公開函式有型別標註；`README.md` 能讓人從零跑起來；設定用 YAML、資料用 jsonl、座標一律 0–1。
- 測試：CPU 可跑的單元測試涵蓋解析、比對、評分、API 流程（`uv run pytest`）；GPU 行為用小批次實測。
- 分支：`feat/* → dev → main`，PR 以 `dev` 為 base。`uv` 管環境；`ruff` 格式與 lint。

## 10. 現況（2026-10-23）

- 已有：模型移植與 base 分數；自動出題（兩階段 teacher、批次）；一次一題的檢查頁；UI 訓練管線（訓→評→比）；④ 分數表／看差異／單張試玩；圖池 hydranet 300、LOCO 200（Data Core T1）。
- 標註批次的進度只在記憶體：server 重啟或取消會中斷，但出好的題都在；資料集頁會標「中斷」並給「接著跑」（只補沒出題的圖、沒作答的題、沒比對的題）。**跑標註時不要重啟 server。**
- 進行中：依本文件改成「資料集」為中心的介面（先出稿）；Photon 相容的 adapter 預設；驗證集切分；Orin 匯出。
- 待 Paul：NVR RTSP 連線資訊（工廠 CCTV 新畫面）。
