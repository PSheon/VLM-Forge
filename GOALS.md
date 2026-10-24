# GOALS — 唯一決議入口

VLM-Forge 的目標、範圍與規格。與程式、其他文件或其他 session 牴觸時以本文件為準；改方向先改本文件。只寫現行決議，過時內容直接刪除（歷史在 git）。
簽核：Paul（重新 grill Q1–Q16，2026-10-23）。Repo 由 Claude 全權負責，細節自行決定，只在 v1 完成點與真正卡住時通知 Paul。

## 1. 目標

Paul 個人的 **Moondream 3.1 微調工具**，跑在 GB300（資料在那裡）。一條線走完：

**上傳照片／指定資料夾 → AI 導師先標一輪 → 人確認 → 存成資料集 → 微調 → 測試 → 匯出到 Orin**

成功的樣子：在自己的工廠／倉儲影像上，微調後的 3.1 在 detect 與巡檢問答（有沒有人、有沒有戴安全帽／背心、出口有沒有被擋、門是否開、地上有無雜物、堆高機是否在動…）明顯贏過原版。內部使用（playground、機器人、之後當 SyncVision 的 teacher），不出貨。

**v1 完成點**：用 ≥200 張審核過的圖訓出一個 adapter，在 CCTV held-out 的 detect 與 query 都 ≥ base，而且 ①→③ 全程在 UI 完成、不碰終端機。

## 2. 不做

Data Core 匯出包、GCS、`stage` 分支與 gate 儀式、PRO 6000 kestrel parity、RTSP 擷取（等有 NVR 連線資訊再談）、vLLM（列研究項目）、帳號／權限／指派、多 GPU、通用 benchmark、GGUF／量化匯出、RLHF、多模型抽象層。不做的功能不留程式（RTSP 擷取、舊的佇列／匯出端點已刪，要做時重寫）。

## 3. 模型與部署

| 項目 | 決議 |
| --- | --- |
| 權重 | Hugging Face `moondream/moondream3.1-9B-A2B`（公開、不需登入），`model-cache/moondream31/`；tokenizer `moondream/starmie-v1` |
| 推論與訓練底層 | `src/vlmforge/model/`：純 PyTorch 移植（vision／text／region／LoRA），GB300 可跑；base 在三套 held-out 的分數與 SyncVision 記錄的 SyncVLM 一致（detect 0.368／0.370） |
| 部署目標 | **Orin AGX，用 Moondream 官方 Photon 跑 base + adapter**（kestrel 有 Jetson aarch64 kernel） |
| adapter 格式 | 官方 `lora_state_dict` 版面（`dense.{i}.up_a/up_b/down_a/down_b`、`moe.{j}.*`，alpha/r 已折進 B），**只掛文字 MLP**，rank 8–32；**region head 預設不訓**（訓了就只能在 Forge 跑，是進階選項） |
| 出題／比對模型 | Qwen3-VL-32B-Instruct（`model-cache/judge`，載入需 ~70 GB），8 張一批；GPU 被佔住放不下時退到 Qwen3-VL-8B-Instruct（`model-cache/judge-8b`，~22 GB），系統 pill 標「用 8B 出題」，重啟後記憶體夠就回 32B。CCTV 考卷上 8B 不比 32B 差（detect 0.51 vs 0.46，query 都 0.97，`runs/teacher-eval/`） |
| GPU | 與 SyncVision 共用 GB300：SyncVision 優先；Forge 訓練 ≤ 60 GB；一律 `numactl --membind=0` |

## 4. 資料集

- **資料集是一等公民**：有名字（例如「工廠-1025」）、建立／更新時間，沒有版本號。
- 圖可混來源、陸續加：上傳、拖曳、zip、影片／GIF（每 N 秒抽一張，預設 5 秒）、GB300 資料夾路徑（含裡面的影片）。held-out 的圖永遠不能進資料集。
- **幾乎一樣的圖不重複收**：和資料集裡已有的某張幾乎一模一樣（64×36 彩色縮圖的第 99 百分位差 ≤ 12，即畫面裡沒有東西移動）就略過；門檻依工廠 CCTV 校準，有東西移動、出現或燈號變色都會留下。已在資料集裡的重複圖可一鍵選取後移除。
- **AI 導師兩種模式**：①自動出題（預設）——導師看圖決定物件（含框）、3–4 題一般問答（數量、是非、屬性、空間、動作）、2–3 題巡檢問答、一段 caption，外加**每張圖都問的物件**（預設 person、forklift、helmet，資料集可改）：導師沒列出的就出一題「0 個框」的負樣本——Moondream 在沒人的 CCTV 畫面會亂畫人（真考卷 person F1 0.09，導師 0.64），只問看得到的東西教不會「沒有」；②指定題目——資料集設定裡寫死要問的物件與問題，每張圖都問。
- 學生（Moondream）答同一批題；比對：框用 IoU 配對、點看是否落在導師框內、文字比數字／是非／字串、caption 由 judge 查幻覺。一致→自動通過；不一致→人看。
- 標籤：文字用導師的答案、框／點用學生的（學它自己的框風格）；人改過的以人為準。匯出 = 資料集的 `jsonl`（座標 0–1）＋圖 zip。
- 每個資料集自動切 **10% 當驗證集**（不進訓練）。
- 設定改了只影響之後的圖；選取舊圖可**重新標註**：機器出的題（自動通過、待看）換成新設定出的題，人看過、改過、丟棄或自己加的題保留且不重複出。

## 5. 檢查（人）

- **每題分流**：一致的題自動通過（即使同一張圖有不一致的題），從中抽 **10%（至少 20 題）** 留給人抽查；人只看不一致＋抽查。重跑分流不重抽已抽中的題。
- **一次一題、每按一次就存**，看完一張自動換下一張；全程鍵盤（鍵位見 `docs/ui-spec.md` §4）。改過沒存要有明顯提示；存了要有提示。
- 導師漏掉的物件或問題，檢查時可**自己加一題**，照平常方式作答。
- **導師可信度**：抽查看滿 30 題、直接同意 ≥ 90% 才信任自動通過的題；之前先看抽查。
- **多人**：進站 dialog 填名字（記在瀏覽器，右上可改），每筆記名字與時間，認領後鎖 10 分鐘；不做帳號、權限、指派。

## 6. 訓練

- UI 一顆按鈕；只露名稱、步數、學習率。預設 r=16、alpha=32、lr 2e-4、步數依資料量（≈ 每張圖 2 步，100–1000）、batch 8、replay 0.5、region head 關。預設值由 Claude 依實驗調整，不逐項問。選到的資料集驗證集沒有確認題時，從訓練圖再切約 10%（≥ 5 張）當驗證。
- 訓完自動在 held-out 與驗證集評估、和 base 比；過關＝**工廠 CCTV 考卷**每個 skill ≥ base − 1 點且至少一個進步（CrowdHuman／機器狗只當參考，不算進去）；一次只跑一個訓練。

## 7. 測試

- **分數在 GB300 算**（③ 測試頁）：CCTV held-out（105 張／210 題，真考卷）＋每個資料集的驗證集；CrowdHuman 與機器狗 gold 只當參考。指標：detect／point F1 macro、query exact（數字、是非）、count MAE；顯示 base vs 新模型與差值，逐題看差異，單張試玩。
- **Orin 只跑延遲腳本**：同一組圖，base vs adapter，每題毫秒；不重算分數。
- ③ 提供「匯出到 Orin」：adapter 檔＋載入範例。

## 8. 介面

- **從資料出發、三步**（Paul 2026-10-23）：**① 資料集 → ② 訓練 → ③ 測試**。資料集頁沒有分頁：圖片格是主畫面，檢查是從縮圖進去的子畫面，AI 導師設定與匯出在右側動作欄。和市面工具一致（Roboflow／Label Studio／Encord：標註與審核都在資料集／專案裡，訓練另起一段引用資料集）。每步一個畫面、一個主按鈕，所有頁面主按鈕同位置（版面規則見 `docs/ui-spec.md` §1）；步驟列顯示各步進度。名詞只用「預標」「模型」「不一致（請你看）」「自動通過」「抽查」。
- 桌機 1440 為主、平板可用、不做手機；跟系統深淺色；繁中。
- 介面改動要用 Playwright 實際看過畫面（`npm run e2e`＋截圖，亮／暗、1024px）才合併。

## 9. 工程規範

- Python AI 專案最佳實踐：乾淨的分層（`model/` 純模型、`train.py`／`evaluate.py`／`autogen.py`／`judge.py` 各司其職、`server/` 只做 API 與工作區、`web/` 只做畫面）；每個模組有說明用途與資料格式的 docstring；公開函式有型別標註；`README.md` 能讓人從零跑起來；設定用 YAML、資料用 jsonl、座標一律 0–1。
- 測試：CPU 可跑的單元測試涵蓋解析、比對、評分、API 流程（`uv run pytest`）；GPU 行為用小批次實測。
- 分支：`feat/* → dev → main`，PR 以 `dev` 為 base。`uv` 管環境；`ruff` 格式與 lint。
- **單一來源**：決議只寫在本文件；`docs/ui-spec.md` 只寫畫面細節、引用這裡的數字；版本紀錄只在 `CHANGELOG.md`。沒有呼叫者的程式碼、端點與文件直接刪。
- **發版**：每次版本號變動同步 README 截圖、CHANGELOG、GitHub 描述與 topics，打 tag 並發 release。

## 10. 現況（2026-10-24）

- ①→③ 全程在 UI 跑通（工廠-測試 120 張 → 2845 題 → 檢查 → 訓練 → 分數／看差異／試一張／Orin 包）；版本內容見 `CHANGELOG.md`。
- 標註批次的進度只在記憶體：server 重啟或取消會中斷，但出好的題都在，資料集頁給「接著跑」。**跑標註或訓練時不要重啟 server。**
- 下一步：Paul 看完 ≥ 30 題抽查 → 用確認過的題訓第一個正式 run，看 CCTV detect／query 是否 ≥ base（v1 完成點）；Orin 實機驗證 `local_adapter.py`。
- 待 Paul：NVR RTSP 連線資訊（工廠 CCTV 新畫面）。
