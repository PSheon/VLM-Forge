# VLM-Forge 介面規格

依 `GOALS.md` §8 寫成的畫面規劃（設計文件，決議仍以 `GOALS.md` 為準）。改介面先改這份。
對應程式：`web/src/App.tsx`（骨架）、`web/src/pages/{Datasets,Review,Train,Test}.tsx`、`web/src/components/{Canvas,ui}.tsx`。

## 0. 原則與名詞

- 從資料出發、三步：① 資料集 → ② 訓練 → ③ 測試。資料集頁四個分頁：圖片／檢查／設定／匯出（第 3、4 節）；檢查不是獨立的一步，和 Roboflow／Label Studio／Encord 一樣住在資料集裡。每步一個畫面、一個主按鈕（橘色）。
- 每按一次就存：改了就提示「還沒存」，存了就提示「已存」。
- 名詞：預標（AI 導師）、模型（Moondream）、一致／不一致、未比對、自動通過、抽查、已確認、已修正、丟棄。不出現 judge／teacher／student／triage。
- 顏色：橘＝目前答案（可改）、藍虛線＝預標、紅＝不一致、綠＝一致／通過、灰＝未比對或已存。
- 桌機 1440 為主、平板可用、不做手機；跟系統深淺色；繁中；數字等寬。

## 1. 全站骨架

| 元件 | 規則 |
| --- | --- |
| 步驟列 | 四步各一行小字進度，每 5 秒更新；完成的步驟加 ✓ |
| 名字 | 右上虛線框，點開 dialog 改；每筆確認記名字 |
| GPU | 「閒置」或「工作中：標註 X 12/20」；一次只跑一件 GPU 工作 |
| 缺東西 | 「模型權重缺」「出題模型缺」「GPU 只剩 N GB」「出題模型：8B（小）」橘色 pill，點了顯示怎麼補 |
| 斷線 | 紅 pill「連不到 server」，每 5 秒重試，畫面不清空 |
| `?` | 當前畫面的快捷鍵說明 |

## 2. 進站與名字

第一次開：dialog「你是誰？」，空白不能送；記在瀏覽器。右上可改名。名字出現在每筆紀錄、所有圖的認領標記、run 卡。

## 3. ① 資料集

- 清單卡片：名稱、圖／題／已確認／驗證、模式、更新時間；狀態 pill：標註中 n/m、還有 N 張要看、全部確認、還沒標、空的、上次標註失敗。
- 新資料集 dialog：名稱必填且不重複（即時檢查）、備註。
- 詳情：拖曳區（多張、zip）、資料夾路徑（不存在／含 held-out 拒絕）、縮圖格（角標 val／不一致數／✓）、AI 導師三選一（自動／自動＋指定／只指定）、指定題目即時解析（`skill: 內容`，顯示有效題數與看不懂的行號）、進階（巡檢主題、側重、skill、只抽幾張）。
- 標註按鈕文字隨狀態：標註新加的 N 張／沒有新圖／排隊中／出題模型缺。進度卡：四階段、進度條、ETA、取消；失敗紅卡附原因；學生沒作答時橘字並繼續。中斷（server 重啟／取消）：紅卡列出「N 張還沒出題 · M 題模型還沒答 · K 題還沒比對」＋「接著跑」；清單 pill「標註中斷，可接著跑」。
- 批次紀錄、匯出 jsonl＋zip（下載連結）、改名、刪除（輸入名稱確認）。

## 4. ② 檢查

- 進頁自動取下一張（先不一致、再抽查）；可依資料集篩選；一次一題，只畫這題的框／點；存一題跳下一題，一張看完自動換圖；全部看完→「去訓練」。
- 題目條：每題一點（紅／綠／灰／淡），當前放大；點了跳題（未存改動先問）。
- 題目卡：狀態、題目大字、比對理由（中文含數字）、格式提示（數量題用數字、是非題 yes/no）。
- 主按鈕隨狀態「沒問題」／「存修正」（Enter；Space 留給拖動畫面）；丟棄（D）；跳過（N，釋放認領）；剩下全存（Shift+Enter，3 秒內可復原）。
- 圖工具列：縮放、適應、移動、預標顯示切換、H 看原圖。
- 認領：10 分鐘；所有圖 overlay 顯示別人認領中（灰＋名字）。

### 4.3 detect
拖曳新框、點框選取、拖框移動、拖角縮放、Delete 刪、Tab 選下一框、方向鍵微移（Shift 10px）、1 用預標框、2 用模型框、Z 還原；0 框時若預標或模型有框，提示「確認這張沒有 X？」；超過 12 個實例的物件不出題。

### 4.4 point
點一下加點、拖點移動、Delete 刪、Tab、方向鍵；預標藍框＝物件範圍；一致＝模型的點落在預標框內。

### 4.5 query
答案框（E 聚焦、框內 Enter 存、Esc 離開）、「1 預標」「2 模型」按鈕、格式提示、不一致理由「模型「…」vs 預標「…」」。

### 4.6 caption
長文框；judge 標出圖裡沒有的片語做成紅 chips，✕ 直接從文字刪；長度 short/normal/long；一致＝judge 沒挑出片語。

### 4.7 所有圖（G）
篩選：待看／自動通過／已看過／全部；角標＝不一致題數／抽／✓；別人認領中灰掉；已看過可重審。

## 5. ③ 訓練

- 只列有已確認題的資料集；勾選；合計與 skill 分佈。
- 名稱自動 ft-MMDD-HHMM；步數＝圖數×2（100–1000）；學習率 2e-4；進階：rank、region head（開了 Orin 載不了）。
- 開始前紅字擋住：0 筆、名稱重複、權重缺、另一個訓練在跑；黃字不擋：有圖沒看、某 skill < 20 筆、region head。
- 執行中：階段、步數進度、ETA、VRAM、loss 曲線（train 實線、val 虛線＋評估點）、取消；完成：核對表 指標｜base｜新｜Δ｜PASS/FAIL；失敗：最後 20 行 log；歷史可刪。

## 6. ④ 測試

- 分數表：列＝考卷×skill（真考卷、各資料集驗證集、參考集灰字）；欄＝base、每個 run；格＝分數＋Δ（綠紅）＋「差異」。
- 看差異：run／考卷／篩選（只看變化／變好／變差／全部）；空間題並排兩張疊圖（正解虛線）；文字題三行＋diff；←/→ 換題。
- 單張試玩：任一來源的圖、四個 skill、和 base 比的模型；結果可「存成資料集的題」。
- 匯出到 Orin：下載 adapter 包（adapter.pt、local_adapter.py、README、bench_orin.py、scores.json）；region head 警告；載入範例用 `local_adapter.Tuned`（官方 client 的 adapter 只認雲端 id，包裡自附 kestrel AdapterProvider）。

## 7. 回饋總表

| 時機 | 文案 | 樣式／停留 |
| --- | --- | --- |
| 改框／點／文字、1/2 | 框已改（還沒存，Enter 存） | 橘 toast 1.8 s＋卡片下常駐「✎ 改過了，還沒存」 |
| 存 | 已存 ✓／已存（你的修正）✓ | 深色 1.8 s |
| 丟棄 | 已丟棄 | 橘 |
| 全存 | 這張 N 題全部存了 ✓ ［復原］ | 深色 3 s |
| 看完一張 | 這張看完了 ✓ | 1.2 s 後換圖 |
| 離開未存 | 有改動還沒存：Enter 存，或 Z 還原（換題時 dialog：丟掉／留下） | — |
| 加圖 | 加入 N 張 | 深色 |
| 標註開始 | 開始標註 N 張，先載入出題模型… | 深色 2.5 s |
| 錯誤 | 一行原因＋怎麼辦 | 紅，常駐到點掉 |

## 8. 多人

名字必填；每筆記 reviewer＋時間；取圖鎖 10 分鐘，跳過即釋放；兩人同存後者覆蓋。不做帳號、權限、指派。

## 9. 邊界

大圖用 2048px 副本與 200px 縮圖，座標 0–1；重複圖以 sha 判斷；取消／重啟保留已出的題；權重或出題模型缺時 pill 常駐、相關按鈕灰掉；GPU 被別的訓練佔住時出題模型載不了（32B 需約 70 GB；有下載 8B 時門檻降到 22 GB，並顯示「出題模型：8B（小）」pill），錯誤直接寫明剩多少；預標超過 12 個實例不出題；壞 JSON 修復或略過；held-out 不能進資料集；不到 10 張的資料集沒有驗證集。

## 10. 畫面 → API

| 畫面 | 端點 |
| --- | --- |
| 骨架 | `GET /api/status`、`GET /api/review/summary` |
| ① | `GET/POST /api/datasets`、`GET/POST/DELETE /api/datasets/{id}`、`GET /api/datasets/{id}/grid`、`POST …/upload`、`POST …/folder`、`POST …/label`、`POST /api/queues/{qid}/cancel`、`POST …/export`、`GET …/download/{jsonl|zip}` |
| ② | `GET /api/review/next-image`、`GET /api/review/images`、`GET /api/review/image/{image_id}`、`POST /api/items/{id}/review`、`POST /api/review/image`、`POST /api/items/{id}/reopen` |
| ③ | `GET/POST /api/train`、`GET /api/train/{name}`、`POST /api/train/{name}/cancel`、`DELETE /api/train/{name}` |
| ④ | `GET /api/test/summary`、`GET /api/heldout/diff`、`POST /api/compare`、`POST /api/datasets/{id}/items`、`GET /api/train/{name}/orin.zip` |

## 11. 驗收

①拖 3 張＋1 zip 計數正確、指定題目打錯行有紅字、標註 20 張進度／ETA／取消、匯出可訓練；②四種 skill 各改一題全程鍵盤、未存跳題會問、兩個瀏覽器拿到不同圖、看完自動換圖與出口；③紅字條件、進度／曲線／取消、PASS/FAIL；④表格與 Δ、看差異篩選、試玩兩模型、Orin 包五個檔。
