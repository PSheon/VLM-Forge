# Changelog

All notable changes to VLM-Forge. Versions follow [SemVer](https://semver.org/); dates are when the version reached `main`.

## [Unreleased]

### Fixed
- **Count answers given by pointing.** Moondream 3 sometimes answers "How many forklifts are there?" with `There is` + a grounded point (the engine does the same); the point was dropped and the answer read "There is ". `Moondream.query_grounded()` keeps the points, the agreement rule counts them, and the UI shows `There is⟨1 個點⟩`. Held-out scoring keeps the plain `query()`, byte-identical to the engine.
- **Re-running triage** no longer re-draws the spot check (it had auto-passed 128 of 149 sampled questions on a half-reviewed batch).
- The AI-teacher settings' save row stays in view while the action column scrolls; side-by-side canvases keep their controls on one line.
- **Training always validates when it can.** A run whose datasets had no confirmed question in their val split trained with no val curve at all (ft-1023-1911). It now holds out ~10 % of the train images (≥ 5 images) instead, and the train page's summary says where validation comes from (datasets return `val_rows`), with a warning when there is none.

### Added
- **Ctrl+Z reopens the last saved question** (any image, a whole 「剩下全存」 counts as one step); reopening an agreed question restores its auto label instead of leaving it empty.
- **Pace** in the review title row: seconds per question over the recent saves (pauses over 2 min ignored) and the time left for what is still pending.
- **Point snapping.** A click inside a pre-label box that has no point yet places the point at the box centre; `Alt` places it exactly where you clicked.
- **Long jobs in the tab title** — 「(標註 72/120) VLM-Forge」, 「(訓練 120/226)」 — and a toast plus a desktop notification (when the tab is in the background and you allowed it on 標註 / 開始訓練) when one finishes or fails.
- `web/e2e/smoke.mjs` (`npm run e2e`): the UI contract of docs/ui-spec §1 as a read-only Playwright check.
- **AI 導師可信度.** The dataset page shows how far the spot check got and the share of re-checked agreed questions you left as they were (≥ 90 % with ≥ 30 seen = auto-passed questions are trustworthy). 「只看抽查」 (dataset card or the review tool row) serves only images with spot-check questions and steps only through those; `GET /api/review/next-image?spot=true`, datasets return `spot`.

## [0.3.0] — 2026-10-23

A layout pass: one visual language, more room for the image, fewer buttons.

### Changed
- **Colour roles.** Orange means only the action to take and your answer on the canvas. Current / selected states are neutral, warnings amber, the focus ring blue.
- **Type and spacing scales** as tokens (20 / 16 / 14 / 13 / 12 px; 4 / 8 / 12 / 16 / 24 px), used everywhere.
- **Top bar.** One short state per step (「119 張待看」「1381 筆可訓」「1 個模型可比」) and a single system chip — GPU, weights, teacher model, last error — whose popover closes on an outside click or `Esc`.
- **Review gets a taller image** (≈ 470 → 560 px at 1440 × 900): the file name and 「這張還有 N 題」 moved into the title row, the canvas controls float over the image, the per-question instructions moved into 「? 快捷鍵」.
- **Every action column** starts with a tool row level with the title row (「? 快捷鍵」 on the right, page tools such as 「所有圖」 on the left), fills the window, and pins its footer (建立 / 刪除) to the bottom.
- **Dataset list** as rows with a thumbnail strip, review progress and the numbers (the API returns `thumbs`).
- **Scores.** The target suite (factory CCTV) is a full-width card; validation splits and reference suites sit below; one 「看差異」 per model column.
- **Widths.** A 340 px action column at 1101–1366 px; one column with the primary action pinned to the bottom of the window at ≤ 1100 px.

### Added
- Loading skeletons for the dataset list, the image grid, the train cards and the scores.

### Removed
- Redundant buttons: the canvas 「移動」 and 「刪除選取」 (Space / middle / right drag pans, `Delete` deletes), the 「去訓練」 link on the dataset page (the step bar has it), duplicate 「看差異」 links, and the train page's second orange button. Key labels appear only on the primary button; the key bar lists the rest.

## [0.2.0] — 2026-10-23

Review is the bottleneck, so this release is about looking at fewer questions, faster.

### Added
- **Per-question triage.** Agreed questions leave the human queue even inside images that have disagreements (10 % / ≥ 20 spot sample). On a 120-image warehouse set: 1 469 questions to look at instead of 2 813. `POST /api/queues/{id}/triage` re-runs it on an existing batch.
- **Quick answers.** Yes/no questions save on `Y` / `N`; count questions take digits and `↑↓`.
- **Focus zoom.** Spatial questions zoom to the union of pre-label and model boxes; `F` = whole image. The canvas follows the image's aspect ratio (no letterbox).
- **Pre-label coverage.** A pre-label box turns green ✓ once the answer covers it; a point outside every box is red.
- **Detect → point sync.** Saving a detect question can save the matching point question too (one point per box centre, on by default, `P` toggles, undo in the toast).
- **Point editing.** 24 px grab area, numbered points, `3` adds a point at every uncovered pre-label box (detect: adds missing boxes), `Backspace` removes the last point/box.
- **Key bar.** A sticky line of the keys that work on the current question; the key you press lights up.
- **Question list** for the current image in the review column; click to jump.
- **Synced views.** 看差異 and 試一張 zoom and pan both canvases together.
- **Train page** rebuilt as ① pick datasets (cards with confirmed counts and skill bars) → ② settings → history, with a "what this run will train" summary next to the button.

### Changed
- **One design language on every page:** content left, a 400 px action column right whose first element is the page's only primary button; no tab bars; sub-views (review, diff, playground) open from the content and go back with `←` or `Esc`; settings, export and the Orin bundle are collapsible sections in the action column. Shared `PageHead`, `Section`, `useEscBack` in `web/src/components/ui.tsx`.
- Skip image is `S` (`N` is "no").
- The train page lists every dataset; ones without confirmed questions are greyed with the reason.

### Fixed
- Primary buttons now sit at the same height on every page.

## [0.1.0] — 2026-10-23

First version that runs data → review → train → test entirely in the UI: datasets with an AI-teacher labelling flywheel (Qwen3-VL-32B, 8B fallback), one-question-at-a-time review, LoRA training on Moondream 3.1's text MLPs (Photon-compatible), held-out scoring and gate, item-by-item diff, playground, Orin bundle with a local adapter loader, resumable labelling batches, motion and robust image loading.

[0.3.0]: https://github.com/PSheon/VLM-Forge/releases/tag/v0.3.0
[0.2.0]: https://github.com/PSheon/VLM-Forge/releases/tag/v0.2.0
[0.1.0]: https://github.com/PSheon/VLM-Forge/releases/tag/v0.1.0
