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
forge --help
```

## 分支

`feat/* → dev → stage → main`。PR 以 `dev` 為 base；進 `stage` 要過 GOALS §4 的 gate。
