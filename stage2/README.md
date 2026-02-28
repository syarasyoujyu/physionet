# Stage 2: ECG再構成（波形セグメンテーション）

このフォルダは「（歪み補正後を想定した）ECG画像(256×256パッチ) → 波形(誘導線)マスク(2値)」を学習するための最小実装です。

`0011/` の `xxxx.json` には、各リードの `plotted_pixels`（波形が描かれた画素座標列）が入っているため、
これを用いて教師マスクを自動生成します（Pillowで線分として描画）。

## 学習

```bash
python3 -m stage2.train \
  --data-root . \
  --out-dir runs/stage2 \
  --max-data-num 1000 \
  --init-from runs/stage2/checkpoints/best.pt \
  --log-level INFO \
  --log-interval 50 \
  --epochs 200 \
  --lr 0.001 \
  --batch-size 16 \
  --patch-size 256
```
