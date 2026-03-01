# Stage 1: ECG正規化（グリッド交点セグメンテーション）

このフォルダは「歪んだECG画像(256×256パッチ) → グリッド交点マスク(2値)」を学習するための最小実装です。

## データ（`0011/`の形式）

- `xxxx.png` : ECG画像（例: `0011/37579740.png`）
- `xxxx.json`: メタデータ（例: `0011/37579740.json`）

本リポジトリの例データには、明示的なグリッド交点マスクが付属していません。
ただし `xxxx.json` には generator が使った `x_grid` / `y_grid` が入っているため、
デフォルトではその情報から主要グリッド交点マスクを直接再構成します。
augment 時に `rotate_applied` が入っている JSON では、その回転も反映します。

もし JSON の格子情報が信頼できないデータであれば、`--label-source auto` で
画像からグリッド線を抽出→コーナー検出に戻せます（OpenCV必須）。
手元に正解マスクがある場合は、`--label-source file --mask-suffix _grid_mask.png` を指定してください。

## 学習

```bash
python3 -m stage1.train \
  --data-root . \
  --out-dir runs/stage1 \
  --max-data-num 1000 \
  --init-from runs/stage1/checkpoints/best.pt \
  --log-level INFO \
  --log-interval 50 \
  --epochs 300 \
  --lr 0.005 \
  --batch-size 16 \
  --patch-size 256
```

## 推論（可視化）

推論結果とGT（交点マスク）をmatplotlibで並べて保存します。

```bash
python3 -m stage1.inference \
  --data-root . \
  --checkpoint runs/stage1/checkpoints/best.pt \
  --out-dir runs/stage1/infer \
  --num-samples 8 \
  --crop center
```

交点マスクは非常に疎（点）なので、可視化では `--dot-radius` で点を膨張して見やすくしています。

## 出力

- `runs/stage1/checkpoints/best.pt` : ベストモデル
