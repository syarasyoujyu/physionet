# Stage 1: ECG正規化（グリッド交点セグメンテーション）

このフォルダは「歪んだECG画像(256×256パッチ) → グリッド交点マスク(2値)」を学習するための最小実装です。

## データ（`0011/`の形式）

- `xxxx.png` : ECG画像（例: `0011/37579740.png`）
- `xxxx.json`: メタデータ（例: `0011/37579740.json`）

本リポジトリの例データには、明示的なグリッド交点マスクが付属していません。
そのためデフォルトでは、画像からグリッド線を抽出→コーナー検出で交点ラベルを自動生成します（OpenCV必須）。

もし手元に正解マスクがある場合は、`--label-source file --mask-suffix _grid_mask.png` のように指定してください。

## 依存関係

- `torch`
- `numpy`
- `pillow`
- `opencv-python`（`--label-source auto` の場合）
- `tqdm`

## 学習

```bash
python3 -m stage1.train \
  --data-root . \
  --out-dir runs/stage1 \
  --epochs 300 \
  --lr 0.005 \
  --batch-size 16 \
  --patch-size 256
```

## 出力

- `runs/stage1/checkpoints/best.pt` : ベストモデル

