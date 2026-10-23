# Pytorch_img

PyTorchで画像処理を行う練習用のリポジトリです。

現在の PyTorch 2.x / torchvision (weights API) / PyYAML 6 系で動くように更新しています。

# Installation

Python 3.10 以上を想定しています。

```bash
pip install -r requirements.txt
```

個別に入れる場合は以下の通りです。なお `sklearn` という名前の
pip パッケージは廃止されたダミーなので、必ず `scikit-learn` を使います。

```bash
pip install torch torchvision
pip install tqdm scikit-learn PyYAML Pillow matplotlib
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```

# Usage

画像を、コード側 (`utils/utils.py` の `make_dataLoader`) が参照する
`dataset/` 以下に下記のように配置します。

```bash
dataset/train/pos/hoge1.png
dataset/train/neg/hoge2.png
dataset/val/pos/hoge3.png
dataset/val/neg/hoge4.png
```

クラス名は `config.yaml` の `CLASS_NAMES` (リスト形式) で定義します。

## Preprocessing

Test Time Augmentationは `transforms.FiveCrop()` で行なっている。
