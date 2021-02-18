# Pytorch_img
 
 Pytorchで画像処理を行う練習用のリポジトリです。
 
# Installation
 
ローカルで学習する場合は以下のライブラリをインストールしてください。
 
```bash
pip install torch
pip install torchvision
pip install tqdm
pip install sklearn
```
 
# Usage

画像を下記のように配置します。
```bash
tongue_dataset/train/pos/hoge1.png
tongue_dataset/train/neg/hoge2.png
tongue_dataset/val/pos/hoge3.png
tongue_dataset/val/neg/hoge4.png
```

 ## Preprocessing
 
 Test Time Augmentationは```transforms.FiveCrop()```で行なっている。