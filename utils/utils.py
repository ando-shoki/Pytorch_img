from glob import glob
from itertools import chain
from os.path import join
from pathlib import Path

import torch
import torch.utils.data as data
import yaml
from PIL import Image
from torchvision import transforms

from utils.preprocessing import gamma


def get_classNames(config_path="./config.yaml"):
    """クラス名のリストを返す。

    config.yaml の CLASS_NAMES は現在のリスト形式と、旧来の文字列形式
    (``pos neg``) のどちらにも対応する。以前は文字列前提で
    ``.split()`` を呼ぶ側に依存していた。
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    class_names = config["CLASS_NAMES"]
    if isinstance(class_names, str):
        return class_names.split()
    return [str(name) for name in class_names]


# 画像データのパスを作成
def make_datapath_list(phase, dataset_path):
    class_names = get_classNames()
    folders = [join(dataset_path, phase, class_name) for class_name in class_names]
    image_lists = []
    ext_list = ["jpg", "jpeg", "png", "bmp"]
    for folder in folders:
        image_list = list(
            chain.from_iterable(
                [glob(join(folder, f"*.{ext}")) for ext in ext_list]
            )
        )
        # glob の返却順に依存しないようソートして再現性を持たせる
        image_lists += sorted(image_list)

    return image_lists


# 画像前処理
# 訓練用のみ無作為な処理でデータを水増しする
class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        resize, scale=(0.5, 1.0)
                    ),  # r*rに画素値を変形後、scaleで指定した比率に無作為に切り抜く
                    transforms.RandomHorizontalFlip(),  # ランダムに水平方向に反転
                    transforms.Lambda(gamma),  # ガンマ処理
                    transforms.ToTensor(),  # テンソルに変換
                    transforms.Normalize(mean, std),  # 標準化
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),  # 画素値の変更 (トリミングの要素は無い点に注意)
                    transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                    transforms.Lambda(gamma),  # ガンマ処理
                    transforms.FiveCrop(120),
                    # 処理後 -> [5, 3, r, r]
                    # stack: リストを渡すと結合する
                    transforms.Lambda(
                        lambda crops: torch.stack(
                            [
                                transforms.Normalize(mean, std)(
                                    transforms.ToTensor()(crop)
                                )
                                for crop in crops
                            ]
                        )
                    ),
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        return self.data_transform[phase](img)


class LoadDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase="train"):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定
        self.class_names = get_classNames()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path).convert("RGB")  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)  # torch.Size([3, 224, 224])

        # ラベルはパスの要素 (ディレクトリ名) から決める。
        # 以前は enumerate の変数で index を上書きし、パスにクラス名が
        # 含まれない場合に label が未定義になっていた。
        path_parts = Path(img_path).parts
        label = None
        for class_index, class_name in enumerate(self.class_names):
            if class_name in path_parts:
                label = class_index
                break
        if label is None:
            raise ValueError(
                f"パスからクラス名を特定できません: {img_path} "
                f"(想定クラス: {self.class_names})"
            )

        return img_transformed, label


def make_dataLoader():
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    batch_size = 32
    # trainとvalの画像へのパスを作成
    train_list = make_datapath_list(phase="train", dataset_path="./dataset/")
    val_list = make_datapath_list(phase="val", dataset_path="./dataset/")

    # Datasetを作成する
    train_dataset = LoadDataset(
        file_list=train_list, transform=ImageTransform(size, mean, std), phase="train"
    )

    val_dataset = LoadDataset(
        file_list=val_list, transform=ImageTransform(size, mean, std), phase="val"
    )

    # DataLoaderを作成する
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    return dataloaders_dict
