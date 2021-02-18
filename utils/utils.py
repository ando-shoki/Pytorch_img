import yaml
from os.path import join
from glob import glob
from itertools import chain
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.utils.data as data
from utils.preprocessing import gamma

def get_classNames():
    # class名の読み込み
    with open('./config.yaml') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        class_names = data["CLASS_NAMES"]
    return class_names

# 画像データのパスを作成
def make_datapath_list(phase,dataset_path):
    
    class_names = get_classNames().split()
    folders = [join(dataset_path, phase, class_name) for class_name in class_names]
    image_lists = []
    ext_list = ["jpg", "png","bmp"]
    for folder in folders:
        image_list = list(chain.from_iterable([glob(join(folder, f"*.{ext}")) for ext in ext_list]))
        image_lists += image_list
    
    return image_lists

# 画像前処理
# 訓練用のみ無作為な処理でデータを水増しする
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)), # r*rに画素値を変形後、scaleで指定した比率に無作為に切り抜く
                transforms.RandomHorizontalFlip(),  # ランダムに水平方向に反転
                transforms.Lambda(gamma),  # ガンマ処理
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # 画素値の変更(トリミングの要素は無い点に注意！)
                transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                transforms.Lambda(gamma),  # ガンマ処理
                transforms.FiveCrop(120),
                # 処理後->[5, 3,r,r]
                # stack : リストを渡すと結合する
                transforms.Lambda(lambda crops : torch.stack( 
                [transforms.Normalize(mean, std)(
                        transforms.ToTensor()(crop)
                            )for crop in crops
                ])
                )
                #transforms.ToTensor(),  # テンソルに変換
                #transforms.Normalize(mean, std)  # 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

class LoadDataset(data.Dataset):
    
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list  # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定
        self.class_names = get_classNames().split()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path).convert("RGB")  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(
            img, self.phase)  # torch.Size([3, 224, 224])
        
        #ラベルの指定
        class_names = self.class_names
        for index,class_name in enumerate(class_names):
            if f"/{class_name}/" in img_path:
                label = index
                break
        # 画像のラベルをファイル名から抜き出す
        return img_transformed, label

def make_dataLoader():
    
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    batch_size = 32
    # trainとvalの画像へのパスを作成
    train_list = make_datapath_list(phase="train", dataset_path='./dataset/')
    val_list = make_datapath_list(phase="val", dataset_path='./dataset/')
    


    # Datasetを作成する
    train_dataset = LoadDataset(
        file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')

    val_dataset = LoadDataset(
        file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')


    # DataLoaderを作成する
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    return dataloaders_dict
