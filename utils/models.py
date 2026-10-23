import yaml
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import AlexNet_Weights, ResNet18_Weights


def _num_classes(config_path="./config.yaml"):
    """config.yaml からクラス数を返す。

    CLASS_NAMES は現在のリスト形式 (``- pos``) と、旧来の
    文字列形式 (``pos neg``) のどちらにも対応する。
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    class_names = config["CLASS_NAMES"]
    if isinstance(class_names, str):
        return len(class_names.split())
    return len(list(class_names))


def _build_optimizer(net, group_specs):
    """パラメータ名の prefix ごとにグループ分けして SGD を作る。

    group_specs: [(prefix のタプル, 学習率), ...] を学習対象グループの
    順に並べたもの。どの prefix にも一致しないパラメータは凍結する。
    空のグループは SGD に渡さない (空の params はエラーになるため)。
    """
    groups = [[] for _ in group_specs]

    for name, param in net.named_parameters():
        for index, (prefixes, _lr) in enumerate(group_specs):
            if name.startswith(prefixes):
                param.requires_grad = True
                groups[index].append(param)
                print(f"params_to_update_{index + 1}に格納: {name}")
                break
        else:
            param.requires_grad = False
            print("学習しない:", name)

    print("fine_tuning設定完了")

    param_groups = [
        {"params": params, "lr": lr}
        for params, (_prefixes, lr) in zip(groups, group_specs)
        if params
    ]
    if not param_groups:
        raise ValueError("学習対象のパラメータが1つもありません")

    return optim.SGD(param_groups, momentum=0.9)


# ResNet18
def finetune_resnet18(use_pretrained=True):
    # 現在の torchvision では pretrained=True ではなく weights= を使う
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    net = models.resnet18(weights=weights)

    # 最後の出力層のユニットを入れ替える (1000 -> num_classes)
    num_classes = _num_classes()
    net.fc = nn.Linear(in_features=512, out_features=num_classes)

    net.train()
    print("学習済みのパラメータをロードし、訓練モードに設定完了")

    # 学習させる層を prefix で指定する
    # 旧コードは "layaer2.0" のような綴りミスと、完全一致判定により
    # layer2 以降と fc が学習対象から外れていた
    optimizer = _build_optimizer(
        net,
        [
            (("layer1.",), 1e-4),
            (("layer2.", "layer3.", "layer4."), 5e-4),
            (("fc.",), 1e-3),
        ],
    )

    return net, optimizer


# AlexNet
def finetune_alexnet(use_pretrained=True):
    weights = AlexNet_Weights.DEFAULT if use_pretrained else None
    net = models.alexnet(weights=weights)

    # AlexNet では classifier[6] が最終層なので出力次元を変更する
    num_classes = _num_classes()
    net.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    net.train()
    print("学習済みのパラメータをロードし、訓練モードに設定完了")

    optimizer = _build_optimizer(
        net,
        [
            (("features.",), 1e-4),
            (("classifier.1.", "classifier.4."), 5e-4),
            (("classifier.6.",), 1e-3),
        ],
    )

    return net, optimizer
