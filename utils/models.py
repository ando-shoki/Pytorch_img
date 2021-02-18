import yaml
import torch.nn as nn
import torch.optim as optim
from torchvision import models

#ResNet18
def finetune_resnet18():
    
    use_pretrained = True  # 学習済みのパラメータを使用
    net = models.resnet18(pretrained=use_pretrained)

    # 最後の出力層のユニットを入れ替える
    with open("./config.yaml") as f:
        config = yaml.load(f)
        num_classes = len(config['CLASS_NAMES'].split())
    # 出力次元を1000->num_classesに変更
    net.fc =  nn.Linear(in_features=512, out_features=num_classes)
    #　学習モードに変更
    net.train()
    print('学習済みのパラメータをロードし、訓練モードに設定完了')

    #以下fine turning
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    # 学習させる層のパラメータ名を指定
    update_param_names_1 = ["layer1.0", "layer1.1"]
    update_param_names_2 = ["layaer2.0", "layer2.1", "layaer3.0", "layer3.1",
                           "layaer4.0", "layer4.1"]
    update_param_names_3 = ["fc"]

    # パラメータごとに各リストに格納する
    for name, param in net.named_parameters():
        if update_param_names_1[0] in name:
            param.requires_grad = True
            params_to_update_1.append(param)
            print("params_to_update_1に格納：", name)

        elif name in update_param_names_2:
            param.requires_grad = True
            params_to_update_2.append(param)
            print("params_to_update_2に格納：", name)

        elif name in update_param_names_3:
            param.requires_grad = True
            params_to_update_3.append(param)
            print("params_to_update_3に格納：", name)

        else:
            param.requires_grad = False
            print("学習しない：", name)
        
        print('fine_tuning設定完了')
        # Optimizer設定
    optimizer = optim.SGD([
        {'params': params_to_update_1, 'lr': 1e-4},
        {'params': params_to_update_2, 'lr': 5e-4},
        {'params': params_to_update_3, 'lr': 1e-3}
    ], momentum=0.9)

    return net,optimizer

# AlexNet
def finetune_alexnet():
    # AlexNetのインスタンスを生成
    use_pretrained = True  # 学習済みのパラメータを使用
    net = models.alexnet(pretrained=use_pretrained)

    # AlexNetの最後の出力層のユニットを入れ替える
    with open("./config.yaml") as f:
        config = yaml.load(f)
        num_classes = len(config['CLASS_NAMES'].split())
    # AlexNetでは6層目がclassiferなので出力次元を変更する
    net.classifier[6] = nn.Linear(in_features = 4096, out_features = num_classes)
    
    net.train()
    print('学習済みのパラメータをロードし、訓練モードに設定完了')

    #以下fine turning
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    # 学習させる層のパラメータ名を指定
    # classiferは1, 4, 6で全て
    update_param_names_1 = ["features"]
    update_param_names_2 = ["classifier.1.weight",
                            "classifier.1.bias", "classifier.4.weight", "classifier.4.bias"]
    update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

    # パラメータごとに各リストに格納する
    for name, param in net.named_parameters():
        if update_param_names_1[0] in name:
            param.requires_grad = True
            params_to_update_1.append(param)
            print("params_to_update_1に格納：", name)

        elif name in update_param_names_2:
            param.requires_grad = True
            params_to_update_2.append(param)
            print("params_to_update_2に格納：", name)

        elif name in update_param_names_3:
            param.requires_grad = True
            params_to_update_3.append(param)
            print("params_to_update_3に格納：", name)

        else:
            param.requires_grad = False
            print("学習しない：", name)
        
        print('fine_tuning設定完了')
        # Optimizer設定
    optimizer = optim.SGD([
        {'params': params_to_update_1, 'lr': 1e-4},
        {'params': params_to_update_2, 'lr': 5e-4},
        {'params': params_to_update_3, 'lr': 1e-3}
    ], momentum=0.9)

    return net,optimizer