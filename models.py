from functools import partial

import torch
import torchvision
from torch import nn
from torchvision.models import EfficientNet
from torchvision.ops import Conv2dNormActivation

import birdnet
from birdnet import BirdNET
from enum import Enum

class ClassificationNetworkType(Enum):
    RESNET50 = "resnet50"
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    WIDE_RESNET50 = "wide-resnet50"
    BIRD_NET = "bird-net"
    EFFICIENT_NET_S = "efficient-net-s"


def build_classification_network(
        network_type: ClassificationNetworkType,
        num_classes: int,
        is_multilabel: bool = True,
        pretrained_path: str = None
):

    if pretrained_path is None:
        if network_type == ClassificationNetworkType.RESNET18:
            classification_network = torchvision.models.resnet18(num_classes=num_classes)
            classification_network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif network_type == ClassificationNetworkType.RESNET34:
            classification_network = torchvision.models.resnet34(num_classes=num_classes)
            classification_network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif network_type == ClassificationNetworkType.RESNET50:
            classification_network = torchvision.models.resnet50(num_classes=num_classes)
            classification_network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif network_type == ClassificationNetworkType.WIDE_RESNET50:
            classification_network = torchvision.models.wide_resnet50_2(num_classes=num_classes)
            classification_network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                     bias=False)
        elif network_type == ClassificationNetworkType.BIRD_NET:
            classification_network = BirdNET(embedding_dimension=num_classes)
        elif network_type == ClassificationNetworkType.EFFICIENT_NET_S:
            classification_network = torchvision.models.efficientnet_v2_s(num_classes=num_classes)
            classification_network.features[0] = Conv2dNormActivation(
                1, classification_network.features[0].out_channels, kernel_size=3, stride=2,
                norm_layer=partial(nn.BatchNorm2d, eps=1e-03), activation_layer=nn.SiLU
            )
        else:
            raise NotImplementedError(f"Embedding network for type {network_type} not implemented")

        classification_network.kwargs = {
            'network_type': network_type,
            'num_classes': num_classes,
            'is_multilabel': is_multilabel
        }

        return classification_network
    else:
        print("Using pretrained weights...")

        kwargs, state_dict = torch.load(pretrained_path)

        assert (kwargs['network_type'] == network_type)

        classification_network = build_classification_network(**kwargs)
        classification_network.load_state_dict(state_dict)

        if network_type == ClassificationNetworkType.RESNET18:
            classification_network.fc = nn.Linear(classification_network.fc.in_features, num_classes)
        elif network_type == ClassificationNetworkType.RESNET34:
            classification_network.fc = nn.Linear(classification_network.fc.in_features, num_classes)
        elif network_type == ClassificationNetworkType.RESNET50:
            classification_network.fc = nn.Linear(classification_network.fc.in_features, num_classes)
        elif network_type == ClassificationNetworkType.WIDE_RESNET50:
            classification_network.fc = nn.Linear(classification_network.fc.in_features, num_classes)
        elif network_type == ClassificationNetworkType.BIRD_NET:
            classification_network.conv4 = nn.Conv2d(in_channels=int(birdnet.FILTERS[-1] * birdnet.RESNET_K * 2), out_channels=num_classes, kernel_size=1)
        elif isinstance(classification_network, EfficientNet):
            classification_network.classifier[-1] = nn.Linear(classification_network.classifier[-1].in_features, out_features=num_classes)
        else:
            raise NotImplementedError(f"Embedding network for type {network_type} not implemented")

        return classification_network
