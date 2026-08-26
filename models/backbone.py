from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ResNetBackbone(nn.Module):

    def __init__(self, pretrained: bool = True, freeze_stages: int = 2, feature_dim: int = 2048):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.pool = net.avgpool
        self.feature_dim = feature_dim
        assert net.fc.in_features == feature_dim

        self._freeze(freeze_stages)

    def _freeze(self, n_stages: int):
        stages = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        for stage in stages[:max(0, min(n_stages, len(stages)))]:
            for p in stage.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return x.view(b, t, self.feature_dim)
    
