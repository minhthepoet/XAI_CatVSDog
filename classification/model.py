import torch.nn as nn
from torchvision import models


class ResNet18BinaryClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(args):
    return ResNet18BinaryClassifier(dropout=args.dropout)
