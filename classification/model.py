import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class CustomResNetBackbone(nn.Module):
    # Lightweight ResNet-style backbone for binary dog/cat classification.
    def __init__(self, block_counts=(3, 4, 6, 3), base_width: int = 64):
        super().__init__()
        self.in_channels = base_width

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_stage(base_width, block_counts[0], stride=1)
        self.layer2 = self._make_stage(base_width * 2, block_counts[1], stride=2)
        self.layer3 = self._make_stage(base_width * 4, block_counts[2], stride=2)
        self.layer4 = self._make_stage(base_width * 8, block_counts[3], stride=2)
        self.out_channels = base_width * 8

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class DogCatResNetClassifier(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        # Smaller than ResNet18/34: fewer blocks and narrower channels.
        self.backbone = CustomResNetBackbone(block_counts=(2, 2, 2, 2), base_width=32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.out_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        pooled = self.pool(features)
        return self.head(pooled)


def build_model(args):
    return DogCatResNetClassifier(dropout=args.dropout)
