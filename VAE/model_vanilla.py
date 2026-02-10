from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


LAYER_CHANNELS: List[int] = [32, 32, 64, 128, 256]
LAYER_NAMES: List[str] = ["stem", "layer1", "layer2", "layer3", "layer4"]
LAYER_HW: Dict[str, int] = {
    "stem": 56,
    "layer1": 56,
    "layer2": 28,
    "layer3": 14,
    "layer4": 7,
}


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(x)


class VanillaVAE(nn.Module):
    """
    Vanilla VAE for image -> merged activation map.
    q(z|x) and p(y|z) where y is merged activations.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        target_hw: int = 56,
        out_channels: int = 512,
        recon_loss: str = "mse",
        merge_mode: str = "merge",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_hw = target_hw
        self.out_channels = out_channels
        self.recon_loss_type = recon_loss
        self.merge_mode = merge_mode

        self.encoder = nn.Sequential(
            ConvBlock(3, 64, stride=2),     # 224 -> 112
            ConvBlock(64, 128, stride=2),   # 112 -> 56
            ConvBlock(128, 256, stride=2),  # 56 -> 28
            ConvBlock(256, 384, stride=2),  # 28 -> 14
            ConvBlock(384, 512, stride=2),  # 14 -> 7
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_hidden = nn.Linear(512, 512)
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)

        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.SiLU(inplace=True),
        )
        self.dec_up1 = UpBlock(512, 384)  # 7 -> 14
        self.dec_up2 = UpBlock(384, 256)  # 14 -> 28
        self.dec_up3 = UpBlock(256, 192)  # 28 -> 56
        self.dec_refine = ConvBlock(192, 128, stride=1)
        self.out_head = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x).flatten(1)
        h = F.silu(self.fc_hidden(h))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(z.size(0), 512, 7, 7)
        h = self.dec_up1(h)
        h = self.dec_up2(h)
        h = self.dec_up3(h)
        h = self.dec_refine(h)
        y_hat = self.out_head(h)
        if y_hat.shape[-2:] != (self.target_hw, self.target_hw):
            y_hat = F.interpolate(y_hat, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)
        return y_hat

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

    def split_merged_to_layers(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        c0 = 0
        for layer_name, c in zip(LAYER_NAMES, LAYER_CHANNELS):
            t = y[:, c0:c0 + c]
            c0 += c
            hw = LAYER_HW[layer_name]
            if t.shape[-2:] != (hw, hw):
                t = F.interpolate(t, size=(hw, hw), mode="bilinear", align_corners=False)
            out[layer_name] = t
        return out

    def reconstruction_loss(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        losses = []
        pred_layers = self.split_merged_to_layers(y_hat)

        if self.merge_mode == "merge":
            target_layers = self.split_merged_to_layers(y_true)
        else:
            target_layers = y_true

        for layer_name in LAYER_NAMES:
            pred = pred_layers[layer_name]
            target = target_layers[layer_name]
            if self.recon_loss_type == "smooth_l1":
                l = F.smooth_l1_loss(pred, target, reduction="mean")
            else:
                l = F.mse_loss(pred, target, reduction="mean")
            losses.append(l)
        return torch.stack(losses).mean()

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y_hat = self.decode(z)
        return y_hat, mu, logvar
