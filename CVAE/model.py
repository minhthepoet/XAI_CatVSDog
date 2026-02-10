import torch
import torch.nn as nn
import torch.nn.functional as F

ACT_LAYER_ORDER = ["stem", "layer1", "layer2", "layer3", "layer4"]
ACT_LAYER_CHANNELS = {
    "stem": 32,
    "layer1": 32,
    "layer2": 64,
    "layer3": 128,
    "layer4": 256,
}
ACT_LAYER_HW = {
    "stem": 56,
    "layer1": 56,
    "layer2": 28,
    "layer3": 14,
    "layer4": 7,
}


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 224 -> 112
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 112 -> 56
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.proj(h)


class ActsEncoder(nn.Module):
    def __init__(self, in_channels: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h = self.net(y).flatten(1)
        return self.proj(h)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        target_hw: int = 56,
        y_channels: int = 512,
        h_dim: int = 256,
        beta: float = 0.2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.target_hw = target_hw
        self.y_channels = y_channels
        self.h_dim = h_dim
        self.beta = beta
        self.layer_order = ACT_LAYER_ORDER
        self.layer_channels = ACT_LAYER_CHANNELS
        self.layer_hw = ACT_LAYER_HW

        self.image_encoder = ImageEncoder(out_dim=h_dim)
        self.acts_encoder = ActsEncoder(in_channels=y_channels, out_dim=h_dim)

        self.posterior = nn.Sequential(
            nn.Linear(h_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
        )
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(h_dim + latent_dim, 256 * 7 * 7),
            nn.ReLU(inplace=True),
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(256, 256),
                UpBlock(256, 192),
                UpBlock(192, 128),
            ]
        )
        self.out_head = nn.Conv2d(128, y_channels, kernel_size=3, padding=1)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_per_sample.mean()

    @staticmethod
    def recon_mse_loss(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_hat, y_true, reduction="mean")

    @staticmethod
    def denormalize_merged(
        y_merged: torch.Tensor,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if mean is None or std is None:
            return y_merged
        return y_merged * (std + eps) + mean

    def merged_to_layer_dict(
        self,
        y_merged: torch.Tensor,
        restore_original_hw: bool = True,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
    ):
        y = self.denormalize_merged(y_merged, mean=mean, std=std)
        parts = {}
        c_start = 0
        for layer in self.layer_order:
            c = self.layer_channels[layer]
            chunk = y[:, c_start : c_start + c]
            c_start += c

            if restore_original_hw:
                hw = self.layer_hw[layer]
                if chunk.shape[-2:] != (hw, hw):
                    chunk = F.interpolate(
                        chunk,
                        size=(hw, hw),
                        mode="bilinear",
                        align_corners=False,
                    )
            parts[layer] = chunk
        return parts

    def decode(self, hx: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = torch.cat([hx, z], dim=1)
        h = self.decoder_fc(h).view(h.size(0), 256, 7, 7)

        for block in self.up_blocks:
            if h.shape[-1] < self.target_hw:
                h = F.interpolate(h, scale_factor=2.0, mode="bilinear", align_corners=False)
            h = block(h)

        if h.shape[-2:] != (self.target_hw, self.target_hw):
            h = F.interpolate(h, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)

        y_hat = self.out_head(h)
        return y_hat

    def forward(
        self,
        x_img: torch.Tensor,
        y_merged: torch.Tensor = None,
        return_layer_dict: bool = False,
        restore_original_hw: bool = True,
    ):
        hx = self.image_encoder(x_img)

        if y_merged is not None:
            hy = self.acts_encoder(y_merged)
            post_h = self.posterior(torch.cat([hx, hy], dim=1))
            mu = self.mu_head(post_h)
            logvar = self.logvar_head(post_h)
            z = self.reparameterize(mu, logvar)
        else:
            mu = torch.zeros(x_img.size(0), self.latent_dim, device=x_img.device, dtype=x_img.dtype)
            logvar = torch.zeros_like(mu)
            z = torch.randn_like(mu)

        y_hat = self.decode(hx, z)
        if return_layer_dict:
            y_hat_layers = self.merged_to_layer_dict(
                y_hat,
                restore_original_hw=restore_original_hw,
            )
            return y_hat, mu, logvar, y_hat_layers
        return y_hat, mu, logvar
