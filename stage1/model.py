from __future__ import annotations

import torch
from torch import nn


def _group_norm(num_channels: int) -> nn.GroupNorm:
    groups = 32
    while groups > 1 and (num_channels % groups) != 0:
        groups //= 2
    return nn.GroupNorm(num_groups=max(1, groups), num_channels=num_channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = _group_norm(in_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.norm2 = _group_norm(out_ch)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ResBlock(in_ch, out_ch),
            ResBlock(out_ch, out_ch),
        )
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.block(x)
        return h, self.down(h)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ResBlock(out_ch + skip_ch, out_ch),
            ResBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetRes(nn.Module):
    """
    U-Net + Residual Blocks (ResBlocks), SiLU activation.
    入力: (B, 3, H, W)
    出力: (B, 1, H, W) logits
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.stem = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        self.down1 = Down(c1, c1)
        self.down2 = Down(c1, c2)
        self.down3 = Down(c2, c3)
        self.down4 = Down(c3, c4)

        self.mid = nn.Sequential(ResBlock(c4, c4), ResBlock(c4, c4))

        self.up4 = Up(c4, c4, c3)
        self.up3 = Up(c3, c3, c2)
        self.up2 = Up(c2, c2, c1)
        self.up1 = Up(c1, c1, c1)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)
        s4, x = self.down4(x)
        x = self.mid(x)
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)
