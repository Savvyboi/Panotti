"""AcousticMobileNet model wrapping MobileNetV3-Small for single-channel Log-Mel spectrograms."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

from config import CLASSES


class AcousticMobileNet(nn.Module):
    """Single-channel MobileNetV3-Small classifier for [B, 1, 128, 100] Log-Mel tensors."""

    def __init__(self, num_classes: int = len(CLASSES), dropout: float = 0.2) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None, num_classes=num_classes, dropout=dropout)

        orig_conv: nn.Conv2d = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            dilation=orig_conv.dilation,
            groups=1,
            bias=orig_conv.bias is not None,
        )
        self.backbone = backbone
        # torchvision MobileNetV3 defaults to bn momentum=0.01 (tuned for 1.2M ImageNet);
        # 0.05 stabilizes running stats cleanly within 20 epochs on acoustic corpora.
        for mod in self.backbone.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.momentum = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw [B, num_classes] logits for input spectrogram [B, 1, 128, 100]."""
        return self.backbone(x)
