"""AcousticMobileNet model wrapping MobileNetV3-Small for single-channel Log-Mel spectrograms."""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from config import CLASSES


class AcousticMobileNet(nn.Module):
    """Single-channel MobileNetV3-Small classifier for [B, 1, 128, 100] Log-Mel tensors."""

    def __init__(
        self,
        num_classes: int = len(CLASSES),
        dropout: float = 0.2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        backbone: nn.Module
        if pretrained:
            try:
                backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
                in_features = backbone.classifier[3].in_features
                if isinstance(backbone.classifier[2], nn.Dropout):
                    backbone.classifier[2].p = float(dropout)
                backbone.classifier[3] = nn.Linear(in_features, num_classes)
            except Exception as exc:
                warnings.warn(
                    f"Falling back to random initialization (pretrained weights unavailable: {exc})",
                    stacklevel=2,
                )
                pretrained = False
                backbone = mobilenet_v3_small(
                    weights=None, num_classes=num_classes, dropout=dropout
                )
        else:
            backbone = mobilenet_v3_small(
                weights=None, num_classes=num_classes, dropout=dropout
            )

        orig_conv: nn.Conv2d = backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            dilation=orig_conv.dilation,
            groups=1,
            bias=orig_conv.bias is not None,
        )
        if pretrained and orig_conv.weight.shape[1] == 3:
            with torch.no_grad():
                new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
                if orig_conv.bias is not None and new_conv.bias is not None:
                    new_conv.bias.copy_(orig_conv.bias)

        backbone.features[0][0] = new_conv
        self.backbone = backbone
        # torchvision MobileNetV3 defaults to bn momentum=0.01 (tuned for 1.2M ImageNet);
        # 0.05 stabilizes running stats cleanly within 20 epochs on acoustic corpora.
        for mod in self.backbone.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.momentum = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw [B, num_classes] logits for input spectrogram [B, 1, 128, 100]."""
        return self.backbone(x)
