# LRASPP MobilenetV2 by MAGNA
""" Contains an LRASPP model with a mobilenet_v2 backbone that is compatiple with TDA4. """
from typing import Any, Optional, Dict
from collections import OrderedDict

from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.models.mobilenetv2 import MobileNetV2, MobileNet_V2_Weights, mobilenet_v2
from torchvision.models._utils import IntermediateLayerGetter, handle_legacy_interface
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model

class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, scale_factor=(
            4, 4), mode="bilinear", align_corners=False)
        l = self.low_classifier(low)
        h = self.high_classifier(x)
        out = l + h
        return out

class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int, optional): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """

    def __init__(
        self, backbone: nn.Module, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = 128
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.backbone = backbone
        self.classifier = LRASPPHead(
            low_channels, high_channels, num_classes, inter_channels)

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(out, scale_factor=(
            8, 8), mode="bilinear", align_corners=False)
        result = OrderedDict()
        result["out"] = out

        return result


def _lraspp_mobilenetv2(backbone: MobileNetV2, num_classes: int) -> LRASPP:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [
        0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels
    backbone = IntermediateLayerGetter(
        backbone, return_layers={str(low_pos): "low", str(high_pos): "high"})

    return LRASPP(backbone, low_channels, high_channels, num_classes)


@register_model()
@handle_legacy_interface(
    weights_backbone=("pretrained_backbone",
                      MobileNet_V2_Weights.IMAGENET1K_V1),
)
def lraspp_mobilenet_v2(
    *,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V2_Weights] = MobileNet_V2_Weights.IMAGENET1K_V1,
    **kwargs: Any,
) -> LRASPP:
    """Constructs a Lite R-ASPP Network model with a MobileNetV2 backbone.

    .. betastatus:: segmentation module

    Args:
        num_classes (int, optional): number of output classes of the model (including the background).
        aux_loss (bool, optional): If True, it uses an auxiliary loss.
        weights_backbone (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The pretrained
            weights for the backbone.
        **kwargs: parameters passed to the ``torchvision.models.segmentation.LRASPP``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/lraspp.py>`_
            for more details about this class.
    """
    if kwargs.pop("aux_loss", False):
        raise NotImplementedError("This model does not use auxiliary loss")

    weights_backbone = MobileNet_V2_Weights.verify(weights_backbone)

    if num_classes is None:
        num_classes = 21

    backbone = mobilenet_v2(weights=weights_backbone)
    model = _lraspp_mobilenetv2(backbone, num_classes)

    return model
