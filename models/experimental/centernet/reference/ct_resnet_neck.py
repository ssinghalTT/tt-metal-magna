import torch
import torch.nn as nn
from typing import Tuple


class CTResNetNeck(nn.Module):
    def __init__(self, parameters=None, init_cfg=None) -> None:
        print("ctresnet neck")
        super().__init__()
        self.fp16_enabled = False
        self.parameters = parameters
        self.deconv_layers = self._make_deconv_layer()

    def _make_deconv_layer(self) -> nn.Sequential:
        """use deconv layers to upsample backbone's output."""
        inplanes = [256, 256, 128, 128, 64, 64]
        conv_in = [512, 256, 256, 128, 128, 64]
        conv_out = [256, 256, 128, 128, 64, 64]
        kernel = 4
        stride = 1
        layers = []
        for i in range(1):
            print(i)
            # feat_channels = num_deconv_filters[i]
            if i % 2 == 0:
                kernel = 3
                stride = 1
            else:
                kernel = 4
                stride = 2
            if i % 2 == 0:
                conv_module = nn.Conv2d(conv_in[i], conv_out[i], kernel_size=kernel, stride=stride, padding=1)
                conv_module.weight = nn.Parameter(self.parameters[f"neck.deconv_layers.{i}.conv.weight"])
            else:
                conv_module = nn.ConvTranspose2d(conv_in[i], conv_out[i], kernel_size=kernel, stride=stride, padding=1)
                conv_module.weight = nn.Parameter(self.parameters[f"neck.deconv_layers.{i}.conv.weight"])
            layers.append(conv_module)
            bn = nn.BatchNorm2d(inplanes[i])  # .eval()
            bn.weight = nn.Parameter(self.parameters[f"neck.deconv_layers.{i}.bn.weight"])
            bn.bias = nn.Parameter(self.parameters[f"neck.deconv_layers.{i}.bn.bias"])
            bn.running_mean = self.parameters[f"neck.deconv_layers.{i}.bn.running_mean"]
            bn.running_var = self.parameters[f"neck.deconv_layers.{i}.bn.running_var"]

            layers.append(bn)
            relu = nn.ReLU(inplace=True)
            layers.append(relu)
            # upsample_module = ConvModule(
            #     feat_channels,
            #     feat_channels,
            #     num_deconv_kernels[i],
            #     stride=2,
            #     padding=1,
            #     conv_cfg=dict(type='deconv'),
            #     norm_cfg=dict(type='BN'))
            # layers.append(upsample_module)
            # self.in_channels = feat_channels

        return nn.Sequential(*layers)

    def forward(self, x) -> Tuple[torch.Tensor]:
        outs = self.deconv_layers(x[-1])
        return (outs,)
