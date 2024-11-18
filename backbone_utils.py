import torch
import torch.nn as nn
from resnet_cbam import resnet50_cbam, resnet101_cbam, resnet152_cbam
from feature_pyramid_network import FeaturePyramidNetwork
from collections import OrderedDict
from typing import Dict

# 获得中间输出
class IntermediateLayerGetter(nn.ModuleDict):
    __annotations__ = {
        "return_layers": Dict[str, str],
    }
    def __init__(self, model, return_layers) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks):
        super(BackboneWithFPN, self).__init__()
        # 返回每个return_layers需要的输出结构
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

# 如果需要其他网络，可自行添加
def resnet50_cbam_fpn_backbone(pretrain, returned_layers=None, extra_blocks=None):
    backbone = resnet50_cbam(pretrained=pretrain)
    # FPN从backbone返回的层数，如果没有特殊要求，则返回最后四层
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    # 对于resnet50来说，最小层不能低于1，最高层不能比4大
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    # 构造resnet中每个layer对应字典
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    # 获得每个layer输出的尺寸
    # 在resnet中为 [256, 512, 1024, 2048]
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # resnet fpn网络最后输出的channel均为256
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

"""
test sample:

    backbonewithfpn = resnet50_cbam_fpn_backbone(pretrain=False)
    sample = torch.rand(size=(3, 3, 224, 224))
    out = backbonewithfpn(sample)
    print([(k ,v.shape) for k, v in out.items()])
    
"""
