import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        # inner_block为backbone到FPN的连接1x1卷积
        # layer_blocks为FPN每层输出到目标检测网络的输出conv
        self.inner_blokcs = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blokcs.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def get_result_from_inner_blocks(self, x, index):
        """
            从对应index的inner_blocks中获取结果
        """
        num_blocks = len(self.inner_blokcs)
        # index为-1时，返回最后一层
        if index < 0:
            index += num_blocks
        i = 0
        out = x
        for module in self.inner_blokcs:
            if i == index:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x, index):
        """
            从对应index的inner_blocks中获取结果
        """
        num_blocks = len(self.layer_blocks)
        # index为-1时，返回最后一层
        if index < 0:
            index += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == index:
                out = module(x)
            i += 1
        return out

    def forward(self, x):
        # 这里输入的x是一个包含Backbone网络每层输出的Dict
        # key为层名，value为输出值
        names = list(x.keys())
        x = list(x.values())
        # 最后一层的输出经过
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # results存储FPN层的每层的输出，给后面目标检测网络使用
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for index in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[index], index)
            # 从FPN到的上层向下层融合，需要进行线性插值，扩大尺寸
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            # results最后的顺序应该是最底层在最前面，所以插入在最前面
            results.insert(0, self.get_result_from_layer_blocks(last_inner, index))
        # 最后返回的输出函数为Dict形式，方便后续处理
        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out
