import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../")
from utils.warpper import Conv1d, BatchNorm1d
 
class Dynamic_weight_network_DFN(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.output_dim = 16
        self.mask_conv_num = 3
        conv_block = conv_with_kaiming_uniform("BN", activation=True)

        before_embedding_conv_num = 1
        before_embedding_tower = []
        for i in range(before_embedding_conv_num-1):
            before_embedding_tower.append(conv_block(input_channel, input_channel))
        before_embedding_tower.append(conv_block(input_channel, self.output_dim))
        self.add_module("before_embedding_tower", nn.Sequential(*before_embedding_tower))
        self.controller = nn.Conv1d(self.output_dim, output_channel, kernel_size=1)
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

    def forward(self, input):
        before_embedding_feature = self.before_embedding_tower(torch.unsqueeze(input, dim=-1).permute(2,1,0))
        controller = self.controller(before_embedding_feature).permute(2,1,0).squeeze(dim=-1)
        
        return controller

def conv_with_kaiming_uniform(norm=None, activation=None, use_sep=False):
    def make_conv(in_channels, out_channels):
        conv_func = Conv1d
        if use_sep:
            assert in_channels == out_channels
            groups = in_channels
        else:
            groups = 1

        conv = conv_func(in_channels,
                         out_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0,
                         groups=groups,
                         bias=(norm is None))

        nn.init.kaiming_uniform_(conv.weight, a=1)
        if norm is None:
            nn.init.constant_(conv.bias, 0)

        module = [conv,]
        if norm is not None and len(norm) > 0:
            norm_module = BatchNorm1d(out_channels)
            module.append(norm_module)
        if activation is not None:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv 
 
class DynamicFilterLayer(nn.Module): 
    def __init__(self, filter_size, stride=1, pad=0, flip_filters=False, grouping=False):
        super(DynamicFilterLayer, self).__init__()
        self.filter_size = filter_size #tuple 3
        self.stride = stride           #tuple 2
        self.pad = pad                 #tuple 2
        self.flip_filters = flip_filters
        self.grouping = grouping

    def forward(self, _input, **kwargs):
        points = _input[0]
        filters = _input[1]

        input_channel = points.shape[-1]
        if points.dim() == 2:
            points = points.unsqueeze(0).permute(0, 2, 1)
            for i, filter_num in enumerate(self.filter_size):
                filter_size = input_channel * filter_num
                filter_weight = filters[:filter_size].view(filter_num, input_channel, 1)
                filters = filters[filter_size:]
                filter_bias = filters[:filter_num].view(-1)
                filters = filters[filter_num:]

                points = F.conv1d(points, filter_weight, filter_bias, padding = self.pad)
                if i < (len(self.filter_size) -1):
                    points = F.relu(points)

                input_channel = filter_num
        else:
            n_mask = points.shape[1]
            num_instances = points.shape[0]
            points = points.permute(0, 2, 1).reshape(1, -1, n_mask)
            for i, filter_num in enumerate(self.filter_size):
                filter_size = input_channel * filter_num
                filter_weight = filters[:, :filter_size].reshape(num_instances * filter_num, input_channel, 1)
                filters = filters[:, filter_size:]
                filter_bias = filters[:, :filter_num].reshape(-1)
                filters = filters[:, filter_num:]

                points = F.conv1d(points, filter_weight, filter_bias, padding = self.pad, groups=num_instances)
                if i < (len(self.filter_size) -1):
                    points = F.relu(points)

                input_channel = filter_num
            points = points.squeeze()
        
        output = torch.sigmoid(points)
        return output



if __name__ == "__main__":
    DyConv = DynamicFilterLayer(filter_size=[8,8,1], stride=1)
    input = torch.randn(10,11)
    filters = torch.randn(177)
    _input = (input, filters)
    output = DyConv(_input)
    print(output)