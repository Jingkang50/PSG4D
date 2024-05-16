import os
import sys

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule
import functools
from collections import OrderedDict

from backbone.transformer import TransformerEncoder
sys.path.append("../")
# import data.scannetv2_inst 
import dknet_ops


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)

        return output


class VGGBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        self.conv_layers = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        return self.conv_layers(input)


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1, add_transformer=False):

        super().__init__()

        self.nPlanes = nPlanes

        if block == 'ResidualBlock':
            block = ResidualBlock
        elif block == 'VGGBlock':
            block = VGGBlock
            
        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)
        self.add_transformer = add_transformer

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1, add_transformer = add_transformer)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

        if  len(nPlanes)<=2 and add_transformer:
            d_model = 128
            self.before_transformer_linear = nn.Linear(nPlanes[0], d_model)
            self.transformer = TransformerEncoder(d_model=d_model, N=2, heads=4, d_ff=64)
            self.after_transformer_linear = nn.Linear(d_model, nPlanes[0])
        else:
            self.before_transformer_linear = None
            self.transformer = None
            self.after_transformer_linear = None

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))

            output = self.blocks_tail(output)

        if self.before_transformer_linear:
            batch_ids = output.indices[:, 0]
            xyz = output.indices[:, 1:].float()
            feats = output.features
            before_params_feats = self.before_transformer_linear(feats)
            feats = self.transformer(xyz=xyz, features=before_params_feats, batch_ids=batch_ids)
            feats = self.after_transformer_linear(feats)
            output = output.replace_feature(feats)

        return output

if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES']= '0'
    norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
    block_reps = 2
    block = ResidualBlock
    m = 16

    dataset = data.scannetv2_inst.Dataset()
    dataset.trainLoader()
    dataset.valLoader()
    trainloader = dataset.train_data_loader

    for i, data in enumerate(trainloader):
        torch.cuda.empty_cache()

        coords = data['locs'].cuda()                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
        voxel_coords = data['voxel_locs'].cuda()              # (M, 1 + 3), long, cuda
        p2v_map = data['p2v_map'].cuda()                      # (N), int, cuda
        v2p_map = data['v2p_map'].cuda()                      # (M, 1 + maxActive), int, cuda
        coords_float = data['locs_float'].cuda()              # (N, 3), float32, cuda
        feats = data['feats'].cuda()                          # (N, C), float32, cuda
        spatial_shape = data['spatial_shape']
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = dknet_ops.voxelization(feats, v2p_map, 4)  # (M, C), float, cuda

        input_ = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, 4)
        #print(voxel_feats.shape, voxel_coords.shape, spatial_shape.shape)
        input_conv = spconv.SparseSequential(
                spconv.SubMConv3d(6, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )
        input_conv.cuda()
        unet = UBlock([m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], norm_fn, block_reps, block, indice_key_id=1, add_transformer=True)
        unet.cuda()
        
        with torch.no_grad():
            output = input_conv(input_)
            output = unet(output)
            output_feats = output.features[p2v_map.long()]
            print(output_feats.shape, output.features[:].shape)