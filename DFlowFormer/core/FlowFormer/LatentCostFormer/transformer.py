import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from utils.utils import coords_grid, bilinear_sampler, upflow8
from ..common import FeedForward, pyramid_retrieve_tokens, sampler, sampler_gaussian_fix, retrieve_tokens, MultiHeadAttention, MLP
from ..encoders import twins_svt_large_context, twins_svt_large
from ...position_encoding import PositionEncodingSine, LinearPositionEncoding
from .twins import PosConv
from .encoder import MemoryEncoder
from .decoder import MemoryDecoder
from .cnn import BasicEncoder

class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == 'twins':
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == 'basicencoder':
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn='instance')


    def forward(self, image1, image2, output=None, flow_init=None, tgt_sparsity =None, mhidden=None, tau=0.01):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        # print('image1', image1.shape)
        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            from thop import profile
            # print('self.context_encoder')
            # print('image1', image1.shape)
            # flops, params = profile(self.context_encoder, inputs=(image1,))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 2) + 'M')
            # print('*' * 60)
            context = self.context_encoder(image1)
        # exit(0)
        # print('self.memory_encoder')
        # print('context', context.shape)
        # flops, params = profile(self.memory_encoder, inputs=(image1, image2, data, context,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        # exit(0)
        cost_memory = self.memory_encoder(image1, image2, data, context)

        # print('self.memory_decoder')
        # print('cost_memory', cost_memory.shape)
        # exit(0)
        # flops, params = profile(self.memory_decoder, inputs=(cost_memory, context, data, flow_init,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        # exit(0)
        flow_predictions = self.memory_decoder(cost_memory, context, data, flow_init=flow_init, tgt_sparsity=tgt_sparsity, mhidden=mhidden, tau=tau)
        # print('here')
        return flow_predictions

    def forward_time(self, image1, image2, output=None, flow_init=None, tgt_sparsity =None, mhidden=None, tau=0.01):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        # print('image1', image1.shape)
        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            context = self.context_encoder(image1)

        cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder.forward_time(cost_memory, context, data, flow_init=flow_init, tgt_sparsity=tgt_sparsity, mhidden=mhidden, tau=tau)
        return flow_predictions
