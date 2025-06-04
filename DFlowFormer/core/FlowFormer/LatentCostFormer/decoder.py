import loguru
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from utils.utils import coords_grid, bilinear_sampler, upflow8
from .attention import MultiHeadAttention, LinearPositionEmbeddingSine, ExpPositionEmbeddingSine
from typing import Optional, Tuple

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .gru import BasicUpdateBlock, GMAUpdateBlock
from .gma import Attention

def initialize_flow(img):
    """ Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = coords_grid(N, H, W).to(img.device)
    mean_init = coords_grid(N, H, W).to(img.device)

    # optical flow computed as difference: flow = mean1 - mean0
    return mean, mean_init

class CrossAttentionLayer(nn.Module):
    # def __init__(self, dim, cfg, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
    def __init__(self, qk_dim, v_dim, query_token_dim, tgt_token_dim, add_flow_token=True, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0., pe='linear'):
        super(CrossAttentionLayer, self).__init__()

        head_dim = qk_dim // num_heads
        self.scale = head_dim ** -0.5
        self.query_token_dim = query_token_dim
        self.pe = pe

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = MultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = nn.Linear(query_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, qk_dim, bias=True), nn.Linear(tgt_token_dim, v_dim, bias=True)

        self.proj = nn.Linear(v_dim*2, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout)
        )
        self.add_flow_token = add_flow_token
        self.dim = qk_dim
    def forward(self, query, key, value, memory, query_coord, patch_size, size_h3w3):
        """
            query_coord [B, 2, H1, W1]
        """
        B, _, H1, W1 = query_coord.shape

        if key is None and value is None:
            key = self.k(memory)
            value = self.v(memory)

        # [B, 2, H1, W1] -> [BH1W1, 1, 2]
        query_coord = query_coord.contiguous()
        query_coord = query_coord.view(B, 2, -1).permute(0, 2, 1)[:,:,None,:].contiguous().view(B*H1*W1, 1, 2)
        if self.pe == 'linear':
            query_coord_enc = LinearPositionEmbeddingSine(query_coord, dim=self.dim)
        elif self.pe == 'exp':
            query_coord_enc = ExpPositionEmbeddingSine(query_coord, dim=self.dim)

        short_cut = query
        query = self.norm1(query)

        if self.add_flow_token:
            q = self.q(query+query_coord_enc)
        else:
            q = self.q(query_coord_enc)
        k, v = key, value

        x = self.multi_head_attn(q, k, v)

        x = self.proj(torch.cat([x, short_cut],dim=2))
        x = short_cut + self.proj_drop(x)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x, k, v

class MemoryDecoderLayer(nn.Module):
    def __init__(self, dim, cfg):
        super(MemoryDecoderLayer, self).__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size # for converting coords into H2', W2' space

        query_token_dim, tgt_token_dim = cfg.query_latent_dim, cfg.cost_latent_dim
        qk_dim, v_dim = query_token_dim, query_token_dim
        self.cross_attend = CrossAttentionLayer(qk_dim, v_dim, query_token_dim, tgt_token_dim, add_flow_token=cfg.add_flow_token, dropout=cfg.dropout)

    def forward(self, query, key, value, memory, coords1, size, size_h3w3):
        """
            x:      [B*H1*W1, 1, C]
            memory: [B*H1*W1, H2'*W2', C]
            coords1 [B, 2, H2, W2]
            size: B, C, H1, W1
            1. Note that here coords0 and coords1 are in H2, W2 space.
               Should first convert it into H2', W2' space.
            2. We assume the upper-left point to be [0, 0], instead of letting center of upper-left patch to be [0, 0]
        """
        x_global, k, v = self.cross_attend(query, key, value, memory, coords1, self.patch_size, size_h3w3)
        B, C, H1, W1 = size
        C = self.cfg.query_latent_dim
        x_global = x_global.view(B, H1, W1, C).permute(0, 3, 1, 2)
        return x_global, k, v

class ReverseCostExtractor(nn.Module):
    def __init__(self, cfg):
        super(ReverseCostExtractor, self).__init__()
        self.cfg = cfg

    def forward(self, cost_maps, coords0, coords1):
        """
            cost_maps   -   B*H1*W1, cost_heads_num, H2, W2
            coords      -   B, 2, H1, W1
        """
        BH1W1, heads, H2, W2 = cost_maps.shape
        B, _, H1, W1 = coords1.shape

        assert (H1 == H2) and (W1 == W2)
        assert BH1W1 == B*H1*W1

        cost_maps = cost_maps.reshape(B, H1* W1*heads, H2, W2)
        coords = coords1.permute(0, 2, 3, 1)
        corr = bilinear_sampler(cost_maps, coords) # [B, H1*W1*heads, H2, W2]
        corr = rearrange(corr, 'b (h1 w1 heads) h2 w2 -> (b h2 w2) heads h1 w1', b=B, heads=heads, h1=H1, w1=W1, h2=H2, w2=W2)
        
        r = 4
        dx = torch.linspace(-r, r, 2*r+1)
        dy = torch.linspace(-r, r, 2*r+1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords0.device)
        centroid = coords0.permute(0, 2, 3, 1).reshape(BH1W1, 1, 1, 2)
        delta = delta.view(1, 2*r+1, 2*r+1, 2)
        coords = centroid + delta
        corr = bilinear_sampler(corr, coords)
        corr = corr.view(B, H1, W1, -1).permute(0, 3, 1, 2)
        return corr

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

class IterMask(nn.Module):
    def __init__(self, input_dim=512):
        super(IterMask, self).__init__()
        self.iter_mask_0 = nn.Sequential(
            nn.Conv2d(input_dim + input_dim // 8, input_dim // 8, 3, 1, 1),
        )
        self.iter_mask_1 = nn.Sequential(
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim // 8, 3, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, tgt_sparsity):
        x = self.iter_mask_0(x)
        mhidden = x * tgt_sparsity
        x = self.iter_mask_1(mhidden)
        inc = x[:,:1]
        x = x[:,1:]
        return x, mhidden, inc


class MemoryDecoder(nn.Module):

    def __init__(self, cfg):
        super(MemoryDecoder, self).__init__()
        dim = self.dim = cfg.query_latent_dim
        self.cfg = cfg

        self.flow_token_encoder = nn.Sequential(
            nn.Conv2d(81*cfg.cost_heads_num, dim, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1)
        )
        self.proj = nn.Conv2d(256, 256, 1)
        self.depth = cfg.decoder_depth
        self.decoder_layer = MemoryDecoderLayer(dim, cfg)
        
        if self.cfg.gma:
            self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
            self.att = Attention(args=self.cfg, dim=128, heads=1, max_pos_size=160, dim_head=128)
        else:
            self.update_block = BasicUpdateBlock(self.cfg, hidden_dim=128)

        N_freqs = 3
        freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        funcs = [torch.sin, torch.cos]
        itr_embedding_l = []
        for i in range(8):
            x = torch.ones(1) * i
            out = []
            out += [x]
            for freq in freq_bands:
                for func in funcs:
                    out += [func(freq * x)]
            out = torch.cat(out, -1).view(-1, 1, 1)
            itr_embedding_l.append(out)
        itr_embedding = torch.stack(itr_embedding_l, dim=0)
        # iter_embedding = torch.unsqueeze(iter_embedding, dim=1)
        self.itr_embedding = nn.Parameter(itr_embedding, requires_grad=False)
        self.iter_mask = IterMask(input_dim=128+4+7)
        
    def upsample_flow(self, flow, mask, tgt_sparsity =None, mhidden=None, tau=0.01):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def encode_flow_token(self, cost_maps, coords):
        """
            cost_maps   -   B*H1*W1, cost_heads_num, H2, W2
            coords      -   B, 2, H1, W1
        """
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        r = 4
        dx = torch.linspace(-r, r, 2*r+1)
        dy = torch.linspace(-r, r, 2*r+1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid = coords.reshape(batch*h1*w1, 1, 1, 2)
        delta = delta.view(1, 2*r+1, 2*r+1, 2)
        coords = centroid + delta
        corr = bilinear_sampler(cost_maps, coords)
        corr = corr.view(batch, h1, w1, -1).permute(0, 3, 1, 2)
        return corr

    def forward(self, cost_memory, context, data={}, flow_init=None, tgt_sparsity =None, mhidden=None, tau=0.01):
        """
            memory: [B*H1*W1, H2'*W2', C]
            context: [B, D, H1, W1]
        """
        cost_maps = data['cost_maps']
        coords0, coords1 = initialize_flow(context)

        if flow_init is not None:
            #print("[Using warm start]")
            coords1 = coords1 + flow_init

        #flow = coords1

        flow_predictions = []

        # from thop import profile
        # print('self.proj')
        # flops, params = profile(self.proj, inputs=(context,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        # exit(0)
        context = self.proj(context)
        net, inp = torch.split(context, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        if self.cfg.gma:
            # print('self.att')
            # flops, params = profile(self.att,
            #                         inputs=(inp,))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 2) + 'M')
            # print('*' * 60)
            # exit(0)
            # print('here')
            attention = self.att(inp)

        size = net.shape
        key, value = None, None

        flow_predictions_wo_skip = []
        inc_l = []
        iter_mask = None
        sparsity_l = []
        iters_base = self.depth / 8
        for idx in range(self.depth):
            # print(idx)
            coords1 = coords1.detach()

            cost_forward = self.encode_flow_token(cost_maps, coords1)

            # print('self.flow_token_encoder')
            # flops, params = profile(self.flow_token_encoder, inputs=(cost_forward,))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 2) + 'M')
            # print('*' * 60)
            # exit(0)
            query = self.flow_token_encoder(cost_forward)
            query = query.permute(0, 2, 3, 1).contiguous().view(size[0]*size[2]*size[3], 1, self.dim)


            # print('self.decoder_layer')
            # flops, params = profile(self.decoder_layer, inputs=(query, key, value, cost_memory, coords1, size, data['H3W3'],))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 2) + 'M')
            # print('*' * 60)
            # exit(0)
            cost_global, key, value = self.decoder_layer(query, key, value, cost_memory, coords1, size, data['H3W3'])

            if self.cfg.only_global:
                corr = cost_global
            else:
                corr = torch.cat([cost_global, cost_forward], dim=1)

            flow = coords1 - coords0

            bs, _, h, w = net.shape
            net_past = net.clone()
            key_past = key.clone().view(bs, -1, 8, 64)
            value_past = value.clone().view(bs, -1, 8, 64)

            if self.cfg.gma:
                # print('self.update_block')
                # flops, params = profile(self.update_block,
                #                         inputs=(net, inp, corr, flow, attention,))
                # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
                # print('Params = ' + str(params / 1000 ** 2) + 'M')
                # print('*' * 60)
                # exit(0)
                net, up_mask_wo_skip, delta_flow_wo_skip = self.update_block(net, inp, corr, flow, attention)
            else:
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # if self.training:
            if iter_mask is not None:
                # print('here')
                # print('net', net.shape)
                # print('key', key.shape)
                # print('value', value.shape)
                # print('delta_flow', delta_flow.shape)
                # print('iter_mask', iter_mask.shape)
                net = net_past * (1 - iter_mask) + net * iter_mask

                key = key_past * (1 - iter_mask) + key.view(bs, -1, 8, 64) * iter_mask
                value = value_past * (1 - iter_mask) + value.view(bs, -1, 8, 64) * iter_mask
                key = key.view(-1, 8, 64)
                value = value.view(-1, 8, 64)
                delta_flow = delta_flow_wo_skip * iter_mask
            else:
                delta_flow = delta_flow_wo_skip.clone()
            # else:
            #     delta_flow = delta_flow_wo_skip.clone()

            itr_embedding = self.itr_embedding[ int(idx // iters_base):int(idx // iters_base)+1].repeat(bs, 1, h, w)
            # print('net', net.shape)
            # print('flow', flow.shape)
            # print('delta_flow', delta_flow.shape)
            # print('mhidden', mhidden.shape)
            # print('itr_embedding', itr_embedding.shape)
            # print('self.iter_mask')
            # flops, params = profile(self.iter_mask,
            #                         inputs=(torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1), tgt_sparsity,))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 2) + 'M')
            # print('*' * 60)
            iter_mask_next, mhidden, inc = self.iter_mask(
                torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
                tgt_sparsity)
            up_mask = self.update_block.forward_mask(net)
            if iter_mask is not None:
                if self.training == True:
                    iter_mask_next = gumbel_softmax(iter_mask_next, 1, tau)[:, :1, ...]
                else:
                    iter_mask_next = (iter_mask_next[:, :1, ...] > iter_mask_next[:, 1:, ...]).float()
            else:
                if self.training == True:
                    iter_mask_next = gumbel_softmax(iter_mask_next, 1, tau)[:, :1, ...]
                else:
                    iter_mask_next = (iter_mask_next[:, :1, ...] > iter_mask_next[:, 1:, ...]).float()

            # flow = delta_flow
            coords1_wo_skip = coords1.clone() + delta_flow_wo_skip
            coords1 = coords1 + delta_flow
            if iter_mask is not None:
                sparsity_l.append(iter_mask)

            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up_wo_skip = self.upsample_flow(coords1_wo_skip - coords0, up_mask_wo_skip)
            flow_predictions.append(flow_up)
            flow_predictions_wo_skip.append(flow_up_wo_skip)
            inc_l.append(inc)
            iter_mask = iter_mask_next
        # exit(0)

        sparsity = torch.stack(sparsity_l, dim=3)

        if self.training:
            return flow_predictions, sparsity, flow_predictions_wo_skip, inc_l
        else:
            return flow_predictions[-1], coords1-coords0, sparsity


    def forward_time(self, cost_memory, context, data={}, flow_init=None, tgt_sparsity =None, mhidden=None, tau=0.01):
        """
            memory: [B*H1*W1, H2'*W2', C]
            context: [B, D, H1, W1]
        """
        cost_maps = data['cost_maps']
        coords0, coords1 = initialize_flow(context)

        if flow_init is not None:
            #print("[Using warm start]")
            coords1 = coords1 + flow_init

        #flow = coords1


        context = self.proj(context)
        net, inp = torch.split(context, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        if self.cfg.gma:
            attention = self.att(inp)

        size = net.shape
        key, value = None, None

        iter_mask = None
        iters_base = self.depth / 8
        for idx in range(self.depth):
            # print(idx)
            if iter_mask is None or iter_mask.view(1).item() > 0.5:
                coords1 = coords1.detach()

                cost_forward = self.encode_flow_token(cost_maps, coords1)
                query = self.flow_token_encoder(cost_forward)
                query = query.permute(0, 2, 3, 1).contiguous().view(size[0]*size[2]*size[3], 1, self.dim)

                cost_global, key, value = self.decoder_layer(query, key, value, cost_memory, coords1, size, data['H3W3'])

                if self.cfg.only_global:
                    corr = cost_global
                else:
                    corr = torch.cat([cost_global, cost_forward], dim=1)

                flow = coords1 - coords0

                net, delta_flow = self.update_block.forward_time(net, inp, corr, flow, attention)

                bs, _, h, w = net.shape
                itr_embedding = self.itr_embedding[ int(idx // iters_base):int(idx // iters_base)+1].repeat(bs, 1, h, w)

                iter_mask_next, mhidden, inc = self.iter_mask(
                    torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
                    tgt_sparsity)
            else:
                flow = coords1 - coords0
                bs, _, h, w = net.shape
                itr_embedding = self.itr_embedding[int(idx // iters_base):int(idx // iters_base) + 1].repeat(bs, 1, h,
                                                                                                             w)
                delta_flow = flow * 0.0
                iter_mask_next, mhidden, inc = self.iter_mask(
                    torch.cat([net, flow, delta_flow, mhidden, itr_embedding], dim=1),
                    tgt_sparsity)

            iter_mask_next = (iter_mask_next[:, :1, ...] > iter_mask_next[:, 1:, ...]).float()
            coords1 = coords1 + delta_flow
            iter_mask = iter_mask_next

        up_mask = self.update_block.forward_mask(net)
        flow_up = self.upsample_flow(coords1 - coords0, up_mask)
        # print('final', time.time() - time_start)
        return flow_up