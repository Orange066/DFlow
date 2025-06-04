import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

from module import KPAEnc, KPAFlowDec, KPAEnc


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

class KPAFlow(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('----- Model: KPA-Flow -----')

        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = KPAFlowDec(self.args, chnn=hdim)

        self.sc = 13
        self.trans = KPAEnc(args, 256, self.sc)
        self.zero = nn.Parameter(torch.zeros(12), requires_grad=False)

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

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [N, 2, H, 8, W, 8]
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, image1, image2, tgt_sparsity =None, mhidden=None, iters=12, flow_init=None, upsample=True, test_mode=False, tau=0.01):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            # from thop import profile
            # print('self.fnet')
            # flops, params = profile(self.fnet, inputs=([image1, image2],))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 2) + 'M')
            # print('*' * 60)
            # exit(0)
            fmap1, fmap2 = self.fnet([image1, image2])

        # print('self.trans(fmap1)')
        # flops, params = profile(self.trans, inputs=(fmap1,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        # print('self.trans(fmap2)')
        # flops, params = profile(self.trans, inputs=(fmap2,))
        # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')
        # print('*' * 60)
        # exit(0)
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        fmap1 = self.trans(fmap1)
        fmap2 = self.trans(fmap2)

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            # print('self.cnet')
            # flops, params = profile(self.cnet, inputs=(image1,))
            # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
            # print('Params = ' + str(params / 1000 ** 2) + 'M')
            # print('*' * 60)
            # exit(0)
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        flow_predictions_wo_skip = []
        inc_l = []
        iter_mask = None
        sparsity_l = []
        iters_base = iters / 8
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                # print('self.update')
                # flops, params = profile(self.update_block, inputs=(net, inp, corr, flow, itr,))
                # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
                # print('Params = ' + str(params / 1000 ** 2) + 'M')
                # print('*' * 60)
                # exit(0)
                net, up_mask, delta_flow, iter_mask_next, mhidden, inc, up_mask_wo_skip, delta_flow_wo_skip = self.update_block(net, inp, corr, flow, itr, iter_mask,
                                                                                         tgt_sparsity=tgt_sparsity,
                                                                                         itr_embedding= self.itr_embedding[ int(itr // iters_base):int(itr // iters_base)+1],
                                                                                         mhidden=mhidden,)


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

            # F(t+1) = F(t) + \Delta(t)
            coords1_wo_skip = coords1.clone() + delta_flow_wo_skip
            coords1 = coords1 + delta_flow
            if iter_mask is not None:
                sparsity_l.append(iter_mask)

            flow = coords1 - coords0

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                flow_up_wo_skip =self.upsample_flow(coords1_wo_skip - coords0, up_mask_wo_skip)

            flow_predictions.append(flow_up)
            flow_predictions_wo_skip.append(flow_up_wo_skip)
            inc_l.append(inc)
            iter_mask = iter_mask_next


        sparsity = torch.stack(sparsity_l, dim=3)

        if test_mode:
            return flow, flow_up, sparsity

        return flow_predictions, self.zero, sparsity, flow_predictions_wo_skip, inc_l

    def forward_time(self, image1, image2, tgt_sparsity =None, mhidden=None, iters=12, flow_init=None, upsample=True, test_mode=False, tau=0.01):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):

            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        fmap1 = self.trans(fmap1)
        fmap2 = self.trans(fmap2)

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        iter_mask = None
        iters_base = iters / 8
        for itr in range(iters):

            if iter_mask is None or iter_mask.view(1).item() > 0.5:
            # count = count + 1
                coords1 = coords1.detach()
                corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                net, delta_flow, iter_mask_next, mhidden = self.update_block.forward_time(net, inp, corr, flow, itr, iter_mask,
                                                                                         tgt_sparsity=tgt_sparsity,
                                                                                         itr_embedding= self.itr_embedding[ int(itr // iters_base):int(itr // iters_base)+1],
                                                                                         mhidden=mhidden,)


            iter_mask_next = (iter_mask_next[:, :1, ...] > iter_mask_next[:, 1:, ...]).float()
            coords1 = coords1 + delta_flow
            iter_mask = iter_mask_next


        up_mask = self.update_block.forward_time_mask(net)
        flow_up = self.upsample_flow(coords1 - coords0, up_mask)
        # print('final', time.time() - time_start)
        return flow_up

    def forward_time_flow_low(self, image1, image2, tgt_sparsity =None, mhidden=None, iters=12, flow_init=None, upsample=True, test_mode=False, tau=0.01):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):

            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        fmap1 = self.trans(fmap1)
        fmap2 = self.trans(fmap2)

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        iter_mask = None
        iters_base = iters / 8
        for itr in range(iters):

            if iter_mask is None or iter_mask.view(1).item() > 0.5:
            # count = count + 1
                coords1 = coords1.detach()
                corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                net, delta_flow, iter_mask_next, mhidden = self.update_block.forward_time(net, inp, corr, flow, itr, iter_mask,
                                                                                         tgt_sparsity=tgt_sparsity,
                                                                                         itr_embedding= self.itr_embedding[ int(itr // iters_base):int(itr // iters_base)+1],
                                                                                         mhidden=mhidden,)


            iter_mask_next = (iter_mask_next[:, :1, ...] > iter_mask_next[:, 1:, ...]).float()
            coords1 = coords1 + delta_flow
            iter_mask = iter_mask_next

        up_mask = self.update_block.forward_time_mask(net)
        flow_up = self.upsample_flow(coords1 - coords0, up_mask)
        # print('final', time.time() - time_start)
        return coords1 - coords0
