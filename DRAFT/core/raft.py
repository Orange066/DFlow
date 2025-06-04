
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
import time

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

def resize(x, scale_factor, mode="bilinear"):
    if mode == 'nearest':
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    else:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

        self.max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

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
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def eliminate_noise(self, mask):
        tensor_dialate = self.max_pool(mask)
        # tensor_erode = -self.max_pool(-tensor_dialate)
        # return tensor_dilate
        return tensor_dialate

    def get_spatial_parameters(self):
        spatial_parameters=self.update_block.get_spatial_mask_parameters()
        return spatial_parameters

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
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
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

                net, up_mask, delta_flow, iter_mask_next, mhidden, inc, up_mask_wo_skip, delta_flow_wo_skip = self.update_block(net, inp, corr, flow, iter_mask,
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

            # iter_mask_next = torch.ones_like(iter_mask_next)

            # if self.training == True:
            #     spatial_mask_next = gumbel_softmax(spatial_mask_next, 1, tau)[:, :1, ...]
            # else:
            #     spatial_mask_next = (spatial_mask_next[:, :1, ...] > spatial_mask_next[:, 1:, ...]).float()
            # spatial_mask_next = self.eliminate_noise(spatial_mask_next)
            # F(t+1) = F(t) + \Delta(t)
            coords1_wo_skip = coords1.clone() + delta_flow_wo_skip
            coords1 = coords1 + delta_flow
            if iter_mask is not None:
                # coords1 = coords1 * (1-spatial_mask) + (coords1 + delta_flow)*spatial_mask
                sparsity_l.append(iter_mask)
                # spatial_mask_up = resize(spatial_mask, 8)
                # # upsample predictions
                # if up_mask is None:
                #     flow_up = upflow8((coords1 - coords0)*spatial_mask_up)+ flow_up*(1-spatial_mask_up)
                # else:
                #     flow_up = self.upsample_flow((coords1 - coords0)*spatial_mask, up_mask)*spatial_mask_up + flow_up*(1-spatial_mask_up)
            # else:
            #     coords1 = coords1 + delta_flow

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
            return coords1 - coords0, flow_up, sparsity
            # return coords1 - coords0, flow_up, sparsity, inc_l, flow_predictions_wo_skip

        return flow_predictions, sparsity, flow_predictions_wo_skip, inc_l

    def forward_inc(self, image1, image2, tgt_sparsity =None, mhidden=None, iters=12, flow_init=None, upsample=True, test_mode=False, tau=0.01):
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
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
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
        iters_base = iters / 6
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):

                net, up_mask, delta_flow, iter_mask_next, mhidden, inc = self.update_block.forward_inc(net, inp, corr, flow, iter_mask,
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

            # if self.training == True:
            #     spatial_mask_next = gumbel_softmax(spatial_mask_next, 1, tau)[:, :1, ...]
            # else:
            #     spatial_mask_next = (spatial_mask_next[:, :1, ...] > spatial_mask_next[:, 1:, ...]).float()
            # spatial_mask_next = self.eliminate_noise(spatial_mask_next)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            if iter_mask is not None:
                # coords1 = coords1 * (1-spatial_mask) + (coords1 + delta_flow)*spatial_mask
                sparsity_l.append(iter_mask)
                # spatial_mask_up = resize(spatial_mask, 8)
                # # upsample predictions
                # if up_mask is None:
                #     flow_up = upflow8((coords1 - coords0)*spatial_mask_up)+ flow_up*(1-spatial_mask_up)
                # else:
                #     flow_up = self.upsample_flow((coords1 - coords0)*spatial_mask, up_mask)*spatial_mask_up + flow_up*(1-spatial_mask_up)
            # else:
            #     coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)
            inc_l.append(inc)
            iter_mask = iter_mask_next

        return flow_predictions, inc_l


    # def forward_time(self, image1, image2, tgt_sparsity =None, mhidden=None, iters=12, flow_init=None, upsample=True, test_mode=False, tau=0.01):
    #     """ Estimate optical flow between pair of frames """
    #
    #     image1 = 2 * (image1 / 255.0) - 1.0
    #     image2 = 2 * (image2 / 255.0) - 1.0
    #
    #     image1 = image1.contiguous()
    #     image2 = image2.contiguous()
    #
    #     hdim = self.hidden_dim
    #     cdim = self.context_dim
    #
    #     # run the feature network
    #     with autocast(enabled=self.args.mixed_precision):
    #         fmap1, fmap2 = self.fnet([image1, image2])
    #
    #     fmap1 = fmap1.float()
    #     fmap2 = fmap2.float()
    #     if self.args.alternate_corr:
    #         corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
    #     else:
    #         corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
    #
    #     # run the context network
    #     with autocast(enabled=self.args.mixed_precision):
    #         cnet = self.cnet(image1)
    #         net, inp = torch.split(cnet, [hdim, cdim], dim=1)
    #         net = torch.tanh(net)
    #         inp = torch.relu(inp)
    #
    #     coords0, coords1 = self.initialize_flow(image1)
    #
    #     if flow_init is not None:
    #         coords1 = coords1 + flow_init
    #
    #     iter_mask = None
    #     iters_base = iters / 8
    #     flow = coords1 - coords0
    #     itr = 0
    #     while itr < iters:
    #         # print('itr', itr)
    #         coords1 = coords1.detach()
    #         corr = corr_fn(coords1) # index correlation volume
    #
    #         with autocast(enabled=self.args.mixed_precision):
    #             net, delta_flow, iter_mask_next, mhidden = self.update_block.forward_time(net, inp, corr, flow, iter_mask,
    #                                                                                      tgt_sparsity=tgt_sparsity,
    #                                                                                      itr_embedding= self.itr_embedding[int(itr // iters_base):int(itr // iters_base)+1],
    #                                                                                      mhidden=mhidden,)
    #
    #         coords1 = coords1 + delta_flow
    #         flow = coords1 - coords0
    #         iter_mask = iter_mask_next
    #         if iter_mask.view(1).item() < 0.5:
    #             # print('hrer')
    #             itr_clone = itr
    #             for itr_next in range(itr+1, iters):
    #                 # print(itr_next)
    #                 iter_mask, mhidden = self.update_block.forward_time_iter_mask(net, flow, mhidden, self.itr_embedding[int(itr_next // iters_base):int(itr_next // iters_base)+1], tgt_sparsity)
    #                 if iter_mask.view(1).item() > 0.5:
    #                     itr = itr_next
    #                     break
    #                 if itr_next == iters-1:
    #                     itr = itr_next-1
    #                     break
    #         itr = itr + 1
    #
    #         # print(itr, time.time()-time_start)
    #
    #     up_mask = self.update_block.forward_time_mask(net)
    #     flow_up = self.upsample_flow(coords1 - coords0, up_mask)
    #
    #     return flow_up
    #

    def forward_time(self, image1, image2,itr_embedding =None, tgt_sparsity =None, mhidden=None, iters=12, flow_init=None, upsample=True, test_mode=False, tau=0.01):
        """ Estimate optical flow between pair of frames """
        # time_start = time.time()
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
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

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
        # count = -1
        # print('first', time.time()-time_start)
        # skip_num = -1
        # skip_num_count = 0
        for itr in range(iters):

            # if skip_num == int(itr // iters_base) and skip_num_count == 2 and iter_mask.view(1).item() < 0.5:
            #     continue
            # if itr!=0:
            #     print(itr, iter_mask.view(1).item(), int(itr // iters_base))
            # time_start = time.time()
            if iter_mask is None or iter_mask.view(1).item() > 0.5:
            # count = count + 1
                coords1 = coords1.detach()
                corr = corr_fn(coords1) # index correlation volume

                flow = coords1 - coords0
            # print('corr', time.time() - time_start)
            # time_start = time.time()
            with autocast(enabled=self.args.mixed_precision):
                # print('i', itr_embedding.shape)
                net, delta_flow, iter_mask_next, mhidden = self.update_block.forward_time(net, inp, corr, flow, iter_mask,
                                                                                         tgt_sparsity=tgt_sparsity,
                                                                                         itr_embedding= self.itr_embedding[int(itr // iters_base):int(itr // iters_base)+1],
                                                                                         mhidden=mhidden,)
            # time_start = time.time()
            iter_mask_next = (iter_mask_next[:, :1, ...] > iter_mask_next[:, 1:, ...]).float()
            # if iter_mask_next.view(1).item() < 0.5:
            #     if skip_num == int(itr // iters_base):
            #         skip_num_count = skip_num_count + 1
            #     else:
            #         skip_num = int(itr // iters_base)
            #         skip_num_count = 0
            # else:
            #     skip_num = -1
            #     skip_num_count = 0
            coords1 = coords1 + delta_flow
            iter_mask = iter_mask_next
            # print(time.time()-time_start)
            # print('flow', time.time() - time_start)

        # time_start = time.time()
        up_mask = self.update_block.forward_time_mask(net)
        flow_up = self.upsample_flow(coords1 - coords0, up_mask)
        # print('final', time.time() - time_start)
        return flow_up

    # def forward_time(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
    #     """ Estimate optical flow between pair of frames """
    #     # print('*' * 60)
    #     time_start = time.time()
    #     image1 = 2 * (image1 / 255.0) - 1.0
    #     image2 = 2 * (image2 / 255.0) - 1.0
    #
    #     image1 = image1.contiguous()
    #     image2 = image2.contiguous()
    #
    #     hdim = self.hidden_dim
    #     cdim = self.context_dim
    #
    #     with autocast(enabled=self.args.mixed_precision):
    #         fmap1, fmap2 = self.fnet([image1, image2])
    #
    #     fmap1 = fmap1.float()
    #     fmap2 = fmap2.float()
    #     if self.args.alternate_corr:
    #         corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
    #     else:
    #         corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
    #
    #     with autocast(enabled=self.args.mixed_precision):
    #         cnet = self.cnet(image1)
    #         net, inp = torch.split(cnet, [hdim, cdim], dim=1)
    #         net = torch.tanh(net)
    #         inp = torch.relu(inp)
    #
    #     coords0, coords1 = self.initialize_flow(image1)
    #
    #     if flow_init is not None:
    #         coords1 = coords1 + flow_init
    #
    #     print('first', time.time()-time_start)
    #     for itr in range(iters):
    #         time_start = time.time()
    #         coords1 = coords1.detach()
    #         corr = corr_fn(coords1)  # index correlation volume
    #
    #         flow = coords1 - coords0
    #         print('corr', time.time() - time_start)
    #         time_start = time.time()
    #         with autocast(enabled=self.args.mixed_precision):
    #
    #             net, delta_flow = self.update_block.forward_time(net, inp, corr, flow)
    #
    #         coords1 = coords1 + delta_flow
    #         print('flow', time.time() - time_start)
    #     time_start = time.time()
    #     up_mask = self.update_block.forward_time_mask(net)
    #     flow_up = self.upsample_flow(coords1 - coords0, up_mask)
    #     print('final', time.time() - time_start)
    #
    #     return flow_up


