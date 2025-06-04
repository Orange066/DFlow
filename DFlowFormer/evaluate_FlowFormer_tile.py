import sys

# from attr import validate
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submission import get_cfg as get_submission_cfg
# from configs.kitti_submission import get_cfg as get_kitti_cfg
from configs.things_eval import get_cfg as get_things_cfg
from configs.small_things_eval import get_cfg as get_small_things_cfg
from core.utils.misc import process_cfg
import core.datasets
from utils import flow_viz
from utils import frame_utils

from core.FlowFormer import build_flowformer
from raft import RAFT

from utils.utils import InputPadder, forward_interpolate
import imageio
import itertools
import cv2

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


TRAIN_SIZE = [432, 960]

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti432':
            self._pad = [0, 0, 0, 432 - self.ht]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        elif mode == 'kitti376':
            self._pad = [0, 0, 0, 376 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]

import math
def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights

@torch.no_grad()
def create_sintel_submission(model, output_path='sintel_submission_multi8_768', sigma=0.05):
    """ Create submission for the Sintel leaderboard """
    print("no warm start")
    #print(f"output path: {output_path}")
    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    # tgt_sparsity_l = [0.8,0.8] # 63_0
    # tgt_sparsity_l = [0.98, 0.98]  # 79_0

    tgt_sparsity_l = [0.885, 0.8]  # 63+1
    for ii, dstype in enumerate(['final', "clean"]):
        test_dataset = core.datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        epe_list = []
        sparsity_list = []
        tgt_sparsity_tmp = tgt_sparsity_l[ii]
        for test_id in range(len(test_dataset)):
            if (test_id+1) % 100 == 0:
                print(f"{test_id} / {len(test_dataset)}")
                # break
            image1, image2, (sequence, frame) = test_dataset[test_id]
            image1, image2 = image1[None].cuda(), image2[None].cuda()

            flows = 0
            flow_count = 0

            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                bs, _, hh, ww = image1_tile.shape
                mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, hh // 8, ww // 8).cuda()
                tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)


                flow_pre, flow_low, sparsity = model(image1_tile, image2_tile, tgt_sparsity=tgt_sparsity, mhidden=mhidden)

                padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)

                if idx == 0:
                    sparsity_total = sparsity
                else:
                    sparsity_total = sparsity_total + sparsity

            sparsity_total = sparsity_total / len(hws)
            flow_pre = flows / flow_count
            flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
            sparsity_list.append(sparsity_total.view(1, -1).cpu().numpy())

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)

        sparsity = np.mean(np.concatenate(sparsity_list, axis=1))
        flop = 20.818860288 + 284.954290752 + (0.21233664 + 0.06096384 + 0.98205696 + 20.3461632) + (
                    0.21233664 + 0.06096384 + 0.1327104 + 20.3461632) * 31 * sparsity + 2.871936 + 31 * 0.154885031

        print("Sintel %f, flop %f" % (sparsity, flop * 4))

@torch.no_grad()
def create_sintel_submission_time(model, output_path='sintel_submission_multi8_768', sigma=0.05):
    """ Create submission for the Sintel leaderboard """
    print("no warm start")
    #print(f"output path: {output_path}")
    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    # tgt_sparsity_l = [0.8,0.8]
    tgt_sparsity_l = [0.885, 0.8]  # 63+1
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    time_l = []
    for ii, dstype in enumerate(['final', "clean"]):
        test_dataset = core.datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        tgt_sparsity_tmp = tgt_sparsity_l[ii]
        warm_skip=False
        for test_id in range(len(test_dataset)):
            if (test_id+1) % 100 == 0:
                print(f"{test_id} / {len(test_dataset)}")
                # break
            image1, image2, (sequence, frame) = test_dataset[test_id]
            image1, image2 = image1[None].cuda(), image2[None].cuda()

            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                bs, _, hh, ww = image1_tile.shape
                mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, hh // 8, ww // 8).cuda()
                tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
                if warm_skip == True:
                    warm_id = 0
                    while (warm_id < 300):
                        _ = model.forward_time(image1_tile, image2_tile, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
                        warm_id = warm_id + 1
                    warm_skip = False
                starter.record()
                _ = model.forward_time(image1_tile, image2_tile, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
                ender.record()
                torch.cuda.synchronize()
                # print(starter.elapsed_time(ender))
                time_l.append(starter.elapsed_time(ender))

        print(dstype, 'time:', np.mean(time_l) * 4)

@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission', sigma=0.05):
    """ Create submission for the Sintel leaderboard """

    IMAGE_SIZE = [432, 1242]

    print(f"output path: {output_path}")
    print(f"image size: {IMAGE_SIZE}")
    print(f"training size: {TRAIN_SIZE}")

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, (432, 1242), TRAIN_SIZE, sigma)
    model.eval()
    test_dataset = core.datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tgt_sparsity_tmp = 1.025
    # tgt_sparsity_tmp = 1.062
    sparsity_list =[]
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:   # fix the height=432, adaptive ajust the width
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 432
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        padder = InputPadder(image1.shape, mode='kitti432') # padding the image to height of 432
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            bs, _, hh, ww = image1_tile.shape
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, hh // 8, ww // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)


            flow_pre, flow_low, sparsity = model(image1_tile, image2_tile, tgt_sparsity=tgt_sparsity, mhidden=mhidden)

            padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

            if idx == 0:
                sparsity_total = sparsity
            else:
                sparsity_total = sparsity_total + sparsity


        sparsity_total = sparsity_total / len(hws)
        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).permute(1, 2, 0).cpu().numpy()
        sparsity_list.append(sparsity_total.view(1, -1).cpu().numpy())

        # output_filename = os.path.join(output_path, frame_id)
        # frame_utils.writeFlowKITTI(output_filename, flow)
        #
        # flow_img = flow_viz.flow_to_image(flow)
        # image = Image.fromarray(flow_img)
        # if not os.path.exists(f'vis_kitti_3patch'):
        #     os.makedirs(f'vis_kitti_3patch/flow')
        #     os.makedirs(f'vis_kitti_3patch/image')
        #
        # image.save(f'vis_kitti_3patch/flow/{test_id}.png')
        # imageio.imwrite(f'vis_kitti_3patch/image/{test_id}_0.png', image1[0].cpu().permute(1, 2, 0).numpy())
        # imageio.imwrite(f'vis_kitti_3patch/image/{test_id}_1.png', image2[0].cpu().permute(1, 2, 0).numpy())

    sparsity = np.mean(np.concatenate(sparsity_list, axis=1))
    flop = 20.818860288 + 284.954250752 + (0.21233664 + 0.06096384 + 0.98205696 + 20.3461632) + (
            0.21233664 + 0.06096384 + 0.1327104 + 20.3461632) * 23 * sparsity + 2.871936 + 23 * 0.154885031

    print("KITTI %f, flop %f" % (sparsity, flop * 4))

@torch.no_grad()
def create_kitti_submission_time(model, output_path='kitti_submission', sigma=0.05):
    """ Create submission for the Sintel leaderboard """
    # iter 0.154885031
    # 20.818860288 + 284.954250752 + (0.21233664 + 0.06096384 + 0.98205696 + 20.3461632) + (0.21233664 + 0.06096384 + 0.1327104 + 20.3461632) * 23 + 2.871936
    IMAGE_SIZE = [432, 1242]

    print(f"output path: {output_path}")
    print(f"image size: {IMAGE_SIZE}")
    print(f"training size: {TRAIN_SIZE}")

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, (432, 1242), TRAIN_SIZE, sigma)
    model.eval()
    test_dataset = core.datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # tgt_sparsity_tmp = 1.025
    tgt_sparsity_tmp = 1.062
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    time_l = []
    warm_skip=False
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:   # fix the height=432, adaptive ajust the width
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 432
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE)

        padder = InputPadder(image1.shape, mode='kitti432') # padding the image to height of 432
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        # print('len(hws)', len(hws))
        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            bs, _, hh, ww = image1_tile.shape
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, hh // 8, ww // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
            if warm_skip == True:
                warm_id = 0
                while (warm_id < 300):
                    _ = model.forward_time(image1_tile, image2_tile, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
                    warm_id = warm_id + 1
                warm_skip = False
            starter.record()
            _ = model.forward_time(image1_tile, image2_tile, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
            ender.record()
            torch.cuda.synchronize()
            # print(starter.elapsed_time(ender))
            time_l.append(starter.elapsed_time(ender))

    print('time:', np.mean(time_l)*4)

@torch.no_grad()
def validate_kitti(model, sigma=0.05, step=-1):
    print(cfg.latentcostformer.decoder_depth)
    # 15.83712
    # 15.28268544
    # 367.33888512
    # 0.4169088
    # iter_mask 0.101105531
    # mask 1.874736
    #  13.528985856 + 154.5897344 + 367.75579392(0.13860864 + 0.03979584 + 0.64106496(0.0866304) + 15.1562592)
    # origin: 13.528985856 + 154.5897344 + (0.13860864 + 0.03979584 + 0.64106496 + 13.2815232) +  (0.13860864 + 0.03979584 + 0.0866304 + 13.2815232) * 23 + 1.874736

    flops_l = []
    for i in range(cfg.latentcostformer.decoder_depth + 2):
        if i == 1:
            flops = 13.528985856 + 154.5897344 + (0.13860864 + 0.03979584 + 0.64106496 + 13.2815232) + 1.874736
        elif i == 0:
            flops = 13.528985856 + 154.5897344 + 1.874736
        else:
            flops = 13.528985856 + 154.5897344 + (0.13860864 + 0.03979584 + 0.64106496 + 13.2815232) + ( 0.13860864 + 0.03979584 + 0.0866304 + 13.2815232) * (i-1) + 1.874736
        flops_l.append(flops)

    IMAGE_SIZE = [376, 1242]
    TRAIN_SIZE = [376, 720]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    val_dataset = core.datasets.KITTI(split='training')

    sparsity_list = []
    out_list, epe_list = [], []
    # tgt_sparsity_tmp = 0.638 # training high
    # tgt_sparsity_tmp = 0.638  # training high
    # tgt_sparsity_tmp = 0.452  # training high
    # tgt_sparsity_tmp = 0.75  # training high
    # tgt_sparsity_tmp = 0.74  # training high
    tgt_sparsity_tmp = 0.905  # training low
    # tgt_sparsity_tmp = 0.55 # testing
    # tgt_sparsity_tmp = 0.3  # 200
    # tgt_sparsity_tmp = 0.72  # 330
    # tgt_sparsity_tmp = 0.58  # 290
    f = open('./flops_kitti_' + str(tgt_sparsity_tmp) + '.txt', 'w')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 376
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        padder = InputPadder(image1.shape, mode='kitti376')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flows = 0
        flow_count = 0
        # print(hws)
        sparsity_total = 0.0
        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]

            bs, _, hh, ww = image1_tile.shape
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, hh // 8, ww // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
            # flow_pre, _, sparsity = model(image1, image2, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
            flow_pre, flow_low, sparsity = model(image1_tile, image2_tile, tgt_sparsity=tgt_sparsity, mhidden=mhidden)

            padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

            if idx == 0:
                sparsity_total = sparsity
            else:
                sparsity_total = sparsity_total + sparsity

        sparsity_total = sparsity_total / len(hws)
        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()


        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        sparsity_list.append(sparsity_total.view(1, -1).cpu().numpy())


        sparsity_tmp = torch.mean(sparsity_total).cpu().numpy()
        flops_tmp = 13.528985856 + 154.5897344 + (0.13860864 + 0.03979584 + 0.64106496 + 13.2815232) + (
                0.13860864 + 0.03979584 + 0.0866304 + 13.2815232) * 23 * sparsity_tmp + 1.874736 + 23 * 0.101105531
        # print(val_id, flops_tmp)
        for flops_idx, flops in enumerate(flops_l):
            # print(flops_tmp)
            if flops > flops_tmp:
                f.write(str(flops_idx) + ' ' + str(flops_tmp) + '\n')
                # print(val_id, flops_idx, epe[val].mean().item())
                break

        if val_id < 10 and step is not None:
            save_path = './look/kitti/' + str(step).zfill(10) + '/'
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            image1 = (image1.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
            image2 = (image2.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_image1.png', image1[0])
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_image2.png', image2[0])
            overlaid = cv2.addWeighted(image1[0].astype('uint8'), 0.5, image2[0].astype('uint8'), 0.5, 1.0)
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_overlaid.png', overlaid)
            flow_gt = flow_gt.permute(1, 2, 0).detach().cpu().numpy()
            # print('flow_pre', flow_pre.shape)
            flow_pr = flow_pre[0].permute(1, 2, 0).detach().cpu().numpy()
            flow_gt = (flow2rgb(flow_gt) * 255).astype('uint8')
            flow_pr = (flow2rgb(flow_pr) * 255).astype('uint8')
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_flow_gt.png', flow_gt)
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_flow_pr.png', flow_pr)
            sparsity = sparsity_total.repeat(1, 1, 64, 1, 64)
            bs, c, h, iter, w = sparsity.shape
            iter_split = (iter + 1) // 4
            # print('sparsity', sparsity.shape)
            sparsity_0 = sparsity[:, :, :, :iter_split].clone().view(bs, c, h, iter_split * w)
            sparsity_1 = sparsity[:, :, :, iter_split:iter_split * 2].clone().view(bs, c, h, iter_split * w)
            sparsity_2 = sparsity[:, :, :, iter_split * 2:iter_split * 3].clone().view(bs, c, h, iter_split * w)
            # print('sparsity[:, :, :, iter_split*3:].clone()', sparsity[:, :, :, iter_split*3:].clone().shape)
            sparsity_3 = sparsity[:, :, :, iter_split * 3:].clone().view(bs, c, h, (iter_split - 1) * w)
            tmp = torch.ones_like(sparsity_3[:, :, :, :w])
            sparsity_3 = torch.cat([sparsity_3, tmp], dim=3)
            sparsity = torch.cat([sparsity_0, sparsity_1, sparsity_2, sparsity_3], dim=2)

            # sparsity = sparsity.view(bs, c, h, iter * w)
            sparsity = (sparsity.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_sparsity.png', sparsity[0])

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    sparsity = np.mean(np.concatenate(sparsity_list, axis=1))
    flop = 13.528985856 + 154.5897344 + (0.13860864 + 0.03979584 + 0.64106496 + 13.2815232) + (
                0.13860864 + 0.03979584 + 0.0866304 + 13.2815232) * 23 * sparsity + 1.874736 + 23 * 0.101105531

    print("Validation KITTI: %f(epe), %f(f1), %f(sparsity), flop %f, flop %f" % (epe, f1, sparsity, flop*4, flop))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_kitti_time(model, sigma=0.05, step=-1):
    print(cfg.latentcostformer.decoder_depth)
    # 15.83712
    # 15.28268544
    # 367.33888512
    # 0.4169088
    # iter_mask 0.101105531
    # mask 1.874736
    #  13.528985856 + 154.5897344 + 367.75579392(0.13860864 + 0.03979584 + 0.64106496(0.0866304) + 15.1562592)
    # origin: 13.528985856 + 154.5897344 + (0.13860864 + 0.03979584 + 0.64106496 + 13.2815232) +  (0.13860864 + 0.03979584 + 0.0866304 + 13.2815232) * 23 + 1.874736

    IMAGE_SIZE = [376, 1242]
    TRAIN_SIZE = [376, 720]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    val_dataset = core.datasets.KITTI(split='training')

    sparsity_list = []
    out_list, epe_list = [], []
    # tgt_sparsity_tmp = 0.638 # training high
    tgt_sparsity_tmp = 0.905  # training low
    # tgt_sparsity_tmp = 0.55 # testing
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    time_l = []
    warm_skip = False
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 376
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        padder = InputPadder(image1.shape, mode='kitti376')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flows = 0
        flow_count = 0
        # print(hws)
        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]


            bs, _, hh, ww = image1_tile.shape
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, hh // 8, ww // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
            # flow_pre, _, sparsity = model(image1, image2, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
            if warm_skip == True:
                warm_id = 0
                while (warm_id < 300):
                    _ = model.forward_time(image1_tile, image2_tile)
                    warm_id = warm_id + 1
                warm_skip =False

            starter.record()
            flow_pre = model.forward_time(image1_tile, image2_tile, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
            ender.record()
            torch.cuda.synchronize()
            # print(starter.elapsed_time(ender))
            time_l.append(starter.elapsed_time(ender))

            padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()


        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    time_avarage = np.mean(time_l)

    print("Validation KITTI: %f, %f, %f" % (epe, f1, time_avarage))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_sintel(model, sigma=0.05, step=-1):
    """ Peform validation using the Sintel (train) split """
    print(cfg.latentcostformer.decoder_depth)
    # exit(0)
    # update 23.2180992
    # mask 2.871936
    # iter_mask 0.154885031
    # origin: 20.818860288 + 284.954290752 + (0.21233664 + 0.06096384 + 0.98205696 + 20.3461632) + (0.21233664 + 0.06096384 + 0.1327104 + 20.3461632) * 31 + 2.871936

    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    results = {}
    flops_l = []
    for i in range(cfg.latentcostformer.decoder_depth + 2):
        if i == 1:
            flops = 20.818860288 + 284.954290752 + (0.21233664 + 0.06096384 + 0.98205696 + 20.3461632) + 2.871936
        else:
            flops = 20.818860288 + 284.954290752 + (0.21233664 + 0.06096384 + 0.98205696 + 20.3461632) + (0.21233664 + 0.06096384 + 0.1327104 + 20.3461632) * (i-1) + 2.871936
        flops_l.append(flops)

    # tgt_sparsity_l = [0.4, 0.4] # training high
    # tgt_sparsity_l = [0.65, 0.65]  # training high
    tgt_sparsity_l = [0.8, 0.92, ]  # training low
    # tgt_sparsity_l = [0.85, 0.85, ]  # testing
    # tgt_sparsity_l = [0.5, 0.5, ]  # testing
    # tgt_sparsity_l = [0.65, 0.65, ]  # testing
    for ii, dstype in enumerate([ 'final', "clean"]) :
        tgt_sparsity_tmp = tgt_sparsity_l[ii]
        val_dataset = core.datasets.MpiSintel(split='training', dstype=dstype)

        epe_list = []
        sparsity_list = []

        f = open('./flops_' + dstype + '_' + str(tgt_sparsity_tmp) + '.txt', 'w')
        for val_id in range(len(val_dataset)):
            if val_id % 50 == 0:
                print(val_id)

            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            flows = 0
            flow_count = 0

            # print(hws)
            sparsity_total = 0.0
            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                bs, _, hh, ww = image1_tile.shape
                mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, hh // 8, ww // 8).cuda()
                tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
                # flow_pre, _, sparsity = model(image1, image2, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
                flow_pre, flow_low, sparsity = model(image1_tile, image2_tile, tgt_sparsity=tgt_sparsity,
                                                     mhidden=mhidden)

                padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)
                # print(idx, weights[idx])
                if idx == 0:
                    sparsity_total = sparsity
                else:
                    sparsity_total = sparsity_total + sparsity
            # print('flow_count', flow_count)
            # exit(0)
            sparsity_total = sparsity_total / len(hws)
            flow_pre = flows / flow_count
            flow_pre = flow_pre[0].cpu()

            epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            sparsity_list.append(sparsity_total.view(1, -1).cpu().numpy())

            sparsity_tmp = torch.mean(sparsity_total).cpu().numpy()
            flops_tmp = 20.818860288 + 284.954290752 + (0.21233664 + 0.06096384 + 0.98205696 + 20.3461632) + (0.21233664 + 0.06096384 + 0.1327104 + 20.3461632) * 31 * sparsity_tmp + 2.871936 + 31 * 0.154885031
            for flops_idx, flops in enumerate(flops_l):
                # print(flops_tmp)
                if flops > flops_tmp:
                    f.write(str(flops_idx) + ' ' + str(flops_tmp) + '\n')
                    break

            if val_id < 10 and step is not None:
                save_path = './look/' +dstype +  '/' + str(step).zfill(10) + '/'
                if os.path.exists(save_path) == False:
                    os.makedirs(save_path)
                image1 = (image1.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
                image2 = (image2.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
                cv2.imwrite(save_path + str(val_id).zfill(4) + '_image1.png', image1[0])
                cv2.imwrite(save_path + str(val_id).zfill(4) + '_image2.png', image2[0])
                flow_gt = flow_gt.permute(1, 2, 0).detach().cpu().numpy()
                flow_pr = flow_pre.permute(1, 2, 0).detach().cpu().numpy()
                flow_gt = (flow2rgb(flow_gt) * 255).astype('uint8')
                flow_pr = (flow2rgb(flow_pr) * 255).astype('uint8')
                cv2.imwrite(save_path + str(val_id).zfill(4) + '_flow_gt.png', flow_gt)
                cv2.imwrite(save_path + str(val_id).zfill(4) + '_flow_pr.png', flow_pr)

                sparsity = sparsity_total.repeat(1, 1, 64, 1, 64)
                bs, c, h, iter, w = sparsity.shape
                sparsity = sparsity.view(bs, c, h, iter * w)
                sparsity = (sparsity.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                cv2.imwrite(save_path + str(val_id).zfill(4) + '_sparsity.png', sparsity[0])

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        sparsity = np.mean(np.concatenate(sparsity_list))

        flop = 20.818860288 + 284.954290752 + (0.21233664 + 0.06096384 + 0.98205696 + 20.3461632) + (0.21233664 + 0.06096384 + 0.1327104 + 20.3461632) * 31 * sparsity + 2.871936 + 31 * 0.154885031
        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, sparsity %f, flop %f" % (
            dstype, epe, px1, px3, px5, sparsity, flop*4))
        results[dstype] = np.mean(epe_list)
        results[dstype + '-sparsity'] = sparsity

    return results

@torch.no_grad()
def validate_sintel_time(model, sigma=0.05, step=-1):
    """ Peform validation using the Sintel (train) split """
    print(cfg.latentcostformer.decoder_depth)
    # exit(0)
    # update 23.2180992
    # mask 2.871936
    # iter_mask 0.154885031
    # origin: 20.818860288 + 284.954290752 + (0.21233664 + 0.06096384 + 0.98205696 + 20.3461632) + (0.21233664 + 0.06096384 + 0.1327104 + 20.3461632) * 31 + 2.871936

    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    results = {}
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # tgt_sparsity_l = [0.4, 0.4] # training high
    tgt_sparsity_l = [0.8, 0.92, ]  # training low
    # tgt_sparsity_l = [0.85, 0.85, ]  # testing
    # tgt_sparsity_l = [0.5, 0.5, ] # testing
    for ii, dstype in enumerate(['final', "clean"]):
        val_dataset = core.datasets.MpiSintel(split='training', dstype=dstype)

        epe_list = []
        tgt_sparsity_tmp = tgt_sparsity_l[ii]
        warm_skip = False
        time_l = []
        for val_id in range(len(val_dataset)):
            if val_id % 50 == 0:
                print(val_id)

            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            flows = 0
            flow_count = 0

            # print(hws)
            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
                bs, _, hh, ww = image1_tile.shape
                mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, hh // 8, ww // 8).cuda()
                tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
                # flow_pre, _, sparsity = model(image1, image2, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
                if warm_skip == True:
                    warm_id = 0
                    while (warm_id < 300):
                        _ = model.forward_time(image1_tile, image2_tile)
                        warm_id = warm_id + 1
                    warm_skip = False
                starter.record()
                flow_pre = model.forward_time(image1_tile, image2_tile, tgt_sparsity=tgt_sparsity,
                                                     mhidden=mhidden)
                ender.record()
                torch.cuda.synchronize()
                time_l.append(starter.elapsed_time(ender))

                padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)

            flow_pre = flows / flow_count
            flow_pre = flow_pre[0].cpu()

            epe = torch.sum((flow_pre - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())


        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)
        time_l = np.mean(time_l)
        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, time: %f" % (dstype, epe, px1, px3, px5, time_l))
        results[dstype] = np.mean(epe_list)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='load model')
    parser.add_argument('--eval', help='eval benchmark')
    parser.add_argument('--small', action='store_true', help='use small model')
    args = parser.parse_args()

    exp_func = None
    cfg = None
    if args.eval == 'sintel_submission':
        # exp_func = create_sintel_submission
        exp_func = create_sintel_submission_time
        cfg = get_submission_cfg()
    elif args.eval == 'kitti_submission':
        # exp_func = create_kitti_submission
        exp_func = create_kitti_submission_time
        cfg = get_submission_cfg()
        cfg.latentcostformer.decoder_depth = 24
    elif args.eval == 'sintel_validation':
        exp_func = validate_sintel
        # exp_func = validate_sintel_time
        if args.small:
            cfg = get_small_things_cfg()
        else:
            cfg = get_things_cfg()
    elif args.eval == 'kitti_validation':
        exp_func = validate_kitti
        # exp_func = validate_kitti_time
        if args.small:
            cfg = get_small_things_cfg()
        else:
            cfg = get_things_cfg()
        cfg.latentcostformer.decoder_depth = 24
    else:
        print(f"EROOR: {args.eval} is not valid")
    cfg.update(vars(args))

    print(cfg)
    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    exp_func(model.module)
