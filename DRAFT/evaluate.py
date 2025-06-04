
import sys

sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import core.datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate
import cv2
import torch.nn as nn


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    tgt_sparsity_tmp = 0.86
    sparsity_list = []
    for dstype in ['clean', 'final']:
        test_dataset = core.datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
            bs, _, h, w = image1.shape
            # print(image1.shape)
            # exit(0)
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)

            flow_low, flow_pr, sparsity = model(image1, image2, iters=iters, test_mode=True, flow_init=flow_prev,
                                                tgt_sparsity=tgt_sparsity,
                                                mhidden=mhidden)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence

            sparsity_list.append(sparsity.view(1, -1).cpu().numpy())

        sparsity = np.mean(np.concatenate(sparsity_list, axis=1))
        flop = 61.34464512 + 30.80208384 + 19.86112128 + 19.86112128 * 31 * sparsity + 2.11043712 + 31 * 0.168270151

        print("Validation Sintel " + dstype + ": %f, flop %f" % (sparsity, flop))


@torch.no_grad()
def create_sintel_submission_time(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    tgt_sparsity_tmp = 0.86
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for dstype in ['clean', 'final']:
        test_dataset = core.datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        time_l = []
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
            bs, _, h, w = image1.shape
            # print(image1.shape)
            # exit(0)
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)

            if test_id == 0:
                warm_id = 0
                while (warm_id < 300):
                    # print(warm_id)
                    _ = model.forward_time(image1, image2, iters=iters, test_mode=True, mhidden=mhidden,
                                           tgt_sparsity=tgt_sparsity)
                    warm_id = warm_id + 1
            starter.record()
            _ = model.forward_time(image1, image2, iters=iters, test_mode=True, tgt_sparsity=tgt_sparsity,
                                   mhidden=mhidden)
            ender.record()
            torch.cuda.synchronize()
            time_l.append(starter.elapsed_time(ender))

        print(dstype, np.mean(time_l))


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = core.datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sparsity_list = []
    tgt_sparsity_tmp = 0.85

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        bs, _, h, w = image1.shape
        # print(image1.shape)
        # exit(0)
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        _, flow_pr, sparsity = model(image1, image2, iters=iters, test_mode=True, tgt_sparsity=tgt_sparsity,
                                     mhidden=mhidden)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        sparsity_list.append(sparsity.view(1, -1).cpu().numpy())
    sparsity = np.mean(np.concatenate(sparsity_list, axis=1))
    flop = 63.889053696 + 32.079670272 + 20.684906424 + 20.684906424 * 23 * sparsity + 2.197972296 + 23 * 0.175249535

    print("Validation KITTI: %f, flop %f" % (sparsity, flop))


@torch.no_grad()
def create_kitti_submission_time(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = core.datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sparsity_list = []
    tgt_sparsity_tmp = 0.85
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    time_l = []
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        bs, _, h, w = image1.shape
        # print(image1.shape)
        # exit(0)
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        if test_id == 0:
            warm_id = 0
            while (warm_id < 300):
                # print(warm_id)
                _ = model.forward_time(image1, image2, iters=iters, test_mode=True, mhidden=mhidden,
                                       tgt_sparsity=tgt_sparsity)
                warm_id = warm_id + 1
        starter.record()
        _ = model.forward_time(image1, image2, iters=iters, test_mode=True, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
        ender.record()
        torch.cuda.synchronize()
        time_l.append(starter.elapsed_time(ender))

    print('time:', np.mean(time_l))


@torch.no_grad()
def validate_chairs(model, iters=24, step=None):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []
    sparsity_list = []
    val_dataset = core.datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr, sparsity = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())
        sparsity_list.append(sparsity.view(1, -1).cpu().numpy())
        if val_id < 10 and step is not None:
            save_path = './look/chairs/' + str(step).zfill(10) + '/'
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            image1 = (image1.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
            image2 = (image2.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_image1.png', image1[0])
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_image2.png', image2[0])
            flow_gt = flow_gt.permute(1, 2, 0).detach().cpu().numpy()
            flow_pr = flow_pr[0].permute(1, 2, 0).detach().cpu().numpy()
            flow_gt = (flow2rgb(flow_gt) * 255).astype('uint8')
            flow_pr = (flow2rgb(flow_pr) * 255).astype('uint8')
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_flow_gt.png', flow_gt)
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_flow_pr.png', flow_pr)
            bs, c, h, iter, w = sparsity.shape
            sparsity = sparsity.view(bs, c, h, iter * w)
            sparsity = (sparsity.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_sparsity.png', sparsity[0])

    epe = np.mean(np.concatenate(epe_list))
    sparsity = np.mean(np.concatenate(sparsity_list))
    print("Validation Chairs EPE: %f, sparsity: %f" % (epe, sparsity))
    return {'chairs': epe, 'chairs-sparsity': sparsity}


@torch.no_grad()
def validate_sintel(model, iters=32, step=None):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    flops_l = []
    for i in range(iters + 2):
        flops = 61.34464512 + 30.80208384 + 19.86112128 * i + 2.11043712
        flops_l.append(flops)

    # tgt_sparsity_l = [0.62, 0.64] # training high
    # tgt_sparsity_l = [0.85, 0.87]  # training high
    # tgt_sparsity_l = [0.76]
    tgt_sparsity_l = [0.99, 1.04] # training low
    # tgt_sparsity_l = [0.74, 0.76]  # testing
    for ii, dstype in enumerate(['clean', 'final']):
        # for ii, dstype in enumerate(['final']):
        # for dstype in ['clean']:
        val_dataset = core.datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        sparsity_list = []
        # tgt_sparsity_tmp = 0.585
        # tgt_sparsity_tmp = 0.57
        # tgt_sparsity_tmp = 0.64
        # tgt_sparsity_tmp = 0.62
        # tgt_sparsity_tmp = 0.8
        # tgt_sparsity_tmp = 0.65
        # tgt_sparsity_tmp = 0.8
        # tgt_sparsity_tmp = 0.76
        tgt_sparsity_tmp = tgt_sparsity_l[ii]
        f = open('./flops_' + dstype + '_' + str(tgt_sparsity_tmp) + '.txt', 'w')
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            bs, _, h, w = image1.shape
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
            flow_low, flow_pr, sparsity = model(image1, image2, iters=iters, test_mode=True, tgt_sparsity=tgt_sparsity,
                                                mhidden=mhidden)

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            # print('epe', epe.mean())
            # exit(0)
            epe_list.append(epe.view(-1).numpy())
            sparsity_list.append(sparsity.view(1, -1).cpu().numpy())
            # 61.34464512 + 30.80208384 + 19.86112128 + 19.86112128 * 31 * 0.181866 + 2.11043712 + 31 * 0.17892864
            sparsity_tmp = torch.mean(sparsity).cpu().numpy()
            flops_tmp = 61.34464512 + 30.80208384 + 19.86112128 + 19.86112128 * 31 * sparsity_tmp + 2.11043712 + 31 * 0.168270151
            for flops_idx, flops in enumerate(flops_l):
                # print(flops_tmp)
                if flops > flops_tmp:
                    f.write(str(flops_idx) + ' ' + str(flops_tmp) + '\n')
                    break

            if val_id < 10 and step is not None:
                save_path = './look/' + dstype + '/' + str(step).zfill(10) + '/'
                if os.path.exists(save_path) == False:
                    os.makedirs(save_path)
                image1 = (image1.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
                image2 = (image2.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
                cv2.imwrite(save_path + str(val_id).zfill(4) + '_image1.png', image1[0])
                cv2.imwrite(save_path + str(val_id).zfill(4) + '_image2.png', image2[0])
                flow_gt = flow_gt.permute(1, 2, 0).detach().cpu().numpy()
                flow_pr = flow_pr[0].permute(1, 2, 0).detach().cpu().numpy()
                flow_gt = (flow2rgb(flow_gt) * 255).astype('uint8')
                flow_pr = (flow2rgb(flow_pr) * 255).astype('uint8')
                cv2.imwrite(save_path + str(val_id).zfill(4) + '_flow_gt.png', flow_gt)
                cv2.imwrite(save_path + str(val_id).zfill(4) + '_flow_pr.png', flow_pr)

                sparsity = sparsity.repeat(1, 1, 64, 1, 64)
                bs, c, h, iter, w = sparsity.shape
                sparsity = sparsity.view(bs, c, h, iter * w)
                sparsity = (sparsity.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                cv2.imwrite(save_path + str(val_id).zfill(4) + '_sparsity.png', sparsity[0])

        f.close()
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)
        sparsity = np.mean(np.concatenate(sparsity_list))
        flop = 61.34464512 + 30.80208384 + 19.86112128 + 19.86112128 * 31 * sparsity + 2.11043712 + 31 * 0.168270151
        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, sparsity %f, flop %f" % (
        dstype, epe, px1, px3, px5, sparsity, flop))
        results[dstype] = np.mean(epe_list)
        results[dstype + '-sparsity'] = sparsity

    return results


def validate_sintel_time(model, iters=32, step=None):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    # tgt_sparsity_tmp = 0.64 # final
    # tgt_sparsity_tmp = 0.62 # clean
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # tgt_sparsity_l = [0.62, 0.64] # training high
    # tgt_sparsity_l = [0.76]
    tgt_sparsity_l = [0.99, 1.04] # training low
    # tgt_sparsity_l = [0.74, 0.76]  # testing
    for ii, dstype in enumerate(['clean', 'final']):
        # for ii, dstype in enumerate(['final']):
        # for dstype in ['clean']:
        val_dataset = core.datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        # tgt_sparsity_tmp = 0.585
        # tgt_sparsity_tmp = 0.57
        # tgt_sparsity_tmp = 0.64
        # tgt_sparsity_tmp = 0.62
        # tgt_sparsity_tmp = 0.8
        # tgt_sparsity_tmp = 0.65
        # tgt_sparsity_tmp = 0.8
        time_l = []
        tgt_sparsity_tmp = tgt_sparsity_l[ii]
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            bs, _, h, w = image1.shape
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
            if val_id == 0:
                warm_id = 0
                while (warm_id < 300):
                    # print(warm_id)
                    _ = model.forward_time(image1, image2, iters=iters, test_mode=True, mhidden=mhidden,
                                           tgt_sparsity=tgt_sparsity)
                    warm_id = warm_id + 1

            starter.record()
            flow_pr = model.forward_time(image1, image2, iters=iters, test_mode=True, tgt_sparsity=tgt_sparsity,
                                         mhidden=mhidden)
            ender.record()
            torch.cuda.synchronize()
            time_l.append(starter.elapsed_time(ender))
            flow = padder.unpad(flow_pr[0]).cpu()
            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)
        time_l = np.mean(time_l)
        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, time: %f" % (dstype, epe, px1, px3, px5, time_l))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24, step=None):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = core.datasets.KITTI(split='training')

    out_list, epe_list = [], []
    out2_list, epe2_list = [], []
    sparsity_list = []
    sparsity2_list = []
    flops_l = []
    for i in range(iters + 2):
        flops = 63.889053696 + 32.079670272 + 20.684906424 * i + 2.197972296
        flops_l.append(flops)
    # tgt_sparsity_tmp = 1.0  # ablation visual
    # tgt_sparsity_tmp = 0.63 # training-high motivation medium
    # tgt_sparsity_tmp = 0.8  # training-high motivation low
    # tgt_sparsity_tmp = 0.9  # training-high motivation low low
    # tgt_sparsity_tmp = 0.84  # 370
    # tgt_sparsity_tmp = 0.74  # 320
    # tgt_sparsity_tmp = 0.485  # training-high motivation high
    tgt_sparsity_tmp = 0.99 # training-low
    # tgt_sparsity_tmp = 0.5  # testing
    # tgt_sparsity_tmp = 0.8  # motivation
    inc_average_l = []
    for i in range(iters):
        inc_average_l.append([])

    f = open('./flops_kitti_' + str(tgt_sparsity_tmp) + '.txt', 'w')
    # print('len(val_dataset)', len(val_dataset))
    for val_id in range(len(val_dataset)):
        # if val_id != 136:
        #     continue
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        # tgt_sparsity_tmp = 0.5
        bs, _, h, w = image1.shape
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        flow_low, flow_pr, sparsity = model(image1, image2, iters=iters, test_mode=True, tgt_sparsity=tgt_sparsity,
                                            mhidden=mhidden)
        # 0.01831525506451726

        flow_prs, inc_l = model.forward_inc(image1, image2, iters=iters, test_mode=True,
                                            tgt_sparsity=tgt_sparsity, mhidden=mhidden)
        for inc_i, inc in enumerate(inc_l):
            inc_average_l[inc_i].append(inc.view(1).item())
        for inc_i, inc in enumerate(inc_l):
            if inc < 0.008861981108784675 or inc_i == iters - 1:
                flow2 = padder.unpad(flow_prs[inc_i][0]).cpu()
                epe2 = torch.sum((flow2 - flow_gt) ** 2, dim=0).sqrt()
                mag2 = torch.sum(flow_gt ** 2, dim=0).sqrt()

                epe2 = epe2.view(-1)
                mag2 = mag2.view(-1)
                val = valid_gt.view(-1) >= 0.5

                out2 = ((epe2 > 3.0) & ((epe2 / mag2) > 0.05)).float()
                epe2_list.append(epe2[val].mean().item())
                out2_list.append(out2[val].cpu().numpy())
                sparsity2 = 63.889053696 + 32.079670272 + 20.684906424 * (inc_i + 1) + 2.197972296 + (
                            inc_i + 1) * 0.175249535
                sparsity2_list.append(sparsity2)
                break

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        sparsity_list.append(sparsity.view(1, -1).cpu().numpy())
        # print(epe[val].mean().item())

        sparsity_tmp = torch.mean(sparsity).cpu().numpy()
        # 63.889053696 + 32.079670272 + 20.684906424 + 20.684906424 * 23 * 0.622826 + 2.197972296 + 23 * 0.186350112
        flops_tmp = 63.889053696 + 32.079670272 + 20.684906424 + 20.684906424 * 23 * sparsity_tmp + 2.197972296 + 23 * 0.175249535
        # print(flops_tmp)

        for flops_idx, flops in enumerate(flops_l):
            # print(flops_tmp)
            if flops > flops_tmp:
                f.write(str(flops_idx) + ' ' + str(flops_tmp) + '\n')
                break

        if val_id < 137 and step is not None:
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
            flow_pr = flow_pr[0].permute(1, 2, 0).detach().cpu().numpy()
            flow_gt = (flow2rgb(flow_gt) * 255).astype('uint8')
            flow_pr = (flow2rgb(flow_pr) * 255).astype('uint8')
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_flow_gt.png', flow_gt)
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_flow_pr.png', flow_pr)
            sparsity = sparsity.repeat(1, 1, 64, 1, 64)
            bs, c, h, iter, w = sparsity.shape
            # print(sparsity.shape)
            sparsity_0 = sparsity[:, :, :, :6].clone().view(bs, c, h, 6 * w)
            sparsity_1 = sparsity[:, :, :, 6:12].clone().view(bs, c, h, 6 * w)
            sparsity_2 = sparsity[:, :, :, 12:18].clone().view(bs, c, h, 6 * w)
            sparsity_3 = sparsity[:, :, :, 18:].clone().view(bs, c, h, 5 * w)
            # sparsity_0 = sparsity[:, :, :, :12].clone().view(bs, c, h, 12 * w)
            # sparsity_1 = sparsity[:, :, :, 12:24].clone().view(bs, c, h, 12 * w)
            # sparsity_2 = sparsity[:, :, :, 24:36].clone().view(bs, c, h, 12 * w)
            # sparsity_3 = sparsity[:, :, :, 36:].clone().view(bs, c, h, 11 * w)
            tmp = torch.ones_like(sparsity_3[:, :, :, :w])
            sparsity_3 = torch.cat([sparsity_3, tmp], dim=3)
            sparsity = torch.cat([sparsity_0, sparsity_1, sparsity_2, sparsity_3], dim=2)

            # sparsity = sparsity.view(bs, c, h, iter * w)
            sparsity = (sparsity.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
            cv2.imwrite(save_path + str(val_id).zfill(4) + '_sparsity.png', sparsity[0])
        # exit(0)
    f.close()

    # for inc_i, inc in enumerate(inc_average_l):
    #     print(inc_i, np.mean(inc))

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    epe2_list = np.array(epe2_list)
    out2_list = np.concatenate(out2_list)
    epe2 = np.mean(epe2_list)
    f12 = 100 * np.mean(out2_list)
    flop2 = np.mean(sparsity2_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    sparsity = np.mean(np.concatenate(sparsity_list, axis=1))
    # flop = 63.889053696 + 32.079670272 + 20.684906424 + 20.684906424 * 23 * sparsity + 2.197972296 + 23 * 0.175249535
    flop = 63.889053696 + 32.079670272 + 20.684906424 + 20.684906424 * 23 * sparsity + 2.197972296 + 23 * 0.175249535

    print("Validation KITTI: %f(epe), %f(f1), %f(sparsity), flop %f" % (epe, f1, sparsity, flop))
    return {'kitti-epe': epe, 'kitti-f1': f1, 'kitti-sparsity': sparsity}


@torch.no_grad()
def validate_kitti_time(model, iters=24, step=None):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = core.datasets.KITTI(split='training')

    out_list, epe_list = [], []
    out2_list, epe2_list = [], []
    sparsity_list = []
    sparsity2_list = []
    flops_l = []
    time_l = []
    tgt_sparsity_tmp = 0.63 # training-high
    # tgt_sparsity_tmp = 0.99 # training-low
    # tgt_sparsity_tmp = 0.5  # testing
    # N_freqs = 3
    # freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
    # funcs = [torch.sin, torch.cos]
    # itr_embedding_l = []
    # for i in range(8):
    #     x = torch.ones(1) * i
    #     out = []
    #     out += [x]
    #     for freq in freq_bands:
    #         for func in funcs:
    #             out += [func(freq * x)]
    #     out = torch.cat(out, -1).view(-1, 1, 1)
    #     out = out.repeat(1,47,156)
    #     itr_embedding_l.append(out)
    # itr_embedding = torch.stack(itr_embedding_l, dim=0)
    # # iter_embedding = torch.unsqueeze(iter_embedding, dim=1)
    # itr_embedding = nn.Parameter(itr_embedding, requires_grad=False).cuda()
    # print('itr_embedding', itr_embedding.shape)
    # exit(0)
    # print('len(val_dataset)', len(val_dataset))
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        # image1 = image1[None]
        # image2 = image2[None]

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)
        # print('image1', image1.shape)

        # tgt_sparsity_tmp = 0.5

        bs, _, h, w = image1.shape
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        # N_freqs = 3
        # freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        # funcs = [torch.sin, torch.cos]
        # itr_embedding_l = []
        # for i in range(8):
        #     x = torch.ones(1) * i
        #     out = []
        #     out += [x]
        #     for freq in freq_bands:
        #         for func in funcs:
        #             out += [func(freq * x)]
        #     out = torch.cat(out, -1).view(-1, 1, 1)
        #     out = out.repeat(1, h//8, w//8)
        #     itr_embedding_l.append(out)
        # itr_embedding = torch.stack(itr_embedding_l, dim=0)
        # # iter_embedding = torch.unsqueeze(iter_embedding, dim=1)
        # itr_embedding = nn.Parameter(itr_embedding, requires_grad=False).cuda()

        # if val_id == 0:
        #     warm_id = 0
        #     while (warm_id < 300):
        #         # print(warm_id)
        #         _ = model.forward_time(image1, image2, iters=iters, test_mode=True, mhidden=mhidden,
        #                                tgt_sparsity=tgt_sparsity)
        #         warm_id = warm_id + 1

        # time_start = time.time()
        # print('image1', image1.shape)
        # print(val_id)
        starter.record()
        flow_pr = model.forward_time(image1, image2, iters=iters, test_mode=True, mhidden=mhidden,
                                     tgt_sparsity=tgt_sparsity)
        # print('*'*10)
        # exit(0)
        ender.record()
        torch.cuda.synchronize()
        # print(count, count/23, s  parsity.mean())
        # time_end = time.time()
        # if val_id >=30:
        # time_l.append(time_end-time_start)
        time_l.append(starter.elapsed_time(ender))
        # print(np.mean(time_l))
        # 0.01831525506451726

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    time_avarage = np.mean(time_l)

    print("Validation KITTI: %f, %f, %f" % (epe, f1, time_avarage))
    return {'kitti-epe': epe, 'kitti-f1': f1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    # model.load_state_dict(torch.load(args.model))

    pretrained_dict = torch.load(args.model)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=False)
    # create_kitti_submission(model.module)
    # create_sintel_submission_time(model.module, warm_start=True)
    # create_kitti_submission_time(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module, step=-1)

        elif args.dataset == 'sintel':
            validate_sintel(model.module, step=-1)
            # validate_sintel_time(model.module, step=-1)

        elif args.dataset == 'kitti':
            validate_kitti(model.module, step=-1)
            # validate_kitti_time(model.module, step=-1)


