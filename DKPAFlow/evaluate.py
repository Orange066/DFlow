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

from KPAFlow import KPAFlow

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
    tgt_sparsity_tmp=0.9
    sparsity_list = []
    for dstype in ['clean', 'final']:
        test_dataset = core.datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            # if sequence != sequence_prev:
            #     flow_prev = None
            
            if (sequence != sequence_prev) or (dstype == 'final' and sequence in ['market_4', ]) or dstype == 'clean':
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
            bs, _, h, w = image1.shape
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)

            flow_low, flow_pr, sparsity = model(image1, image2, iters=iters, test_mode=True, flow_init=flow_prev, tgt_sparsity=tgt_sparsity,
                                         mhidden=mhidden)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence

            sparsity_list.append(sparsity.view(1, -1).cpu().numpy())

        sparsity = np.mean(np.concatenate(sparsity_list, axis=1))
        flop = 61.34464512 + 1.6613376 * 2 + 30.80208384 + 22.684352512 + 22.684352512 * 31 * sparsity + 3.120128 + 31 * 0.168270151

        print("Validation Sintel " + dstype + ": %f, flop %f" % (sparsity, flop))


@torch.no_grad()
def create_sintel_submission_time(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    tgt_sparsity_tmp = 0.9
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for dstype in ['clean', 'final']:
        test_dataset = core.datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        time_l = []
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            # if sequence != sequence_prev:
            #     flow_prev = None

            if (sequence != sequence_prev) or (dstype == 'final' and sequence in ['market_4', ]) or dstype == 'clean':
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
            bs, _, h, w = image1.shape
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
            if test_id == 0:
                warm_id = 0
                while (warm_id < 300):
                    # print(warm_id)
                    _ = model.forward_time_flow_low(image1, image2, iters=iters, test_mode=True, mhidden=mhidden,
                                           tgt_sparsity=tgt_sparsity)
                    warm_id = warm_id + 1
            starter.record()
            flow_low = model.forward_time_flow_low(image1, image2, iters=iters, test_mode=True, flow_init=flow_prev,
                                                tgt_sparsity=tgt_sparsity,
                                                mhidden=mhidden)
            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            ender.record()
            torch.cuda.synchronize()
            time_l.append(starter.elapsed_time(ender))
            sequence_prev = sequence

        print(dstype, np.mean(time_l))

@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = core.datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sparsity_list = []
    # tgt_sparsity_tmp = 0.75
    # tgt_sparsity_tmp = 0.95 # 77
    tgt_sparsity_tmp = 1.055
    # tgt_sparsity_tmp = 1.02
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        bs, _, h, w = image1.shape
        # print(image1.shape)
        # exit(0)
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        _, flow_pr, sparsity = model(image1, image2, iters=iters, test_mode=True, tgt_sparsity = tgt_sparsity, mhidden=mhidden)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        sparsity_list.append(sparsity.view(1, -1).cpu().numpy())
    sparsity = np.mean(np.concatenate(sparsity_list, axis=1))
    flop = 63.889053696 + 1.727791104 * 2 + 32.079670272 + 23.72376576 + 23.72376576 * 23 * sparsity + 3.2495424 + 23 * 0.175249535

    print("Validation KITTI: %f, flop %f" % (sparsity, flop))

@torch.no_grad()
def create_kitti_submission_time(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = core.datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # tgt_sparsity_tmp = 0.75
    # tgt_sparsity_tmp = 0.95 # 77
    tgt_sparsity_tmp = 1.055
    # tgt_sparsity_tmp = 1.08
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    time_l = []
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        bs, _, h, w = image1.shape
        # print(image1.shape)
        # exit(0)
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        if test_id == 0:
            warm_id = 0
            while(warm_id < 300):
                # print(warm_id)
                _ = model.forward_time(image1, image2, iters=iters, test_mode=True, mhidden=mhidden, tgt_sparsity=tgt_sparsity)
                warm_id = warm_id + 1

        starter.record()
        _ = model.forward_time(image1, image2, iters=iters, test_mode=True, tgt_sparsity = tgt_sparsity, mhidden=mhidden)
        ender.record()
        torch.cuda.synchronize()
        time_l.append(starter.elapsed_time(ender))

    print('time:', np.mean(time_l))

@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = core.datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32, step=None):
    """ Peform validation using the Sintel (train) split """
    # 61.34464512 + 1.6613376 * 2 + 30.80208384 + 32 * 25.804480512 ( 3.7013504 + 0.60870656 + 2.0446272 + 0.372621312 + 13.84660992 + 2.11043712 + 3.120128)
    # 61.34464512 + 1.6613376 * 2 + 30.80208384 + 32 * 22.684352512 + 3.120128
    model.eval()
    results = {}
    flops_l = []
    for i in range(iters+2):
        flops = 61.34464512 + 1.6613376 * 2 + 30.80208384 + i * 22.684352512 + 3.120128
        flops_l.append(flops)

    # tgt_sparsity_l = [0.725, 0.685]
    # tgt_sparsity_l = [0.88, 0.88] # testing
    # tgt_sparsity_l = [0.725]  # clean
    # tgt_sparsity_l = [0.685]  # final
    # tgt_sparsity_l = [0.725, 0.685]  # trainging-high
    # tgt_sparsity_l = [0.9, 0.86]  # trainging-high
    tgt_sparsity_l = [1.03, 1.0]  # trainging-low
    for ii, dstype in enumerate(['clean', 'final']):
    # for ii, dstype in enumerate(['clean']):
    # for ii, dstype in enumerate(['final']):
        val_dataset = core.datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        sparsity_list = []
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
            flow_low, flow_pr, sparsity = model(image1, image2, iters=iters, test_mode=True,tgt_sparsity = tgt_sparsity,mhidden=mhidden)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            sparsity_list.append(sparsity.view(1, -1).cpu().numpy())
            # 61.34464512 + 30.80208384 + 19.86112128 + 19.86112128 * 31 * 0.181866 + 2.11043712 + 31 * 0.17892864
            sparsity_tmp = torch.mean(sparsity).cpu().numpy()
            flops_tmp = 61.34464512 + 1.6613376 * 2 + 30.80208384 + 22.684352512 + 22.684352512 * 31 * sparsity_tmp + 3.120128 + 31 * 0.168270151
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
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        sparsity = np.mean(np.concatenate(sparsity_list))
        flop = 61.34464512 + 1.6613376 * 2 + 30.80208384 + 22.684352512 + 22.684352512 * 31 * sparsity + 3.120128 + 31 * 0.168270151
        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, sparsity %f, flop %f" % (
        dstype, epe, px1, px3, px5, sparsity, flop))
        results[dstype] = np.mean(epe_list)
        results[dstype + '-sparsity'] = sparsity

    return results

@torch.no_grad()
def validate_sintel_time(model, iters=32, step=None):
    """ Peform validation using the Sintel (train) split """
    # 61.34464512 + 1.6613376 * 2 + 30.80208384 + 32 * 25.804480512 ( 3.7013504 + 0.60870656 + 2.0446272 + 0.372621312 + 13.84660992 + 2.11043712 + 3.120128)
    # 61.34464512 + 1.6613376 * 2 + 30.80208384 + 32 * 22.684352512 + 3.120128
    model.eval()
    results = {}

    # tgt_sparsity_l = [0.725, 0.685] # trainging-high
    # tgt_sparsity_l = [1.03, 1.0]  # trainging-low
    tgt_sparsity_l = [0.88, 0.88] # testing
    # tgt_sparsity_l = [0.88, 0.88]
    # tgt_sparsity_l = [1.1]
    # for ii, dstype in enumerate(['clean', 'final']):
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # for ii, dstype in enumerate(['final']):
    for ii, dstype in enumerate(['clean', 'final']):
        val_dataset = core.datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        tgt_sparsity_tmp = tgt_sparsity_l[ii]
        time_l = []
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
                    _ = model.forward_time(image1, image2, iters=iters, test_mode=True,tgt_sparsity = tgt_sparsity,mhidden=mhidden)
                    warm_id = warm_id + 1

            starter.record()
            flow_pr = model.forward_time(image1, image2, iters=iters, test_mode=True,tgt_sparsity = tgt_sparsity,mhidden=mhidden)
            ender.record()
            torch.cuda.synchronize()
            time_l.append(starter.elapsed_time(ender))
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
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

@torch.no_grad()
def validate_kitti(model, iters=24, step=None):
    """ Peform validation using the KITTI-2015 (train) split """
    # 63.889053696 + 1.727791104 * 2 + 32.079670272 + 24 * 26.97330816 (3.85487232 + 0.633954048 + 2.12943276 + 0.4866048 + 14.420929536 + 2.197972296 + 3.2495424)
    # 63.889053696 + 1.727791104 * 2 + 32.079670272 + 24 * 23.72376576 + 3.2495424
    model.eval()
    val_dataset = core.datasets.KITTI(split='training')

    out_list, epe_list = [], []
    sparsity_list = []
    flops_l = []
    for i in range(iters + 2):
        flops = 63.889053696 + 1.727791104 * 2 + 32.079670272 + i * 23.72376576 + 3.2495424
        flops_l.append(flops)
    # tgt_sparsity_tmp = 0.5
    # tgt_sparsity_tmp = 0.52 # training-high medium
    # tgt_sparsity_tmp = 0.313  # training-high low
    # tgt_sparsity_tmp = 0.65  # training-high high
    tgt_sparsity_tmp = 1.05  # training-low
    # tgt_sparsity_tmp = 0.95 # testing 1
    # tgt_sparsity_tmp = 0.68 # testing 1
    # tgt_sparsity_tmp = 0.42 # 250

    f = open('./flops_kitti_' + str(tgt_sparsity_tmp) + '.txt', 'w')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        bs, _, h, w = image1.shape
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        flow_low, flow_pr, sparsity = model(image1, image2, iters=iters, test_mode=True, tgt_sparsity = tgt_sparsity, mhidden=mhidden)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        sparsity_list.append(sparsity.view(1, -1).cpu().numpy())

        sparsity_tmp = torch.mean(sparsity).cpu().numpy()
        # 63.889053696 + 32.079670272 + 20.684906424 + 20.684906424 * 23 * 0.622826 + 2.197972296 + 23 * 0.186350112
        flops_tmp = 63.889053696 + 1.727791104 * 2 + 32.079670272 + 23.72376576 + 23.72376576 * 23 * sparsity_tmp + 3.2495424 + 23 * 0.175249535
        for flops_idx, flops in enumerate(flops_l):
            # print(flops_tmp)
            if flops > flops_tmp:
                f.write(str(flops_idx) + ' ' + str(flops_tmp) + '\n')
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
    f.close()
    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)
    sparsity = np.mean(np.concatenate(sparsity_list, axis=1))
    # flop = 63.889053696 + 32.079670272 + 20.684906424 + 20.684906424 * 23 * sparsity + 2.197972296 + 23 * 0.175249535
    flop = 63.889053696 + 1.727791104 * 2 + 32.079670272 + 23.72376576 + 23.72376576 * 23 * sparsity + 3.2495424 + 23 * 0.175249535
    print("Validation KITTI: %f(epe), %f(f1), %f(sparsity), flop %f" % (epe, f1, sparsity, flop))
    return {'kitti-epe': epe, 'kitti-f1': f1, 'kitti-sparsity': sparsity}

@torch.no_grad()
def validate_kitti_time(model, iters=24, step=None):
    """ Peform validation using the KITTI-2015 (train) split """
    # 63.889053696 + 1.727791104 * 2 + 32.079670272 + 24 * 26.97330816 (3.85487232 + 0.633954048 + 2.12943276 + 0.4866048 + 14.420929536 + 2.197972296 + 3.2495424)
    # 63.889053696 + 1.727791104 * 2 + 32.079670272 + 24 * 23.72376576 + 3.2495424
    model.eval()
    val_dataset = core.datasets.KITTI(split='training')

    out_list, epe_list = [], []
    # tgt_sparsity_tmp = 0.5
    # tgt_sparsity_tmp = 0.52 # training-high
    tgt_sparsity_tmp = 1.05  # training-low
    # tgt_sparsity_tmp = 0.95 # testing
    # tgt_sparsity_tmp = 0.68 # testing
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    time_l = []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        bs, _, h, w = image1.shape
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)

        if val_id == 0:
            warm_id = 0
            while (warm_id < 300):
                _ = model.forward_time(image1, image2, iters=iters, test_mode=True, tgt_sparsity = tgt_sparsity, mhidden=mhidden)
                warm_id = warm_id + 1
        starter.record()
        flow_pr = model.forward_time(image1, image2, iters=iters, test_mode=True, tgt_sparsity = tgt_sparsity, mhidden=mhidden)
        ender.record()
        torch.cuda.synchronize()
        time_l.append(starter.elapsed_time(ender))

        flow = padder.unpad(flow_pr[0]).cpu()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(KPAFlow(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_sintel_submission_time(model.module, warm_start=True)
    # create_kitti_submission(model.module)
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


