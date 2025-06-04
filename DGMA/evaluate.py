import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import imageio

import core.datasets
from network import RAFTGMA
from utils import flow_viz
from utils import frame_utils

from utils.utils import InputPadder, forward_interpolate
import cv2

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

@torch.no_grad()
def create_sintel_submission(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    # tgt_sparsity_list = [0.89, 0.86]
    # for ii, dstype in enumerate(['clean', 'final']):
    tgt_sparsity_list = [0.87, 0.86]
    for ii, dstype in enumerate(['clean', 'final']):
        test_dataset = core.datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        sparsity_list = []
        tgt_sparsity_tmp = tgt_sparsity_list[ii]
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))
            bs, _, h, w = image1.shape
            # print(image1.shape)
            # exit(0)
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
            flow_low, flow_pr, sparsity = model(image1, image2, iters=32, test_mode=True, tgt_sparsity=tgt_sparsity,
                                         mhidden=mhidden, flow_init=flow_prev)
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
        flop = 61.34464512 + 30.80208384 + 0.23068672 + 22.42707456 + 22.42707456 * 31 * sparsity + 3.120128 + 11 * 0.168270151
        print("Validation %s: %f, flop %f" % (dstype, sparsity, flop))

@torch.no_grad()
def create_sintel_submission_time(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    # tgt_sparsity_list = [0.89, 0.86]
    # for ii, dstype in enumerate(['clean', 'final']):
    tgt_sparsity_list = [0.87, 0.86]
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for ii, dstype in enumerate(['clean', 'final']):
        test_dataset = core.datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        sparsity_list = []
        tgt_sparsity_tmp = tgt_sparsity_list[ii]
        time_l = []
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))
            bs, _, h, w = image1.shape
            # print(image1.shape)
            # exit(0)
            mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
            if test_id == 0:
                warm_id = 0
                while (warm_id < 300):
                    # print(warm_id)
                    _ = model(image1, image2, iters=32, test_mode=True, mhidden=mhidden,
                                           tgt_sparsity=tgt_sparsity)
                    warm_id = warm_id + 1
            starter.record()
            flow_low = model(image1, image2, iters=32, test_mode=True, tgt_sparsity=tgt_sparsity,
                                         mhidden=mhidden, flow_init=flow_prev)
            ender.record()
            torch.cuda.synchronize()
            time_l.append(starter.elapsed_time(ender))

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()


            sequence_prev = sequence
        print(dstype, np.mean(time_l))

@torch.no_grad()
def create_sintel_submission_vis(model, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=32, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            # Visualizations
            flow_img = flow_viz.flow_to_image(flow)
            image = Image.fromarray(flow_img)
            if not os.path.exists(f'vis_test/RAFT/{dstype}/'):
                os.makedirs(f'vis_test/RAFT/{dstype}/flow')

            if not os.path.exists(f'vis_test/ours/{dstype}/'):
                os.makedirs(f'vis_test/ours/{dstype}/flow')

            if not os.path.exists(f'vis_test/gt/{dstype}/'):
                os.makedirs(f'vis_test/gt/{dstype}/image')

            # image.save(f'vis_test/ours/{dstype}/flow/{test_id}.png')
            image.save(f'vis_test/RAFT/{dstype}/flow/{test_id}.png')
            imageio.imwrite(f'vis_test/gt/{dstype}/image/{test_id}.png', image1[0].cpu().permute(1, 2, 0).numpy())
            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = core.datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sparsity_list = []
    # tgt_sparsity_tmp = 0.75
    tgt_sparsity_tmp = 0.91
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        bs, _, h, w = image1.shape
        # print(image1.shape)
        # exit(0)
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        _, flow_pr, sparsity = model(image1, image2, iters=24, test_mode=True, tgt_sparsity=tgt_sparsity,
                                     mhidden=mhidden)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        sparsity_list.append(sparsity.view(1, -1).cpu().numpy())
    sparsity = np.mean(np.concatenate(sparsity_list, axis=1))
    flop = 63.889053696 + 32.079670272 + 0.240254976 + 23.357288448 + 23.357288448 * 23 * sparsity + 3.2495424 + 23 * 0.175249535

    print("Validation KITTI: %f, flop %f" % (sparsity, flop))

@torch.no_grad()
def create_kitti_submission_time(model, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = core.datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sparsity_list = []
    # tgt_sparsity_tmp = 0.75
    tgt_sparsity_tmp = 0.91
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    time_l = []
    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        bs, _, h, w = image1.shape
        # print(image1.shape)
        # exit(0)
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h // 8, w // 8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        if test_id == 0:
            warm_id = 0
            while(warm_id < 300):
                # print(warm_id)
                _ = model(image1, image2, iters=24, test_mode=True, mhidden=mhidden, tgt_sparsity=tgt_sparsity)
                warm_id = warm_id + 1
        starter.record()
        _, = model(image1, image2, iters=24, test_mode=True, tgt_sparsity=tgt_sparsity,
                                     mhidden=mhidden)
        ender.record()
        torch.cuda.synchronize()
        time_l.append(starter.elapsed_time(ender))

    print('time:', np.mean(time_l))


@torch.no_grad()
def create_kitti_submission_vis(model, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        # Visualizations
        flow_img = flow_viz.flow_to_image(flow)
        image = Image.fromarray(flow_img)
        if not os.path.exists(f'vis_kitti'):
            os.makedirs(f'vis_kitti/flow')
            os.makedirs(f'vis_kitti/image')

        image.save(f'vis_kitti/flow/{test_id}.png')
        imageio.imwrite(f'vis_kitti/image/{test_id}_0.png', image1[0].cpu().permute(1, 2, 0).numpy())
        imageio.imwrite(f'vis_kitti/image/{test_id}_1.png', image2[0].cpu().permute(1, 2, 0).numpy())


@torch.no_grad()
def validate_chairs(model, iters=6):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs_epe': epe}


@torch.no_grad()
def validate_things(model, iters=6):
    """ Perform evaluation on the FlyingThings (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        epe_list = []
        val_dataset = datasets.FlyingThings3D(dstype=dstype, split='validation')
        print(f'Dataset length {len(val_dataset)}')
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel(model, iters=6, step=None):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    flops_l = []
    for i in range(iters+2):
        # 25.54720256
        flops = 61.34464512 + 30.80208384 + 0.23068672 + 22.42707456 * i + 3.120128
        flops_l.append(flops)

    tgt_sparsity_l =  [0.95, 0.93]  # training-low
    # tgt_sparsity_l = [0.65, 0.52] # training-high
    # tgt_sparsity_l = [0.86, 0.73] # training-high
    # tgt_sparsity_l = [0.87, 0.76]  # training-high
    # tgt_sparsity_l =[0.8,0.8]  # testing
    # tgt_sparsity_l = [0.9]  # testing
    for ii, dstype in enumerate(['clean', 'final']):
    # for ii, dstype in enumerate(['clean']):
    # for ii, dstype in enumerate(['final']):
        val_dataset = core.datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        sparsity_list = []
        # tgt_sparsity_tmp = 0.585
        # tgt_sparsity_tmp = 0.57
        # tgt_sparsity_tmp = 0.64
        # tgt_sparsity_tmp = 0.62
        # tgt_sparsity_tmp = 0.8
        # tgt_sparsity_tmp = 0.65
        tgt_sparsity_tmp = tgt_sparsity_l[ii]
        # tgt_sparsity_tmp = 0.78
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

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            sparsity_list.append(sparsity.view(1, -1).cpu().numpy())
            sparsity_tmp = torch.mean(sparsity).cpu().numpy()
            flops_tmp = 61.34464512 + 30.80208384 + 0.23068672 + 22.42707456 + 22.42707456 * 31 * sparsity_tmp + 3.120128 + 31* 0.168270151
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
        flop = 61.34464512 + 30.80208384 + 0.23068672 + 22.42707456 + 22.42707456 * 31 * sparsity + 3.120128 + 31 * 0.168270151
        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, sparsity %f, flop %f" % (
        dstype, epe, px1, px3, px5, sparsity, flop))
        results[dstype] = np.mean(epe_list)
        results[dstype + '-sparsity'] = sparsity

    return results

@torch.no_grad()
def validate_sintel_time(model, iters=6, step=None):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    # tgt_sparsity_l = [0.65, 0.52] # training-high
    tgt_sparsity_l =  [0.95, 0.93]  # training-low
    # tgt_sparsity_l =[0.8,0.8]  # testing
    torch.cuda.synchronize()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for ii, dstype in enumerate(['clean', 'final']):
        val_dataset = core.datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []
        sparsity_list = []
        # tgt_sparsity_tmp = 0.585
        # tgt_sparsity_tmp = 0.57
        # tgt_sparsity_tmp = 0.64
        # tgt_sparsity_tmp = 0.62
        tgt_sparsity_tmp = tgt_sparsity_l[ii]
        # tgt_sparsity_tmp = 0.65 # training-high
        # tgt_sparsity_tmp = 0.78
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
                    _ = model.forward_time(image1, image2, iters=iters, test_mode=True, tgt_sparsity=tgt_sparsity,
                                                mhidden=mhidden)
                    warm_id = warm_id + 1
            starter.record()
            flow_pr = model.forward_time(image1, image2, iters=iters, test_mode=True, tgt_sparsity=tgt_sparsity,
                                                mhidden=mhidden)
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
def validate_sintel_occ(model, iters=6):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['albedo', 'clean', 'final']:
    # for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
        epe_list = []
        epe_occ_list = []
        epe_noc_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _, occ, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            epe_noc_list.append(epe[~occ].numpy())
            epe_occ_list.append(epe[occ].numpy())

        epe_all = np.concatenate(epe_list)

        epe_noc = np.concatenate(epe_noc_list)
        epe_occ = np.concatenate(epe_occ_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        epe_occ_mean = np.mean(epe_occ)
        epe_noc_mean = np.mean(epe_noc)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Occ epe: %f, Noc epe: %f" % (epe_occ_mean, epe_noc_mean))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def separate_inout_sintel_occ():
    """ Peform validation using the Sintel (train) split """
    dstype = 'clean'
    val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
    # coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    # coords = torch.stack(coords[::-1], dim=0).float()
    # return coords[None].expand(batch, -1, -1, -1)

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, occ, occ_path = val_dataset[val_id]
        _, h, w = image1.size()
        coords = torch.meshgrid(torch.arange(h), torch.arange(w))
        coords = torch.stack(coords[::-1], dim=0).float()

        coords_img_2 = coords + flow_gt
        out_of_frame = (coords_img_2[0] < 0) | (coords_img_2[0] > w) | (coords_img_2[1] < 0) | (coords_img_2[1] > h)
        occ_union = out_of_frame | occ
        in_frame = occ_union ^ out_of_frame

        # Generate union of occlusions and out of frame
        # path_list = occ_path.split('/')
        # path_list[-3] = 'occ_plus_out'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, occ_union.int().numpy() * 255)

        # Generate out-of-frame
        # path_list = occ_path.split('/')
        # path_list[-3] = 'out_of_frame'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, out_of_frame.int().numpy() * 255)

        # # Generate in-frame occlusions
        # path_list = occ_path.split('/')
        # path_list[-3] = 'in_frame_occ'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, in_frame.int().numpy() * 255)



@torch.no_grad()
def validate_kitti(model, iters=6, step=None):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = core.datasets.KITTI(split='training')

    out_list, epe_list = [], []
    sparsity_list = []
    flops_l = []
    for i in range(iters + 2): # 26.606830848
        flops = 63.889053696 + 32.079670272 + 0.240254976 + 23.357288448 * i + 3.2495424
        flops_l.append(flops)
    # tgt_sparsity_tmp =  0.595`
    # tgt_sparsity_tmp = 0.63
    tgt_sparsity_tmp = 0.96 # training-low
    # tgt_sparsity_tmp = 0.6 # training-high # high
    # tgt_sparsity_tmp = 0.37  # training-high # low
    # tgt_sparsity_tmp = 0.48  # training-high # medium
    # tgt_sparsity_tmp = 0.678  # 380
    # tgt_sparsity_tmp = 0.52  # 286
    # tgt_sparsity_tmp = 0.6  # testing
    # tgt_sparsity_tmp = 0.76
    # tgt_sparsity_tmp = 0.8
    # tgt_sparsity_tmp = 0.75
    # tgt_sparsity_tmp = 0.65
    # tgt_sparsity_tmp = 0.55
    # tgt_sparsity_tmp = 0.5
    f = open('./flops_kitti_' + str(tgt_sparsity_tmp) + '.txt', 'w')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        bs, _, h, w = image1.shape
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h//8, w//8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        flow_low, flow_pr, sparsity= model(image1, image2, iters=iters, test_mode=True, tgt_sparsity = tgt_sparsity, mhidden=mhidden)
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
        # 26.606830848
        # 63.889053696 + 32.079670272 + 20.684906424 + 20.684906424 * 23 * 0.622826 + 2.197972296 + 23 * 0.186350112
        flops_tmp = 63.889053696 + 32.079670272 + 0.240254976 + 23.357288448 + 23.357288448 * 23 * sparsity_tmp + 3.2495424 + 23 * 0.175249535
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
            cv2.imwrite(save_path+str(val_id).zfill(4)+'_image1.png', image1[0])
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
            # print('sparsity', sparsity.shape)
            sparsity_0 = sparsity[:, :, :, :3].clone().view(bs, c, h, 3 * w)
            sparsity_1 = sparsity[:, :, :, 3:6].clone().view(bs, c, h, 3 * w)
            sparsity_2 = sparsity[:, :, :, 6:9].clone().view(bs, c, h, 3 * w)
            sparsity_3 = sparsity[:, :, :, 9:].clone().view(bs, c, h, 2 * w)
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
    flop = 63.889053696 + 32.079670272 + 0.240254976 + 23.357288448 + 23.357288448 * 23 * sparsity + 3.2495424 + 23 * 0.175249535

    print("Validation KITTI: %f(epe), %(f1), %f(sparsity), flop %f" % (epe, f1, sparsity, flop))

    # print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti_epe': epe, 'kitti_f1': f1, 'kitti-sparsity': sparsity}

@torch.no_grad()
def validate_kitti_time(model, iters=6, step=None):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = core.datasets.KITTI(split='training')

    out_list, epe_list = [], []

    # tgt_sparsity_tmp = 0.76
    tgt_sparsity_tmp = 0.96 # training-low
    # tgt_sparsity_tmp = 0.6  # testing
    # tgt_sparsity_tmp = 0.75
    # tgt_sparsity_tmp = 0.65
    # tgt_sparsity_tmp = 0.6 # training-high
    # tgt_sparsity_tmp = 0.5
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
        mhidden = torch.zeros(bs, (128 + 4 + 7) // 8, h//8, w//8).cuda()
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        if val_id == 0:
            warm_id = 0
            while (warm_id < 300):
                _ = model.forward_time(image1, image2, iters=iters, test_mode=True, tgt_sparsity = tgt_sparsity, mhidden=mhidden)
                warm_id = warm_id + 1
        starter.record()
        flow_pr= model.forward_time(image1, image2, iters=iters, test_mode=True, tgt_sparsity = tgt_sparsity, mhidden=mhidden)
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
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    args = parser.parse_args()

    if args.dataset == 'separate':
        separate_inout_sintel_occ()
        sys.exit()

    model = torch.nn.DataParallel(RAFTGMA(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model, warm_start=True)
    # create_sintel_submission_time(model, warm_start=True)
    # create_sintel_submission_vis(model, warm_start=True)
    # create_kitti_submission(model)
    # create_kitti_submission_time(model)
    # create_kitti_submission_vis(model)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module, iters=args.iters)

        elif args.dataset == 'things':
            validate_things(model.module, iters=args.iters)

        elif args.dataset == 'sintel':
            # validate_sintel(model.module, iters=args.iters)
            validate_sintel(model.module, iters=32)
            # validate_sintel_time(model.module, iters=args.iters)
            # validate_sintel_time(model.module, iters=32)
        elif args.dataset == 'sintel_occ':
            validate_sintel_occ(model.module, iters=args.iters)

        elif args.dataset == 'kitti':
            # validate_kitti(model.module, iters=args.iters)
            # validate_kitti_time(model.module, iters=args.iters)
            # validate_kitti_time(model.module, iters=24)
            validate_kitti(model.module, iters=24)