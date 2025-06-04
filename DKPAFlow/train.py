from __future__ import print_function, division
import sys

sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
from KPAFlow import KPAFlow
import evaluate
import core.datasets
# from datasets import fetch_dataloader

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


# os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'
def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def sequence_loss(flow_preds, flow_predictions_wo_skip, inc_l, flow_gt, valid, sparsity, tgt_sparsity, gamma=0.8,
                  weight_sparsity=0.1, relu=nn.ReLU(), max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0
    inc_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    bs, _, _, _ = flow_preds[0].shape
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()
    # print('len(flow_preds)', len(flow_preds))
    # print('len(flow_predictions_wo_skip)', len(flow_predictions_wo_skip))
    for i in range(n_predictions - 1):
        # x != x
        # if  torch.sum(flow_preds[i]!=flow_preds[i]) != 0:
        # print(torch.sum(flow_preds[i]!=flow_preds[i]))
        # exit(0)
        inc = inc_l[i]
        abs_0 = torch.mean((flow_preds[i] - flow_gt).abs(), dim=1, keepdim=True)
        abs_1 = torch.mean((flow_predictions_wo_skip[i + 1] - flow_gt).abs(), dim=1, keepdim=True)
        diff = torch.abs(abs_0 - abs_1 - inc)[valid[:, None]]
        diff = diff.mean()

        inc_loss = inc_loss + diff

        # epe
        # epe_0 = torch.sum((flow_preds[i] - flow_gt) ** 2, dim=1).sqrt()
        # # epe_0 = epe_0.view(-1)[valid.view(-1)].mean()
        # epe_1 = torch.sum((flow_predictions_wo_skip[i+1] - flow_gt) ** 2, dim=1).sqrt()
        # # epe_1 = epe_1.view(-1)[valid.view(-1)].mean()
        # # print('valid', torch.sum(valid))
        # inc = inc_l[i]
        # # print('epe_0 - epe_1', (epe_0 - epe_1).shape)
        # # print('inc', inc.shape)
        # diff = torch.abs(epe_0 - epe_1 - inc[:, 0])[valid]
        # # print('diff0', diff, diff.shape)
        # diff = diff.mean()
        # # print('diff', diff)
        # # print('inc', inc)
        #
        # inc_loss = inc_loss + diff

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    # sparsity_loss = torch.abs(sparsity.mean() - tgt_sparsity) * weight_sparsity
    sparsity_loss = relu(sparsity.mean() - tgt_sparsity) * weight_sparsity * 1.0

    metrics = {
        '0_epe': epe.mean().item(),
        '1_1px': (epe < 1).float().mean().item(),
        '2_3px': (epe < 3).float().mean().item(),
        '3_5px': (epe < 5).float().mean().item(),
        '4_sparsity': sparsity.mean().item(),
        '5_flow_loss': flow_loss.item(),
        '6_inc_loss': inc_loss.item(),
    }

    return flow_loss + sparsity_loss + inc_loss * .1, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def write_sparsity_map(self, sparsity_map, image1, image2, flow, flow_predictions):
        if self.writer is None:
            self.writer = SummaryWriter()

        # bs, c, h, iter, w = sparsity_map.shape
        # sparsity_map = sparsity_map.view(bs, c, h, iter * w)
        # sparsity_map = (sparsity_map.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
        image1 = (image1.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
        image2 = (image2.permute(0, 2, 3, 1).detach().cpu().numpy()).astype('uint8')
        flow = flow.permute(0, 2, 3, 1).detach().cpu().numpy()

        flow_predictions = torch.stack(flow_predictions, dim=3)
        sparsity_map_tmp = torch.ones_like(flow_predictions)[:, :1, :, :-1]
        bs, c, h, iter, w = flow_predictions.shape
        flow_predictions = flow_predictions.view(bs, c, h, iter * w)
        flow_predictions = flow_predictions.permute(0, 2, 3, 1).detach().cpu().numpy()

        sparsity_map = sparsity_map_tmp * sparsity_map
        sparsity_map = sparsity_map.view(bs, 1, h, (iter - 1) * w)
        sparsity_map = (sparsity_map.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')

        for i in range(4):
            flow_tmp = flow2rgb(flow[i, :, :, :])
            flow_predictions_tmp = flow2rgb(flow_predictions[i, :, :, :])
            self.writer.add_image(str(i) + '/sparsity', sparsity_map[i], self.total_steps, dataformats='HWC')
            self.writer.add_image(str(i) + '/image1', image1[i], self.total_steps, dataformats='HWC')
            self.writer.add_image(str(i) + '/image2', image2[i], self.total_steps, dataformats='HWC')
            self.writer.add_image(str(i) + '/flow', flow_tmp, self.total_steps, dataformats='HWC')
            self.writer.add_image(str(i) + '/flow_predictions', flow_predictions_tmp, self.total_steps,
                                  dataformats='HWC')

    def close(self):
        self.writer.close()


def train(args):
    relu = nn.ReLU()
    model = nn.DataParallel(KPAFlow(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        print('here')
        print(args.restore_ckpt)
        # model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        pretrained_dict = torch.load(args.restore_ckpt)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # for p in model.parameters():
    for k, v in model.named_parameters():
        if 'iter_mask' not in k:
            v.requires_grad = False  # 固定参数
    for k, v in model.named_parameters():
        print(k, v.requires_grad)
    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = core.datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    VAL_MAP = 200
    add_noise = True

    should_keep_training = True
    num_steps_weight = args.num_steps * 0.025
    num_steps_tau = args.num_steps * 0.5
    while should_keep_training:
        mhidden = torch.zeros(args.batch_size, (128 + 4 + 7) // 8, args.image_size[0] // 8,
                              args.image_size[1] // 8).cuda()
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            # print('image', image1.device)
            # print('model', model.device)
            tau = max(1 - (total_steps / num_steps_tau), 0.4)
            bs = image1.shape[0]
            tgt_sparsity_tmp = np.random.uniform(0.2, 1.0)
            # tgt_sparsity_tmp = 0.5
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
            flow_predictions, _, sparsity, flow_predictions_wo_skip, inc_l = model(image1, image2, iters=args.iters,
                                                                                tau=tau, tgt_sparsity=tgt_sparsity,
                                                                                mhidden=mhidden)

            weight_sparsity = 5.
            # weight_sparsity = min((total_steps) / num_steps_weight, 1) * weight_sparsity
            loss, metrics = sequence_loss(flow_predictions, flow_predictions_wo_skip, inc_l, flow, valid, sparsity,
                                          tgt_sparsity_tmp, args.gamma, weight_sparsity, relu)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            # if total_steps == num_steps_weight:
            #     for k, v in model.named_parameters():
            #         v.requires_grad = True

            if total_steps % VAL_MAP == 0:
                logger.write_sparsity_map(sparsity, image1, image2, flow, flow_predictions)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module, step=total_steps))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module, step=total_steps))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module, step=total_steps))

                logger.write_dict(results)

                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='KPA', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--dataset', help="dataset for evaluation")

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)