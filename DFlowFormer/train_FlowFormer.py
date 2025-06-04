from __future__ import print_function, division
import sys
# sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from core import optimizer
import evaluate_FlowFormer as evaluate
import evaluate_FlowFormer_tile as evaluate_tile
import core.datasets as datasets
from core.loss import sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
from loguru import logger as loguru_logger

# from torch.utils.tensorboard import SummaryWriter
from core.utils.logger import Logger

# from core.FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer

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

#torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(cfg):
    print(cfg.log_dir)
    # exit(0)
    model = nn.DataParallel(build_flowformer(cfg))
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        print('here')
        print(cfg.restore_ckpt)
        # model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        pretrained_dict = torch.load(cfg.restore_ckpt)
        model_dict = model.state_dict()
           # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
           # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    for k, v in model.named_parameters():
        if 'iter_mask' not in k:
            v.requires_grad = False  # 固定参数
    for k, v in model.named_parameters():
        print(k, v.requires_grad)
    model.cuda()
    model.train()

    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    add_noise = False

    should_keep_training = True
    num_steps_tau = cfg.trainer.num_steps * 0.5
    relu = nn.ReLU()
    while should_keep_training:
        mhidden = torch.zeros(cfg.batch_size, (128 + 4 + 7) // 8, cfg.image_size[0] // 8,
                              cfg.image_size[1] // 8).cuda()
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if cfg.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            output = {}
            tau = max(1 - (total_steps / num_steps_tau), 0.4)
            bs = image1.shape[0]
            tgt_sparsity_tmp = np.random.uniform(0.2, 1.0)
            tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
            flow_predictions, sparsity, flow_predictions_wo_skip, inc_l = model(image1, image2, output, tau=tau, tgt_sparsity=tgt_sparsity, mhidden=mhidden)
            weight_sparsity = 5.
            loss, metrics = sequence_loss(flow_predictions, flow_predictions_wo_skip, inc_l, flow, valid, sparsity, tgt_sparsity_tmp, cfg, weight_sparsity, relu)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            metrics.update(output)
            logger.push(metrics)

            ### change evaluate to functions

            if total_steps % cfg.val_freq == cfg.val_freq - 1:
                PATH = '%s/%d_%s.pth' % (cfg.log_dir, total_steps+1, cfg.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in cfg.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module, step=total_steps))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module, step=total_steps, cfg_=cfg))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module, step=total_steps, cfg_=cfg))

                logger.write_dict(results)

                model.train()
            
            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = cfg.log_dir + '/final'
    torch.save(model.state_dict(), PATH)

    PATH = f'checkpoints/{cfg.stage}.pth'
    torch.save(model.state_dict(), PATH)

    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='flowformer', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    args = parser.parse_args()

    if args.stage == 'chairs':
        from configs.default import get_cfg
    elif args.stage == 'things':
        from configs.things import get_cfg
    elif args.stage == 'sintel':
        from configs.sintel import get_cfg
    elif args.stage == 'kitti':
        from configs.kitti import get_cfg
    elif args.stage == 'autoflow':
        from configs.autoflow import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / 'log.txt'), encoding="utf8")
    loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(cfg)
