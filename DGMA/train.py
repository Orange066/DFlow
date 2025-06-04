from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from network import RAFTGMA

from utils import flow_viz
import core.datasets
import evaluate

from torch.cuda.amp import GradScaler

# exclude extremly large displacements
MAX_FLOW = 400


def convert_flow_to_image(image1, flow):
    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)
    flow_image = cv2.resize(flow_image, (image1.shape[3], image1.shape[2]))
    return flow_image


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_loss(flow_preds, flow_predictions_wo_skip, inc_l, flow_gt, valid, sparsity, tgt_sparsity, gamma=0.8, weight_sparsity=0.1, relu=nn.ReLU(), max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    inc_loss = 0.0
    # exclude invalid pixels and extremely large displacements
    valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < MAX_FLOW)
    bs, _, _, _ = flow_preds[0].shape
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    for i in range(n_predictions-1):
        inc = inc_l[i]
        abs_0 = torch.mean((flow_preds[i] - flow_gt).abs(), dim=1, keepdim=True)
        abs_1 = torch.mean((flow_predictions_wo_skip[i+1] - flow_gt).abs(), dim=1, keepdim=True)
        diff = torch.abs(abs_0 - abs_1 - inc)[valid[:, None]]
        diff = diff.mean()

        inc_loss = inc_loss + diff

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    sparsity_loss = relu(sparsity.mean() - tgt_sparsity) * weight_sparsity * 10.0
    # print(flow_loss)
    metrics = {
        '0_epe': epe.mean().item(),
        '1_1px': (epe < 1).float().mean().item(),
        '2_3px': (epe < 3).float().mean().item(),
        '3_5px': (epe < 5).float().mean().item(),
        '4_sparsity': sparsity.mean().item(),
        '5_flow_loss': flow_loss.item(),
        '6_inc_loss': inc_loss.item(),
    }

    return flow_loss+sparsity_loss+inc_loss*1.0, metrics
    # return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        # print the training status
        print(training_str + metrics_str + time_left_hms)

        # logging running loss to total loss
        self.train_epe_list.append(np.mean(self.running_loss_dict['0_epe']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []

            self.running_loss_dict[key].append(metrics[key])

        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            self._print_training_status()
            self.running_loss_dict = {}


def main(args):

    model = nn.DataParallel(RAFTGMA(args), device_ids=args.gpus)

    print(f"Parameter Count: {count_parameters(model)}")


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

    for k, v in model.named_parameters():
        if 'iter_mask' not in k:
            v.requires_grad = False  # 固定参数
    for k, v in model.named_parameters():
        print(k, v.requires_grad)
    model.cuda()
    model.train()

    # if args.stage != 'chairs':
    #     model.module.freeze_bn()

    train_loader = core.datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args)

    while logger.total_steps <= args.num_steps:
        train(model, train_loader, optimizer, scheduler, logger, scaler, args)
        if logger.total_steps >= args.num_steps:
            plot_train(logger, args)
            plot_val(logger, args)
            break

    PATH = args.output+f'/{args.name}.pth'
    torch.save(model.state_dict(), PATH)
    return PATH


def train(model, train_loader, optimizer, scheduler, logger, scaler, args):
    mhidden = torch.zeros(args.batch_size, (128 + 4 + 7) // 8, args.image_size[0] // 8,
                          args.image_size[1] // 8).cuda()
    num_steps_tau = args.num_steps * 0.5
    relu = nn.ReLU()
    VAL_MAP = 200
    for i_batch, data_blob in enumerate(train_loader):
        tic = time.time()

        image1, image2, flow, valid = [x.cuda() for x in data_blob]

        optimizer.zero_grad()

        tau = max(1 - (logger.total_steps / num_steps_tau), 0.4)
        bs = image1.shape[0]
        tgt_sparsity_tmp = np.random.uniform(0.2, 1.0)
        tgt_sparsity = torch.tensor(tgt_sparsity_tmp).cuda().repeat(bs).view(-1, 1, 1, 1)
        flow_pred, sparsity, flow_predictions_wo_skip, inc_l = model(image1, image2, iters=args.iters, tau=tau, tgt_sparsity=tgt_sparsity, mhidden=mhidden)

        weight_sparsity = 5.
        loss, metrics = sequence_loss(flow_pred, flow_predictions_wo_skip, inc_l, flow, valid, sparsity, tgt_sparsity_tmp, args.gamma, weight_sparsity, relu)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        toc = time.time()

        metrics['time'] = toc - tic
        logger.push(metrics)

        # Validate
        if logger.total_steps % args.val_freq == args.val_freq - 1:
            validate(model, args, logger)
            plot_train(logger, args)
            plot_val(logger, args)
            PATH = args.output + f'/{logger.total_steps+1}_{args.name}.pth'
            torch.save(model.state_dict(), PATH)

        # if logger.total_steps % VAL_MAP == 0:
        #     logger.write_sparsity_map(sparsity, image1, image2, flow, flow_pred)

        if logger.total_steps >= args.num_steps:
            break


def validate(model, args, logger):
    model.eval()
    results = {}

    # Evaluate results
    for val_dataset in args.validation:
        if val_dataset == 'chairs':
            results.update(evaluate.validate_chairs(model.module, args.iters, logger.total_steps))
        elif val_dataset == 'sintel':
            results.update(evaluate.validate_sintel(model.module, args.iters, logger.total_steps))
        elif val_dataset == 'kitti':
            results.update(evaluate.validate_kitti(model.module, args.iters, logger.total_steps))

    # Record results in logger
    for key in results.keys():
        if key not in logger.val_results_dict.keys():
            logger.val_results_dict[key] = []
        logger.val_results_dict[key].append(results[key])

    logger.val_steps_list.append(logger.total_steps)
    model.train()


def plot_val(logger, args):
    for key in logger.val_results_dict.keys():
        # plot validation curve
        plt.figure()
        plt.plot(logger.val_steps_list, logger.val_results_dict[key])
        plt.xlabel('x_steps')
        plt.ylabel(key)
        plt.title(f'Results for {key} for the validation set')
        plt.savefig(args.output+f"/{key}.png", bbox_inches='tight')
        plt.close()


def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_epe_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output+"/train_epe.png", bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output', type=str, default='checkpoints', help='output directory to save checkpoints and plots')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--model_name', default='', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    main(args)
