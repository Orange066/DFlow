import torch
import torch.nn as nn
MAX_FLOW = 400


def sequence_loss(flow_preds, flow_predictions_wo_skip, inc_l, flow_gt, valid, sparsity, tgt_sparsity, cfg, weight_sparsity=0.1, relu=nn.ReLU()):
    """ Loss function defined over sequence of flow predictions """

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)    
    flow_loss = 0.0
    inc_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    bs, _ ,_, _ = flow_preds[0].shape
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

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for ii, t in enumerate(flow_gt_thresholds):
        e = epe[flow_gt_length < t]
        metrics.update({
                f"{ii+7}_{t}-th-5px": (e < 5).float().mean().item()
        })


    return flow_loss + sparsity_loss + inc_loss * .1, metrics

