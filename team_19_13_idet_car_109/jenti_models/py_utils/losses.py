import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat

def _off_loss(off, gt_off, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_off)

    off    = off[mask]
    gt_off = gt_off[mask]

    off_loss = nn.functional.smooth_l1_loss(off, gt_off, reduction="sum")
    off_loss = off_loss / (num + 1e-4)
    return off_loss


def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)

def _focal_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class CornerNet_Focal_Loss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, off_weight=1, focal_loss=_focal_loss):
        super(CornerNet_Focal_Loss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.off_weight  = off_weight
        self.focal_loss  = focal_loss

    def forward(self, outs, targets):
        tl_heats = outs[0]
        br_heats = outs[1]

        gt_tl_heat  = targets[0]
        gt_br_heat  = targets[1]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)


        loss = focal_loss
        return loss.unsqueeze(0)

class CornerNet_Offset_Loss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, off_weight=1, focal_loss=_focal_loss):
        super(CornerNet_Offset_Loss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.off_weight  = off_weight
        self.focal_loss  = focal_loss
        self.off_loss    = _off_loss



    def forward(self, outs, targets):
        tl_heats = outs[0]
        br_heats = outs[1]
        #tl_tags  = outs[2]
        #br_tags  = outs[3]
        tl_offs  = outs[2]
        br_offs  = outs[3]

        gt_tl_heat  = targets[0]
        gt_br_heat  = targets[1]
        gt_mask     = targets[2]
        gt_tl_off   = targets[3]
        gt_br_off   = targets[4]
        gt_tl_ind   = targets[5]
        gt_br_ind   = targets[6]

        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)

        off_loss = 0
        tl_offs  = [_tranpose_and_gather_feat(tl_off, gt_tl_ind) for tl_off in tl_offs]
        br_offs  = [_tranpose_and_gather_feat(br_off, gt_br_ind) for br_off in br_offs]
        for tl_off, br_off in zip(tl_offs, br_offs):
            off_loss += self.off_loss(tl_off, gt_tl_off, gt_mask)
            off_loss += self.off_loss(br_off, gt_br_off, gt_mask)
        off_loss = self.off_weight * off_loss

        loss = (focal_loss + off_loss) / max(len(tl_heats), 1)
        return loss.unsqueeze(0)