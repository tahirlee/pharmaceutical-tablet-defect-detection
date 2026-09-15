import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.distrubute_utils import is_main_process, reduce_value
from utils.lr_methods import warmup


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017, https://arxiv.org/abs/1708.02002).

    Standard (weighted) CrossEntropyLoss rescales every sample's loss by its
    class's weight, but treats every sample within a class the same. Focal
    loss additionally down-weights samples the model is ALREADY confident
    and correct on, and concentrates gradient on hard/ambiguous samples --
    exactly the ones sitting near the decision boundary (e.g. P(defective)
    in the 0.3-0.6 range), rather than ones it's already getting right.

    alpha: optional per-class weight tensor (same role as class_weights /
           CrossEntropyLoss's `weight` -- pass compute_class_weights(...)
           output here to keep countering class imbalance on top of the
           hard-example focusing).
    gamma: focusing strength. gamma=0 reduces to plain (weighted) CE.
           Higher gamma -> more aggressively down-weights easy samples.
           2.0 is the standard default from the paper; try 1.0-3.0.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # model's predicted probability for the true class
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def build_loss_function(loss_type='ce', class_weights=None, focal_gamma=2.0, label_smoothing=0.0):
    """
    Central place to construct the training loss, so train_one_epoch doesn't
    need to know about every loss variant's constructor arguments.

    loss_type: 'ce' (weighted CrossEntropyLoss, weight=None falls back to
               plain unweighted CE) or 'focal' (FocalLoss above).
    class_weights: per-class weight tensor from compute_class_weights, or
                   None. Used as `weight`/`alpha` in either loss.
    focal_gamma: only used when loss_type == 'focal'.
    label_smoothing: only used when loss_type == 'ce'. Ignored for focal
                      loss (F.cross_entropy inside FocalLoss doesn't apply
                      it here to keep the pt/focal_term math well-defined).
    """
    if loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp=False, lr_method=None,
                     class_weights=None, loss_type='ce', focal_gamma=2.0, label_smoothing=0.0):
    model.train()
    # class_weights (if provided) is a tensor of length num_classes, already
    # moved to `device`, used to counter class imbalance / harder-class bias
    # (e.g. the model under-predicting "defective") -- see compute_class_weights
    # in train.py. loss_type selects between plain/weighted CrossEntropyLoss
    # ('ce', the previous default) and FocalLoss ('focal', see above) which
    # additionally focuses training on hard/ambiguous samples.
    loss_function = build_loss_function(
        loss_type=loss_type, class_weights=class_weights,
        focal_gamma=focal_gamma, label_smoothing=label_smoothing,
    )
    train_loss = torch.zeros(1).to(device)
    acc_num = torch.zeros(1).to(device)

    optimizer.zero_grad()

    lr_scheduler = None
    if epoch == 0 and lr_method == warmup:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup(optimizer, warmup_iters, warmup_factor)

    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    # 创建一个梯度缩放标量，以最大程度避免使用fp16进行运算时的梯度下溢
    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.amp.GradScaler(device.type, enabled=enable_amp)

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        with torch.amp.autocast(device.type, enabled=enable_amp):
            pred = model(images.to(device))
            loss = loss_function(pred, labels.to(device))

            pred_class = torch.max(pred, dim=1)[1]
            acc_num += torch.eq(pred_class, labels.to(device)).sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += reduce_value(loss, average=True).detach()

        # 在进程中打印平均loss
        if is_main_process():
            info = '[epoch{}]: learning_rate:{:.5f}'.format(
                epoch + 1,
                optimizer.param_groups[0]["lr"]
            )
            data_loader.desc = info  # tqdm 成员 desc

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    return train_loss.item() / (step + 1), acc_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证集样本个数
    num_samples = len(data_loader.dataset)
    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred_class = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred_class, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)
    val_acc = sum_num.item() / num_samples

    return val_acc