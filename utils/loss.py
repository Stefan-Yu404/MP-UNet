"""
The shape of pred and target must be the same during calculation, that is, target is the onehot encoded Tensor.
        True_Positives = (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1).sum(dim=1)
        True_Negatives = ((1 - pred[:, i]) * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1)
        False_Positives = ((1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1)
        False_Negatives = (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1)

        TverskyLoss, DiceLoss
"""

import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1
        Tversky = 0.
        a = 0.3
        b = 0.7
        bce_weight = 1.0

        # the definition of Tversky: TP / (TP + a * FN + b * FP)
        for i in range(pred.size(1)):
            TP = (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1)
            FP = ((1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1)
            FN = (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1)

            Tversky += TP / (TP + a * FN + b * FP + smooth)

        Tversky = Tversky / pred.size(1)
        TverskyLoss = torch.clamp((1 - Tversky).mean(), 0, 1)

        sigmoid = torch.nn.Sigmoid()
        out = sigmoid(pred)
        BinaryCrossEnergyLoss = torch.nn.BCELoss()
        BECLoss = BinaryCrossEnergyLoss(out, target)

        return TverskyLoss + bce_weight * BECLoss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1
        dice = 0.

        # the definition of dice: 2 * TP / (2 * TP + FP + FN)
        for i in range(pred.size(1)):
            TP = (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1)
            FP = ((1 - pred[:, i]) * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1)
            FN = (pred[:, i] * (1 - target[:, i])).sum(dim=1).sum(dim=1).sum(dim=1)

            dice += 2 * TP / (2 * TP + FP + FN + smooth)

        dice = dice / pred.size(1)
        diceloss = torch.clamp((1 - dice).mean(), 0, 1)

        return diceloss
