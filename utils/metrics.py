import torch
import numpy as np
import math


class AverageMeter(object):
    """Computes and stores the average and current value for calculate average meter"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)


class DiceAverage(object):
    """Computes and stores the average and current value for calculate average dice"""
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dicelist = []
        for class_index in range(targets.size()[1]):
            TP = torch.sum((logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))
            FP = torch.sum((1 - logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))
            FN = torch.sum((logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))

            dice = (2 * TP) / (2 * TP + FP + FN + 1e-7)
            dicelist.append(dice.item())
        return np.asarray(dicelist)


class AccuracyAverage(object):
    """Computes and stores the average and current value for calculate average accuracy"""
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = AccuracyAverage.get_accuracy(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_accuracy(logits, targets):
        accuracylist = []
        for class_index in range(targets.size()[1]):
            TP = torch.sum((logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))
            TN = torch.sum((1 - logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))
            FP = torch.sum((1 - logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))
            FN = torch.sum((logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))

            accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)
            accuracylist.append(accuracy.item())
        return np.asarray(accuracylist)


class PrecisionAverage(object):
    """Computes and stores the average and current value for calculate average precision"""
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = PrecisionAverage.get_precision(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_precision(logits, targets):
        precisionlist = []
        for class_index in range(targets.size()[1]):
            TP = torch.sum((logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))
            FP = torch.sum((1 - logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))

            precision = (TP) / (TP + FP + 1e-7)
            precisionlist.append(precision.item())
        return np.asarray(precisionlist)


class SpecificityAverage(object):
    """Computes and stores the average and current value for calculate average specificity"""
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = SpecificityAverage.get_specificity(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_specificity(logits, targets):
        specificitylist = []
        for class_index in range(targets.size()[1]):
            TN = torch.sum((1 - logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))
            FP = torch.sum((1 - logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))

            specificity = (TN) / (TN + FP + 1e-7)
            specificitylist.append(specificity.item())
        return np.asarray(specificitylist)


class IOUAverage(object):
    """Computes and stores the average and current value for calculate average iou"""
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = IOUAverage.get_IOU(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_IOU(logits, targets):
        ioulist = []
        for class_index in range(targets.size()[1]):
            TP = torch.sum((logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))
            FP = torch.sum((1 - logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))
            FN = torch.sum((logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))

            iou = (TP) / (TP + FP + FN + 1e-7)
            ioulist.append(iou.item())
        return np.asarray(ioulist)


class MatthewsAverage(object):
    """Computes and stores the average and current value for calculate average MCC"""
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = MatthewsAverage.get_matthews(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_matthews(logits, targets):
        matthewslist = []
        for class_index in range(targets.size()[1]):
            TP = torch.sum((logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))
            TN = torch.sum((1 - logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))
            FP = torch.sum((1 - logits[:, class_index, :, :, :]) * (targets[:, class_index, :, :, :]))
            FN = torch.sum((logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))

            matthews = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-7)
            matthewslist.append(matthews.item())
        return np.asarray(matthewslist)
