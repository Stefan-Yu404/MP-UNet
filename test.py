import torch
import torch.nn as nn
import SimpleITK as sitk
import os
import numpy as np
import config
from tqdm import tqdm
from utils import logger, common
from utils.common import to_one_hot_3d
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataset.dataset_lits_test import Test_Datasets
from models import UNet, MPUNet
from utils.metrics import DiceAverage, AccuracyAverage, PrecisionAverage, SpecificityAverage, IOUAverage, MatthewsAverage


def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    test_acc = AccuracyAverage(args.n_labels)
    test_pre = PrecisionAverage(args.n_labels)
    test_spe = SpecificityAverage(args.n_labels)
    test_iou = IOUAverage(args.n_labels)
    test_MCC = MatthewsAverage(args.n_labels)

    target = to_one_hot_3d(img_dataset.label, args.n_labels)
    
    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            data = data.to(device)
            output = model(data)
            output = nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale, 1//args.xy_down_scale, 1//args.xy_down_scale), mode='trilinear', align_corners=False)
            img_dataset.update_result(output.detach().cpu())

    pred = img_dataset.recompone_result()
    pred = torch.argmax(pred, dim=1)
    pred_img = common.to_one_hot_3d(pred, args.n_labels)

    test_dice.update(pred_img, target)
    test_acc.update(pred_img, target)
    test_pre.update(pred_img, target)
    test_spe.update(pred_img, target)
    test_iou.update(pred_img, target)
    test_MCC.update(pred_img, target)

    test_log = OrderedDict({'Test_dice_liver': test_dice.avg[1], 'Test_accuracy_liver': test_acc.avg[1],
                           'Test_precision_liver': test_pre.avg[1], 'Test_specificity_liver': test_spe.avg[1],
                        'Test_iou_liver': test_iou.avg[1], 'Test_Matthews correlation coefficient_liver': test_MCC.avg[1]})

    # test_dice = OrderedDict({'Dice_liver': test_dice.avg[1]})
    if args.n_labels == 3: test_dice.update({'Dice_tumor': test_dice.avg[2]})
    
    pred = np.asarray(pred.numpy(), dtype='uint8')

    if args.postprocess:
        pass
    pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))

    return test_log, pred


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('C:\\Users\\zyu\\Desktop\\ZY2\\Codes\\experiments\\MPUNet_final', args.save)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MPUNet(in_channel=1, out_channel=args.n_labels, training=False).to(device)
    # model = UNet(in_channel=1, out_channel=args.n_labels, training=False).to(device)

    model = model.to(device)
    ckpt = torch.load('{}/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])
    test_log = logger.Test_Logger(save_path, "test_log")

    # data info
    result_save_path = '{}/result'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    
    datasets = Test_Datasets(args.test_data_path, args=args)
    for img_dataset, file_idx in datasets:
        test_dice, pred_img = predict_one_img(model, img_dataset, args)
        test_log.update(file_idx, test_dice)
        sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-'+file_idx+'.gz'))
