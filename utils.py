# -*- coding: utf-8 -*-
# @Time    : 2018/11/14 21:54
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.viridis


def save_checkpoint_v2(state, is_acc_best, is_miu_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_acc_best:
        best_filename = os.path.join(output_directory, 'model_acc_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if is_miu_best:
        best_filename = os.path.join(output_directory, 'model_miou_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)


def adjust_learning_rate(optimizer, epoch, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    if epoch % 10 == 0:
        lr = lr_init / 10.0
    else:
        lr = lr_init
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# ploy策略的学习率更新
def update_ploy_lr(optimizer, initialized_lr, current_step, max_step, power=0.9):
    lr = initialized_lr * ((1 - float(current_step) / max_step) ** (power))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)
