# -*- coding: utf-8 -*-
# @Time    : 2018/10/30 16:09
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import os
import argparse
import shutil

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn

import utils
from dataloaders.dataloader import voc_dataloader
from pspnet import PSPNet
import numpy as np

import metrics

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 默认使用GPU 1
torch.backends.cudnn.benchmark = True

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def get_args():
    parser = argparse.ArgumentParser(
        """PSPNet""")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--backend", type=str, default="resnet34", help="Feature extractor")
    parser.add_argument("--max_classes", type=int, default=22, help="The number of image class")
    parser.add_argument("--alpha", type=float, default=0.4, help="The number of image class")
    parser.add_argument("--batch_size", type=int, default=32

                        , help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training "
                             "will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_path", type=str, default="/home/data/model/wangxin//VOCAug/",
                        help="the root folder of dataset")
    parser.add_argument("--pre_trained_model", type=str, default="")
    parser.add_argument("--log_path", type=str, default="tensorboard2")
    parser.add_argument("--saved_path", type=str, default="results2")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')

    args = parser.parse_args()
    return args


args = get_args()


def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")

    train_loader = voc_dataloader(args.data_path, batch_size=args.batch_size, isTrain=True)
    val_loader = voc_dataloader(args.data_path, isTrain=False)

    print("=> data loaders created.")
    return train_loader, val_loader


def main():
    global args
    # 如果有多GPU 使用多GPU训练
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        args.batch_size = args.batch_size * torch.cuda.device_count()
    else:
        print("Let's use", torch.cuda.current_device())

    if args.resume:
        assert os.path.isfile(args.resume), \
            "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(args.resume))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args)
        args.resume = True
    else:
        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ")
        model = PSPNet(n_classes=args.max_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256,
                       backend='resnet34')
        print("=> model created.")
        start_epoch = 0
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay)
        best_result = np.inf

        # create results folder, if not already exists
        output_directory = os.path.join(args.saved_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    log_path = os.path.join(args.log_path, "{}".format('vocaug'))
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    # for multi-gpu training
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    # define loss function
    weights = torch.ones(args.max_classes)
    weights[0] = 0
    seg_criterion = nn.NLLLoss2d(weight=weights.cuda()).cuda()
    cls_criterion = nn.BCEWithLogitsLoss(weight=weights.cuda()).cuda()

    # criterion = [seg_criterion, cls_criterion]

    is_best = False

    for epoch in range(start_epoch, args.epochs):
        loss = train(train_loader, model, seg_criterion, optimizer, epoch, logger)  # train for one epoch

        if loss < best_result:
            best_result = loss
            is_best = True
        else:
            is_best = False

        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best , epoch, output_directory)

        if (epoch + 1) % 10 == 0:
            validate(val_loader, model, epoch, logger)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    model.train()

    batch_num = len(train_loader)
    epoch_losses = []

    max_step = len(train_loader) * args.epochs

    for i, (image, label) in enumerate(train_loader):
        current_step = epoch * batch_num + i
        # 更新学习率
        lr = utils.update_ploy_lr(optimizer, args.lr, current_step, max_step)  # power默认为0.9

        image, label = image.cuda(), label.cuda()
        torch.cuda.synchronize()
        pred, _ = model(image)

        # print('pred size = ', pred.size())
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        epoch_losses.append(loss.item())

        if (i + 1) % args.print_freq == 0:
            print('Train Epoch: {0} [{1}/{2}]\t'
                  'loss={loss:.3f}\t'
                  'avg={avg:.3f}\t'
                  'lr={lr:.6f}\t\n'
                .format(
                epoch, i + 1, len(train_loader), loss=loss.item(), avg=np.mean(epoch_losses), lr=lr))

            logger.add_scalar('Train/loss', loss.item(), current_step)
            logger.add_scalar('Train/AvgLoss', np.mean(epoch_losses), current_step)

    return np.mean(epoch_losses)


def validate(val_loader, model, epoch, logger):
    model.eval()

    accs = []
    mius = []

    for i, (image, label) in enumerate(val_loader):
        image, label = image.cuda(), label.cuda()

        torch.cuda.synchronize()

        with torch.no_grad():
            pred, _ = model(image)
            pred = torch.argmax(pred, dim=1)
            # pred = torch.squeeze(pred)

        # print('val pred size = ', pred.size())
        # print('label size = ', label.size())
        torch.cuda.synchronize()

        pred = torch.squeeze(pred)
        label = torch.squeeze(label)
        acc = metrics.pixel_accuracy(pred, label)
        miu = metrics.mean_IU(pred, label, args.max_classes)

        accs.append(acc)
        mius.append(miu)

        acc_avg = np.mean(accs)
        miu_avg = np.mean(mius)

        if (i + 1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Pixel Acc={acc:.2f}'
                  'Mean IOU={miu:.2f}'.format(
                i + 1, len(val_loader), acc=acc_avg, miu=miu_avg))

    acc_avg = np.mean(accs)
    miu_avg = np.mean(mius)

    print('\n*\n'
          'Pixel Acc={acc_avg:.2f}\n'
          'Mean IOU={miu_avg:.2f}\n'.format(
        acc_avg=acc_avg, miu_avg=miu_avg))

    logger.add_scalar('Test/Pixel Acc', acc_avg, epoch)
    logger.add_scalar('Test/Mean IoU', miu_avg, epoch)

    return acc_avg, miu_avg


if __name__ == '__main__':
    main()
