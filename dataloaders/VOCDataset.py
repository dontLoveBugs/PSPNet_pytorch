# -*- coding: utf-8 -*-
# @Time    : 2018/10/29 20:03
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


import os
from PIL import Image
from torch.utils.data import Dataset
import dataloaders.transforms as transforms
import dataloaders.transform as transform
import torch

EXTENSIONS = ['.jpg', '.png']


def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def image_path(root, basename, extension):
    return os.path.join(root, '{basename}{extension}')


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


class VOC12(Dataset):

    def __init__(self, root, isTrain = True):
        self.images_root = os.path.join(root, 'img')
        self.labels_root = os.path.join(root, 'gt')
        self.list_root = os.path.join(root, 'list')

        # print('image root = ', self.images_root)
        # print('labels root = ', self.labels_root)

        if isTrain:
            list_path = os.path.join(self.list_root, 'train_aug.txt')
            self.input_transform = transforms.Compose([
                transforms.RandomRotation(10),      # 随机旋转
                transforms.CenterCrop(256),
                transforms.RandomHorizontalFlip(),  # 随机翻转
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            self.target_transform = transforms.Compose([
                transforms.CenterCrop(256),
                transform.ToLabel()])
        else:
            list_path = os.path.join(self.list_root, 'val.txt')
            self.input_transform = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
            self.target_transform = transforms.Compose([
                transforms.CenterCrop(256),
                transform.ToLabel()])

        self.filenames = [i_id.strip() for i_id in open(list_path)]

    def __getitem__(self, index):
        filename = self.filenames[index]
        # print('filename = ', filename)

        with open(self.images_root + "/" + str(filename) + '.jpg', 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(self.labels_root + "/" + str(filename) + '.png', 'rb') as f:
            label = load_image(f).convert('P')


        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
            label = torch.squeeze(label)
            label[label == 255] = 0

        # _label = torch.zeros(20)
        #
        # for i in range(20):
        #     # print('test:', (i+1 == label).size())
        #     if torch.sum(i+1 == label) > 0:
        #         _label[i] = 1

        # return image, _label, label

        return image, label


    def __len__(self):
        return len(self.filenames)