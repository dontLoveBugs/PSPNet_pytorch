# -*- coding: utf-8 -*-
# @Time    : 2018/10/28 22:15
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


from dataloaders.VOCDataset import VOC12
from torch.utils.data import DataLoader

def voc_dataloader(data_path, batch_size=16, isTrain=True):
    dataset = VOC12(data_path, isTrain)
    if isTrain:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(dataset, batch_size=1, shuffle=False)



if __name__ == '__main__':
    data_path = 'D:\DATASETS\VOCAug'

    train_loader = voc_dataloader(data_path, batch_size=16, isTrain=True)
    val_loader = voc_dataloader(data_path, isTrain=False)

    # for i, (input, target) in enumerate(val_loader):
    #     if i == 0:
    #         print(input.size())
    #         print(target.size())
    #         input = input.numpy()
    #         input = np.reshape(input, (3, 256, 256))
    #         input = np.transpose(input, (1, 2, 0))
    #         print(input)
    #
    #         target = target.numpy()
    #         target = np.reshape(target, (1, 256, 256))
    #         target = np.transpose(target, (1, 2, 0))
    #         print('target size = ', target.shape)
    #         print(target)
    #
    #        # plt.imshow(input)
    #         cv2.imshow('target', target)
    #         cv2.waitKey()
    #         break

    print('一次epoch迭代数：', len(train_loader))
    # for i, (input, target) in enumerate(train_loader):
    #     print(i)

