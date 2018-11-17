# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 20:11
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com

import torch

target = torch.LongTensor(3, 8, 8).random_(0, 4)

print(target.size())

a = torch.randn(4, 4)
print(a)

print(a[a > 0])

a[a < 0] = 0

print(a)