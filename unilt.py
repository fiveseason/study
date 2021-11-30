# user/bin/python
# -*- coding:utf-8 -*-
import torch
from matplotlib import pyplot as plt


def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


def plot_image(img, lable, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title('{}:{}'.format(name, lable[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(lable, depth=10):
    out = torch.zeros(lable.size(0), depth)
    idx = torch.LongTensor(lable).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out
