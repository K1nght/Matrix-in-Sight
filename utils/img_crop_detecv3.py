import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, to_pil_image
from det_numbers.decode import *

# from keras.models import load_model

# # 定义自己的卷积神经网络
# class HZH(nn.Module):
#     def __init__(self):
#         super(HZH, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, 5)  # input 1 x 28 x 28; output 16 x 24 x 24
#         self.pool = nn.MaxPool2d(2, 2)  # input 16 x 24 x 24; output 16 x 12 x 12
#         self.conv2 = nn.Conv2d(16, 16, 5)  # input 16 x 12 x 12; output 16 x 8 x 8
#         self.fc1 = nn.Linear(16 * 4 * 4, 128)  # input 16 x 4 x 4;  output 128
#         self.fc2 = nn.Linear(128, 64)  # input 128;         output 64
#         self.fc3 = nn.Linear(64, 10)  # input 64;          output 10
#         self.dropout = nn.Dropout(0.1)  # 在训练过程中使用dropout防止过拟合
#
#     def forward(self, x):
#         # 每层卷积后都会经过relu激活函数以及池化操作
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         # 将特征图拉成向量传输给全连接层
#         x = x.view(-1, 16 * 4 * 4)
#         # 使用relue作为全连接层的激活函数
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)  # 加入dropout层防止过拟合
#         x = F.relu(self.fc2(x))
#         # 最后一层全连接层不通过relu激活函数
#         x = self.fc3(x)
#         return x
#
#     def predict(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         # 将特征图拉成向量传输给全连接层
#         x = x.view(-1, 16 * 4 * 4)
#         # 使用relue作为全连接层的激活函数
#         x = F.relu(self.fc1(x))
#         # 预测过程中不使用dropout
#         x = F.relu(self.fc2(x))
#         # 最后一层全连接层不通过relu激活函数
#         x = self.fc3(x)
#
#         return x

class HZH(nn.Module):
    def __init__(self):
        super(HZH, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)  # input 1 x 28 x 28; output 64 x 24 x 24
        self.pool = nn.MaxPool2d(2, 2)  # input 64 x 24 x 24; output 64 x 12 x 12
        # self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(64, 128, 5)  # input 64 x 12 x 12; output 128 x 8 x 8
        # self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # input 16 x 4 x 4;  output 128
        self.fc2 = nn.Linear(256, 128)  # input 128;         output 64
        self.fc3 = nn.Linear(128, 10)  # input 64;          output 10
        self.dropout = nn.Dropout(0.2)  # 在训练过程中使用dropout防止过拟合

    def forward(self, x):
        # 每层卷积后都会经过relu激活函数以及池化操作
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        # 将特征图拉成向量传输给全连接层
        x = x.view(-1, 128 * 4 * 4)
        # 使用relue作为全连接层的激活函数
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)  # 加入dropout层防止过拟合
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # 最后一层全连接层不通过relu激活函数
        x = self.fc3(x)
        return x

    def predict(self, x):
        # 每层卷积后都会经过relu激活函数以及池化操作
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        # 将特征图拉成向量传输给全连接层
        x = x.view(-1, 128 * 4 * 4)
        # 使用relue作为全连接层的激活函数
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)  # 加入dropout层防止过拟合
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # 最后一层全连接层不通过relu激活函数
        x = self.fc3(x)

        # _, pred = torch.max(x.data, 1)
        return x


def img_proce(img, xmin, ymin, xmax, ymax,  model, mnist=False):

    if mnist:
        width_new = 28
        height_new = 28
        x_pad = 0
        y_pad = 0
    else:
        width_new = 192
        height_new = 64
        x_pad = 5
        y_pad = 3

    xmin = xmin-x_pad
    xmax = xmax+x_pad
    ymin = ymin-y_pad
    ymax = ymax+y_pad

    image = img[ymin:ymax, xmin:xmax]


    height, width = image.shape[0], image.shape[1]
    # print(height, width)

    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))

    height_nxt, width_nxt = img_new.shape[0], img_new.shape[1]
    if height_nxt == height_new:
        top = 0
        bottom = 0
        if mnist:
            left = int((width_new-width_nxt)/2)
        else:
            left = 5
        right = width_new - width_nxt - left
        # print(right, left)
    else:
        top = int((height_new - height_nxt) / 2)
        bottom = height_new - height_nxt - top
        right = 0
        left = 0

    if mnist==True:
        gray_img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)#np.mean(gray_img_new) - 10
        # print(gray_img_new)
        ret, thresh = cv2.threshold(gray_img_new, np.mean(gray_img_new)-10, 255, cv2.THRESH_TOZERO_INV)
        for m in range(thresh.shape[0]):
            for n in range(thresh.shape[1]):
                if thresh[m][n] == 0:
                    thresh[m][n] = 255;
        # print('aaa', thresh)
        img_pad = cv2.copyMakeBorder(thresh, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img_pre = 255 - img_pad
        img_pre = img_pre.reshape(1, 28, 28, 1) / 255.0
        img_pre = np.transpose(img_pre, (0, 3, 1, 2))

        model = model.double()
        tens_img = torch.from_numpy(img_pre)
        pre_number = np.argmax(model.predict(tens_img.cuda()).cpu().detach().numpy())

        return pre_number

    else:

        img_pad = cv2.copyMakeBorder(img_new, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img_pre = to_tensor(img_pad)

        output = model(img_pre.unsqueeze(0).cuda())
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)

        pred_str = decode(output_argmax[0])
        return pred_str




if __name__ == '__main__':

    img_path = './det-12-26/86/86.jpg'
    txt_path = './det-12-26/86/86.txt'

    txtFile = open(txt_path)
    txtList = txtFile.readlines()
    img = cv2.imread(img_path)

    model_path = 'num-last.pth'
    model = torch.load(model_path)

    for oneline in txtList:
        idx, index, label, xmin, ymin, xmax, ymax, ratio = oneline.split(' ')
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # print(xmin, ymin, xmax, ymax)
        x_pad = 0
        y_pad = 0
        xmin = xmin-x_pad
        xmax = xmax+x_pad
        ymin = ymin-y_pad
        ymax = ymax+y_pad
        if 'number' in label:
            pre = img_proce(img, xmin, ymin, xmax, ymax, model)
            print(pre)


    # print(img)

    # set_h = 64
    # set_w = 192
    # white_bg = np.ones((set_h, set_w, 3), dtype=np.uint8)*255
    # print(white_bg)
    #
    # cv2.imshow('test', white_bg)
    # cv2.waitKey(0)



