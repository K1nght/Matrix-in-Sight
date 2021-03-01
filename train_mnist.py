import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
torch.manual_seed(0)


# 定义自己的卷积神经网络
class HZH(nn.Module):
    def __init__(self):
        super(HZH, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)  # input 1 x 28 x 28; output 16 x 24 x 24
        self.pool = nn.MaxPool2d(2, 2)  # input 16 x 24 x 24; output 16 x 12 x 12
        self.conv2 = nn.Conv2d(16, 16, 5)  # input 16 x 12 x 12; output 16 x 8 x 8
        self.fc1 = nn.Linear(16 * 4 * 4, 128)  # input 16 x 4 x 4;  output 128
        self.fc2 = nn.Linear(128, 64)  # input 128;         output 64
        self.fc3 = nn.Linear(64, 10)  # input 64;          output 10
        self.dropout = nn.Dropout(0.1)  # 在训练过程中使用dropout防止过拟合

    def forward(self, x):
        # 每层卷积后都会经过relu激活函数以及池化操作
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 将特征图拉成向量传输给全连接层
        x = x.view(-1, 16 * 4 * 4)
        # 使用relue作为全连接层的激活函数
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 加入dropout层防止过拟合
        x = F.relu(self.fc2(x))
        # 最后一层全连接层不通过relu激活函数
        x = self.fc3(x)
        return x

    def predict(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 将特征图拉成向量传输给全连接层
        x = x.view(-1, 16 * 4 * 4)
        # 使用relue作为全连接层的激活函数
        x = F.relu(self.fc1(x))
        # 预测过程中不使用dropout
        x = F.relu(self.fc2(x))
        # 最后一层全连接层不通过relu激活函数
        x = self.fc3(x)

        _, pred = torch.max(x.data, 1)
        return pred


class training_process:
    def __init__(self, train_loader, test_loader, model, criterion, optimizer,
                 epoch=10, report_freq=2000):
        """
            train_loader: 训练数据Dataloader
            test_loader: 测试数据Dataloader
            model: 使用的模型
            criterion: 损失函数
            optimizer: 优化方法
            epoch: 训练轮数
            report_freq: 每report_freq个batch会对train_loss和test_loss进行一次check
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_size = len(train_loader)
        self.test_size = len(test_loader)
        self.epoch = epoch
        self.report_freq = report_freq
        self.loss_train = []
        self.loss_test = []
        self.acc_train = []
        self.acc_test = []

    # 清空已有的训练过程
    def clear(self):
        self.loss_train = []
        self.loss_test = []
        self.acc_train = []
        self.acc_test = []

    # 输出训练过程数据
    def checkpoint(self):
        return self.loss_train, self.loss_test, self.acc_train, self.acc_test

    # 定义训练过程
    def train(self):
        for epoch in tqdm(range(self.epoch)):
            train_loss_sum = 0.0
            for i, train_data in enumerate(self.train_loader):
                # 加载一个batch的数据
                train_inputs, train_labels = train_data
                # train_inputs, train_labels = train_inputs.cuda(), train_labels.cuda()
                # 清优化器梯度
                self.optimizer.zero_grad()
                # 正向传播计算误差
                train_outputs = self.model(train_inputs)
                train_loss = self.criterion(train_outputs, train_labels)
                # 反向传播计算梯度
                train_loss.backward()
                # 优化器更新模型参数
                self.optimizer.step()

                train_loss_sum += train_loss.item()  # 添加训练误差
                if i % self.report_freq == self.report_freq - 1:
                    self.loss_train.append(train_loss_sum / self.report_freq)
                    # 计算测试集误差
                    test_loss_sum = 0.0
                    with torch.no_grad():
                        for test_data in self.test_loader:
                            test_inputs, test_labels = test_data
                            # test_inputs, test_labels = test_inputs.cuda(), test_labels.cuda()
                            test_outputs = self.model(test_inputs)
                            test_loss_sum += self.criterion(test_outputs, test_labels)
                    self.loss_test.append(test_loss_sum / self.test_size)

                    print('[%d, %5d]|training loss: %.6f| testing loss: %.6f' %
                          (epoch + 1, i + 1, train_loss_sum / self.report_freq, test_loss_sum / self.test_size))
                    train_loss_sum = 0.0

            # 计算在训练集以及测试集上的精度
            train_correct = 0
            train_total = 0
            test_total = 0.0
            test_correct = 0
            with torch.no_grad():
                for train_data in self.train_loader:
                    train_inputs, train_labels = train_data
                    train_predicted = self.model.predict(train_inputs)
                    train_total += train_labels.size(0)
                    train_correct += (train_predicted == train_labels).sum().item()
                for test_data in self.test_loader:
                    test_inputs, test_labels = test_data
                    test_predicted = self.model.predict(test_inputs)
                    test_total += test_labels.size(0)
                    test_correct += (test_predicted == test_labels).sum().item()

            self.acc_train.append(train_correct / train_total * 100)
            self.acc_test.append(test_correct / test_total * 100)
            print('Accuracy of the network on the train data: %.3f %%' % (100 * train_correct / train_total))
            print('Accuracy of the network on the test data: %.3f %%' % (100 * test_correct / test_total))


        print('Finished Training')

    # 作出四张图，分别是训练集误差曲线、测试集误差曲线、训练集精度曲线、测试集精度曲线
    def plot_loss_and_acc(self, figname):
        fig = plt.figure(figsize=(10, 16))
        # 训练集误差曲线
        ax1 = fig.add_subplot(211)
        ax1.plot(np.linspace(self.report_freq, self.epoch * 14000, len(self.loss_train), endpoint=True),
                 np.array(self.loss_train))
        ax1.plot(np.linspace(self.report_freq, self.epoch * 14000, len(self.loss_test), endpoint=True),
                 np.array(self.loss_test))
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss of the network on the training and testing data')
        ax1.legend(['Train', 'Test'])
        ax1.grid(True)

        # 训练集和测试集精度曲线
        ax2 = fig.add_subplot(212)
        ax2.plot(np.arange(self.epoch) + 1, np.array(self.acc_train))
        ax2.plot(np.arange(self.epoch) + 1, np.array(self.acc_test))
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy of the network on the training and testing data')
        ax2.legend(['Train', 'Test'])
        ax2.grid(True)

        plt.savefig('./img/%s_loss_acc.png' % figname)
        plt.show()


# 首先通过torchvision.datasets.MNIST直接导入MNIST数据集
train_set = torchvision.datasets.MNIST(root='./data',
                                      train=True, download=False)# , transform=transform

# 查看数据集大小维度以及计算训练数据的 均值 方差
print("The size of the dataset is ", list(train_set.train_data.size()))
train_mean = train_set.train_data.float().mean()/255
train_std = train_set.train_data.float().std()/255
print("The mean and the std of the training dataset is respectively %.4f and %.4f." % (train_mean, train_std))

# 构建预训练transform函数
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([train_mean],[train_std])])

# 将训练数据和测试数据重新通过transform后导入
train_set = torchvision.datasets.MNIST(root='./data',
                                      train=True, download=False, transform=transform)#
print("Train dataset:\n", train_set)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                          shuffle=True, num_workers=0)
test_set = torchvision.datasets.MNIST(root='./data',
                                     train=False, download=True, transform=transform)#
print("Test dataset:\n", test_set)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                         shuffle=True, num_workers=0)
# 定义数据集中每个label对应的类名
classes = ('0', '1', '2', '3', '4', '5', '6' ,'7', '8', '9')

hzh = HZH()  # 实例化CNN网络
hzh = hzh
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(hzh.parameters(), lr=0.0003)  # 使用Adam优化器
my_training = training_process(train_loader=train_loader, test_loader=test_loader,
                               model=hzh, criterion=criterion, optimizer=optimizer, epoch=10, report_freq=2000)
my_training.train()  # 开始训练
save_model_path = os.path.join('checkpoint', 'mnist-last.pth')  # 保存训练模型
torch.save(hzh, save_model_path)

