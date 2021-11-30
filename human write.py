import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt

from unilt import plot_image, plot_curve, one_hot

batch_size = 256
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),  # 转化为tensor
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 能够将数据分布在0附近
                               ])),
    batch_size=batch_size, shuffle=True)
text_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)  # shuffle打散数据
x, y = next(iter(train_loader))
print(x.shape, y.shape)
plot_image(x, y, 'image sample')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 这里继承nn类，调用它的方法
        self.fc1 = nn.Linear(28 * 28, 256)  # 第一个是数据大小，第二个是节点数（经验决定）
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        # 每一层的定义在这里，最后输出10因为输出层的只能输出10个结果

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #  h1 = xw1 + b1
        x = F.relu(self.fc2(x))
        #  h2 = h1w2 + b2
        x = self.fc3(x)
        #  h3 = h2w3 + b3

        return x


#  创建网络对象
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

train_loss = []
for epoch in range(3):

    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(x.size(0), 28 * 28)
        out = net(x)
        y_onehot = one_hot(y)
        loss = F.mse_loss(out, y_onehot)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 5 == 0:
            print('次数：{}，loss：{}'.format(batch_idx,loss))


plot_curve(train_loss)

total_correct = 0
for x, y in text_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum.float()
    total_correct += correct

total_num = len(text_loader.dataset)
acc = total_correct / total_num
print('acc:', acc)
