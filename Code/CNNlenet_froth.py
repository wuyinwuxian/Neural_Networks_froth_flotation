#!/usr/bin/env python
# coding: utf-8

#Test Accuracy: 90.71%
#Finished Traning,cost 1985.3693 seconds


import gzip, struct
import numpy as np
import torch.utils.data as data
import os
from torchvision import transforms, datasets, utils
from tqdm import tqdm
from time import *


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 冲突的libiomp5md.dll均改名为exlibiomp5md.dll

import torch


sizes=32
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(sizes),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((sizes, sizes)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


data_root = os.path.abspath(os.path.join(os.getcwd(),""))   # get data root path
image_path = os.path.join(data_root, "data")  # flower data set path
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
print(22)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)
batch_size = 32
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])  # number of workers

print('Using {} dataloader workers every process'.format(nw))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=4, shuffle=False,
                                              num_workers=nw)

classes = ('medium','high','low')


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

test_data_iter = iter(test_loader)
X, y = test_data_iter.next()

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

print(' '.join('%5s' % classes[y[j].item()] for j in range(4)))
imshow(utils.make_grid(X))

from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self,siz):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * siz * siz, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x,siz):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * siz * siz)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def calculate(sizes):
    a=sizes-4
    a=a*0.5
    a=a-4
    a = a * 0.5
    a=int(a)
    return a


siz=calculate(sizes)
net = LeNet(siz)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

net.to(device)

# In[23]:

import torch.optim as optim

criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.99))


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        import math
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))


net.apply(weight_init)

# In[24]:


def main():
    begin_time = time()
    plot_loss = []
    plot_iter = []
    plot_accu = []
    plot_epo = []
    iter = 0
    for epoch in range(15):
        total_loss = 0.0
        for batch_idx, (inputs, label) in enumerate(train_loader):
            iter=iter+1
            inputs, label = inputs.to(device), label.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(inputs,siz)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            plot_loss.append(loss.item())
            plot_iter.append(iter)
            # print statistics
            if batch_idx % 64 == 63:
                print('[{}, {:5d}] Loss: {:.3f}'.format(
                    epoch + 1, batch_idx + 1, total_loss / 64))
                total_loss = 0.0
        val_accurate=test()
        plot_accu.append(val_accurate)
        plot_epo.append(epoch)
    fig1 = plt.figure()
    plt.subplot(111)
    plt.plot(plot_iter, plot_loss, color='r', linestyle='-')
    plt.savefig("CNNlenet_froth.jpg")
    fig2 = plt.figure()
    plt.subplot(111)
    plt.plot(plot_epo, plot_accu, color='r', linestyle='-')
    plt.savefig("CNNlenet_froth2.jpg")
    end_time = time()
    cost_time = end_time - begin_time
    print("Finished Traning,cost %.4f seconds" % (cost_time))


def test():
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs, label = inputs.to(device), label.to(device)
            output = net(inputs,siz)
            # sum up batch loss
            test_loss += criterion(output, label.long()).item()
            # get the index of the max log-probability
            predict = output.data.max(1, keepdim=True)[1]
            correct += predict.eq(label.long().data.view_as(predict)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('TESTLoss: {:.3f}'.format(test_loss))
    print('\nTest Accuracy: {:.2f}%\n'.format(
        100.0 * correct / len(test_loader.dataset)))
    acc = 100.0 * correct / len(test_loader.dataset)
    return acc


main()

torch.save(net.state_dict(),'CNNlenet_froth_netparams.pth')
print('Finished!')


