#9620.2129 seconds,94.7%
#1071.6280 seconds,92.4%



import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from time import *





class AlexNet(nn.Module):
    def __init__(self, num_classes=3, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):# 若是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # 用（何）kaiming_normal_法初始化权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)# 初始化偏重为0
            elif isinstance(m, nn.Linear): # 若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)# 正态分布初始化
                nn.init.constant_(m.bias, 0)# 初始化偏重为0



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #这里预处理没有减去均值，如是在用imagenet训练好的VGG上做迁移学习，则需要减去均值
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(),"")) # get data root path
    image_path = os.path.join(data_root, "data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    froth_list = train_dataset.class_to_idx#类别名与数字类别的映射关系字典（class_to_idx）
    cla_dict = dict((val, key) for key, val in froth_list.items())
    # 形成索引与类别名称对应
    json_str = json.dumps(cla_dict, indent=3)#json.dumps将一个Python数据结构转换为JSON
    with open('CNNalexnet_froth_class.json', 'w') as json_file:
        json_file.write(json_str)


    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    classes = ('medium', 'high', 'low')

    print("using {} images for training, {} images fot validation.".format(train_num,
                                                                           val_num))
    test_data_iter = iter(validate_loader)
    test_image, test_label = test_data_iter.next()

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    print(' '.join('%5s' % classes[test_label[j].item()] for j in range(4)))
    imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=3, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 15
    save_path = './CNNalexnet_froth_netparams.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    begin_time = time()
    plot_loss = []
    plot_iter = []
    plot_accu = []
    plot_epo = []
    iterr = 0
    begin_time = time()
    for epoch in range(epochs):
        # # train
        net.train()
        # #AlexNet会使用dropout随机失活神经元，但只想在训练时失活，因此用net.train()、net.eval()管理
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            iterr = iterr + 1
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            plot_loss.append(loss.item())
            plot_iter.append(iterr)
            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, colour='green')
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        plot_accu.append(val_accurate)
        plot_epo.append(epoch)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    fig1 = plt.figure()
    plt.subplot(111)
    plt.plot(plot_iter, plot_loss, color='r', linestyle='-')
    plt.show()
    plt.savefig("CNNalexnet_froth.jpg")
    plt.subplot(111)
    plt.plot(plot_iter, plot_loss, color='r', linestyle='-')
    plt.show()
    plt.savefig("CNNalexnet_froth2.jpg")
    end_time = time()
    cost_time = end_time - begin_time
    print("Finished Traning,cost %.4f seconds" % (cost_time))



if __name__ == '__main__':
    main()
