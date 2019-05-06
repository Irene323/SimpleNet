import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from siamese_net import Net, NetDataset, Net2

class Config():
    margin = 11
    patch_dir = '/home/yirenli/data/liberty/'
    read_dir = '../resources/siamese_200000_train.txt'
    write_dir = '../out/siamese_net2_margin{}/siamese_net2_margin{}_200000_train_epoch50.pkl'.format(margin, margin)
    train_number_epochs = 50
    patch_len = 100000

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, C=11.3137):
        super(ContrastiveLoss, self).__init__()
        self.C = C

    def forward(self, output1, output2, label):
        # print(output1.shape)
        # print(label)
        # euclidean_distance = F.pairwise_distance(output1, output2)
        # euclidean_distance = ((output1 - output2).pow(2).sum(dim=[-1,1], keepdim=True, dtype=torch.float32).pow(0.5))#.reshape(-1,1)
        euclidean_distance = (output1 - output2).pow(2).sum(1).pow(0.5)
        loss_contrastive = (1 - label) * euclidean_distance +\
                           (label) * torch.clamp(self.C - euclidean_distance, min=0.0)
        return loss_contrastive

if __name__ == '__main__':
    net_dataset = NetDataset(Config)
    train_dataloader = DataLoader(net_dataset, shuffle=False)#, batch_size=1, num_workers=0)
    net = Net2().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    counter = []
    loss_history = []
    iteration_number = 0
    loss = 0.0

    for epoch in range(0, Config.train_number_epochs):
        loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            output1, output2 = net(img0, img1)
            # if i % 20 == 0:
            #     print(output1)
            #     print(output2)
            #     print(output3)
            loss_contrastive = criterion.forward(output1, output2, label)
            # print(loss_contrastive.shape)
            # print(loss_contrastive)
            loss += loss_contrastive#.sum()
            # print(loss)
            loss_contrastive.backward()
            optimizer.step()
            if i % 1000 == 999:
                print((int)(i / 1000))
            if i == (Config.patch_len-1):#((int)(Config.patch_len/16))): #((i % Config.patch_len) == (Config.patch_len - 1)):
                iteration_number += Config.patch_len
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss / Config.patch_len))
                counter.append(iteration_number)
                loss_history.append(loss / Config.patch_len)
                torch.save(net.state_dict(), ('../out/siamese_net2_margin{}/siamese_net2_margin{}_200000_train_epoch{}.pkl'.format(Config.margin, Config.margin, epoch)))

    plt.plot(counter, loss_history)
    # plt.show()
    plt.savefig('../out/siamese_net2_margin{}/siamese_net2_margin{}_loss.jpg'.format(Config.margin, Config.margin))
    torch.save(net.state_dict(), Config.write_dir)
