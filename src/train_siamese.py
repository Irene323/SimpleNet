import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from siamese_net import Net, NetDataset, Net2

class Config():
    margin = 10
    patch_dir = '/home/yirenli/data/liberty/'
    read_dir = '../resources/siamese_200000_train.txt'
    write_dir = '../out/siamese_margin{}/siamese_margin{}_200000_train_epoch{}.pkl'
    # pkl_dir = '../out/siamese_margin100/siamese_margin100_200000_train_epoch200.pkl'
    write_bool = True
    train_number_epochs = 200
    patch_len = 195000
    batchsize = 64

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, C=Config.margin):
        super(ContrastiveLoss, self).__init__()
        self.C = C

    def forward(self, output1, output2, label):
        # print(output1.shape)
        # print(label)
        # euclidean_distance = F.pairwise_distance(output1, output2)
        # euclidean_distance = ((output1 - output2).pow(2).sum(dim=[-1,1], keepdim=True, dtype=torch.float32).pow(0.5))#.reshape(-1,1)
        euclidean_distance = (output1 - output2).pow(2).sum(1,keepdim=False).pow(0.5)
        # print('==========================================')
        # print(euclidean_distance)
        # euclidean_distance=euclidean_distance.unsqueeze(1)
        # torch.t(label)
        # print(label)
        # label=label.squeeze(1)
        # print(label)
        loss_contrastive = (1 - label) * euclidean_distance +\
                           (label) * torch.clamp(self.C - euclidean_distance, min=0.0)
        # print(loss_contrastive)
        return loss_contrastive

if __name__ == '__main__':
    net_dataset = NetDataset(Config)
    train_dataloader = DataLoader(net_dataset, shuffle=False, batch_size=Config.batchsize)#, num_workers=0)
    net = Net().cuda()
    # net.load_state_dict(torch.load(Config.pkl_dir))
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
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda().squeeze(1) # for batch_size>1

            output1, output2 = net(img0, img1)
            # if i % 20 == 0:
            #     print(output1)
            #     print(output2)
            #     print(output3)
            loss_contrastive = criterion.forward(output1, output2, label)
            # print(loss_contrastive.shape)
            # print(loss_contrastive)
            loss_one=loss_contrastive.cpu().detach().numpy().sum()
            if np.isnan(loss_one):
                print('loss_one is nan')
            else:
                loss += loss_one
            # print(loss)
            loss_contrastive.sum().backward()
            optimizer.step()
            if i % 1000 == 0:
                print('i: {}'.format(i))
            if i == ((int)(Config.patch_len/Config.batchsize)):#(Config.patch_len-1): #((i % Config.patch_len) == (Config.patch_len - 1)):
                iteration_number += Config.patch_len
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss / Config.patch_len))
                counter.append(iteration_number)
                loss_history.append(loss / Config.patch_len)
                if Config.write_bool and (epoch%10)==0:
                    torch.save(net.state_dict(), (Config.write_dir.format(Config.margin, Config.margin, epoch)))

    plt.plot(counter, loss_history)
    # plt.show()
    plt.savefig((Config.write_dir.rstrip('epoch{}.pkl')+'loss.jpg').format(Config.margin, Config.margin))
    torch.save(net.state_dict(), Config.write_dir.format(Config.margin, Config.margin, Config.train_number_epochs))
