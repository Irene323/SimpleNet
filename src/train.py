import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from triplet_net import Net, NetDataset, Net2

class Config():
    margin = 11.3137
    patch_dir = '/home/yirenli/data/liberty/'
    read_dir = '../resources/merge_200000_train.txt'
    write_dir = '../out/net2_tanh_margin{}/net2_tanh_margin{}_merge_200000_train_epoch{}.pkl'
    pkl_dir = '../out/net2_tanh_margin11.3137/net2_tanh_margin11.3137_merge_200000_train_epoch100.pkl'
    write_bool = True
    train_number_epochs = 200
    patch_len = 43515
    batchsize = 1
    size_average = True

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=Config.margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    # The distance from the baseline (anchor) input to the positive (truthy) input is minimized,
    # and the distance from the baseline (anchor) input to the negative (falsy) input is maximized.
    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1).pow(0.5)
        distance_negative = (anchor - negative).pow(2).sum(1).pow(0.5)
        # print('p:',distance_positive)
        # print('n:',distance_negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


if __name__ == '__main__':
    net_dataset = NetDataset(Config)
    train_dataloader = DataLoader(net_dataset, shuffle=False, batch_size=Config.batchsize)#, num_workers=4)
    net = Net2().cuda()
    # net.load_state_dict(torch.load(Config.pkl_dir))
    criterion = TripletLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    counter = []
    loss_history = []
    iteration_number = 0
    loss = 0.0

    for epoch in range(0, Config.train_number_epochs):
        loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            img0, img1, img2 = data
            img0, img1, img2 = img0.cuda(), img1.cuda(), img2.cuda()

            output1, output2, output3 = net(img0, img1, img2)
            # if i % 20 == 0:
            #     print(output1)
            #     print(output2)
            #     print(output3)
            loss_triplet = criterion.forward(output1, output2, output3, size_average=Config.size_average)

            # loss_one = loss_triplet.cpu().detach().numpy().sum()
            # if np.isnan(loss_one):
            #     print('loss_one is nan')
            # else:
            #     loss += loss_one
            # # print(loss)
            # loss_triplet.sum().backward()

            # print(loss_triplet)
            loss += float(loss_triplet)
            loss_triplet.backward()

            optimizer.step()

            if i % 1000 == 0:
                print('i: {}'.format(i))

            if (i == ((int)((Config.patch_len-1)/Config.batchsize))): #((i % Config.patch_len) == (Config.patch_len - 1)):
                iteration_number += Config.patch_len
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss / Config.patch_len))
                counter.append(iteration_number)
                loss_history.append(loss / Config.patch_len)
                if Config.write_bool and (epoch%10)==0:
                    torch.save(net.state_dict(), (Config.write_dir.format(Config.margin, Config.margin, epoch)))

    plt.plot(counter, loss_history)
    # plt.show()
    plt.savefig((Config.write_dir.rstrip('epoch{}.pkl')+'loss2.jpg').format(Config.margin, Config.margin))
    torch.save(net.state_dict(), Config.write_dir.format(Config.margin, Config.margin, Config.train_number_epochs))
