import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from triplet_net import Net, NetDataset, Net2

class Config():
    patch_dir = 'D:\\dataset\\liberty\\'
    read_dir = '../resources/merge_200000_train.txt'
    write_dir = '../out/net2_margin_2_merge_200000_epoch30.pkl'
    train_number_epochs = 30
    patch_len = 43515

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    # The distance from the baseline (anchor) input to the positive (truthy) input is minimized,
    # and the distance from the baseline (anchor) input to the negative (falsy) input is maximized.
    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        # print('p:',distance_positive)
        # print('n:',distance_negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


if __name__ == '__main__':
    net_dataset = NetDataset(Config)
    train_dataloader = DataLoader(net_dataset, shuffle=False, batch_size=16, num_workers=4)
    net = Net2().cuda()
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
            loss_triplet = criterion.forward(output1, output2, output3)
            loss += float(loss_triplet)
            loss_triplet.backward()
            optimizer.step()
            if i % 96 == 95:
                print((int)(i / 96))
            if (i == ((int)(Config.patch_len/16))): #((i % Config.patch_len) == (Config.patch_len - 1)):
                iteration_number += Config.patch_len
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss / (i + 1)))
                counter.append(iteration_number)
                loss_history.append(loss / (i + 1))
                torch.save(net.state_dict(), ('../out/net2_margin_2_merge_200000_epoch%d.pkl' % (epoch)))

    plt.plot(counter, loss_history)
    plt.show()
    torch.save(net.state_dict(), Config.write_dir)
