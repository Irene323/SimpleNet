import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class NetDataset(Dataset):
    def __init__(self):
        self.patch_dir = 'D:\\dataset\\liberty\\'
        self.i = 0
        self.patches = []
        with open('../resources/merge_200000_rnd.txt', 'r') as f:
            for l in f:
                a = l.strip('\n').split(' ')
                line = [a[0], a[1], a[2]]
                self.patches.append(line)
        # print(json.dumps(patches,indent=4))
        # print(np.array(self.patches))

    def __getitem__(self, item):
        # print(self.patches[self.i])
        if self.i == len(self.patches):
            self.i = 0
        imageID0 = int(int(self.patches[self.i][0]) / 256)  # >> 8
        patchID0 = int(self.patches[self.i][0]) % 256  # & b'11111111
        image_dir0 = self.patch_dir + ('patches%04d.bmp' % (imageID0))

        imageID1 = int(int(self.patches[self.i][1]) / 256)
        patchID1 = int(self.patches[self.i][1]) % 256
        image_dir1 = self.patch_dir + ('patches%04d.bmp' % (imageID1))

        imageID2 = int(int(self.patches[self.i][2]) / 256)
        patchID2 = int(self.patches[self.i][2]) % 256
        image_dir2 = self.patch_dir + ('patches%04d.bmp' % (imageID2))

        self.i += 1

        img0 = cv2.imread(image_dir0)
        img0 = img0[(int(patchID0 / 16) * 64):(int(patchID0 / 16) * 64 + 64),
               ((patchID0 % 16) * 64):((patchID0 % 16) * 64 + 64)]
        # cv2.namedWindow("image0")
        # cv2.imshow("image0", img0)
        # cv2.waitKey(0)

        img1 = cv2.imread(image_dir1)
        img1 = img1[(int(patchID1 / 16) * 64):(int(patchID1 / 16) * 64 + 64),
               ((patchID1 % 16) * 64):((patchID1 % 16) * 64 + 64)]
        # cv2.namedWindow("image1")
        # cv2.imshow("image1", img1)
        # cv2.waitKey(0)

        img2 = cv2.imread(image_dir2)
        img2 = img2[(int(patchID2 / 16) * 64):(int(patchID2 / 16) * 64 + 64),
               ((patchID2 % 16) * 64):((patchID2 % 16) * 64 + 64)]

        img0l = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow("image0l")
        # cv2.imshow("image0l", img0l)
        # cv2.waitKey(0)
        img0l = np.array(img0l, dtype=np.float32)

        img1l = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow("image1l")
        # cv2.imshow("image1l", img1l)
        # cv2.waitKey(0)
        img1l = np.array(img1l, dtype=np.float32)

        img2l = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow("image2l")
        # cv2.imshow("image2l", img2l)
        # cv2.waitKey(0)
        img2l = np.array(img2l, dtype=np.float32)

        npimg0l = img0l.reshape((1, 64, 64))  # xq
        npimg1l = img1l.reshape((1, 64, 64))  # more batches 16/32
        npimg2l = img2l.reshape((1, 64, 64))

        return npimg0l, npimg1l, npimg2l

    def __len__(self):
        # print(len(self.patches))
        return len(self.patches)  # all the matches and randomly choose 2 times nonmatch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # 64 * 64 initially gray
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2, padding=0),
            # filter 29 * 29 smaller kernel, fewer stride
            # nn.MaxPool2d(1)# pooling Reduce the amount of calculation, and will reduce the accuracy
            nn.BatchNorm2d(32),  # Andreas: don't need it
            nn.ReLU()  # non-linear
        )
        self.conv2 = nn.Sequential(  # 29 * 29
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6, stride=3, padding=0),  # 8 * 8
            # nn.MaxPool2d(1)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=4, padding=0),  # 1 * 1
            # nn.MaxPool2d(1)
            # nn.BatchNorm1d(128)
            # nn.ReLU() # Andreas: tanh on the last layer
        )

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # * 100
        output = x.view(x.size()[0], -1)
        return output
        # # With arctan on the last layer to do normalization
        # x = self.conv3(x)
        # x = torch.atan(x / 128 * np.pi) / np.pi * 128.0
        # x = x.view(x.size()[0], -1)
        # return x

    # Contrastive Loss
    # def forward(self, input1, input2):
    #     output1 = self.forward_once(input1)
    #     output2 = self.forward_once(input2)
    #     return output1, output2

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1000.0):
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
    net_dataset = NetDataset()
    train_dataloader = DataLoader(net_dataset, shuffle=True)
    net = Net().cuda()
    criterion = TripletLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    counter = []
    loss_history = []
    iteration_number = 0
    loss = 0.0

    for epoch in range(0, 30):
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
            loss += loss_triplet
            loss_triplet.backward()
            optimizer.step()
            if i % 125 == 124:
                iteration_number += 125
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss / (i + 1)))
                counter.append(iteration_number)
                loss_history.append(loss / (i + 1))

plt.plot(counter,loss_history)
plt.show()
torch.save(net.state_dict(), '../out/margin_1000_merge_200000_rnd_epoch30.pkl')
