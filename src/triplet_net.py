import cv2
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from multiprocessing import Lock


def load_image(p, conf):
    imageID0 = int(int(p) / 256)  # >> 8
    patchID0 = int(p) % 256  # & b'11111111
    image_dir0 = conf.patch_dir + ('patches%04d.bmp' % (imageID0))
    img0 = cv2.imread(image_dir0)
    img0 = img0[(int(patchID0 / 16) * 64):(int(patchID0 / 16) * 64 + 64),
           ((patchID0 % 16) * 64):((patchID0 % 16) * 64 + 64)]
    img0l = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img0l = np.array(img0l, dtype=np.float32)
    npimg0l = img0l.reshape((1, 64, 64))
    return npimg0l


lock = Lock()


class NetDataset(Dataset):
    def __init__(self, conf):
        self.dataset = {}
        self.i = -1
        self.patches = []
        self.conf = conf
        # with open(self.conf.read_dir, 'r') as f:
        #     for l in f:
        #         a = l.strip('\n').split(' ')
        #         line = [a[0], a[1], a[2]]
        #         self.patches.append(line)
        # print(json.dumps(patches,indent=4))
        # print(np.array(self.patches))
        with open(self.conf.read_dir, 'r') as f:
            for l in f:
                if len(self.patches) >= self.conf.patch_len:
                    break
                a = l.strip('\n').split(' ')
                line = [a[0], a[1], a[2]]
                self.patches.append(line)
        # for p in self.patches:
        #     if not p[0] in self.dataset.keys():
        #         self.dataset[p[0]] = load_image(p[0],self.conf)
        #     if not p[1] in self.dataset.keys():
        #         self.dataset[p[1]] = load_image(p[1],self.conf)
        #     if not p[2] in self.dataset.keys():
        #         self.dataset[p[2]] = load_image(p[2], self.conf)
        #     print(self.dataset.__len__())

    def __getitem__(self, item):
        # print(self.patches[self.i])
        self.i += 1
        if self.i == len(self.patches):
            self.i = 0
        temp = []
        for i in range(3):
            lock.acquire()
            if not self.patches[self.i][i] in self.dataset.keys():
                self.dataset[self.patches[self.i][i]] = load_image(self.patches[self.i][i], self.conf)
            temp.append(self.dataset[self.patches[self.i][i]])
            lock.release()
        return temp[0], temp[1], temp[2]
        # imageID0 = int(int(self.patches[self.i][0]) / 256)  # >> 8
        # patchID0 = int(self.patches[self.i][0]) % 256  # & b'11111111
        # image_dir0 = self.conf.patch_dir + ('patches%04d.bmp' % (imageID0))
        #
        # imageID1 = int(int(self.patches[self.i][1]) / 256)
        # patchID1 = int(self.patches[self.i][1]) % 256
        # image_dir1 = self.conf.patch_dir + ('patches%04d.bmp' % (imageID1))
        #
        # imageID2 = int(int(self.patches[self.i][2]) / 256)
        # patchID2 = int(self.patches[self.i][2]) % 256
        # image_dir2 = self.conf.patch_dir + ('patches%04d.bmp' % (imageID2))
        #
        # img0 = cv2.imread(image_dir0)
        # img0 = img0[(int(patchID0 / 16) * 64):(int(patchID0 / 16) * 64 + 64),
        #        ((patchID0 % 16) * 64):((patchID0 % 16) * 64 + 64)]
        # # cv2.namedWindow("image0")
        # # cv2.imshow("image0", img0)
        # # cv2.waitKey(0)
        #
        # img1 = cv2.imread(image_dir1)
        # img1 = img1[(int(patchID1 / 16) * 64):(int(patchID1 / 16) * 64 + 64),
        #        ((patchID1 % 16) * 64):((patchID1 % 16) * 64 + 64)]
        # # cv2.namedWindow("image1")
        # # cv2.imshow("image1", img1)
        # # cv2.waitKey(0)
        #
        # img2 = cv2.imread(image_dir2)
        # img2 = img2[(int(patchID2 / 16) * 64):(int(patchID2 / 16) * 64 + 64),
        #        ((patchID2 % 16) * 64):((patchID2 % 16) * 64 + 64)]
        #
        # img0l = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        # # cv2.namedWindow("image0l")
        # # cv2.imshow("image0l", img0l)
        # # cv2.waitKey(0)
        # img0l = np.array(img0l, dtype=np.float32)
        #
        # img1l = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # # cv2.namedWindow("image1l")
        # # cv2.imshow("image1l", img1l)
        # # cv2.waitKey(0)
        # img1l = np.array(img1l, dtype=np.float32)
        #
        # img2l = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # # cv2.namedWindow("image2l")
        # # cv2.imshow("image2l", img2l)
        # # cv2.waitKey(0)
        # img2l = np.array(img2l, dtype=np.float32)
        #
        # npimg0l = img0l.reshape((1, 64, 64))  # xq
        # npimg1l = img1l.reshape((1, 64, 64))  # more batches 16/32
        # npimg2l = img2l.reshape((1, 64, 64))

        # return npimg0l, npimg1l, npimg2l

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


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        temp_conv = []
        temp_conv.append(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=0))
        temp_conv.append(nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=0))
        temp_conv.append(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0))
        temp_conv.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0))
        temp_conv.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0))
        temp_conv.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0))
        # for tc in temp_conv:
        #     nn.init.uniform_(tc.weight, a=0, b=0)
        #     tc.bias.data.fill_(0)

        self.conv1 = nn.Sequential(
            temp_conv[0],
            # nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            temp_conv[1],
            # nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            temp_conv[2],
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            temp_conv[3],
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            temp_conv[4],
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            temp_conv[5],
            # nn.BatchNorm2d(128),
            nn.Tanh()
        )

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3
