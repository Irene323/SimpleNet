# import cv2
#
# num = 141855
# imageID0 = int(int(num) / 256)
# patchID0 = int(num) % 256
# image_dir0 = 'D:\\dataset\\liberty\\' + ('patches%04d.bmp' % (imageID0))
# img = cv2.imread(image_dir0)
# img = img[(int(patchID0 / 16) * 64):(int(patchID0 / 16) * 64 + 64), ((patchID0 % 16) * 64):((patchID0 % 16) * 64 + 64)]
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.SIFT()
# kp = sift.detect(gray, None)
# print(kp)
#
# img = cv2.drawKeypoints(gray, kp)
#
# cv2.imwrite('sift_keypoints.jpg', img)

# import torch
# import torch.nn.functional as F
# import numpy as np
#
# input1 = torch.tensor([[1., 2.], [3., 4.]])
# print(input1)
# input2 = torch.tensor([[0., 3.], [5., 1.]])
# print(input2)
# euclidean_distance = F.pairwise_distance(input1, input2)
# print(euclidean_distance)
# print(input1-input2)
# print((input1 - input2).pow(2))
# print((input1 - input2).pow(2).sum(axis=1, dim=[2,1], keepdim=True, dtype=np.float32))
# distance_positive = (input1 - input2).pow(2).sum(1).pow(0.5)
# print(distance_positive)

write_dir = '../out/siamese_margin{}/siamese_margin{}_200000_train_epoch{}.pkl'
print(write_dir.rstrip('epoch{}.pkl'))