import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

ture = []
for i in range(5000):
    ture.append(1)
    ture.append(0)
y = np.array(ture)
# print(ture)

# batchnorm_tanh_margin11 = []
# with open('D:\\PycharmProjects\\SimpleNet\\out\\batchnorm_tanh_margin11\\batchnorm_tanh_margin11_merge_200000_test_dist.txt',
#           'r') as r:
#     for l in r:
#         a = l.strip('\n').split(' ')
#         batchnorm_tanh_margin11.append(float(a[0]))
#         batchnorm_tanh_margin11.append(float(a[1]))
# # print(len(score))
# batchnorm_tanh_margin11 = np.array(batchnorm_tanh_margin11)
# fpr_2, tpr_2, thresholds_2 = roc_curve(y, batchnorm_tanh_margin11, pos_label=0)  # 正样本 match 0
# # print(fpr)
# # print(tpr)
# print(thresholds_2)

net2_tanh_margin11 = []
with open('D:\\PycharmProjects\\SimpleNet\\out\\net2_tanh_margin11\\net2_tanh_margin11_merge_200000_test_dist.txt',
          'r') as r:
    for l in r:
        a = l.strip('\n').split(' ')
        net2_tanh_margin11.append(float(a[0]))
        net2_tanh_margin11.append(float(a[1]))
net2_tanh_margin11 = np.array(net2_tanh_margin11)
fpr_11, tpr_11, thresholds_11 = roc_curve(y, net2_tanh_margin11, pos_label=0)  # 正样本 match 0
print(thresholds_11)

# score_sift = []
# with open('D:\\PycharmProjects\\SimpleNet\\resources\\sift\\merge_200000_test_sift_dist.txt', 'r') as r:
#     for l in r:
#         a = l.strip('\n').split(' ')
#         score_sift.append(float(a[0]))
#         score_sift.append(float(a[1]))
# score_sift = np.array(score_sift)
# fpr_sift, tpr_sift, thresholds_sift = roc_curve(y, score_sift, pos_label=0)  # 正样本 match 0
# # print(fpr1)
# # print(tpr1)
# print(thresholds_sift)

score_sift_normalized = []
with open('D:\\PycharmProjects\\SimpleNet\\resources\\sift\\merge_200000_test_sift_dist_normalized.txt', 'r') as r:
    for l in r:
        a = l.strip('\n').split(' ')
        score_sift_normalized.append(float(a[0]))
        score_sift_normalized.append(float(a[1]))
pred_sift_normalized = np.array(score_sift_normalized)
fpr_sift_normalized, tpr_sift_normalized, thresholds2 = roc_curve(y, pred_sift_normalized, pos_label=0)  # 正样本 match 0
# print(fpr2)
# print(tpr2)
print(thresholds2)

print('11  ', auc(fpr_11, tpr_11))
# print('sift', auc(fpr_sift, tpr_sift))
print('s_n ', auc(fpr_sift_normalized, tpr_sift_normalized))

plt.plot(fpr_11, tpr_11, c='red', label='triplet network')
# plt.plot(fpr_sift, tpr_sift, c='blue', label='sift passed param')
plt.plot(fpr_sift_normalized, tpr_sift_normalized, c='blue', label='sift')
plt.legend(loc='best')
plt.show()
