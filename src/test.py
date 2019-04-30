import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from triplet_net import Net, NetDataset, Net2


class Config():
    patch_dir = 'D:\\dataset\\liberty\\'
    read_dir = '../resources/merge_200000_test.txt'
    pkl_dir = '../out/net2_margin2/net2_margin_2_merge_200000_epoch30.pkl'
    out_dir = '../out/txt/net2_merge_200000_test_out.txt'

if __name__ == '__main__':
    net_dataset = NetDataset(Config)
    test_dataloader = DataLoader(net_dataset, shuffle=False)  # , num_workers=4, batch_size=16
    net = Net2().cuda()
    net.load_state_dict(torch.load(Config.pkl_dir))
    countloss = 0

    with open(Config.out_dir, 'w') as w:
        for i, data in enumerate(test_dataloader, 0):
            img0, img1, img2 = data
            img0, img1, img2 = img0.cuda(), img1.cuda(), img2.cuda()

            output1, output2, output3 = net(img0, img1, img2)

            # print(output1.size())
            # if i % 20 == 0:
            #     print(output1.cpu().detach().numpy())
            #     print(output2.cpu().detach().numpy())
            #     print(output3.cpu().detach().numpy())

            # distance_positive = (output1 - output2).pow(2).sum(1)
            # distance_negative = (output1 - output3).pow(2).sum(1)
            #
            # # print('positive:')
            # # print(distance_positive.cpu().detach().numpy())
            # # print('negative:')
            # # print(distance_negative.cpu().detach().numpy())
            #
            # w.write((str)(distance_positive.cpu().detach().numpy()[0]))
            # w.write(' ')
            # w.write((str)(distance_negative.cpu().detach().numpy()[0]))
            # w.write('\n')

            # # 1 for match, 0 for nonmatch
            # if (distance_positive < 500):
            #     w.write('1 ')
            # else:
            #     w.write('0 ')
            # if (distance_negative >= 500):
            #     w.write('0\n')
            # else:
            #     w.write('1\n')

    #         losses = F.relu(distance_positive - distance_negative + 2)
    #         print(losses.cpu().detach().numpy()[0])
    #         if (losses>0):
    #             countloss+=1
    # print(countloss) # model1 merge_200000_test.txt 490 incorrect out of 5000
