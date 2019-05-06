import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from triplet_net import Net, NetDataset, Net2


class Config():
    margin = 11
    patch_dir = '/home/yirenli/data/liberty/'
    read_dir = '../resources/merge_200000_test.txt'
    pkl_dir = '../out/batchnorm_tanh_margin{}/batchnorm_tanh_margin{}_merge_200000_train_epoch30.pkl'.format(margin, margin)
    out_dist_dir = '../out/txt/batchnorm_tanh_margin{}_merge_200000_test_dist.txt'.format(margin)
    out_desc_dir = '../out/txt/batchnorm_tanh_margin{}_merge_200000_test_desc.txt'.format(margin)


if __name__ == '__main__':
    net_dataset = NetDataset(Config)
    test_dataloader = DataLoader(net_dataset, shuffle=False)  # , num_workers=4, batch_size=16
    net = Net2().cuda()
    net.load_state_dict(torch.load(Config.pkl_dir))
    #torch.load(Config.pkl_dir) # This leads to different output!!!!!!!
    # torch.load(Config.pkl_dir, map_location='cpu') # This leads to different output!!!!!!!
    countloss = 0

    with open(Config.out_dist_dir, 'w') as wdist:
        with open(Config.out_desc_dir, 'w') as wdesc:
            for i, data in enumerate(test_dataloader, 0):
                img0, img1, img2 = data
                img0, img1, img2 = img0.cuda(), img1.cuda(), img2.cuda()

                output1, output2, output3 = net(img0, img1, img2)

                # print(output1.size())
                # if i % 20 == 0:
                #     print(output1.cpu().detach().numpy())
                #     print(output2.cpu().detach().numpy())
                #     print(output3.cpu().detach().numpy())

                wdesc.write('{}\n{}\n{}\n{}\n'.format(i, output1.cpu().detach().numpy(), output2.cpu().detach().numpy(),
                                                      output3.cpu().detach().numpy()))

                distance_positive = (output1 - output2).pow(2).sum(1)
                distance_negative = (output1 - output3).pow(2).sum(1)
                #
                # # print('positive:')
                # # print(distance_positive.cpu().detach().numpy())
                # # print('negative:')
                # # print(distance_negative.cpu().detach().numpy())

                wdist.write('{} {}\n'.format(distance_positive.cpu().detach().numpy()[0],
                                             distance_negative.cpu().detach().numpy()[0]))

                # # 1 for match, 0 for nonmatch
                # if (distance_positive < 500):
                #     w.write('1 ')
                # else:
                #     w.write('0 ')
                # if (distance_negative >= 500):
                #     w.write('0\n')
                # else:
                #     w.write('1\n')

                losses = F.relu(distance_positive - distance_negative + Config.margin)
                print(losses.cpu().detach().numpy()[0])
                if (losses>0):
                    countloss+=1
    print(countloss)
    # net2 tanh margin      test error
    # 1                     349
    # 2                     378
    # 5                     368
    # 7                     442
    # 11                    439
    #batch 11               928
