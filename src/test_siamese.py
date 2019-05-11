import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from siamese_net import Net, NetDataset, Net2


class Config():
    margin = 100
    patch_dir = '/home/yirenli/data/liberty/'
    read_dir = '../resources/siamese_200000_test.txt'
    pkl_dir = '../out/siamese_margin{}/siamese_margin{}_200000_train_epoch200.pkl'.format(margin, margin)
    out_dist_dir = '../out/txt/siamese_margin{}_200000_test_dist.txt'.format(margin)
    out_desc_dir = '../out/txt/siamese_margin{}_200000_test_desc.txt'.format(margin)
    patch_len = 5000


if __name__ == '__main__':
    net_dataset = NetDataset(Config)
    test_dataloader = DataLoader(net_dataset, shuffle=False)  # , num_workers=4, batch_size=16
    net = Net().cuda()
    net.load_state_dict(torch.load(Config.pkl_dir))
    # torch.load(Config.pkl_dir) # This leads to different output!!!!!!!
    # torch.load(Config.pkl_dir, map_location='cpu') # This leads to different output!!!!!!!
    # countloss = 0

    with open(Config.out_dist_dir, 'w') as wdist:
        with open(Config.out_desc_dir, 'w') as wdesc:
            for i, data in enumerate(test_dataloader, 0):
                img0, img1, label = data
                img0, img1 = img0.cuda(), img1.cuda()

                output1, output2 = net(img0, img1)

                wdesc.write('{}\n{}\n{}\n'.format(i, output1.cpu().detach().numpy(), output2.cpu().detach().numpy()))

                distance_positive = (output1 - output2).pow(2).sum(1)

                wdist.write('{}\n'.format(distance_positive.cpu().detach().numpy()[0]))
    # print(countloss)
