from __future__ import division

import argparse, datetime, os, math, copy, random
import numpy as np
import cv2

from cv2_transform import transforms
from torch.utils.data import DataLoader
import torch
from torch import nn

from network.tripletloss import TripletLoss
from data_read import ImageFolder, ImageFolderVehicleID, ImageFolderPDukeMTMC
from thop import profile, clever_format
import utils
from cv2_transform import copy_paste

from network.dbn import DBN


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--img-height', type=int, default=384,
                    help='the height of image for input')
parser.add_argument('--img-width', type=int, default=128,
                    help='the width of image for input')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-workers', type=int, default=4,
                    help='the number of workers for data loader')
parser.add_argument('--dataset-root', type=str, default="/data/laishenqi/dataset",
                    help='the number of workers for data loader')
parser.add_argument('--dataset', type=str, default="market1501", help='market1501, dukemtmc, npdetected, nplabeled, msmt17, occluded_duke, p_dukemtmc')
parser.add_argument('--noise-dataset', type=str, default="", help="veri776, vehicleid")
parser.add_argument('--noise-interval', type=int, default=0)
parser.add_argument('--paste-dataset', type=str, default="", help="veri776, vehicleid")
parser.add_argument('--height-ratio', type=float, default=0.0)
parser.add_argument('--instance-num', type=int, default=4)
parser.add_argument('--net', type=str, default="small", help="small,large")
parser.add_argument('--std', type=float, default=0.1, help="std to init the weight and bias in batch norm")
parser.add_argument('--freeze', type=str, default="", help="stem,layer1,layer2,layer3")
parser.add_argument('--norm', type=str2bool, default=True)
parser.add_argument('--gpus', type=str, default="0,1", help='number of gpus to use.')
parser.add_argument('--epochs', type=str, default="5,75")
parser.add_argument('--lr', type=float, default=2.0e-3, help='learning rate. default is 2.0e-3.')
parser.add_argument('--seed', type=int, default=613, help='random seed to use. Default=613.')
parser.add_argument('--lr-decay', type=int, default=0.1)
parser.add_argument('--erasing', type=float, default=0.0)
parser.add_argument('--ema-ratio', type=float, default=1.0)
parser.add_argument('--ema-extra', type=int, default=0)

def get_data_iters(batch_size):
    transform_noise = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Pad(padding=8, padding_mode='symmetric'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_train = transforms.Compose([
        transforms.Resize((opt.img_height, opt.img_width)),
        copy_paste.CopyPaste(probability=1.0, height_ratio=opt.height_ratio, dataset_root=opt.dataset_root, paste_dataset=opt.paste_dataset),
        transforms.Pad(padding=8, padding_mode='symmetric'),
        transforms.RandomCrop((opt.img_height, opt.img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    print(transform_train)

    if opt.noise_dataset == "veri776":
        noise_imgs = ImageFolder(db_path=os.path.join(opt.dataset_root, "VeRi/image_train"), transform=transform_noise, instance_num=opt.instance_num)
    elif opt.noise_dataset == "vehicleid":
        noise_imgs = ImageFolderVehicleID(db_path=os.path.join(opt.dataset_root, "VehicleID_V1.0"), transform=transform_noise, instance_num=opt.instance_num)
    noise_data = DataLoader(noise_imgs, batch_size, shuffle=False, drop_last=True, num_workers=opt.num_workers, pin_memory=True)

    if opt.dataset == "market1501":
        path = "Market-1501-v15.09.15/bounding_box_train"
        train_imgs = ImageFolder(db_path=os.path.join(opt.dataset_root, path), transform=transform_train, instance_num=opt.instance_num)
    elif opt.dataset == "dukemtmc":
        path = "DukeMTMC-reID/bounding_box_train"
        train_imgs = ImageFolder(db_path=os.path.join(opt.dataset_root, path), transform=transform_train, instance_num=opt.instance_num)
    elif opt.dataset == "npdetected":
        path = "cuhk03-np/detected/bounding_box_train"
        train_imgs = ImageFolder(db_path=os.path.join(opt.dataset_root, path), transform=transform_train, instance_num=opt.instance_num)
    elif opt.dataset == "nplabeled":
        path = "cuhk03-np/labeled/bounding_box_train"
        train_imgs = ImageFolder(db_path=os.path.join(opt.dataset_root, path), transform=transform_train, instance_num=opt.instance_num)
    elif opt.dataset == "msmt17":
        path = "MSMT17/bounding_box_train"
        train_imgs = ImageFolder(db_path=os.path.join(opt.dataset_root, path), transform=transform_train, instance_num=opt.instance_num)
    elif opt.dataset == "occluded_duke":
        path = "Occluded_Duke/bounding_box_train"
        train_imgs = ImageFolder(db_path=os.path.join(opt.dataset_root, path), transform=transform_train, instance_num=opt.instance_num)
    elif opt.dataset == "p_dukemtmc":
        path = "P-DukeMTMC-reid"
        train_imgs = ImageFolderPDukeMTMC(db_path=os.path.join(opt.dataset_root, path), transform=transform_train, instance_num=opt.instance_num)

    train_data = DataLoader(train_imgs, batch_size, shuffle=False, drop_last=True, num_workers=opt.num_workers, pin_memory=True)

    return noise_data, train_data


def adjust_lr(epoch, epochs, opt):
    stop_epoch = int(epochs[1] * opt.ema_ratio)
    minlr = opt.lr * 0.01
    dlr = (opt.lr - minlr) / epochs[0]
    if epoch <= epochs[0]:
        lr = minlr + dlr * epoch

    if epoch >= epochs[0] and epoch <= stop_epoch:
        lr = 0.5 * opt.lr * (math.cos(math.pi * (epoch - epochs[0]) / (epochs[1] - epochs[0])) + 1)

    if epoch >= stop_epoch:
        lr = 0.5 * opt.lr * (math.cos(math.pi * (stop_epoch - epochs[0]) / (epochs[1] - epochs[0])) + 1)
    return lr


def train(train_data, trainer, net, criterion1, criterion2, scaler, i):
    _loss = 0.
    train_data.dataset.shuffle_items()
    for data, label in train_data:
        data_list = data.cuda(non_blocking=True)
        label_list = label.cuda(non_blocking=True)
        trainer.zero_grad()
        losses = []
        with torch.cuda.amp.autocast():
            outputs, features = net(data_list)
            for idx, output in enumerate(outputs[i]):
                losses.append(criterion1(output, label_list))
            for feature in features:
                losses.append(criterion2(feature, label_list))
            loss = sum(losses) / len(losses)
        scaler.scale(loss).backward()
        scaler.step(trainer)
        scaler.update()
        _loss += loss.detach().sum().cpu().numpy()
    return _loss


def main(net, batch_size, epochs, opt):
    net_list = []
    net = nn.DataParallel(net)
    net.cuda()

    noise_data, train_data = get_data_iters(batch_size)
    trainer = torch.optim.SGD(params=net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    if len(opt.freeze) > 0:
        for name, param in net.named_parameters():
            for l in opt.freeze.split(","):
                if l in name:
                    param.requires_grad = False

    scaler = torch.cuda.amp.GradScaler()

    prev_time = datetime.datetime.now()
    criterion1 = nn.CrossEntropyLoss().cuda()
    criterion2 = TripletLoss(norm=opt.norm).cuda()
    for epoch in range(1, epochs[1] + opt.ema_extra + 1):
        net.train()

        _noise_loss = 0.
        _loss = 0.
        length = len(train_data)
        lr = adjust_lr(epoch, epochs, opt)
        for param_group in trainer.param_groups:
            param_group['lr'] = lr
        trainer.defaults['lr'] = lr

        if opt.noise_interval > 0:
            if len(opt.noise_dataset) > 0 and epoch < int(epochs[1] * opt.ema_ratio) and epoch % opt.noise_interval == 0:
                _noise_loss += train(noise_data, trainer, net, criterion1, criterion2, scaler, 1)
        _loss += train(train_data, trainer, net, criterion1, criterion2, scaler, 0)

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        __noise_loss = _noise_loss / length
        __loss = _loss / length

        epoch_str = "Epoch {}. Noise loss: {:.5f}, Train loss: {:.5f}, ".format(epoch, __noise_loss, __loss)
        prev_time = cur_time
        print(epoch_str + time_str + ', lr: {:.8f}'.format(trainer.defaults['lr']))

        if epoch >= int(epochs[1] * opt.ema_ratio) and opt.ema_ratio < 1:
            net_list.append(copy.deepcopy(net.module).eval().cpu())
            print("EMA checkpoint number", len(net_list))
            if epoch == (epochs[1] + opt.ema_extra):
                ema_net = utils.all_average(net_list)
                ema_net = utils.bn_update(train_data, ema_net)

    if not os.path.exists("params"):
        os.mkdir("params")

    if opt.ema_ratio == 1:
        torch.save(net.module.state_dict(), 'params/ema.pth')
    else:
        torch.save(ema_net.module.state_dict(), 'params/ema.pth')


if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    batch_size = opt.batch_size
    epochs = [int(i) for i in opt.epochs.split(',')]
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if opt.dataset == "market1501":
        num_classes = 751
    elif opt.dataset == "dukemtmc":
        num_classes = 702
    elif opt.dataset == "npdetected":
        num_classes = 767
    elif opt.dataset == "nplabeled":
        num_classes = 767
    elif opt.dataset == "msmt17":
        num_classes = 1041
    elif opt.dataset == "occluded_duke":
        num_classes = 702
    elif opt.dataset == "p_dukemtmc":
        num_classes = 665

    if opt.noise_dataset == "veri776":
        noise_classes = 576
    elif opt.noise_dataset == "vehicleid":
        noise_classes = 13164

    custom_ops = {}
    net = DBN(num_classes=num_classes, noise_classes=noise_classes, std=opt.std, net=opt.net, erasing=opt.erasing)
    temp = copy.deepcopy(net)
    print(temp)

    input = torch.randn(1, 3, opt.img_height, opt.img_width)
    macs, params = profile(temp, inputs=(input, ), custom_ops=custom_ops)
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)

    del temp
    main(net, batch_size, epochs, opt)
    os.system("python test_{}.py --net {} --img-height {} --img-width {} --batch-size 128 --gpus {} --dataset-root {} --num-classes {} --noise-classes {}".format(opt.dataset, opt.net, opt.img_height, opt.img_width, opt.gpus, opt.dataset_root, num_classes, noise_classes))
