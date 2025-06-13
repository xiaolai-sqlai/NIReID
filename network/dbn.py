import copy
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from network.layer import BatchDrop, BatchErasing
from torchvision.models import resnet50, resnet101, regnet_y_1_6gf, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2


class GlobalAvgPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalAvgPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)


class GlobalMaxPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalMaxPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)


class DBN(nn.Module):
    def __init__(self, num_classes=751, noise_classes=0, num_feat=256, std=0.1, net="regnet_y_1_6gf", erasing=0.0):
        super(DBN, self).__init__()
        if self.training:
            self.erasing = nn.Identity()
            if erasing > 0:
                self.erasing = BatchErasing(smax=erasing)

        if net == "resnet50":
            model = resnet50(pretrained=True)
            pool_num = 2048
        elif net == "resnet101":
            model = resnet101(pretrained=True)
            pool_num = 2048
        elif net == "regnet_y_1_6gf":
            model = regnet_y_1_6gf(pretrained=True)
            pool_num = 888
        elif net == "resnext101_32x8d":
            model = resnext101_32x8d(pretrained=True)
            pool_num = 2048
        elif net == "wide_resnet50_2":
            model = wide_resnet50_2(pretrained=True)
            pool_num = 2048
        elif net == "wide_resnet101_2":
            model = wide_resnet101_2(pretrained=True)
            pool_num = 2048

        if "res" in net:
            self.stem = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool
            )
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3

            model.layer4[0].conv2.stride = (1, 1)
            model.layer4[0].downsample[0].stride = (1, 1)

            self.branch_1 = copy.deepcopy(model.layer4)
            self.branch_2 = copy.deepcopy(model.layer4)
        elif net.startswith("regnet"):
            self.stem = model.stem
            self.layer1 = model.trunk_output.block1
            self.layer2 = model.trunk_output.block2
            self.layer3 = model.trunk_output.block3

            model.trunk_output.block4[0].proj[0].stride = (1, 1)
            model.trunk_output.block4[0].f.b[0].stride = (1, 1)
            self.branch_1 = copy.deepcopy(model.trunk_output.block4)
            self.branch_2 = copy.deepcopy(model.trunk_output.block4)

        self.pool_list = nn.ModuleList()
        self.feat_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.class_list = nn.ModuleList()
        self.noise_class_list = nn.ModuleList()

        for i in range(3):
            self.pool_list.append(GlobalAvgPool2d(p=2))
            feat = nn.Linear(pool_num, num_feat, bias=False)
            init.kaiming_normal_(feat.weight)
            self.feat_list.append(feat)
            bn = nn.BatchNorm1d(num_feat)
            init.normal_(bn.weight, mean=1.0, std=std)
            init.normal_(bn.bias, mean=0.0, std=std)
            self.bn_list.append(bn)

            linear = nn.Linear(num_feat, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)

            if noise_classes > 0:
                noise_linear = nn.Linear(num_feat, noise_classes, bias=False)
                init.normal_(noise_linear.weight, std=0.001)
            else:
                noise_linear = nn.Identity()
            self.noise_class_list.append(noise_linear)

        for i in range(2):
            self.pool_list.append(GlobalMaxPool2d(p=1))
            feat = nn.Linear(pool_num, num_feat, bias=False)
            init.kaiming_normal_(feat.weight)
            self.feat_list.append(feat)
            bn = nn.BatchNorm1d(num_feat)
            init.normal_(bn.weight, mean=1.0, std=std)
            init.normal_(bn.bias, mean=0.0, std=std)
            bn.bias.requires_grad = False
            self.bn_list.append(bn)

            linear = nn.Linear(num_feat, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)

            if noise_classes > 0:
                noise_linear = nn.Linear(num_feat, noise_classes, bias=False)
                init.normal_(noise_linear.weight, std=0.001)
            else:
                noise_linear = nn.Identity()
            self.noise_class_list.append(noise_linear)

    def forward(self, x):
        if self.training:
            x = self.erasing(x)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        # x_chunk = [x1, x2, x1, x2[:, :, 0::2], x2[:, :, 1::2]]
        x_chunk = [x1, x2, x1] + list(torch.chunk(x2, dim=2, chunks=2))

        pool_list = []
        feat_list = []
        bn_list = []
        class_list = []
        noise_class_list = []

        for i in range(5):
            pool = self.pool_list[i](x_chunk[i]).flatten(1)
            pool_list.append(pool)
            feat = self.feat_list[i](pool)
            feat_list.append(feat)
            bn = self.bn_list[i](feat)
            bn_list.append(bn)

            feat_class = self.class_list[i](bn)
            class_list.append(feat_class)

            noise_class = self.noise_class_list[i](bn)
            noise_class_list.append(noise_class)

        if self.training:
            return (class_list, noise_class_list), bn_list[:3]
        return bn_list,
