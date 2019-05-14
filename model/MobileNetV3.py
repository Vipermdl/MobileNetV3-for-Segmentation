import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def hard_sigmoid(x):
    out = (0.2 * x) + 0.5
    out = F.threshold(-out, -1., -1.)
    out = F.threshold(-out, 0., 0.)
    return out


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.activation = nn.ReLU6(inplace)

    def forward(self, x):
        out = self.activation(x + 3.) / 6.
        return out * x


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size)
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        out = hard_sigmoid(out)

        return out * x


class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        padding = (kernal_size - 1) // 2

        self.use_connect = stride == 1 and in_channels == out_channels

        if self.nonLinear == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
            nn.BatchNorm2d(exp_size),
        )

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        # MobileNetV2
        out = self.conv(x)
        out = self.depth_conv(out)

        # Squeeze and Excite
        if self.SE:
            out = self.squeeze_block(out)

        # point-wise conv
        out = self.point_conv(out)

        # connection
        if self.use_connect:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, model_mode="LARGE", num_classes=19, dropout_flag=False):
        super(MobileNetV3, self).__init__()
        self.activation_HS = nn.ReLU6(inplace=True)
        self.num_classes = num_classes
        print("num classes: ", self.num_classes)

        self.model_mode = model_mode

        if model_mode == "LARGE":
            layers = [
                [16, 16, 3, 1, "RE", False, 16],
                [16, 24, 3, 2, "RE", False, 64],
                [24, 24, 3, 1, "RE", False, 72],
                [24, 40, 5, 2, "RE", True, 72],
                [40, 40, 5, 1, "RE", True, 120],

                [40, 40, 5, 1, "RE", True, 120],
                [40, 80, 3, 2, "HS", False, 240],
                [80, 80, 3, 1, "HS", False, 200],
                [80, 80, 3, 1, "HS", False, 184],
                [80, 80, 3, 1, "HS", False, 184],

                [80, 112, 3, 1, "HS", True, 480],
                [112, 112, 3, 1, "HS", True, 672],
                [112, 160, 5, 1, "HS", True, 672],
                # [160, 160, 5, 2, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 960],
            ]
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                h_swish(inplace=True),
            )

            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                self.block.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block = nn.Sequential(*self.block)

            self.r_aspp = R_ASPP_module(in_channels_x=160, in_channels_f=40, num_classes=num_classes)
            if dropout_flag:
                print("have droput layer")
                self.last_conv = nn.Sequential(nn.Dropout2d(0.1, False),
                                               nn.Conv2d(38, num_classes, kernel_size=1, stride=1))
            else:
                self.last_conv = nn.Sequential(nn.Conv2d(38, num_classes, kernel_size=1, stride=1))



        elif model_mode == "SMALL":
            layers = [
                [16, 16, 3, 2, "RE", True, 16],
                [16, 24, 3, 2, "RE", False, 72],
                [24, 24, 3, 1, "RE", False, 88],
                [24, 40, 5, 2, "RE", True, 96],
                [40, 40, 5, 1, "RE", True, 240],
                [40, 40, 5, 1, "RE", True, 240],
                [40, 48, 5, 1, "HS", True, 120],
                [48, 48, 5, 1, "HS", True, 144],
                # [48, 96, 5, 2, "HS", True, 288],
                [48, 96, 5, 1, "HS", True, 288],
                [96, 96, 5, 1, "HS", True, 576],
                [96, 96, 5, 1, "HS", True, 576],
            ]

            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                h_swish(inplace=True),
            )

            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                self.block.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block = nn.Sequential(*self.block)

            self.r_aspp = R_ASPP_module(in_channels_x=96, in_channels_f=24, num_classes=num_classes)
            if dropout_flag:
                print("have droput layer")
                self.last_conv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(38, num_classes, kernel_size=1, stride=1))
            else:
                self.last_conv = nn.Sequential(nn.Conv2d(38, num_classes, kernel_size=1, stride=1))


        self.apply(_weights_init)

    def forward(self, x):
        out = self.init_conv(x)

        for i, layer in enumerate(self.block):
            out = layer(out)
            if i == 2 and self.model_mode =="SMALL":
                feature = out
            if i == 5 and self.model_mode == "LARGE":
                feature = out

        out = self.r_aspp(out, feature)
        out = self.last_conv(out)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        return out


class R_ASPP_module(nn.Module):
    def __init__(self, in_channels_x, in_channels_f, num_classes):
        super(R_ASPP_module, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels_x, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Conv2d(in_channels_x, 128, kernel_size=1, stride=1)

        self.out_conv1 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1)
        self.out_conv2 = nn.Conv2d(in_channels_f, num_classes, kernel_size=1, stride=1)

        self._init_weight()

    def forward(self, x, feature):
        x_temp1 = self.layer1(x)

        x_temp2 = F.avg_pool2d(x, kernel_size=9, stride=(5, 5))

        x_temp2 = self.layer2(x_temp2)
        x_temp2_weight = torch.sigmoid(x_temp2)
        x_temp2_weight = F.interpolate(x_temp2_weight, x_temp1.size()[2:], mode='bilinear', align_corners=False)
        out = x_temp2_weight * x_temp1

        out = F.interpolate(out, feature.size()[2:], mode='bilinear', align_corners=False)
        out = self.out_conv1(out)
        feature = self.out_conv2(feature)

        # out = out + feature   #origin paper
        out = torch.cat((out, feature), dim=1)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


temp = torch.zeros((1, 3, 680, 680))
model = MobileNetV3(model_mode="SMALL")
print(model(temp).shape)
