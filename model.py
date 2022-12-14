"""
modification made on the basis of link:https://github.com/Xiaoccer/MobileFaceNet_Pytorch
"""
from mindspore import dtype as mstype
from mindspore import nn
import mindspore
from mindspore.common.initializer import Zero
from mindspore import Parameter
#import torch
#import torch.nn.functional as F
import math
#from torch.nn import Parameter


class Bottleneck(nn.Cell):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.SequentialCell(
            # pw
            nn.Conv2d(in_channels =inp,out_channels = inp * expansion, kernel_size=1,stride= 1,pad_mode='pad',padding= 0, has_bias=False),
            nn.BatchNorm2d(num_features=inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(in_channels =inp * expansion,out_channels = inp * expansion,kernel_size= 3,stride= stride, pad_mode='pad',padding=1, group=inp * expansion, has_bias=False),
            nn.BatchNorm2d(num_features=inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(in_channels =inp * expansion,out_channels = oup,kernel_size= 1,stride= 1, pad_mode='pad',padding= 0, has_bias=False),
            nn.BatchNorm2d(num_features=oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Cell):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(in_channels =inp, out_channels=oup, kernel_size=k,stride= s,pad_mode='pad',padding= p, group=inp, has_bias=False)
        else:
            self.conv = nn.Conv2d(in_channels =inp, out_channels=oup, kernel_size=k,stride= s,pad_mode='pad',padding= p, has_bias=False)
        self.bn = nn.BatchNorm2d(num_features=oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

Mobilenetv2_bottleneck_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class MobileFaceNet(nn.Cell):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 arcface=None):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(2, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)

        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc_out = nn.Dense(128, num_class)
        self.arcface = arcface
        # init
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.has_bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.SequentialCell(*layers)

    def forward(self, x, label):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        if self.arcface is not None:
            out = self.arcface(feature, label)
        else:
            out = self.fc_out(feature)
        return out, feature


class TgramNet(nn.Cell):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len,has_bias=False,pad_mode='same')#??????
        print(self.conv_extrctor.shape)
        self.conv_encoder = nn.SequentialCell(
            *[nn.SequentialCell(
                nn.LayerNorm((313,)),
               # nn.LeakyReLU(0.2, inplace=True),
                nn.LeakyReLU(0.2),
                 nn.Conv1d(in_channels=mel_bins, out_channels =mel_bins,kernel_size= 3,stride= 1, pad_mode='pad',padding=1, has_bias=False)
               ) for _ in range(num_layer)])
    def forward(self, x):
        out = self.conv_extrctor(x)
        print(out.shape)
       # out = self.conv_encoder(out)
        return out



class STgramMFN(nn.Cell):
    def __init__(self, num_class,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 arcface=None):
        super(STgramMFN, self).__init__()
        self.arcface = arcface
        self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(num_class=num_class,
                                           bottleneck_setting=bottleneck_setting,
                                           arcface=arcface)

    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wav, x_mel, label):
        x_wav = self.tgramnet(x_wav).unsqueeze(1)
        x = mindspore.ops.Concat((x_mel, x_wav), dim=1)
        out, feature = self.mobilefacenet(x, label)
        return out, feature


class ArcMarginProduct(nn.Cell):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
       # print(type(torch.Tensor(out_features, in_features)))
        self.weight = Parameter(mindspore.Tensor(shape=(out_features, in_features), dtype=mindspore.dtype.float32, init=Zero()))
        #print(type(self.weight))
        mindspore.common.initializer.XavierUniform(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0??,180??]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = mindspore.nn.Dense(mindspore.ops.L2Normalize(x),mindspore.ops.L2Normalize(self.weight))
        sine = mindspore.ops.Sqrt(1.0 - mindspore.ops.Pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = mindspore.numpy.where(cosine > 0, phi, cosine)
        else:
            phi =mindspore.numpy.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = mindspore.ops.Zeros(cosine.size(), device=x.device)
        # print(x.device, label.device, one_hot.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output
