"""YOLOv7 Backbone

- Author: Hyunwook Kim
- Contact: wooks527@gmail.com
"""
from torch import nn
from collections import OrderedDict

import torch


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            1,
        )


class Conv(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p),
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=3e-2)
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, k)
        self.cv3 = Conv(c1, c2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        return x[0] + x[1]


class YOLOv7Backbone(nn.Module):
    """YOLOv7 Backbone"""

    def __init__(self, num_classes=10, weights=None):
        super(YOLOv7Backbone, self).__init__()
        self.model = nn.Sequential(
            ReOrg(),  # P0
            Conv(12, 80, k=3, s=1),  # P1
            DownC(80, 80),  # P2
            Conv(160, 64, k=1, s=1),
            Conv(160, 64, k=1, s=1),
            Conv(64, 64, k=3, s=1),
            Conv(64, 64, k=3, s=1),
            Conv(64, 64, k=3, s=1),
            Conv(64, 64, k=3, s=1),
            Conv(64, 64, k=3, s=1),
            Conv(64, 64, k=3, s=1),
            Concat(),
            Conv(320, 160, k=1, s=1),
            Conv(160, 64, k=1, s=1),
            Conv(160, 64, k=1, s=1),
            Conv(64, 64, k=3, s=1),
            Conv(64, 64, k=3, s=1),
            Conv(64, 64, k=3, s=1),
            Conv(64, 64, k=3, s=1),
            Conv(64, 64, k=3, s=1),
            Conv(64, 64, k=3, s=1),
            Concat(),
            Conv(320, 160, k=1, s=1),
            Shortcut(),
            DownC(160, 160),  # P3
            Conv(320, 128, k=1, s=1),
            Conv(320, 128, k=1, s=1),
            Conv(128, 128, k=3, s=1),
            Conv(128, 128, k=3, s=1),
            Conv(128, 128, k=3, s=1),
            Conv(128, 128, k=3, s=1),
            Conv(128, 128, k=3, s=1),
            Conv(128, 128, k=3, s=1),
            Concat(),
            Conv(640, 320, k=1, s=1),
            Conv(320, 128, k=1, s=1),
            Conv(320, 128, k=1, s=1),
            Conv(128, 128, k=3, s=1),
            Conv(128, 128, k=3, s=1),
            Conv(128, 128, k=3, s=1),
            Conv(128, 128, k=3, s=1),
            Conv(128, 128, k=3, s=1),
            Conv(128, 128, k=3, s=1),
            Concat(),
            Conv(640, 320, k=1, s=1),
            Shortcut(),
            DownC(320, 320),  # P4
            Conv(640, 256, k=1, s=1),
            Conv(640, 256, k=1, s=1),
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            Concat(),
            Conv(1280, 640, k=1, s=1),
            Conv(640, 256, k=1, s=1),
            Conv(640, 256, k=1, s=1),
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            Conv(256, 256, k=3, s=1),
            Concat(),
            Conv(1280, 640, k=1, s=1),
            Shortcut(),
            DownC(640, 480),  # P5
            Conv(960, 384, k=1, s=1),
            Conv(960, 384, k=1, s=1),
            Conv(384, 384, k=3, s=1),
            Conv(384, 384, k=3, s=1),
            Conv(384, 384, k=3, s=1),
            Conv(384, 384, k=3, s=1),
            Conv(384, 384, k=3, s=1),
            Conv(384, 384, k=3, s=1),
            Concat(),
            Conv(1920, 960, k=1, s=1),
            Conv(960, 384, k=1, s=1),
            Conv(960, 384, k=1, s=1),
            Conv(384, 384, k=3, s=1),
            Conv(384, 384, k=3, s=1),
            Conv(384, 384, k=3, s=1),
            Conv(384, 384, k=3, s=1),
            Conv(384, 384, k=3, s=1),
            Conv(384, 384, k=3, s=1),
            Concat(),
            Conv(1920, 960, k=1, s=1),
            Shortcut(),
            DownC(960, 640),  # P6
            Conv(1280, 512, k=1, s=1),
            Conv(1280, 512, k=1, s=1),
            Conv(512, 512, k=3, s=1),
            Conv(512, 512, k=3, s=1),
            Conv(512, 512, k=3, s=1),
            Conv(512, 512, k=3, s=1),
            Conv(512, 512, k=3, s=1),
            Conv(512, 512, k=3, s=1),
            Concat(),
            Conv(2560, 1280, k=1, s=1),
            Conv(1280, 512, k=1, s=1),
            Conv(1280, 512, k=1, s=1),
            Conv(512, 512, k=3, s=1),
            Conv(512, 512, k=3, s=1),
            Conv(512, 512, k=3, s=1),
            Conv(512, 512, k=3, s=1),
            Conv(512, 512, k=3, s=1),
            Conv(512, 512, k=3, s=1),
            Concat(),
            Conv(2560, 1280, k=1, s=1),
            Shortcut(),
        )
        self.pool = nn.AvgPool2d(num_classes, 1)

        if weights is not None:
            ckpt = torch.load(weights)
            state_dict = OrderedDict()
            for key, value in ckpt["state_dict"].items():
                if "fc" in key:
                    continue

                new_key = ".".join(key.split(".")[1:])
                state_dict[new_key] = value

            matched = self.load_state_dict(state_dict)
            print(matched)

        self.fc = nn.Linear(1280, num_classes, bias=False)

    def forward(self, x):
        xs = []
        for i, m in enumerate(self.model):
            # Handle non-sequential inputs
            if i in (
                4,
                26,
                48,
                70,
                92,
            ):
                x = xs[-2]
            elif i in (
                13,
                35,
                57,
                79,
                101,
            ):
                x = xs[-11]
            elif i in (
                14,
                36,
                58,
                80,
                102,
            ):
                x = xs[-12]
            elif isinstance(m, Concat):
                x = [xs[-1], xs[-3], xs[-5], xs[-7], xs[-8]]
            elif isinstance(m, Shortcut):
                x = [xs[-1], xs[-11]]

            x = m(x)
            xs.append(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x
