import torch
import torch.nn as nn
import numpy as np
from srm_filter_kernel import all_normalized_hpf_list


class TLU(nn.Module):
    # 激活函数，限制特征图的数值范围，抑制极端激活，防止异常值传播
    # 截断tensor的值，使其范围在[-threshold, threshold]
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output


class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters 加载30个5x5的滤波器
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)

        # numpy->tensor, reshape成pytorch卷积核格式，权重不可训练 始终为高通滤波器
        hpf_weight = nn.Parameter(torch.Tensor(np.array(all_hpf_list_5x5)).view(30, 1, 5, 5), requires_grad=False)

        # 输入通道为1通道(灰度图) 输出为30个通道 对应30个滤波器 [B, 1, H, W]->[B, 30, H, W]
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)

        # 权重固定为hpf_weight，即使用高通滤波
        self.hpf.weight = hpf_weight

        # 截断 threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):
        output = self.hpf(input)
        # output = self.tlu(output)

        return output


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.group1 = HPF()

        # 对RGB三个通道做hpf，特征提取下来共3x30=90个通道
        self.group1_b = nn.Sequential(
            nn.Conv2d(90, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Hardtanh(min_val=-5, max_val=5)
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 降低特征图尺寸(下采样)
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.advpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        # input->[B, 2, C, H, W]
        img_poor = input[:, 0, :, :, :]
        img_rich = input[:, 1, :, :, :]
        a, b, c, d = img_poor.shape

        img_poor = img_poor.reshape(-1, 1, c, d)
        img_poor = self.group1(img_poor)
        img_poor = img_poor.reshape(a, -1, c, d)
        img_poor = self.group1_b(img_poor)

        img_rich = img_rich.reshape(-1, 1, c, d)
        img_rich = self.group1(img_rich)
        img_rich = img_rich.reshape(a, -1, c, d)
        img_rich = self.group1_b(img_rich)

        # 高通滤波器后得到的特征计算残差
        res = img_poor - img_rich

        output = self.group2(res)
        output = self.group3(output)
        output = self.group4(output)
        output = self.group5(output)
        output = self.advpool(output)
        output = output.view(output.size(0), -1)

        out = self.fc2(output)

        return out


def initWeight(module):
    # 初始化权重 Conv2d:kaiming_normal  Linear:normal+constant
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)
