import torch
import torch.nn as nn


FILTERS = [8, 16, 32, 64, 128]
KERNEL_SIZES = [5, 3, 3, 3, 3]
RESNET_K = 4
RESNET_N = 3


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class Resblock(nn.Module):

    def __init__(self, in_channels, filters, kernel_size, stride=1, preactivated=True, block_id=1, name=''):
        super(Resblock, self).__init__()

        self.block_id = block_id
        self.preactivated = preactivated
        self.stride = stride
        self.block_id = block_id
        self.name = name

        # Bottleneck Convolution
        self.conv1 = Conv2dSame(in_channels, out_channels=in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # First Convolution
        self.conv2 = Conv2dSame(in_channels, out_channels=in_channels, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(in_channels)

        # Pooling Layer
        self.pool1 = nn.MaxPool2d(kernel_size=(stride, stride), stride=(stride, stride))

        # Dropout Layer
        self.drop = nn.Dropout()

        # Second Convolution
        self.conv3 = Conv2dSame(in_channels, filters, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm2d(filters)

        # Average Pooling
        self.pool2 = nn.AvgPool2d(kernel_size=(stride, stride), stride=(stride, stride), count_include_pad=False)

        # Shortcut Convolution
        self.conv4 = Conv2dSame(in_channels, filters, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(filters)

        self.relu = nn.ReLU()
        # Pre-activation

    def forward(self, x):

        if self.block_id > 1:
            pre = self.relu(x)
        else:
            pre = x

            # Pre-activated shortcut?
        if self.preactivated:
            x = pre

        # Bottleneck Convolution
        if self.stride > 1:
            pre = self.conv1(pre)
            pre = self.bn1(pre)
            pre = self.relu(pre)

        # First Convolution
        net = self.conv2(pre)
        net = self.bn2(net)
        net = self.relu(net)

        # Pooling layer
        if self.stride > 1:
            net = self.pool1(net)

        # Dropout Layer
        net = self.drop(net)

        # Second Convolution
        net = self.conv3(net)
        net = self.bn3(net)
        net = self.relu(net)

        # Shortcut Layer
        if not list(net.size()) == list(x.size()):

            # Average pooling
            shortcut = self.pool2(x)

            # Shortcut convolution
            shortcut = self.conv4(shortcut)
            shortcut = self.bn4(shortcut)

        else:

            # Shortcut = input
            shortcut = x

        # Merge Layer
        out = net + shortcut

        return out


class BirdNET(nn.Module):

    def __init__(self, embedding_dimension):
        super(BirdNET, self).__init__()

        # Pre-processing stage
        self.conv1 = Conv2dSame(in_channels=1, out_channels=FILTERS[0] * RESNET_K, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(int(FILTERS[0] * RESNET_K))

        # Max Pooling
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Residual Stacks
        self.resStacks = nn.ModuleList()

        for i in range(1, len(FILTERS)):
            self.resStacks.append(Resblock(in_channels=int(FILTERS[i - 1] * RESNET_K),
                                           filters=int(FILTERS[i] * RESNET_K),
                                           kernel_size=KERNEL_SIZES[i],
                                           stride=2,
                                           preactivated=True,
                                           block_id=i,
                                           name='BLOCK ' + str(i) + '-1'))

            for j in range(1, RESNET_N):
                self.resStacks.append(Resblock(in_channels=int(FILTERS[i] * RESNET_K),
                                               filters=int(FILTERS[i] * RESNET_K),
                                               kernel_size=KERNEL_SIZES[i],
                                               preactivated=False,
                                               block_id=i + j,
                                               name='BLOCK ' + str(i) + '-' + str(j + 1)))

        self.bn2 = nn.BatchNorm2d(int(FILTERS[-1] * RESNET_K))

        # Classification Branch
        self.conv2 = nn.Conv2d(in_channels=int(FILTERS[-1] * RESNET_K), out_channels=int(FILTERS[-1] * RESNET_K),
                               kernel_size=(8, 8))
        self.bn3 = nn.BatchNorm2d(int(FILTERS[-1] * RESNET_K))
        self.drop1 = nn.Dropout()
        # Dense Convolution
        self.conv3 = nn.Conv2d(in_channels=int(FILTERS[-1] * RESNET_K), out_channels=int(FILTERS[-1] * RESNET_K * 2),
                               kernel_size=1)
        self.bn4 = nn.BatchNorm2d(int(FILTERS[-1] * RESNET_K * 2))
        self.drop2 = nn.Dropout()
        # Class Convolution
        self.conv4 = nn.Conv2d(in_channels=int(FILTERS[-1] * RESNET_K * 2), out_channels=embedding_dimension, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for block in self.resStacks:
            x = block(x)

        x = self.bn2(x)
        x = self.relu(x)

        # Classification Branch
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.conv4(x)

        x = torch.logsumexp(x, (2, 3))

        return x
