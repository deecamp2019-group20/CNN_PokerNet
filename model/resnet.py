import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnetpokernet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    }


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # 3*3 conv with padding
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    # 1*1 conv with padding
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, groups=1,
            base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        '''Both self.conv1 and self.downsample layers
        downsample the input when stride != 1'''
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # x might normally indicate a image tensor such as 224 * 224 * 3
        # How it comes when the input tensor become much smaller (15 * 4 * 3)?

        # store the input
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottelneck(nn.Module):
    expansion = 4

    def __init__(
            self, inplanes, planes, stride=1, downsample=None, groups=1,
            base_width=64, dilation=1, norm_layer=None):

        super(Bottelneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers
        # downsample the input when stride != 1

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # TODO: Need Redefine the 'num_classes'
    # num_classes here indicate the classes of ImageNet
    def __init__(
            self, block, layers, num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            is_pokernet=False):

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # TODO: Need Redefine the 'inplanes'
        self.inplanes = 64
        # This is just like normal kernel
        self.dilation = 1
        # This indicate that whether the Network is pokernent or not
        self.is_pokernet = is_pokernet

        if replace_stride_with_dilation is None:
            # each element in the tuple indicate that should we replace
            # the 2 * 2 stride width a dilated conv
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_width_dilation should be None '
                'or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # TODO:The input channel MIGHT be 3
        # HERE: try to use a input tensor with 10 channels
        if is_pokernet:
            self.conv1 = nn.Conv2d(
                10, self.inplanes, kernel_size=5,
                stride=1, padding=2, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7,
                stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if is_pokernet:
            self.layer1 = self._make_layer(
                block, 64, layers[0], stride=1
            )
            self.layer2 = self._make_layer(
                block, 128, layers[1], stride=1,
                dilate=replace_stride_with_dilation[0]
            )
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=1,
                dilate=replace_stride_with_dilation[1]
            )
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=1,
                dilate=replace_stride_with_dilation[2]
            )
        else:
            self.layer1 = self._make_layer(
                block, 64, layers[0]
            )
            self.layer2 = self._make_layer(
                block, 128, layers[1], stride=2,
                dilate=replace_stride_with_dilation[0]
            )
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=2,
                dilate=replace_stride_with_dilation[1]
            )
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=2,
                dilate=replace_stride_with_dilation[2]
            )

        # TODO: This avgpool's output size MIGHT need to modified
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero residual
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottelneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        # just transfer stride into the form of dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample to standardize the input
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if not(self.is_pokernet):
            x = self.maxpool(x)

        # Blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Output Flatten
        x = self.avgpool(x)
        # Flatten the tensor, excluding the batch dim
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


def _resnet(
        arch, block, layers, num_classes,
        pretrained, progress, is_pokernet, **kwargs):

    model = ResNet(
                block, layers, num_classes=num_classes,
                is_pokernet=is_pokernet, **kwargs
            )
    if is_pokernet:
        # Is the PokerNet
        return model
    else:
        # Normal ResNet, Not the PokerNet
        if pretrained:
            state_dict = load_state_dict_from_url(
                model_urls[arch], progress=progress)
            model.load_state_dict(state_dict)
        return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition"
    <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download
    """
    return _resnet(
        'resnet18', BasicBlock, [2, 2, 2, 2], 1000,
        pretrained, progress, False, **kwargs
    )


def resnetpokernet(num_classes, pretrained=False, progress=False, **kwargs):
    r""" ResNet-Poker model
    Using ResNet as the backbone to predict the probability of PokerMan

    Args:
        pretrained (bool): If True, returns a model pre-trained on Poker
        progress (bool): If True, displays a progress bar of the download
    """
    return _resnet(
        'resnetpokernet', BasicBlock, [1, 1, 1, 2], num_classes,
        pretrained, progress, True, **kwargs
    )
