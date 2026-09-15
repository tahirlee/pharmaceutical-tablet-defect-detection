import math
import torch
from torch import nn
from torch.hub import load_state_dict_from_url

# Official ImageNet-pretrained DLA checkpoints (Fisher Yu / ucbdrive/dla).
# Only dla34 is defined here since that's the only variant this file builds;
# add more URLs from http://dl.yf.io/dla/models/imagenet/ if you add other
# variants (dla60, dla102, etc.) later.
MODEL_URLS = {
    "dla34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth",
}


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, bias=False,
                              padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1, root_residual=False):
        super(Tree, self).__init__()

        self.level_root = level_root
        self.levels = levels
        self.root_dim = root_dim
        self.downsample = None
        self.project = None

        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            root_dim += in_channels

        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride)
            self.tree2 = block(out_channels, out_channels, stride=1)
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride,
                              root_dim=0,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              root_residual=root_residual)
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual)

        if self.levels == 1:
            x2 = self.tree2(x1)
            out = self.root(x2, x1, *children)
        else:
            children.append(x1)
            out = self.tree2(x1, children=children)
        return out


class DLA(nn.Module):
    def __init__(self, layers, channels, num_classes=1000, block=BasicBlock, residual_root=False, pool_size=7):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes

        self.patchfy_stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True))

        self.stage_0 = self._make_conv_level(channels[0], channels[0], layers[0])

        self.stage_1 = self._make_conv_level(channels[0], channels[1], layers[1], stride=2)

        self.stage_2 = Tree(layers[2], block, channels[1], channels[2], stride=2,
                            level_root=False, root_residual=residual_root)
        self.stage_3 = Tree(layers[3], block, channels[2], channels[3], stride=2,
                            level_root=True, root_residual=residual_root)
        self.stage_4 = Tree(layers[4], block, channels[3], channels[4], stride=2,
                            level_root=True, root_residual=residual_root)
        self.stage_5 = Tree(layers[5], block, channels[4], channels[5], stride=2,
                            level_root=True, root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv_level(self, inplanes, planes, num_layers, stride=1):
        modules = []
        for i in range(num_layers):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        stages_features_list = []
        x = self.patchfy_stem(x)
        for i in range(6):
            x = getattr(self, 'stage_{}'.format(i))(x)
            stages_features_list.append(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x


# --------------------------------------------------------------------------- #
# Pretrained weight loading
# --------------------------------------------------------------------------- #
# The official ucbdrive/dla checkpoint was trained on a class with these
# top-level module names: base_layer, level0, level1, level2, level3, level4,
# level5, avgpool, fc. Every submodule INSIDE those (Tree.tree1/tree2/root,
# Root.conv/bn, BasicBlock.conv1/bn1/conv2/bn2, downsample, project) is named
# identically in both implementations -- only the top-level names differ
# here (patchfy_stem / stage_0..stage_5), so remapping is just a prefix swap.
_KEY_PREFIX_MAP = {
    "base_layer.": "patchfy_stem.",
    "level0.": "stage_0.",
    "level1.": "stage_1.",
    "level2.": "stage_2.",
    "level3.": "stage_3.",
    "level4.": "stage_4.",
    "level5.": "stage_5.",
    # avgpool.* and fc.* keep the same name in both -- no remap needed.
}


def _remap_pretrained_state_dict(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in _KEY_PREFIX_MAP.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped[new_key] = value
    return remapped


def load_pretrained_dla34(model, num_classes, progress=True):
    """Downloads the official ImageNet-pretrained DLA-34 checkpoint, remaps
    its keys onto this implementation's module names, and loads it.

    The checkpoint's fc is a 1000-class Conv2d (ImageNet). If num_classes
    != 1000, fc.weight/fc.bias are dropped from the checkpoint before
    loading (strict=False) so the model's freshly-initialized fc for your
    num_classes is left untouched -- everything else (patchfy_stem through
    stage_5) loads from ImageNet weights.
    """
    state_dict = load_state_dict_from_url(
        MODEL_URLS["dla34"], progress=progress, map_location="cpu"
    )
    state_dict = _remap_pretrained_state_dict(state_dict)

    if num_classes != 1000:
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    expected_missing = {"fc.weight", "fc.bias"} if num_classes != 1000 else set()
    unexpected_missing = set(missing) - expected_missing
    if unexpected_missing:
        raise RuntimeError(
            f"Unexpected missing keys when loading pretrained DLA-34: {unexpected_missing}. "
            f"The key-remapping in _KEY_PREFIX_MAP may be out of date."
        )
    if unexpected:
        raise RuntimeError(
            f"Unexpected extra keys when loading pretrained DLA-34: {unexpected}. "
            f"The key-remapping in _KEY_PREFIX_MAP may be out of date."
        )
    print(f"[dla34] loaded ImageNet-pretrained weights "
          f"({'fc reinitialized for ' + str(num_classes) + ' classes' if num_classes != 1000 else 'fc kept (1000 classes)'}).")
    return model


def dla34(num_classes, pretrained=True, progress=True, **kwargs):  # DLA-34
    model = DLA(layers=[1, 1, 1, 2, 2, 1],
                channels=[16, 32, 64, 128, 256, 512],
                block=BasicBlock, num_classes=num_classes, **kwargs)

    if pretrained:
        load_pretrained_dla34(model, num_classes, progress=progress)

    return model

# net = dla34(5)
# summary(net.to('cuda'), (3, 224,224))