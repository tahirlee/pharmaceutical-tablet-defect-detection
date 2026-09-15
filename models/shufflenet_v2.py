import torch.nn as nn
from torchvision.models import (
    shufflenet_v2_x0_5 as tv_shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
    shufflenet_v2_x1_0 as tv_shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
)

_VARIANTS = {
    "shufflenet_v2_x0_5": (tv_shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1),
    "shufflenet_v2_x1_0": (tv_shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
}


class _StageProxy:
    """
    A lightweight, NON-nn.Module wrapper around a list of submodules.
    Exposes enough of nn.Sequential's interface (iteration, indexing,
    slicing, len, .children(), .parameters()) for external fine-tuning
    code to treat it like a real stage -- WITHOUT registering as a
    duplicate submodule in the model's module tree (which would break
    tools like `thop`'s FLOPs counter -- see the Xception builder for
    the full explanation of this failure mode).
    """

    def __init__(self, modules):
        self._mods = list(modules)

    def __len__(self):
        return len(self._mods)

    def __iter__(self):
        return iter(self._mods)

    def __getitem__(self, idx):
        result = self._mods[idx]
        if isinstance(idx, slice):
            return _StageProxy(result)
        return result

    def children(self):
        return iter(self._mods)

    def parameters(self):
        for m in self._mods:
            yield from m.parameters()


def _wrap_shufflenet_stages(model):
    """
    Expose torchvision's ShuffleNetV2 layout as .layer1 .. .layer4,
    mirroring ResNet's naming, so external fine-tuning code that only
    knows how to handle "ResNet stages" (checking for .layer1..layer4)
    or ".features" works on ShuffleNetV2 too.

    torchvision's ShuffleNetV2 layout:
        conv1, maxpool, stage2, stage3, stage4, conv5, fc
    """
    model.layer1 = _StageProxy([model.conv1, model.maxpool])
    model.layer2 = _StageProxy([model.stage2])
    model.layer3 = _StageProxy([model.stage3])
    model.layer4 = _StageProxy([model.stage4, model.conv5])
    return model


def _build_shufflenet_v2(variant, num_classes, freeze_features=False, unfreeze_last_n_layers=4):
    """
    Build ImageNet-pretrained ShuffleNetV2. REPLACES the previous
    from-scratch implementation -- any old shufflenet*.pth checkpoint is
    not compatible.

    Default behavior: start from ImageNet weights and fine-tune the ENTIRE
    network (freeze_features=False -> every parameter is trainable),
    matching the VGG/ResNet/GoogLeNet/Xception/MobileNet builders' "load
    pretrained, then fine-tune everything" pattern.

    .layer1..layer4 (see _wrap_shufflenet_stages) let external code use
    the same ResNet-style fine-tuning groups:
        0 -> train only fc
        1 -> train layer4 (stage4 + conv5) + fc
        2 -> train layer3 + layer4 + fc
        3 -> train layer2 + layer3 + layer4 + fc
        4 -> train layer1 + layer2 + layer3 + layer4 + fc (== full fine-tune)
    """
    if unfreeze_last_n_layers < 0 or unfreeze_last_n_layers > 4:
        raise ValueError(
            "unfreeze_last_n_layers for ShuffleNetV2 must be between 0 and 4."
        )

    constructor, weights = _VARIANTS[variant]
    model = constructor(weights=weights)
    model = _wrap_shufflenet_stages(model)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # fc always trains.
    for param in model.fc.parameters():
        param.requires_grad = True

    if freeze_features and unfreeze_last_n_layers > 0:
        stages = [model.layer1, model.layer2, model.layer3, model.layer4]
        for stage in stages[-unfreeze_last_n_layers:]:
            for param in stage.parameters():
                param.requires_grad = True

    return model


def shufflenet_v2_x0_5(num_classes, freeze_features=False, unfreeze_last_n_layers=4, **kwargs):
    return _build_shufflenet_v2("shufflenet_v2_x0_5", num_classes, freeze_features, unfreeze_last_n_layers)


def shufflenet_v2_x1_0(num_classes, freeze_features=False, unfreeze_last_n_layers=4, **kwargs):
    return _build_shufflenet_v2("shufflenet_v2_x1_0", num_classes, freeze_features, unfreeze_last_n_layers)