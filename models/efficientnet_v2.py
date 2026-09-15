import torch.nn as nn
from torchvision.models import (
    efficientnet_v2_s as tv_efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m as tv_efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_l as tv_efficientnet_v2_l, EfficientNet_V2_L_Weights,
)

_VARIANTS = {
    "efficientnetv2_s": (tv_efficientnet_v2_s, EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    "efficientnetv2_m": (tv_efficientnet_v2_m, EfficientNet_V2_M_Weights.IMAGENET1K_V1),
    "efficientnetv2_l": (tv_efficientnet_v2_l, EfficientNet_V2_L_Weights.IMAGENET1K_V1),
}


def _build_efficientnet_v2(variant, num_classes, freeze_features=True):
    """
    ImageNet-pretrained EfficientNetV2 (torchvision's own built-in
    implementation, not the hand-written MBConv/FusedMBConv one this
    REPLACES). Internal architecture details differ slightly from the
    original paper's exact configs, but this is the standard, actively
    maintained, pretrained version -- any old efficientnetv2*.pth checkpoint
    from the previous from-scratch implementation is not compatible.

    freeze_features: freezes `model.features`. Classifier is
    [Dropout, Linear] -- only the Linear is replaced.
    """
    constructor, weights = _VARIANTS[variant]
    model = constructor(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def efficientnetv2_s(num_classes, freeze_features=True):
    return _build_efficientnet_v2("efficientnetv2_s", num_classes, freeze_features)


def efficientnetv2_m(num_classes, freeze_features=True):
    return _build_efficientnet_v2("efficientnetv2_m", num_classes, freeze_features)


def efficientnetv2_l(num_classes, freeze_features=True):
    return _build_efficientnet_v2("efficientnetv2_l", num_classes, freeze_features)
