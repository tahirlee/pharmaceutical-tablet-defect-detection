import torch.nn as nn
from torchvision.models import (
    resnet34 as tv_resnet34, ResNet34_Weights,
    resnet50 as tv_resnet50, ResNet50_Weights,
    resnet101 as tv_resnet101, ResNet101_Weights,
    resnext50_32x4d as tv_resnext50_32x4d, ResNeXt50_32X4D_Weights,
    resnext101_32x8d as tv_resnext101_32x8d, ResNeXt101_32X8D_Weights,
)


_VARIANTS = {
    "resnet34": (
        tv_resnet34,
        ResNet34_Weights.IMAGENET1K_V1,
    ),
    "resnet50": (
        tv_resnet50,
        ResNet50_Weights.IMAGENET1K_V1,
    ),
    "resnet101": (
        tv_resnet101,
        ResNet101_Weights.IMAGENET1K_V1,
    ),
    "resnext50_32x4d": (
        tv_resnext50_32x4d,
        ResNeXt50_32X4D_Weights.IMAGENET1K_V1,
    ),
    "resnext101_32x8d": (
        tv_resnext101_32x8d,
        ResNeXt101_32X8D_Weights.IMAGENET1K_V1,
    ),
}


def _build_resnet(
    variant,
    num_classes,
    freeze_features=True,
    unfreeze_last_n_layers=0,
):
    """
    Build ImageNet-pretrained ResNet / ResNeXt.

    Fine-tuning groups:
        0 -> train only fc
        1 -> train layer4 + fc
        2 -> train layer3 + layer4 + fc
        3 -> train layer2 + layer3 + layer4 + fc
        4 -> train layer1 + layer2 + layer3 + layer4 + fc
    """
    if unfreeze_last_n_layers < 0 or unfreeze_last_n_layers > 4:
        raise ValueError(
            "unfreeze_last_n_layers for ResNet must be between 0 and 4."
        )

    constructor, weights = _VARIANTS[variant]
    model = constructor(weights=weights)

    # Freeze pretrained backbone first.
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Replace ImageNet classifier with new 2-class classifier.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # fc always trains.
    for param in model.fc.parameters():
        param.requires_grad = True

    # Explicitly unfreeze the last ResNet stages.
    resnet_stages = [
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]

    if unfreeze_last_n_layers > 0:
        for stage in resnet_stages[-unfreeze_last_n_layers:]:
            for param in stage.parameters():
                param.requires_grad = True

    return model


def resnet34(
    num_classes,
    freeze_features=True,
    unfreeze_last_n_layers=0,
    **kwargs,
):
    return _build_resnet(
        "resnet34",
        num_classes,
        freeze_features,
        unfreeze_last_n_layers,
    )


def resnet50(
    num_classes,
    freeze_features=True,
    unfreeze_last_n_layers=0,
    **kwargs,
):
    return _build_resnet(
        "resnet50",
        num_classes,
        freeze_features,
        unfreeze_last_n_layers,
    )


def resnet101(
    num_classes,
    freeze_features=True,
    unfreeze_last_n_layers=0,
    **kwargs,
):
    return _build_resnet(
        "resnet101",
        num_classes,
        freeze_features,
        unfreeze_last_n_layers,
    )


def resnext50_32x4d(
    num_classes,
    freeze_features=True,
    unfreeze_last_n_layers=0,
    **kwargs,
):
    return _build_resnet(
        "resnext50_32x4d",
        num_classes,
        freeze_features,
        unfreeze_last_n_layers,
    )


def resnext101_32x8d(
    num_classes,
    freeze_features=True,
    unfreeze_last_n_layers=0,
    **kwargs,
):
    return _build_resnet(
        "resnext101_32x8d",
        num_classes,
        freeze_features,
        unfreeze_last_n_layers,
    )
