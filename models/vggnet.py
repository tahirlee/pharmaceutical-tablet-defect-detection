import torch.nn as nn
from torchvision.models import (
    vgg11 as tv_vgg11, VGG11_Weights,
    vgg13 as tv_vgg13, VGG13_Weights,
    vgg16 as tv_vgg16, VGG16_Weights,
    vgg19 as tv_vgg19, VGG19_Weights,
)

_VARIANTS = {
    "vgg11": (tv_vgg11, VGG11_Weights.IMAGENET1K_V1),
    "vgg13": (tv_vgg13, VGG13_Weights.IMAGENET1K_V1),
    "vgg16": (tv_vgg16, VGG16_Weights.IMAGENET1K_V1),
    "vgg19": (tv_vgg19, VGG19_Weights.IMAGENET1K_V1),
}


def _vgg_feature_blocks(model):
    """
    Split `model.features` (one flat Sequential of conv/relu/pool) into its
    5 conv "stages", using each MaxPool2d as a stage boundary. This gives
    an analogue of ResNet's layer1..layer4 for progressive unfreezing.
    Returns a list of nn.Sequential blocks, e.g. [stage1, ..., stage5].
    """
    blocks = []
    current = []
    for layer in model.features:
        current.append(layer)
        if isinstance(layer, nn.MaxPool2d):
            blocks.append(nn.Sequential(*current))
            current = []
    if current:
        blocks.append(nn.Sequential(*current))
    return blocks


def _build_vgg(
    variant,
    num_classes,
    freeze_features=False,
    unfreeze_last_n_layers=5,
):
    """
    Build ImageNet-pretrained VGG.

    Default behavior: start from ImageNet weights and fine-tune the ENTIRE
    network (freeze_features=False -> every parameter, including all conv
    stages, is trainable). Matches the ResNet builder's "load pretrained,
    then fine-tune everything" pattern.

    Set freeze_features=True to freeze the backbone instead, with
    unfreeze_last_n_layers controlling how many of the last conv stages
    (out of 5 total, split on each MaxPool2d) are thawed on top of a
    frozen backbone:
        0 -> train only classifier
        1 -> train stage5 + classifier
        2 -> train stage4 + stage5 + classifier
        3 -> train stage3 + stage4 + stage5 + classifier
        4 -> train stage2..stage5 + classifier
        5 -> train stage1..stage5 + classifier  (== full fine-tune)

    torchvision's VGG classifier is [Linear(25088,4096), ReLU, Dropout,
    Linear(4096,4096), ReLU, Dropout, Linear(4096,1000)] -- only the final
    Linear is replaced.
    """
    if unfreeze_last_n_layers < 0 or unfreeze_last_n_layers > 5:
        raise ValueError(
            "unfreeze_last_n_layers for VGG must be between 0 and 5."
        )

    constructor, weights = _VARIANTS[variant]
    model = constructor(weights=weights)

    if freeze_features:
        # Freeze everything, then selectively thaw the last N conv stages.
        for param in model.parameters():
            param.requires_grad = False

        stages = _vgg_feature_blocks(model)
        if unfreeze_last_n_layers > 0:
            for stage in stages[-unfreeze_last_n_layers:]:
                for param in stage.parameters():
                    param.requires_grad = True
    else:
        # Full fine-tuning: everything trainable (default).
        for param in model.parameters():
            param.requires_grad = True

    # Replace ImageNet classifier head with new num_classes classifier.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # classifier always trains.
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def vgg11(num_classes, freeze_features=False, unfreeze_last_n_layers=5, **kwargs):
    return _build_vgg("vgg11", num_classes, freeze_features, unfreeze_last_n_layers)


def vgg13(num_classes, freeze_features=False, unfreeze_last_n_layers=5, **kwargs):
    return _build_vgg("vgg13", num_classes, freeze_features, unfreeze_last_n_layers)


def vgg16(num_classes, freeze_features=False, unfreeze_last_n_layers=5, **kwargs):
    return _build_vgg("vgg16", num_classes, freeze_features, unfreeze_last_n_layers)


def vgg19(num_classes, freeze_features=False, unfreeze_last_n_layers=5, **kwargs):
    return _build_vgg("vgg19", num_classes, freeze_features, unfreeze_last_n_layers)