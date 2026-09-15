import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small as tv_mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large as tv_mobilenet_v3_large, MobileNet_V3_Large_Weights,
)

_VARIANTS = {
    "mobilenet_v3_small": (tv_mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (tv_mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V1),
}


def _freeze_bn_layers(model):
    """
    Freeze every BatchNorm2d layer: fixes its running_mean/running_var at
    the pretrained ImageNet statistics (no more updates) and freezes its
    own weight/bias (affine params) so they don't train either.

    Why this matters for small-batch fine-tuning: with batch_size=16 and
    especially with heavy augmentation, each BN layer's running stats get
    updated from small, distorted batches every step. Those drifted stats
    get baked into the model and used at inference -- but validation/test
    images are clean (no augmentation), so the model ends up evaluating
    with BatchNorm statistics that don't match the real distribution of
    clean images. This is a well-documented failure mode for small-batch
    fine-tuning of any BatchNorm-heavy pretrained network, and it can look
    exactly like what you're seeing: strong training accuracy, unstable
    validation, and a recall/precision balance that shifts sharply between
    validation and test even on the SAME trained weights.

    We also monkey-patch `.train()` so that whenever the training loop
    calls `model.train()` at the start of each epoch, BatchNorm layers are
    immediately put back into eval mode -- otherwise a plain call to
    `.eval()` once wouldn't survive the next `model.train()` call.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for m in bn_modules:
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    original_train = model.train

    def train_with_frozen_bn(mode=True):
        original_train(mode)
        if mode:
            for m in bn_modules:
                m.eval()
        return model

    model.train = train_with_frozen_bn
    return model


def _build_mobilenet_v3(variant, num_classes, freeze_features=False, freeze_bn=True):
    """
    ImageNet-pretrained MobileNetV3 (torchvision). REPLACES the previous
    from-scratch implementation -- any old mobilenet_v3*.pth checkpoint is
    not compatible.

    Default behavior: start from ImageNet weights and fine-tune the ENTIRE
    network (freeze_features=False -> every parameter, including
    `model.features`, is trainable). Matches the VGG/ResNet/GoogLeNet/
    Xception builders' "load pretrained, then fine-tune everything"
    pattern.

    freeze_features=True freezes `model.features` (the conv backbone)
    instead, training only the classifier head. The classifier is
    [Linear, Hardswish, Dropout, Linear] -- only the final Linear is
    replaced.
    """
    constructor, weights = _VARIANTS[variant]
    model = constructor(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    # classifier always trains.
    for param in model.classifier.parameters():
        param.requires_grad = True

    if freeze_bn:
        model = _freeze_bn_layers(model)

    return model


def mobilenet_v3_small(num_classes, freeze_features=False, freeze_bn=True, **kwargs):
    return _build_mobilenet_v3("mobilenet_v3_small", num_classes, freeze_features, freeze_bn)


def mobilenet_v3_large(num_classes, freeze_features=False, freeze_bn=True, **kwargs):
    return _build_mobilenet_v3("mobilenet_v3_large", num_classes, freeze_features, freeze_bn)