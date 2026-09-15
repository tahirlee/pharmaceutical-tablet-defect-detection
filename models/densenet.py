import torch.nn as nn
from torchvision.models import (
    densenet121 as tv_densenet121, DenseNet121_Weights,
    densenet169 as tv_densenet169, DenseNet169_Weights,
    densenet201 as tv_densenet201, DenseNet201_Weights,
    densenet161 as tv_densenet161, DenseNet161_Weights,
)

# Maps a variant name to its torchvision constructor + ImageNet-pretrained
# weights enum. Keeps a single _build_densenet() implementation below instead
# of duplicating it per variant.
_VARIANTS = {
    "densenet121": (tv_densenet121, DenseNet121_Weights.IMAGENET1K_V1),
    "densenet169": (tv_densenet169, DenseNet169_Weights.IMAGENET1K_V1),
    "densenet201": (tv_densenet201, DenseNet201_Weights.IMAGENET1K_V1),
    "densenet161": (tv_densenet161, DenseNet161_Weights.IMAGENET1K_V1),
}


def _build_densenet(variant: str, num_classes: int):
    """
    Builds a DenseNet initialized with ImageNet-pretrained weights
    (torchvision), with only the final classification layer replaced for
    `num_classes`.

    NOTE ON FREEZING: this always returns the model with the FULL backbone
    trainable (nothing is frozen here). Freeze policy -- full fine-tune vs.
    classifier-only vs. last-N-blocks -- is applied uniformly across every
    architecture afterwards, in configure_finetuning() inside the training
    script, driven by --freeze_features / --no_freeze_features and
    --unfreeze_last_n_layers. That function calls
    set_requires_grad(net, False/True) on the WHOLE model before selectively
    re-enabling parts, so any freezing decision made here would just be
    silently overwritten there. Keeping it out of the builder means there is
    exactly one place in the codebase that decides what trains, for every
    model -- avoids the two disagreeing, which is what caused the earlier
    bug where DenseNet's `--unfreeze_last_n_layers 1` only trained a bare
    closing BatchNorm (norm5) instead of a real block.

    For a full fine-tune run: `--no_freeze_features` (nothing extra needed
    in this file).
    """
    if variant not in _VARIANTS:
        raise ValueError(
            f"Unknown densenet variant '{variant}'. Choose from: {list(_VARIANTS.keys())}"
        )

    constructor, weights = _VARIANTS[variant]
    model = constructor(weights=weights)  # ImageNet-pretrained starting point

    # torchvision's DenseNet classifier is a single Linear(num_features, 1000).
    # Replace it so it outputs num_classes instead.
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


def densenet121(num_classes, **kwargs):
    return _build_densenet("densenet121", num_classes)


def densenet169(num_classes, **kwargs):
    return _build_densenet("densenet169", num_classes)


def densenet201(num_classes, **kwargs):
    return _build_densenet("densenet201", num_classes)


def densenet161(num_classes, **kwargs):
    return _build_densenet("densenet161", num_classes)