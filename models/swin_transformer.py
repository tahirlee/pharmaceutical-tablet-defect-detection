import torch.nn as nn
from torchvision.models import (
    swin_t as tv_swin_t, Swin_T_Weights,
    swin_s as tv_swin_s, Swin_S_Weights,
    swin_b as tv_swin_b, Swin_B_Weights,
)

_VARIANTS = {
    "swin_tiny_patch4_window7_224": (tv_swin_t, Swin_T_Weights.IMAGENET1K_V1),
    "swin_small_patch4_window7_224": (tv_swin_s, Swin_S_Weights.IMAGENET1K_V1),
    "swin_base_patch4_window7_224": (tv_swin_b, Swin_B_Weights.IMAGENET1K_V1),
}


def _replace_swin_head(model, num_classes):
    """
    train_recall_aug_control.py's classifier_modules() only recognizes
    `net.fc` or `net.classifier` as nn.Module instances -- torchvision's
    Swin Transformer natively exposes its head as `.head` (a plain
    nn.Linear), which is why "Cannot find classifier head" was raised.

    Fix: create a genuinely NEW nn.Linear and register it as
    `model.classifier` (satisfies the isinstance check with no
    duplication in the module tree, since it's a brand-new module, not an
    alias of `.head`). The original `.head` is removed, and
    `model.forward` is monkey-patched to call `self.classifier` at the
    same point `self.head` used to be called (re-implementing
    torchvision's own Swin forward pass exactly: patch embed + stages via
    `.features` -> `.norm` -> `.permute` -> `.avgpool` -> `.flatten` ->
    head).

    Note: unlike ViT, Swin's `.features` is ALREADY a native Sequential
    of patch-embedding + stage blocks (torchvision built it that way), so
    configure_finetuning()'s `elif hasattr(net, 'features')` branch works
    automatically here -- no proxy/wrapping needed for that part, only
    for the classifier head naming.
    """
    in_features = model.head.in_features
    del model.head

    model.classifier = nn.Linear(in_features, num_classes)

    def new_forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    model.forward = new_forward.__get__(model, type(model))

    return model


def _build_swin(variant, num_classes, freeze_features=False):
    """
    ImageNet-pretrained Swin Transformer (torchvision). If you had a
    previously-saved checkpoint from a different Swin implementation, it is
    almost certainly NOT compatible with this one -- retrain from this
    pretrained starting point.

    Default behavior: start from ImageNet weights and fine-tune the ENTIRE
    network (freeze_features=False -> every parameter is trainable),
    matching the other builders' "load pretrained, then fine-tune
    everything" pattern.

    See `_replace_swin_head` for why `.classifier` replaces `.head`.
    """
    constructor, weights = _VARIANTS[variant]
    model = constructor(weights=weights)

    model = _replace_swin_head(model, num_classes)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    # classifier always trains.
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def swin_tiny_patch4_window7_224(num_classes, freeze_features=False, **kwargs):
    return _build_swin("swin_tiny_patch4_window7_224", num_classes, freeze_features)


def swin_small_patch4_window7_224(num_classes, freeze_features=False, **kwargs):
    return _build_swin("swin_small_patch4_window7_224", num_classes, freeze_features)


def swin_base_patch4_window7_224(num_classes, freeze_features=False, **kwargs):
    return _build_swin("swin_base_patch4_window7_224", num_classes, freeze_features)