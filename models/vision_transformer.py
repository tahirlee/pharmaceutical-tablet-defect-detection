import torch
import torch.nn as nn
from torchvision.models import (
    vit_b_16 as tv_vit_b_16, ViT_B_16_Weights,
    vit_b_32 as tv_vit_b_32, ViT_B_32_Weights,
    vit_l_16 as tv_vit_l_16, ViT_L_16_Weights,
)

_VARIANTS = {
    "vit_base_patch16_224": (tv_vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
    "vit_base_patch32_224": (tv_vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1),
    "vit_large_patch16_224": (tv_vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1),
}


class _StageProxy:
    """
    A lightweight, NON-nn.Module wrapper exposing enough of nn.Sequential's
    interface (iteration, indexing, slicing, len, .children(), .parameters())
    for external fine-tuning code, WITHOUT registering as a duplicate
    submodule in the model's module tree (which would break tools like
    `thop`'s FLOPs counter -- see the Xception builder for the full
    explanation). Wraps modules that are ALREADY registered elsewhere in
    the real module tree (e.g. model.encoder.layers) -- we're just
    providing an alternate, non-duplicating view onto them.
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


def _replace_vit_head_and_expose_stages(model, num_classes):
    """
    Two compatibility fixes for train_recall_aug_control.py's
    architecture-agnostic fine-tuning code:

    1. classifier_modules() only recognizes `net.fc` or `net.classifier`
       as nn.Module instances -- torchvision's ViT natively exposes
       neither (it uses `.heads`), which is why "Cannot find classifier
       head" was raised. Fix: create a genuinely NEW `nn.Linear` and
       register it as `model.classifier` (satisfies the isinstance check
       with no duplication, since it's a brand-new module, not an alias
       of `.heads`). The original `.heads` submodule is then removed, and
       `model.forward` is monkey-patched to call `self.classifier` at the
       same point `self.heads` used to be called (re-implementing
       torchvision's own ViT forward pass, which is: patchify -> prepend
       class token -> transformer encoder -> take the class-token output
       -> head).

    2. configure_finetuning()'s progressive-unfreeze logic only recognizes
       `.layer1`/`.layer4` (ResNet-style) or `.features` (block-list
       style) -- ViT has neither. Fix: expose `model.features` (via
       `_StageProxy`, so nothing is duplicated in the module tree) as the
       list of transformer encoder blocks (`model.encoder.layers`), so
       `--unfreeze_last_n_layers N` thaws the last N transformer blocks,
       the same way it thaws the last N conv blocks for MobileNet/VGG.
    """
    in_features = model.heads.head.in_features
    del model.heads

    model.classifier = nn.Linear(in_features, num_classes)

    def new_forward(self, x):
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = x[:, 0]
        x = self.classifier(x)
        return x

    model.forward = new_forward.__get__(model, type(model))

    # Expose transformer encoder blocks as .features for progressive
    # unfreezing (these are the SAME modules already registered under
    # model.encoder.layers -- no duplication).
    model.features = _StageProxy(list(model.encoder.layers.children()))

    return model


def _build_vit(variant, num_classes, freeze_features=False):
    """
    ImageNet-pretrained Vision Transformer (torchvision). If you had a
    previously-saved checkpoint from a different ViT implementation, it is
    almost certainly NOT compatible with this one -- retrain from this
    pretrained starting point.

    Default behavior: start from ImageNet weights and fine-tune the ENTIRE
    network (freeze_features=False -> every parameter is trainable),
    matching the VGG/ResNet/GoogLeNet/Xception/MobileNet/ShuffleNetV2
    builders' "load pretrained, then fine-tune everything" pattern.

    See `_replace_vit_head_and_expose_stages` for why `.classifier` and
    `.features` are set up the way they are -- both exist specifically to
    make train_recall_aug_control.py's classifier_modules() and
    configure_finetuning() work on this architecture without modifying
    that file.
    """
    constructor, weights = _VARIANTS[variant]
    model = constructor(weights=weights)

    model = _replace_vit_head_and_expose_stages(model, num_classes)

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


def vit_base_patch16_224(num_classes, freeze_features=False, **kwargs):
    return _build_vit("vit_base_patch16_224", num_classes, freeze_features)


def vit_base_patch32_224(num_classes, freeze_features=False, **kwargs):
    return _build_vit("vit_base_patch32_224", num_classes, freeze_features)


def vit_large_patch16_224(num_classes, freeze_features=False, **kwargs):
    return _build_vit("vit_large_patch16_224", num_classes, freeze_features)