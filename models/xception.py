import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "Xception with ImageNet-pretrained weights requires `timm` "
        "(torchvision has no xception model). Install it with:\n"
        "    pip install timm"
    ) from e


_VARIANTS = {
    "xception": "legacy_xception",
}


class _StageProxy:
    """
    A lightweight, NON-nn.Module wrapper around a list of submodules.

    Why not nn.ModuleList? Because nn.ModuleList *is* an nn.Module, so
    assigning it as `model.layer1 = nn.ModuleList([model.conv1, ...])`
    makes PyTorch register it as a real submodule -- which means conv1
    (etc.) now appears TWICE in the module tree: once as `model.conv1`
    and again as `model.layer1[0]`. That duplication is harmless for
    training, but breaks tools that walk the module tree and attach
    per-module hooks/buffers exactly once (e.g. `thop`'s FLOPs counter
    used in test.py), causing
        AttributeError: 'Conv2d' object has no attribute 'total_ops'

    `_StageProxy` is a plain Python object, so PyTorch's nn.Module
    `__setattr__` does NOT register it as a submodule -- it stays outside
    the module tree entirely. It still exposes `.parameters()` and
    `__iter__`, so any external code (e.g. configure_finetuning()) that
    does `stage.parameters()` or `for module in stage: ...` keeps working
    exactly as if it were a real ResNet stage.
    """

    def __init__(self, modules):
        self._mods = list(modules)

    def parameters(self):
        for m in self._mods:
            yield from m.parameters()

    def __iter__(self):
        return iter(self._mods)


def _wrap_xception_stages(model):
    """
    Expose timm's Xception blocks as .layer1 .. .layer4 (via _StageProxy,
    see above), mirroring ResNet's naming, so external code that only
    knows how to freeze/unfreeze "ResNet stages" (checking for
    .layer1..layer4) works on Xception unmodified.

    timm's xception layout:
        conv1, bn1, act1, conv2, bn2, act2,   (stem)
        block1 .. block12,                     (entry/middle blocks)
        conv3, bn3, act3, conv4, bn4, act4,   (exit)
        fc
    """
    model.layer1 = _StageProxy([
        model.conv1, model.bn1, model.act1,
        model.conv2, model.bn2, model.act2,
        model.block1, model.block2, model.block3, model.block4,
    ])
    model.layer2 = _StageProxy([model.block5, model.block6, model.block7, model.block8])
    model.layer3 = _StageProxy([model.block9, model.block10, model.block11, model.block12])
    model.layer4 = _StageProxy([
        model.conv3, model.bn3, model.act3,
        model.conv4, model.bn4, model.act4,
    ])
    return model


def _build_xception(
    variant,
    num_classes,
    freeze_features=True,
    unfreeze_last_n_layers=0,
):
    """
    Build ImageNet-pretrained Xception (via `timm`, since torchvision has
    no xception model). Structured identically to `_build_resnet` so the
    same external fine-tuning/freezing code works unchanged.

    Fine-tuning groups (same convention as ResNet):
        0 -> train only fc
        1 -> train layer4 + fc
        2 -> train layer3 + layer4 + fc
        3 -> train layer2 + layer3 + layer4 + fc
        4 -> train layer1 + layer2 + layer3 + layer4 + fc  (== full fine-tune)
    """
    if unfreeze_last_n_layers < 0 or unfreeze_last_n_layers > 4:
        raise ValueError(
            "unfreeze_last_n_layers for Xception must be between 0 and 4."
        )

    timm_name = _VARIANTS[variant]
    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    model = _wrap_xception_stages(model)

    # Freeze pretrained backbone first.
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # fc always trains (timm already sized it to num_classes above).
    for param in model.fc.parameters():
        param.requires_grad = True

    # Explicitly unfreeze the last Xception stages.
    xception_stages = [
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]

    if unfreeze_last_n_layers > 0:
        for stage in xception_stages[-unfreeze_last_n_layers:]:
            for param in stage.parameters():
                param.requires_grad = True

    return model


def xception(
    num_classes,
    freeze_features=False,
    unfreeze_last_n_layers=4,
    **kwargs,
):
    return _build_xception(
        "xception",
        num_classes,
        freeze_features,
        unfreeze_last_n_layers,
    )