import torch.nn as nn
from torchvision.models import googlenet as tv_googlenet, GoogLeNet_Weights


def _googlenet_stages(model):
    """
    Group GoogLeNet's flat layer layout into 4 stages (stem, inception3,
    inception4, inception5), analogous to ResNet's layer1..layer4, for
    progressive unfreezing.
    """
    stage1 = nn.ModuleList([model.conv1, model.maxpool1, model.conv2, model.conv3, model.maxpool2])
    stage2 = nn.ModuleList([model.inception3a, model.inception3b, model.maxpool3])
    stage3 = nn.ModuleList([
        model.inception4a, model.inception4b, model.inception4c,
        model.inception4d, model.inception4e, model.maxpool4,
    ])
    stage4 = nn.ModuleList([model.inception5a, model.inception5b])
    return [stage1, stage2, stage3, stage4]


def googlenet(
    num_classes,
    freeze_features=False,
    unfreeze_last_n_layers=4,
    **kwargs,
):
    """
    ImageNet-pretrained GoogLeNet/Inception-v1 (torchvision).

    The pretrained checkpoint was trained with aux_logits=True, so we
    construct with that to load correctly, then disable aux_logits (and
    drop the aux1/aux2 heads) since we only want a single-tensor output for
    our simple training loop -- the aux heads exist for training big models
    from scratch and aren't needed for fine-tuning a frozen/pretrained one.
    (Note: aux1/aux2 are NOT pretrained even when weights=IMAGENET1K_V1 --
    torchvision warns about this -- but that's irrelevant here since we
    discard them.)

    Default behavior: start from ImageNet weights and fine-tune the ENTIRE
    network (freeze_features=False -> every parameter is trainable),
    matching the VGG/ResNet builders' "load pretrained, then fine-tune
    everything" pattern.

    Set freeze_features=True to freeze the backbone instead, with
    unfreeze_last_n_layers controlling how many of the last stages (out of
    4 total: stem, inception3, inception4, inception5) are thawed on top
    of a frozen backbone:
        0 -> train only fc
        1 -> train inception5 + fc
        2 -> train inception4 + inception5 + fc
        3 -> train inception3 + inception4 + inception5 + fc
        4 -> train stem + inception3 + inception4 + inception5 + fc (== full fine-tune)
    """
    if unfreeze_last_n_layers < 0 or unfreeze_last_n_layers > 4:
        raise ValueError(
            "unfreeze_last_n_layers for GoogLeNet must be between 0 and 4."
        )

    model = tv_googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=True)
    model.aux_logits = False
    model.aux1 = None
    model.aux2 = None

    if freeze_features:
        # Freeze everything, then selectively thaw the last N stages.
        for param in model.parameters():
            param.requires_grad = False

        stages = _googlenet_stages(model)
        if unfreeze_last_n_layers > 0:
            for stage in stages[-unfreeze_last_n_layers:]:
                for module in stage:
                    for param in module.parameters():
                        param.requires_grad = True
    else:
        # Full fine-tuning: everything trainable (default).
        for param in model.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # fc always trains.
    for param in model.fc.parameters():
        param.requires_grad = True

    return model