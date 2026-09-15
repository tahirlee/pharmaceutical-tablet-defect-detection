import torch.nn as nn
from torchvision.models import alexnet as tv_alexnet, AlexNet_Weights


def alexnet(num_classes, freeze_features=True):
    """
    Builds AlexNet initialized with ImageNet-pretrained weights (torchvision),
    with only the final classification layer replaced for `num_classes`.

    Keeps the same call signature `classic_models.find_model_using_name`
    already expects: alexnet(num_classes) -> nn.Module

    Note: this REPLACES the previous from-scratch custom AlexNet (which used
    the original-paper channel sizes 96/256/384/384/256 and had no pretrained
    weights available, and was never actually calling _initialize_weights()
    since alexnet() didn't pass init_weights=True). torchvision's AlexNet
    uses different internal channel sizes (64/192/384/256/256), so any
    previously-saved alexnet.pth checkpoint from the old architecture is NOT
    compatible and can't be loaded into this one -- you'll need to retrain
    from this new pretrained starting point.

    freeze_features: if True (default), the pretrained convolutional
    backbone is frozen and only the classifier head is trained. This is
    strongly recommended for small datasets like this one (~1600 training
    images) since fine-tuning all ~57M parameters of AlexNet on that little
    data is a fast route back to the same collapse/overfitting behavior you
    were seeing. Once training is stable and val accuracy is clearly above
    chance, you can try freeze_features=False (or unfreeze just the last
    conv block) for a final fine-tuning pass at a lower learning rate.
    """
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = tv_alexnet(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # torchvision's classifier is:
    # [Dropout, Linear(9216,4096), ReLU, Dropout, Linear(4096,4096), ReLU, Linear(4096,1000)]
    # Replace just the final Linear so it outputs num_classes instead of 1000.
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model