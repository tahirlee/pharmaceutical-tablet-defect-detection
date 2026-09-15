from .alexnet import alexnet
from .vggnet import vgg11, vgg13, vgg16, vgg19
from .zfnet import zfnet
from .googlenet_v1 import googlenet
from .xception import xception
from .resnet import  resnet34, resnet50, resnet101, resnext50_32x4d, resnext101_32x8d
from .densenet import densenet121, densenet161, densenet169, densenet201
from .dla import dla34
from .mobilenet_v3 import mobilenet_v3_small, mobilenet_v3_large
from .shufflenet_v2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0
from .efficientnet_v2 import efficientnetv2_l, efficientnetv2_m, efficientnetv2_s
from .convnext import convnext_tiny, convnext_small, convnext_base, convnext_large, convnext_xlarge

from .vision_transformer import vit_base_patch16_224, vit_base_patch32_224, vit_large_patch16_224
from .swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224

from .proposed_model import proposed_model


cfgs = {
    'alexnet': alexnet,
    'zfnet': zfnet,
    'vgg': vgg16,
    'vgg_tiny': vgg11,
    'vgg_small': vgg13,
    'vgg_big': vgg19,
    'googlenet': googlenet,
    'xception': xception,
    'resnet_small': resnet34,
    'resnet': resnet50,
    'resnet_big': resnet101,
    'resnext': resnext50_32x4d,
    'resnext_big': resnext101_32x8d,
    'densenet_tiny': densenet121,
    'densenet_small': densenet161,
    'densenet': densenet169,
    'densenet_big': densenet201,
    # Direct variant names, so --model densenet121 (etc.) works without
    # needing to know the size-tier alias.
    'densenet121': densenet121,
    'densenet161': densenet161,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'dla': dla34,
    'mobilenet_v3': mobilenet_v3_small,
    'mobilenet_v3_large': mobilenet_v3_large,
    'shufflenet_small':shufflenet_v2_x0_5,
    'shufflenet': shufflenet_v2_x1_0,
    'efficient_v2_small': efficientnetv2_s,
    'efficient_v2': efficientnetv2_m,
    'efficient_v2_large': efficientnetv2_l,
    'convnext_tiny': convnext_tiny,
    'convnext_small': convnext_small,
    'convnext': convnext_base,
    'convnext_big': convnext_large,
    'convnext_huge': convnext_xlarge,

    'vision_transformer_small': vit_base_patch32_224,
    'vision_transformer': vit_base_patch16_224,
    'vision_transformer_big': vit_large_patch16_224,
    'swin_transformer_tiny': swin_tiny_patch4_window7_224,
    'swin_transformer_small': swin_small_patch4_window7_224,
    'swin_transformer': swin_base_patch4_window7_224,
    'apfanet': proposed_model

}

def find_model_using_name(model_name, num_classes, freeze_features=False):
    constructor = cfgs[model_name]
    try:
        # Models with a pretrained-backbone concept (VGG, ResNet, GoogLeNet,
        # Xception, MobileNetV3, DenseNet, etc.) accept freeze_features to
        # control whether the ImageNet backbone is frozen or fine-tuned.
        # Default is False (full fine-tuning) so that any caller -- like
        # test.py -- that builds a model without specifying freeze_features
        # gets a model matching "trainable params == total params" for
        # architecture reporting, unless the caller explicitly asks to
        # freeze. Actual TRAINING runs (train_recall_aug_control.py) pass
        # freeze_features/unfreeze_last_n_layers explicitly via CLI flags,
        # so this default only affects callers that don't specify it.
        return constructor(num_classes, freeze_features=freeze_features)
    except TypeError:
        # Other architectures (from-scratch, no pretrained-backbone concept)
        # don't accept freeze_features at all -- call without it instead of
        # failing. This also means train.py's own try/except fallback for
        # freeze_features is no longer needed, though it's harmless to keep.
        return constructor(num_classes)