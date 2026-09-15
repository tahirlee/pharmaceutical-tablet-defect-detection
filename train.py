import os
import argparse
import math
import shutil
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import classic_models
from utils.lr_methods import warmup
from dataload.dataload_five_flower import Five_Flowers_Load
from utils.train_engin import train_one_epoch, evaluate



# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size for training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay for Adam')
    parser.add_argument('--seed', default=True, action='store_true', help='fix the initialization of parameters')
    parser.add_argument('--tensorboard', default=False, action='store_true', help='use tensorboard for visualization')
    parser.add_argument('--use_amp', default=True, action='store_true', help='training with mixed precision')
    parser.add_argument(
        '--data_path',
        type=str,
        default=os.path.join('datasets', 'pv'),
        help='dataset root, relative to the current working directory'
    )
    parser.add_argument('--num_classes', type=int, default=2, help='the number of classes')
    parser.add_argument('--model', type=str, default="apfanet", help='select a model for training')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--patience', type=int, default=15,
                         help='early stopping patience (epochs with no val F2 improvement)')

    parser.add_argument('--freeze_features', dest='freeze_features', action='store_true',
                         help='freeze the pretrained backbone and only train the classifier head (default)')
    parser.add_argument('--no_freeze_features', dest='freeze_features', action='store_false',
                         help='fine-tune the whole network, including the pretrained backbone')
    parser.set_defaults(freeze_features=True)
    parser.add_argument('--unfreeze_last_n_layers', type=int, default=1)

    parser.add_argument('--freeze_classifier_base', dest='freeze_classifier_base', action='store_true',
                         help='also freeze the two large 4096-unit FC layers in the classifier head, '
                              'training only the final Linear(4096, num_classes) layer. Cuts trainable '
                              'params from ~54M to ~8K -- use this if the classifier head itself is '
                              'still overfitting a small dataset even with the conv backbone frozen. (default)')
    parser.add_argument('--no_freeze_classifier_base', dest='freeze_classifier_base', action='store_false',
                         help='train the full classifier head (both 4096-unit FC layers + final layer)')
    parser.set_defaults(freeze_classifier_base=True)

    parser.add_argument('--use_class_weights', dest='use_class_weights', action='store_true',
                         help='weight CrossEntropyLoss inversely to class frequency, to counter the '
                              'model defaulting to the majority-behaving class (e.g. under-predicting '
                              '"defective") on an imbalanced dataset. (default)')
    parser.add_argument('--no_class_weights', dest='use_class_weights', action='store_false',
                         help='use plain unweighted CrossEntropyLoss')
    parser.set_defaults(use_class_weights=True)

    parser.add_argument('--loss_type', type=str, default='focal', choices=['ce', 'focal'],
                         help="'ce' = (weighted) CrossEntropyLoss. 'focal' (default) = Focal Loss, "
                              "which additionally concentrates training on hard/ambiguous samples "
                              "instead of only rescaling by class frequency -- recommended when "
                              "class-weighted CE alone still leaves low recall on the minority/harder class.")
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                         help='focusing strength for --loss_type focal. 0 reduces to plain weighted CE; '
                              'higher values push harder on hard examples. Paper default / good starting point: 2.0')
    parser.add_argument('--positive_class_index', type=int, default=0,
                         help='numeric index for defective. Use the index assigned to defective in class_indices.json.')
    parser.add_argument('--val_threshold', type=float, default=0.40,
                         help='predict defective if P(defective) >= this value; use the same value in test.py.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                         help='label smoothing for --loss_type ce (ignored for focal). 0 disables it.')

    parser.add_argument('--augmentation', type=str, default='none', choices=['none', 'light', 'heavy'],
                         help='online augmentation for original images. Start with none for a clean baseline. '
                              'When --offline_augmented is set, light is used regardless.')
    parser.add_argument('--offline_augmented', dest='offline_augmented', action='store_true',
                         help='set this when --data_path/Train already contains offline-augmented images '
                              '(see offline_augmentation.py). Switches to a much lighter online transform '
                              '(flip + normalize + light noise only) so distortion does not compound on '
                              'top of images that are already rotated/jittered/blurred on disk.')
    parser.set_defaults(offline_augmented=False)

    return parser


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('Random seed has been fixed.')


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
class AddGaussianNoise(object):
    """
    Adds Gaussian noise to a (already ToTensor'd + Normalize'd) image tensor.
    Applied only during training as an extra augmentation to discourage the
    classifier head from memorizing exact pixel patterns on a small dataset.
    std=0.05 is a mild amount relative to normalized pixel range; raise it
    (e.g. 0.08-0.10) if train/val gap is still large, lower it if training
    loss stops decreasing / becomes unstable.
    """
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


def build_transforms(args):
    """Returns (data_transform dict, transform_name) for logging."""
    heavy_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.90, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.01),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        AddGaussianNoise(std=0.02),
    ])

    # Augmentation redesigned around what actually distinguishes the two
    # classes in this dataset: "normal" = an intact whole tablet/capsule,
    # "defective" = a physically damaged one (crack, chip, split, or surface
    # staining). That changes a few defaults from a generic-classification
    # recipe:
    #   - Crop scale kept fairly tight (0.85-1.0, not 0.7-1.0): tablets sit
    #     roughly centered in each photo and a defect can be a small edge
    #     chip -- cropping too aggressively risks cropping the defect itself
    #     out of frame.
    #   - Vertical flip added: tablet orientation carries no class signal
    #     here (unlike e.g. natural photos), so this is free extra variety.
    #   - Color jitter kept, but hue lowered further and saturation slightly
    #     reduced: some defective samples show color staining, which IS
    #     part of the defect signal -- jittering hue/saturation too hard
    #     risks washing out or fabricating the very cue that matters.
    #   - Added light GaussianBlur: simulates real focus/camera variation
    #     seen across your photos without touching color or shape.

    # Used instead of heavy_train_transform when --offline_augmented is set:
    # the dataset on disk already contains rotated/flipped/jittered/blurred
    # copies (see offline_augmentation.py), so this only does light,
    # non-destructive variety (crop + flip + normalize + a touch of noise)
    # rather than compounding another full round of heavy distortion on top.
    light_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        AddGaussianNoise(mean=0., std=0.02),
    ])

    no_aug_train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.offline_augmented:
        selected_train_transform = light_train_transform
        transform_name = 'light (offline-augmented dataset)'
    elif args.augmentation == 'none':
        selected_train_transform = no_aug_train_transform
        transform_name = 'none (baseline)'
    elif args.augmentation == 'light':
        selected_train_transform = light_train_transform
        transform_name = 'light'
    else:
        selected_train_transform = heavy_train_transform
        transform_name = 'heavy'

    data_transform = {
        "Train": selected_train_transform,
        "Val": transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }
    return data_transform, transform_name


# --------------------------------------------------------------------------- #
# Class weighting
# --------------------------------------------------------------------------- #
def compute_class_weights(dataset, num_classes, device):
    """
    Computes per-class weights (total_samples / (num_classes * class_count))
    for use in a weighted CrossEntropyLoss, to counter class imbalance / a
    model that's biased toward predicting one class (e.g. under-predicting
    "defective" -- low recall on that class even though overall accuracy
    looks okay).

    Tries a few common label-list attribute names on the dataset first
    (fast path, no re-transforming images). If none are found, falls back
    to iterating the dataset via __getitem__ -- works for any dataset class,
    just slower since it re-applies the Train transform once per sample.
    Either way this only runs once at startup, not per epoch.
    """
    labels = None
    for attr_name in ("images_label", "img_label", "labels", "targets", "images_class"):
        if hasattr(dataset, attr_name):
            labels = getattr(dataset, attr_name)
            break

    if labels is None:
        print("compute_class_weights: no label-list attribute found on dataset "
              "(tried images_label/img_label/labels/targets/images_class); "
              "falling back to iterating dataset via __getitem__ (slower, runs once).")
        labels = [dataset[i][1] for i in range(len(dataset))]

    class_counts = torch.zeros(num_classes)
    for label in labels:
        class_counts[int(label)] += 1

    if (class_counts == 0).any():
        raise ValueError(
            f"compute_class_weights: one or more classes have zero training samples: "
            f"{class_counts.tolist()}. Check your dataset."
        )

    class_weights = class_counts.sum() / (num_classes * class_counts)
    print(f"Class counts (train set): {class_counts.tolist()}")
    print(f"Class weights (for loss): {class_weights.tolist()}")
    return class_weights.to(device)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_defect_metrics(model, data_loader, device, positive_class_index, threshold):
    """Evaluate binary defect detection at an explicit operating threshold.

    Returns accuracy, precision, recall, F1, F2, and confusion-counts. F2 is
    used for checkpoint selection because it weights recall twice as much as
    precision, matching a safety/quality-inspection setting.
    """
    if positive_class_index not in (0, 1):
        raise ValueError('--positive_class_index must be 0 or 1 for binary classification.')
    if not 0.0 < threshold < 1.0:
        raise ValueError('--val_threshold must be between 0 and 1.')

    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probabilities = torch.softmax(model(images), dim=1)[:, positive_class_index]
            predicted_positive = probabilities >= threshold
            actual_positive = labels == positive_class_index
            tp += (predicted_positive & actual_positive).sum().item()
            fp += (predicted_positive & ~actual_positive).sum().item()
            tn += (~predicted_positive & ~actual_positive).sum().item()
            fn += (~predicted_positive & actual_positive).sum().item()

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    beta_sq = 4.0  # F2: recall has 2x the importance of precision
    f2 = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'f1': f1, 'f2': f2, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


# --------------------------------------------------------------------------- #
# Fine-tuning policy (architecture-aware, applied identically to every model)
# --------------------------------------------------------------------------- #
def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def classifier_modules(net):
    # torchvision ResNet/ResNeXt: model.fc; most other torchvision models:
    # model.classifier. Return classifier modules that must always train.
    if hasattr(net, 'fc') and isinstance(net.fc, nn.Module):
        return [net.fc]
    if hasattr(net, 'classifier') and isinstance(net.classifier, nn.Module):
        return [net.classifier]
    raise AttributeError(
        'Cannot find classifier head. Add this architecture to classifier_modules().'
    )


def configure_finetuning(net, freeze_features, unfreeze_last_n_layers, freeze_classifier_base):
    """Architecture-aware fine-tuning without relying on .features only.

    ResNet / ResNeXt: 0=fc only; 1=layer4+fc; 2=layer3-4+fc; ...
    ConvNeXt / feature-based networks: last N top-level feature blocks + head.
    """
    if unfreeze_last_n_layers < 0:
        raise ValueError('--unfreeze_last_n_layers must be >= 0.')

    heads = classifier_modules(net)
    if not freeze_features:
        set_requires_grad(net, True)
        print('[fine-tuning] full network is trainable (--no_freeze_features).')
        return

    # Start from classifier-only training, then selectively enable final stages.
    set_requires_grad(net, False)
    for head in heads:
        set_requires_grad(head, True)

    if hasattr(net, 'layer1') and hasattr(net, 'layer4'):
        # ResNet and ResNeXt structure
        stages = [net.layer1, net.layer2, net.layer3, net.layer4]
        if unfreeze_last_n_layers > len(stages):
            raise ValueError('For ResNet/ResNeXt, --unfreeze_last_n_layers must be 0..4.')
        for stage in (stages[-unfreeze_last_n_layers:] if unfreeze_last_n_layers else []):
            set_requires_grad(stage, True)
        print(f'[fine-tuning] ResNet-style: last {unfreeze_last_n_layers} stage(s) + classifier.')

    elif hasattr(net, 'features'):
        # ConvNeXt, EfficientNet, DenseNet, MobileNet, etc.
        blocks = list(net.features.children())
        if unfreeze_last_n_layers > len(blocks):
            raise ValueError(
                f'--unfreeze_last_n_layers is {unfreeze_last_n_layers}, but this model has only {len(blocks)} feature blocks.'
            )
        for block in (blocks[-unfreeze_last_n_layers:] if unfreeze_last_n_layers else []):
            set_requires_grad(block, True)
        print(f'[fine-tuning] features-style: last {unfreeze_last_n_layers} of {len(blocks)} feature blocks + classifier.')

    elif unfreeze_last_n_layers:
        raise AttributeError(
            'This model has neither ResNet stages nor .features. Add its fine-tuning rule to configure_finetuning().'
        )

    # This is relevant to large sequential heads such as AlexNet/VGG, but
    # ResNet fc is a single small layer and remains trainable.
    if freeze_classifier_base and hasattr(net, 'classifier') and isinstance(net.classifier, nn.Sequential):
        linear_layers = [m for m in net.classifier if isinstance(m, nn.Linear)]
        for layer in linear_layers[:-1]:
            set_requires_grad(layer, False)
        if linear_layers:
            set_requires_grad(linear_layers[-1], True)


def build_model(args, device):
    """Creates the model ONCE, ImageNet-pretrained, then applies one
    consistent fine-tuning policy (identical across architectures)."""
    model = classic_models.find_model_using_name(
        args.model, num_classes=args.num_classes
    ).to(device)

    configure_finetuning(
        model, args.freeze_features, args.unfreeze_last_n_layers, args.freeze_classifier_base
    )

    print(f"freeze_features={args.freeze_features}  unfreeze_last_n_layers={args.unfreeze_last_n_layers}")
    print(f"freeze_classifier_base={args.freeze_classifier_base}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} total")

    return model


# --------------------------------------------------------------------------- #
# Output paths
# --------------------------------------------------------------------------- #
def setup_output_paths(args):
    save_path = os.path.join(os.getcwd(), 'trained_models', 'weights', args.model)
    os.makedirs(save_path, exist_ok=True)
    return {
        'log_file': os.path.join(save_path, f"{args.model}.txt"),
        'checkpoint_file': os.path.join(save_path, f"{args.model}.pth"),
        'last_checkpoint_file': os.path.join(save_path, f"{args.model}_last_checkpoint.pth"),
    }


def setup_tensorboard(args):
    log_path = os.path.join('results', 'Tensorboard', args.model)
    print(f'Start Tensorboard with "tensorboard --logdir={log_path}"')
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    print(f"tensorboard log save in {log_path}")
    return SummaryWriter(log_path)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

    if args.seed:
        seed_torch()

    tb_writer = setup_tensorboard(args) if args.tensorboard else None

    data_transform, transform_name = build_transforms(args)
    print(f"offline_augmented={args.offline_augmented}, augmentation={args.augmentation} "
          f"-> using {transform_name} train transform")

    train_dataset = Five_Flowers_Load(os.path.join(args.data_path, 'Train'), transform=data_transform["Train"])
    val_dataset = Five_Flowers_Load(os.path.join(args.data_path, 'Val'), transform=data_transform["Val"])

    if args.num_classes != train_dataset.num_class:
        raise ValueError(f"dataset have {train_dataset.num_class} classes, but input {args.num_classes}")

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
        num_workers=nw, collate_fn=train_dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
        num_workers=nw, collate_fn=val_dataset.collate_fn,
    )

    model = build_model(args, device)

    pg = [p for p in model.parameters() if p.requires_grad]
    # weight_decay adds L2 regularization; cheap insurance against overfitting
    # on small datasets like this one.
    optimizer = optim.Adam(pg, lr=args.lr, weight_decay=args.weight_decay)

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset, args.num_classes, device)

    if args.loss_type == 'focal':
        print(f"loss_type=focal (gamma={args.focal_gamma}), class_weights as alpha: "
              f"{class_weights.tolist() if class_weights is not None else None}")
    else:
        print(f"loss_type=ce, label_smoothing={args.label_smoothing}, weight: "
              f"{class_weights.tolist() if class_weights is not None else None}")

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    paths = setup_output_paths(args)
    best_f2 = -1.0
    no_improve_epochs = 0  # counter for early stopping based on defective F2

    try:
        for epoch in range(args.epochs):
            mean_loss, train_acc = train_one_epoch(
                model=model, optimizer=optimizer, data_loader=train_loader,
                device=device, epoch=epoch, use_amp=args.use_amp, lr_method=warmup,
                class_weights=class_weights, loss_type=args.loss_type,
                focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
            )
            scheduler.step()

            # Select checkpoints by defect-detection F2, not accuracy.
            # F2 prioritizes recall, so it penalizes missed defective items more strongly.
            val = evaluate_defect_metrics(
                model, val_loader, device, args.positive_class_index, args.val_threshold
            )
            line = (
                f'[epoch {epoch + 1}] train_loss: {mean_loss:.3f} train_acc: {train_acc:.3f} | '
                f'val_acc: {val["accuracy"]:.3f} defect_precision: {val["precision"]:.3f} '
                f'defect_recall: {val["recall"]:.3f} F1: {val["f1"]:.3f} F2: {val["f2"]:.3f} '
                f'(TP={val["tp"]}, FP={val["fp"]}, TN={val["tn"]}, FN={val["fn"]})'
            )
            print(line)
            with open(paths['log_file'], 'a') as f:
                f.write(line + '\n')

            if tb_writer is not None:
                tb_writer.add_scalar('train_loss', mean_loss, epoch)
                tb_writer.add_scalar('train_acc', train_acc, epoch)
                tb_writer.add_scalar('val_accuracy', val['accuracy'], epoch)
                tb_writer.add_scalar('val_defect_precision', val['precision'], epoch)
                tb_writer.add_scalar('val_defect_recall', val['recall'], epoch)
                tb_writer.add_scalar('val_defect_f2', val['f2'], epoch)
                tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            if val['f2'] > best_f2:
                best_f2 = val['f2']
                no_improve_epochs = 0
                torch.save(model.state_dict(), paths['checkpoint_file'])
                print(f'New best validation defect F2: {best_f2:.3f}, model saved.')
            else:
                no_improve_epochs += 1
                print(f'No F2 improvement for {no_improve_epochs} epoch(s). Best F2: {best_f2:.3f}')

            if no_improve_epochs >= args.patience:
                print(f'Early stopping triggered after {epoch + 1} epochs '
                      f'(no improvement in validation F2 for {args.patience} consecutive epochs).')
                with open(paths['log_file'], 'a') as f:
                    f.writelines(f'Early stopping triggered at epoch {epoch + 1}\n')
                break

    except Exception as e:
        print(f"Training interrupted by exception: {e}")
        torch.save(model.state_dict(), paths['last_checkpoint_file'])
        print(f"Saved emergency checkpoint to {paths['last_checkpoint_file']}")
        raise


if __name__ == '__main__':
    opt = build_arg_parser().parse_args()
    main(opt)