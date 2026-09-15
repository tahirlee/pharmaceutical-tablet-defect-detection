#!/usr/bin/env python3
"""Create defect-safe offline augmentations for a two-class tablet dataset.

Input layout:
  datasets/pharmacy/Train/normal/*.jpg
  datasets/pharmacy/Train/defective/*.jpg

Output layout (new, never validation/test):
  datasets/pharmacy/Train_augmented/normal/*
  datasets/pharmacy/Train_augmented/defective/*

Each source image is copied unchanged and then receives N mild variants. This
never augments Val or test: doing so would contaminate evaluation.
"""
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def mild_defect_safe_augment(image, rng):
    """Geometry/light variation that should not erase small cracks or chips.

    Deliberately NOT used: random crop, blur, Gaussian noise, cutout/erasing,
    hue or saturation shift. Those can hide or fabricate defect evidence.
    """
    img = image.copy().convert('RGB')
    if rng.random() < 0.50:
        img = ImageOps.mirror(img)
    if rng.random() < 0.50:
        img = ImageOps.flip(img)

    # The source gallery already contains many orientations. Keep rotation mild.
    angle = rng.uniform(-12.0, 12.0)
    fill = tuple(int(x) for x in img.resize((1, 1)).getpixel((0, 0)))
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)

    # Mild illumination/camera variation only; preserve colour/staining cues.
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.93, 1.07))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.93, 1.07))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='datasets/pharmacy/Train',
                        help='original TRAIN directory; must contain class folders')
    parser.add_argument('--output_dir', default='datasets/pharmacy/Train_augmented',
                        help='new output TRAIN directory')
    parser.add_argument('--copies_per_image', type=int, default=2,
                        help='number of augmented copies per original image')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--jpeg_quality', type=int, default=95)
    parser.add_argument('--overwrite', action='store_true',
                        help='delete output_dir first (use deliberately)')
    args = parser.parse_args()

    if args.copies_per_image < 0:
        raise ValueError('--copies_per_image must be >= 0')
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'{output_dir} already exists. Pick another output_dir or add --overwrite.')
        shutil.rmtree(output_dir)

    class_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f'No class folders found in {input_dir}')
    output_dir.mkdir(parents=True)
    rng = random.Random(args.seed)
    total_original, total_augmented = 0, 0

    for class_dir in class_dirs:
        destination = output_dir / class_dir.name
        destination.mkdir()
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            print(f'WARNING: no supported image files in {class_dir}')
            continue
        for index, source in enumerate(files):
            # Copy originals exactly, with a stable unique prefix to prevent collisions.
            stem = f'{index:05d}_{source.stem}'
            shutil.copy2(source, destination / f'{stem}_original{source.suffix.lower()}')
            total_original += 1
            with Image.open(source) as opened:
                image = opened.convert('RGB')
                for copy_index in range(args.copies_per_image):
                    augmented = mild_defect_safe_augment(image, rng)
                    # Save JPEG for predictable RGB training input and reasonable disk size.
                    augmented.save(destination / f'{stem}_aug{copy_index + 1}.jpg', quality=args.jpeg_quality)
                    total_augmented += 1
        print(f'{class_dir.name}: {len(files)} originals -> {len(files) * (1 + args.copies_per_image)} output images')

    print(f'Complete. Originals copied: {total_original}; augmented copies: {total_augmented}; total: {total_original + total_augmented}')
    print(f'Use --data_path {output_dir.parent} only after renaming/linking output_dir to Train, or pass your loader the output path directly.')
    print('Do NOT augment Val or test. Tune the defect threshold on Val and evaluate once on test.')

if __name__ == '__main__':
    main()
