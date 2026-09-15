import os
import argparse
import json
import time
import torch
from PIL import Image
from torchvision import transforms

import classic_models
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=os.path.join('datasets', 'pillQC'))
parser.add_argument('--test_subfolder', type=str, default='test')
parser.add_argument('--num_classes', type=int, default=2, help='the number of classes')
parser.add_argument('--model', type=str, default='apfanet', help='must match the model used for training')
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--pos_label', type=str, default='defective',
                     help='class name treated as the positive class for precision/recall/F1')
parser.add_argument('--val_threshold', type=float, default=0.4,
                )
parser.add_argument('--input_size', type=int, default=224, help='input image size (square) used for efficiency benchmarking')
parser.add_argument('--bench_warmup', type=int, default=10, help='number of warm-up forward passes before timing')
parser.add_argument('--bench_runs', type=int, default=100, help='number of timed forward passes to average over')
parser.add_argument('--bench_batch_size', type=int, default=1,
                     help='batch size used for the FPS/latency benchmark (1 = single-image latency, '
                          'matches typical deployment; try a larger value e.g. 16 to see batched throughput)')
opt = parser.parse_args()

def compute_efficiency_metrics(model, device, input_size=224, batch_size=1, num_warmup=10, num_runs=100):
    """
    Reports:
      - total_params / trainable_params: model size and how much of it was
        actually being trained (reflects whatever freeze_features /
        freeze_classifier_base state the checkpoint was trained with).
      - gflops: multiply-accumulate operations for one forward pass at
        batch_size=1, reported as GFLOPs (billions of FLOPs, using the
        common convention of 1 MAC ~= 2 FLOPs). Independent of `batch_size`
        arg below -- FLOPs are always measured per single sample.
      - avg_latency_ms / fps: wall-clock time for a forward pass of
        `batch_size` images, averaged over `num_runs` timed iterations
        after `num_warmup` untimed warm-up iterations (warm-up avoids
        counting one-time CUDA kernel compilation / cache effects).
        fps = images processed per second = batch_size / avg_latency_seconds.
    """
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    gflops = None
    if THOP_AVAILABLE:
        try:
            flop_input = torch.randn(1, 3, input_size, input_size).to(device)
            macs, _ = profile(model, inputs=(flop_input,), verbose=False)
            gflops = (macs * 2) / 1e9  # MACs -> FLOPs (x2), then to GFLOPs
        except Exception as e:
            print(f"Could not compute GFLOPs via thop: {e}")
    else:
        print("thop not installed -- GFLOPs will be skipped. Install with: pip install thop")

    bench_input = torch.randn(batch_size, 3, input_size, input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(bench_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(bench_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

    avg_latency_s = sum(timings) / len(timings)
    avg_latency_ms = avg_latency_s * 1000
    fps = batch_size / avg_latency_s if avg_latency_s > 0 else float('inf')

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "gflops": gflops,
        "batch_size": batch_size,
        "avg_latency_ms": avg_latency_ms,
        "fps": fps,
    }


def print_efficiency_report(metrics):
    print("\n---- Computational Efficiency ----")
    print("Total params:      {:,}".format(metrics["total_params"]))
    print("Trainable params:  {:,}".format(metrics["trainable_params"]))
    if metrics["gflops"] is not None:
        print("GFLOPs (per image): {:.3f}".format(metrics["gflops"]))
    else:
        print("GFLOPs (per image): N/A (install thop: pip install thop)")
    print("Benchmark batch size: {}".format(metrics["batch_size"]))
    print("Avg inference latency: {:.3f} ms/batch".format(metrics["avg_latency_ms"]))
    print("Throughput: {:.2f} FPS (images/sec)".format(metrics["fps"]))


def main():
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print(opt)

    if not 0.0 < opt.val_threshold < 1.0:
        raise ValueError('--val_threshold must be between 0 and 1.')

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ---- Paths, all relative to current working directory ----
    test_folder = os.path.join(opt.data_path, opt.test_subfolder)
    assert os.path.exists(test_folder), "Test folder '{}' does not exist.".format(test_folder)

    json_path = os.path.join(opt.data_path, 'class_indices.json')
    assert os.path.exists(json_path), "File '{}' does not exist.".format(json_path)

    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

    if opt.num_classes != 2:
        raise ValueError(
            f"This script's threshold-based prediction rule (pos_idx vs. 1-pos_idx) assumes "
            f"exactly 2 classes; got --num_classes {opt.num_classes}."
        )

    # Resolve which class index corresponds to opt.pos_label (e.g. "defective"),
    # instead of assuming it's always index 1. class_indict maps
    # {"0": "className0", "1": "className1", ...}.
    pos_idx = None
    for idx_str, class_name in class_indict.items():
        if class_name == opt.pos_label:
            pos_idx = int(idx_str)
            break
    if pos_idx is None:
        raise ValueError(
            f"pos_label '{opt.pos_label}' not found in class_indices.json values: "
            f"{list(class_indict.values())}"
        )
    neg_idx = 1 - pos_idx  # valid because num_classes == 2, checked above

    # Weights path matches exactly how train.py saves them:
    # os.path.join(os.getcwd(), 'trained_models/weights', args.model, f"{args.model}.pth")
    weights_path = os.path.join(os.getcwd(), 'trained_models', 'weights', opt.model, f'{opt.model}.pth')
    assert os.path.exists(weights_path), "File '{}' does not exist.".format(weights_path)

    # ---- Build the model exactly the same way training did ----
    model = classic_models.find_model_using_name(opt.model, num_classes=opt.num_classes).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # ---- Computational efficiency benchmark (params / GFLOPs / latency / FPS) ----
    metrics = compute_efficiency_metrics(
        model, device,
        input_size=opt.input_size,
        batch_size=opt.bench_batch_size,
        num_warmup=opt.bench_warmup,
        num_runs=opt.bench_runs,
    )
    print_efficiency_report(metrics)

    true_labels = []
    predicted_labels = []
    predicted_probs = []  # probability of the positive class, useful for ROC/AUC if needed later

    for class_folder in os.listdir(test_folder):
        class_folder_path = os.path.join(test_folder, class_folder)
        if os.path.isdir(class_folder_path):
            for image_file in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_file)
                assert os.path.isfile(image_path), "File '{}' does not exist.".format(image_path)

                img = Image.open(image_path).convert("RGB")
                img_tensor = data_transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    pos_class_prob = probs[0, pos_idx].item()  # P(opt.pos_label), resolved above -- not hardcoded

                # Threshold at opt.val_threshold, matching how train.py selected this
                # checkpoint (evaluate_defect_metrics uses the same rule), instead of a
                # plain argmax -- argmax on 2 classes is equivalent to a fixed 0.50
                # threshold, which is a different operating point than what the
                # checkpoint was chosen for whenever --val_threshold != 0.50.
                predicted_class = pos_idx if pos_class_prob >= opt.val_threshold else neg_idx

                true_labels.append(class_folder)
                predicted_labels.append(class_indict[str(predicted_class)])
                predicted_probs.append(pos_class_prob)

               # print("Image: {:20}   True Class: {:10}   Predicted Class: {:10}   P({}): {:.4f}".format(
               #     image_file, class_folder, class_indict[str(predicted_class)], opt.pos_label, pos_class_prob
               # ))

    # ---- Binary metrics ----
    # class_indict values are the class NAMES (e.g. "Normal"/"Defective"), same strings used as folder names,
    # so true_labels and predicted_labels are directly comparable.
    labels_sorted = ['defective', 'normal']

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label=opt.pos_label, average='binary')
    recall = recall_score(true_labels, predicted_labels, pos_label=opt.pos_label, average='binary')
    f1 = f1_score(true_labels, predicted_labels, pos_label=opt.pos_label, average='binary')
    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=labels_sorted
    )

    print("\n---- Results ----")
    print("Classes: {}".format(labels_sorted))
    print("Positive class (for precision/recall/F1): {}".format(opt.pos_label))
    print("Decision threshold: P({}) >= {:.2f}".format(opt.pos_label, opt.val_threshold))
    print("Accuracy:  {:.4f}".format(accuracy))
    print("Precision: {:.4f}".format(precision))
    print("Recall:    {:.4f}".format(recall))
    print("F1-score:  {:.4f}".format(f1))


    print("\nConfusion Matrix")
    print("Rows = True, Columns = Predicted")
    print("             defective  normal")
    print(cm)

    print("class_indices.json:", class_indict)
    print("Positive label:", opt.pos_label)
    print("Positive index:", pos_idx)

    print("\nTest folders:")
    for class_folder in os.listdir(test_folder):
        class_folder_path = os.path.join(test_folder, class_folder)
        if os.path.isdir(class_folder_path):
            print(class_folder, ":", len(os.listdir(class_folder_path)))


if __name__ == '__main__':
    main()