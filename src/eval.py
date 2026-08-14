"""验证 YOLO 检测模型，并输出总体和各类别的性能指标。

示例：
    python 4_验证模型性能.py
    python 4_验证模型性能.py --model runs/train/yolov85/weights/best.pt --data data_物流.yaml --device 0
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 YOLO 模型并输出各类别性能指标")
    parser.add_argument(
        "--model",
        default=Path("转Onnx") /  "best.pt",
        type=Path,
        help="模型权重 .pt 文件路径",
    )
    parser.add_argument("--data", default=Path("data_物流.yaml"), type=Path, help="数据集 YAML 文件路径")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="要验证的数据集划分")
    parser.add_argument("--imgsz", default=640, type=int, help="验证图片尺寸")
    parser.add_argument("--batch", default=16, type=int, help="批大小")
    parser.add_argument("--device", default=None, help="设备，例如 0、0,1、cpu")
    parser.add_argument("--workers", default=2, type=int, help="数据加载进程数")
    parser.add_argument("--plots", action="store_true", help="保存 PR 曲线、混淆矩阵等图表")
    return parser.parse_args()


def read_dataset_config(data_file: Path) -> dict[str, Any]:
    with data_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"数据集配置无效：{data_file}")
    return config


def resolve_dataset_path(value: str | Path, dataset_root: Path, yaml_directory: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root_relative = dataset_root / path
    return root_relative if root_relative.exists() else yaml_directory / path


def find_images(source: Any, dataset_root: Path, yaml_directory: Path) -> list[Path]:
    """解析 data.yaml 中的目录、图片文件、txt 列表或路径列表。"""
    if isinstance(source, (list, tuple)):
        images: list[Path] = []
        for item in source:
            images.extend(find_images(item, dataset_root, yaml_directory))
        return images

    source_path = resolve_dataset_path(str(source), dataset_root, yaml_directory)
    if source_path.is_dir():
        return sorted(path for path in source_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if source_path.suffix.lower() == ".txt":
        return [
            resolve_dataset_path(line.strip(), dataset_root, source_path.parent)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [source_path] if source_path.suffix.lower() in IMAGE_SUFFIXES else []


def label_path_for_image(image_path: Path) -> Path | None:
    """将 .../images/.../image.jpg 映射为 .../labels/.../image.txt。"""
    parts = image_path.parts
    image_index = next((index for index in range(len(parts) - 1, -1, -1) if parts[index] == "images"), None)
    if image_index is None:
        return None
    return Path(*parts[:image_index], "labels", *parts[image_index + 1 :]).with_suffix(".txt")


def count_validation_labels(images: list[Path]) -> tuple[int, Counter[int], Counter[int]]:
    """返回验证图片数、包含各类的图片数及各类标注实例数。"""
    image_count_by_class: Counter[int] = Counter()
    instance_count_by_class: Counter[int] = Counter()

    for image_path in images:
        label_path = label_path_for_image(image_path)
        if label_path is None or not label_path.is_file():
            continue

        classes_in_image: set[int] = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                class_id = int(float(fields[0]))
            except ValueError:
                continue
            classes_in_image.add(class_id)
            instance_count_by_class[class_id] += 1

        for class_id in classes_in_image:
            image_count_by_class[class_id] += 1

    return len(images), image_count_by_class, instance_count_by_class


def metric_by_class(metric_values: Any, class_indices: Any) -> dict[int, float]:
    return {int(class_id): float(value) for class_id, value in zip(class_indices, metric_values)}


def format_row(name: str, images: int, instances: int, precision: float, recall: float, map50: float, map50_95: float) -> str:
    return (
        f"{name:<12}{images:>8}{instances:>11}"
        f"{precision:>11.3f}{recall:>10.3f}{map50:>11.3f}{map50_95:>12.3f}"
    )


def print_metrics_table(metrics: Any, class_names: Any, image_total: int, image_counts: Counter[int], instance_counts: Counter[int]) -> None:
    names = class_names if isinstance(class_names, dict) else dict(enumerate(class_names))
    box = metrics.box
    class_indices = box.ap_class_index
    precision = metric_by_class(box.p, class_indices)
    recall = metric_by_class(box.r, class_indices)
    map50 = metric_by_class(box.ap50, class_indices)
    map50_95 = metric_by_class(box.ap, class_indices)

    print("\nClass          Images  Instances      Box(P          R      mAP50  mAP50-95)")
    print(format_row("all", image_total, sum(instance_counts.values()), box.mp, box.mr, box.map50, box.map))
    for class_id, class_name in names.items():
        print(
            format_row(
                str(class_name),
                image_counts[class_id],
                instance_counts[class_id],
                precision.get(class_id, 0.0),
                recall.get(class_id, 0.0),
                map50.get(class_id, 0.0),
                map50_95.get(class_id, 0.0),
            )
        )


def main() -> None:
    args = parse_arguments()
    data_file = args.data.resolve()
    model_file = args.model.resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"未找到数据集配置文件：{data_file}")
    if not model_file.is_file():
        raise FileNotFoundError(f"未找到模型权重文件：{model_file}")

    dataset_config = read_dataset_config(data_file)
    if args.split not in dataset_config:
        raise KeyError(f"数据集配置中没有 '{args.split}' 划分：{data_file}")

    dataset_root = Path(dataset_config.get("path", data_file.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_file.parent / dataset_root).resolve()
    images = find_images(dataset_config[args.split], dataset_root, data_file.parent)
    image_total, image_counts, instance_counts = count_validation_labels(images)

    model = YOLO(model_file)
    metrics = model.val(
        data=str(data_file),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=args.plots,
        verbose=False,
    )
    print_metrics_table(metrics, model.names, image_total, image_counts, instance_counts)


if __name__ == "__main__":
    main()
