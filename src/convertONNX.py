import argparse
import shutil
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="将 YOLO 模型导出为 ONNX 并用 ONNX Runtime 加载验证"
    )
    parser.add_argument(
        "model",
        type=str,
        help="待转换的 YOLO 模型文件路径（如 models/best.pt）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="导出 ONNX 的输出路径（文件路径或目录），默认与模型同目录",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="导出输入尺寸（默认 640）",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset 版本（默认 11）",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="是否对导出的 ONNX 模型进行简化（默认关闭）",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="是否导出支持动态输入尺寸的 ONNX 模型（默认关闭）",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA 设备 ID（默认 0）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ==========================
    # 1️⃣ 加载 YOLO 模型并导出 ONNX
    # ==========================
    model = YOLO(args.model)  # 加载训练好的权重
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )  # 导出为 ONNX 格式（输出到 .pt 同目录）

    # 可选：将导出的 ONNX 移动到指定输出位置
    if args.output:
        target = Path(args.output)
        if target.suffix:  # 视为完整文件名
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(onnx_path, target)
        else:  # 视为输出目录
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(onnx_path, target / Path(onnx_path).name)
        onnx_path = str(target)

    print(f"✅ 模型已导出为: {onnx_path}")

    # ==========================
    # 2️⃣ 使用 ONNX Runtime 载入模型
    # ==========================
    providers = [
        ("CUDAExecutionProvider", {"device_id": args.device}),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(onnx_path, providers=providers)
    print("当前推理设备:", session.get_providers())


if __name__ == "__main__":
    main()
