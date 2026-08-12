from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import _env_fix  # 必须在导入 torch/ultralytics 之前导入：剔除 PATH 中冲突的 cuDNN 目录

from _callbacks import print_per_class_metrics  # 每个 epoch 验证后按类别打印指标

# 项目根目录（本文件位于 src/ 下，其上一级即项目根），
# 所有路径基于根目录解析，避免受运行时的当前工作目录影响
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def discover_latest_checkpoint(search_root: Path) -> Path:
    """Return the newest checkpoint stored as a training run's last.pt."""
    candidates = [
        path
        for path in search_root.rglob("last.pt")
        if path.is_file() and path.parent.name == "weights"
    ]
    if not candidates:
        raise FileNotFoundError(f"未找到可续训的 checkpoint: {search_root.resolve()}")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def validate_checkpoint(checkpoint: Path) -> Path:
    """Validate and normalize a checkpoint path before loading it."""
    resolved = checkpoint.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint 文件不存在: {resolved}")
    if resolved.suffix.lower() != ".pt":
        raise ValueError(f"checkpoint 必须是 .pt 文件: {resolved}")
    if resolved.name != "last.pt":
        print(
            f"提示: checkpoint 不是 last.pt（{resolved.name}）。"
            "续训通常应使用 last.pt 从中断点继续；best.pt 也可恢复，但会从对应轮次重新开始。"
        )
    return resolved


def resume_training(checkpoint: Path, yolo_factory: Callable | None = None) -> None:
    """Resume the Ultralytics run saved in ``checkpoint``."""
    checkpoint = validate_checkpoint(checkpoint)
    if yolo_factory is None:
        from ultralytics import YOLO

        yolo_factory = YOLO

    print(f"续训 checkpoint: {checkpoint}")
    print(
        f"修改时间: {datetime.fromtimestamp(checkpoint.stat().st_mtime).isoformat(sep=' ', timespec='seconds')}"
    )
    model = yolo_factory(checkpoint)
    model.add_callback("on_fit_epoch_end", print_per_class_metrics)
    model.train(resume=True)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从中断的 Ultralytics YOLO checkpoint 继续训练"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="要续训的 last.pt 路径；省略时自动选择最新文件",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "train",
        help="自动搜索 checkpoint 的根目录（相对路径基于项目根解析）",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    try:
        checkpoint = args.checkpoint
        if checkpoint is not None:
            # 相对路径一律基于项目根解析，避免受当前工作目录影响
            checkpoint = PROJECT_ROOT / checkpoint if not checkpoint.is_absolute() else checkpoint
        else:
            search_root = (
                args.search_root
                if args.search_root.is_absolute()
                else PROJECT_ROOT / args.search_root
            )
            checkpoint = discover_latest_checkpoint(search_root)
        resume_training(checkpoint)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"错误: {error}") from error


if __name__ == "__main__":
    main()
