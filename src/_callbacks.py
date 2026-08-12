"""训练回调：每个 epoch 验证结束后按类别打印检测指标。

与 `_env_fix.py` 同风格的下划线前缀内部模块，供 `train.py` / `train_continue.py` 共用，
避免两处各自维护一份实现导致漂移。
"""

from typing import Any


def print_per_class_metrics(trainer: Any) -> None:
    """Print detection metrics for every class after an epoch validation."""
    validator = getattr(trainer, "validator", None)
    metrics = getattr(validator, "metrics", None)
    box = getattr(metrics, "box", None)
    if box is None:
        return

    names = getattr(validator, "names", None)
    if names is None:
        names = getattr(getattr(trainer, "model", None), "names", {})
    if not isinstance(names, dict):
        names = dict(enumerate(names))

    class_indices = getattr(box, "ap_class_index", [])

    def values_by_class(values: Any) -> dict[int, float]:
        return {
            int(class_id): float(value)
            for class_id, value in zip(class_indices, values)
        }

    precision = values_by_class(getattr(box, "p", []))
    recall = values_by_class(getattr(box, "r", []))
    map50 = values_by_class(getattr(box, "ap50", []))
    map50_95 = values_by_class(getattr(box, "ap", []))

    epoch = int(getattr(trainer, "epoch", 0)) + 1
    print(f"\nEpoch {epoch} per-class validation metrics:")
    print(f"{'Class':<20}{'P':>10}{'R':>10}{'mAP50':>10}{'mAP50-95':>12}")
    for class_id, class_name in names.items():
        print(
            f"{str(class_name):<20}"
            f"{precision.get(class_id, 0.0):>10.3f}"
            f"{recall.get(class_id, 0.0):>10.3f}"
            f"{map50.get(class_id, 0.0):>10.3f}"
            f"{map50_95.get(class_id, 0.0):>12.3f}"
        )
