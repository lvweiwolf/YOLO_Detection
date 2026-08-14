import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # 忽略警告信息

import _env_fix  # 必须在导入 torch/ultralytics 之前导入：剔除 PATH 中冲突的 cuDNN 目录

from _callbacks import print_per_class_metrics  # 每个 epoch 验证后按类别打印指标

from ultralytics import YOLO  # 导入YOLO模块

# 项目根目录（本文件位于 src/ 下，其上一级即项目根），
# 所有路径基于根目录解析，避免受运行时的当前工作目录影响
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    # 初始化模型，加载 models/ 下的预训练权重
    model = YOLO(
        model=str(PROJECT_ROOT / "models" / "yolov8n.pt")
    )  # 加载yolov8n预训练权重

    model.add_callback("on_fit_epoch_end", print_per_class_metrics)

    # 训练模型，使用以下参数
    model.train(
        data=str(
            PROJECT_ROOT / "models" / "data.yaml"
        ),  # 数据集配置文件路径，包含训练集和验证集路径，以及类别信息

        
        imgsz=640,  # 输入图片大小，通常为32的倍数，影响训练速度和精度
        epochs=10000,  # 训练的轮数
        patience=500,
        device="0",  # 使用的设备，'cpu' 或 'Gpu'，可以指定 GPU 设备编号例如 'cuda:0'
        batch=32,  # 每次训练输入的图片数量（批大小）。
        workers=6,

        # int8 = True,
        # 以下是一些图像增强相关的参数
        close_mosaic=100,  # 在一定数量的训练周期之后关闭 Mosaic 增强。Mosaic 是一种数据增强方法，结合多个图像生成一个复合图像
        project=str(
            PROJECT_ROOT / "runs" / "train"
        ),  # 训练输出根目录（绝对路径，避免受工作目录影响）

        name="yolov8",  # 实验名称，保存时会生成一个文件夹名为 runs/train/yolov8
        single_cls=False,  # 是否只训练一个类别（即检测单一物体），设置为True时会忽略类别信息
        cache=False,  # 是否将数据集缓存到内存中，以提高训练速度

        # 额外的超参数（其余参数使用默认值）
        # 以下是YOLOv12训练过程中的一些参数设置：
        # 默认模型的优化器参数（包括学习率、动量、权重衰减等）
        # 这些可以在模型构建时指定，通常是根据数据集和训练的不同设置来优化
        optimizer="auto",  # 优化器，默认选择最合适的优化器
        lr0=0.005,  # 初始学习率
        lrf=0.01,  # 学习率的最终调整值
        momentum=0.937,  # 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=4.0,  # 预热阶段的轮数（用于逐步提高学习率）
        warmup_momentum=0.8,  # 预热阶段的动量
        warmup_bias_lr=0.0,  # 预热阶段的偏置学习率
        
        # 以下是一些框架训练的额外参数：
        box=7.5,  # 目标框的损失权重
        cls=0.5,  # 类别损失的权重
        dfl=1.5,  # DFL损失的权重（检测焦点对齐损失）
        pose=12.0,  # 姿态损失的权重
        kobj=1.0,  # 目标标签损失权重
        nbs=64,  # 每个批次的训练图像数量（根据硬件计算能力动态调整）

        # 数据增强相关的参数：
        hsv_h=0.015,  # 色调的随机变化范围
        hsv_s=0.7,  # 饱和度的随机变化范围
        hsv_v=0.4,  # 亮度的随机变化范围
        degrees=0.0,  # 随机旋转角度的范围
        translate=0.1,  # 随机平移范围
        scale=0.5,  # 随机缩放范围
        shear=0.0,  # 随机剪切角度的范围
        perspective=0.0,  # 随机透视变换的范围
        flipud=0.0,  # 上下翻转概率
        fliplr=0.5,  # 左右翻转概率
        bgr=0.0,  # BGR通道的扰动
        mosaic=1.0,  # Mosaic数据增强的比例
        mixup=0.0,  # MixUp数据增强的比例
        copy_paste=0.1,  # Copy-paste数据增强的概率
        copy_paste_mode="flip",  # Copy-paste增强的方式（翻转）
        auto_augment="randaugment",  # 自动增强的策略

        # 训练过程中的其他调试参数：
        save=True,  # 是否保存模型权重
        save_period=-1,  # 保存周期，-1表示只在训练结束时保存
        save_json=False,  # 是否保存训练过程中的JSON日志
        save_hybrid=False,  # 是否保存混合精度的权重

        # 评估过程参数：
        val=True,  # 是否进行验证评估
        split="val",  # 数据集分割，验证集的名称
        save_txt=False,  # 是否保存检测框的预测结果
        save_conf=False,  # 是否保存预测的置信度
        save_crop=False,  # 是否保存裁剪后的目标框
        show=False,  # 是否显示实时检测结果
        show_labels=True,  # 是否显示标签
        show_conf=True,  # 是否显示置信度
        show_boxes=True,  # 是否显示检测框

        # 其他调试和优化参数
        plots=True,  # 是否绘制训练过程中的图形（如损失曲线）
        nms=False,  # 是否禁用非最大抑制（NMS）v
        format="torchscript",  # 导出格式
        simplify=True,  # 是否简化模型（优化大小）
        opset=None,  # 通过ONNX导出的opset版本
        workspace=None,  # 计算图空间大小
    )


if __name__ == "__main__":
    main()
