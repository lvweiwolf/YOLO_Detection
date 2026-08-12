from pathlib import Path

import onnxruntime as ort
import numpy as np
import cv2

# 项目根目录（本文件位于 src/ 下，其上一级即项目根），
# 所有路径基于根目录解析，避免受运行时的当前工作目录影响
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型与测试图片路径（按需修改）
ONNX_MODEL = str(PROJECT_ROOT / "runs" / "train" / "yolov8" / "weights" / "best.onnx")
TEST_IMAGE = str(PROJECT_ROOT / "test.jpg")

providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]


def main():
    # 尝试优先使用 GPU
    session = ort.InferenceSession(ONNX_MODEL, providers=providers)
    print("当前推理设备:", session.get_providers())

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 图像预处理
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        raise FileNotFoundError(f"无法读取测试图片: {TEST_IMAGE}")

    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized[:, :, ::-1].transpose(2, 0, 1) / 255.0
    img_input = np.expand_dims(img_input, 0).astype(np.float32)

    # 推理
    outputs = session.run([output_name], {input_name: img_input})
    output = np.asarray(outputs[0])
    print("输出 shape:", output.shape)


if __name__ == "__main__":
    main()
