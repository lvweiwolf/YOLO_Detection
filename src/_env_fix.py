"""环境修复：过滤 PATH 中与 CUDA 库自带的 cuDNN 冲突的目录。

背景：本机 PATH 中安装了独立的 cuDNN（如 C:\\Program Files\\NVIDIA\\CUDNN\\v9.24\\bin\\12.9\\x64），
与 torch 2.13.0+cu130 自带的 cuDNN 9.2.0 混装时，cuDNN 9 的 dispatcher 会加载到
版本不匹配的子库，运行时报 CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH。
onnxruntime-gpu 的 CUDA provider 同理。

注意：本模块必须在任何 CUDA 相关库（torch / ultralytics / onnxruntime）导入之前被导入，
它会将 PATH 中含 "cudnn" 的目录剔除（仅影响当前进程，不修改系统环境）。
"""

import os
import re


def _filter_cudnn_from_path() -> None:
    """从当前进程的 PATH 中剔除含 cudnn 的目录。"""
    paths = [
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and not re.search(r"cudnn", p, re.IGNORECASE)
    ]
    os.environ["PATH"] = os.pathsep.join(paths)


_filter_cudnn_from_path()
