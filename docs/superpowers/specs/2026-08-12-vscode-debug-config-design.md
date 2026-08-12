# 设计：VSCode 训练调试配置

日期：2026-08-12

## 背景与目标

项目 `YOLO_Detection` 当前没有任何 VSCode 调试配置，训练/推理脚本只能从命令行运行，无法在 VSCode 中设置断点调试。

目标：新增 `.vscode/launch.json`，提供两个调试启动项，分别针对训练脚本 `src/train.py` 与推理脚本 `src/eval.py`，支持在 VSCode 中直接 F5 断点调试。

## 范围

- 仅新增 `.vscode/launch.json` 一个文件。
- 仅包含两个调试启动项：`训练 (train.py)`、`推理 (eval.py)`。
- 不包含"当前文件"通用调试项、不修改任何 Python 脚本、不新增 `settings.json`。

## 设计

### 文件位置

`.vscode/launch.json`（`.vscode/` 未被 `.gitignore` 忽略，随仓库版本管理）。

### 配置内容

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "训练 (train.py)",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/train.py",
      "python": "${workspaceFolder}/.venv/Scripts/python.exe",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": true
    },
    {
      "name": "推理 (eval.py)",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/src/eval.py",
      "python": "${workspaceFolder}/.venv/Scripts/python.exe",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": true
    }
  ]
}
```

### 关键取舍

| 字段 | 取值 | 理由 |
|------|------|------|
| `python` | `${workspaceFolder}/.venv/Scripts/python.exe` | 与 AGENTS.md 约定一致（Windows 下 `.venv/Scripts/python`），不写本机绝对路径，换机器/换用户仍可用 |
| `cwd` | `${workspaceFolder}` | 项目根；脚本本身基于 `PROJECT_ROOT` 解析路径，此项保证 `test.jpg` 等相对定位也稳妥 |
| `type` | `python` | 兼容新旧版本 Python 扩展，行为一致 |
| `justMyCode` | `true` | 断点默认只停在自己的代码；如需调试 ultralytics 库内部，改此字段即可（一行改动） |

## 验证方式

- 用 VSCode 打开项目根目录，从运行面板选择 `训练 (train.py)` / `推理 (eval.py)` 启动。
- 在两个脚本的 `main()` 入口处设置断点，F5 启动后确认断点命中、变量可查看。
- 启动项名称与设计一致。

## 不做的事（YAGNI）

- 不添加 CLI 参数（`args`）：两个脚本当前均无 argparse 参数，暂不需要。
- 不添加"当前文件"通用调试项：用户明确只要求 train.py 与 eval.py。
- 不添加 `settings.json` / 环境变量配置：无实际需求。
