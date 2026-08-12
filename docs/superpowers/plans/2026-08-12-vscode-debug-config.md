# VSCode 训练调试配置 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建 `.vscode/launch.json`，提供针对 `src/train.py` 与 `src/eval.py` 的两个 VSCode 调试启动项。

**Architecture:** 单文件交付：在项目根新建 `.vscode/launch.json`，使用 Python 扩展调试器（`type: "python"`），解释器引用 `${workspaceFolder}/.venv/Scripts/python.exe`（Windows + uv 环境），`cwd` 设为项目根。不修改任何 Python 脚本。

**Tech Stack:** VSCode Python Debugger（`ms-python.python` 扩展）、JSON。

## Global Constraints

- 解释器路径一律用 `${workspaceFolder}/.venv/Scripts/python.exe`，不写本机绝对路径（AGENTS.md 约定：Windows 下 Python 为 `.venv/Scripts/python`）。
- `"type"` 使用 `"python"`（兼容新旧版 Python 扩展）。
- 仅两个调试项：`训练 (train.py)` 与 `推理 (eval.py)`；不加"当前文件"项、不加 `args`。
- `.vscode/` 不在 `.gitignore` 中，文件需纳入版本管理并提交。
- JSON 必须语法合法；配置字段与设计文档 `docs/superpowers/specs/2026-08-12-vscode-debug-config-design.md` 逐项一致。

---

### Task 1: 创建 .vscode/launch.json 并验证

**Files:**
- Create: `.vscode/launch.json`

**Interfaces:**
- Consumes: 设计文档 `docs/superpowers/specs/2026-08-12-vscode-debug-config-design.md` 中的 JSON 配置（无需代码接口）。
- Produces: 合法的 `launch.json`，含两个配置项；供用户在 VSCode 运行面板选择启动。

- [ ] **Step 1: 创建 `.vscode/launch.json`**

用 write_file 创建文件，内容（与设计文档完全一致）：

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

- [ ] **Step 2: 验证 JSON 语法与内容**

运行（项目根下）：

```bash
python -c "import json; d=json.load(open('.vscode/launch.json', encoding='utf-8')); assert d['version']=='0.2.0'; assert len(d['configurations'])==2; print('OK:', [c['name'] for c in d['configurations']])"
```

Expected: 输出 `OK: ['训练 (train.py)', '推理 (eval.py)']`，无异常。

- [ ] **Step 3: 确认配置项与设计一致**

人工核对（read_file 打开 `.vscode/launch.json`）：两个配置项的 `program` 分别指向 `src/train.py` / `src/eval.py`；`python`、`cwd`、`console`、`justMyCode`、`type` 与设计文档 JSON 逐字段一致。

- [ ] **Step 4: 提交**

```bash
git add .vscode/launch.json
git commit -m "feat: 添加 VSCode 训练/推理调试配置"
```

Expected: 提交成功，提交仅包含 `.vscode/launch.json` 一个文件（`models/data.yaml` 的既有改动不要 `git add`）。

---

## 验证方式（交付后）

- 在 VSCode 打开项目根目录，运行面板选择 `训练 (train.py)` / `推理 (eval.py)` 启动。
- 在 `src/train.py` / `src/eval.py` 的 `main()` 入口设置断点，F5 启动后断点命中、变量可查看。
- 注：train.py 断点调试需数据集就位（见 AGENTS.md Notes），eval.py 需 `runs/train/yolov8/weights/best.onnx` 与 `test.jpg` 存在；若缺文件属环境问题而非配置问题。
