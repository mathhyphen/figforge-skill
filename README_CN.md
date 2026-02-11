# FigForge - 科学图表生成器

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenClaw Skill](https://img.shields.io/badge/OpenClaw-Skill-green.svg)](https://docs.openclaw.ai)

> **OpenClaw 优化的科学图表生成工具 (v2.1.0)**
> 
> 支持完整工作流：文本 → OpenClaw分析 → MODULE LIST → 图像

---

## 🎯 快速开始

### 方式1：完整工作流（推荐）

```bash
# 分析文本并生成图像
python scripts/run_complete.py -i input.txt -o figure.png
```

### 方式2：仅图像生成

```bash
# 从已有 MODULE LIST 生成图像
python scripts/run.py -m module_list.txt -o figure.png
```

### OpenClaw 直接使用

```bash
openclaw run figforge -i my_paper.txt -o figure.png
```

---

## 📋 环境要求

- Python 3.8+
- `google-genai` 包
- Gemini API Key（用于图像生成）

```bash
pip install google-genai python-dotenv click
```

---

## 🔧 配置

### 环境变量

| 变量 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `GEMINI_API_KEY` | ✅ 是 | - | Google Gemini API 密钥 |
| `IMAGE_MODEL` | ❌ 否 | `models/gemini-3-pro-image-preview` | 图像生成模型 |
| `OUTPUT_DIR` | ❌ 否 | `outputs` | 输出目录 |

### 默认模型

**图像生成默认使用**: `models/gemini-3-pro-image-preview`

如需使用其他模型，可设置环境变量或命令行参数：
```bash
# 使用 Gemini 2.0 Flash
export IMAGE_MODEL="gemini-2.0-flash-exp-image-generation"

# 或在命令行指定
python scripts/run.py -m module_list.txt --image-model gemini-2.0-flash-exp-image-generation
```

---

## 🎨 工作流程

### 完整工作流 (v2.1.0)

```
输入文本 (Input Text)
    ↓
OpenClaw Agent 分析 (Analysis: Kimi/GLM/Qwen)
    ↓
MODULE LIST (结构化描述)
    ↓
Gemini 图像生成 (Image Generation)
    ↓
科学图表 (Scientific Figure)
```

### 图像专用模式 (v2.0)

```
MODULE LIST (预生成)
    ↓
Gemini 图像生成 (Image Generation)
    ↓
科学图表 (Scientific Figure)
```

---

## 📖 使用示例

### 示例1：完整工作流

```bash
# 设置 API 密钥
export GEMINI_API_KEY="your-gemini-api-key"

# 生成图像
python scripts/run_complete.py -i methodology.txt -o results/figure1.png
```

### 示例2：使用预生成的 MODULE LIST

```bash
python scripts/run.py -m module_list.txt -o figure.png
```

### 示例3：自定义模型

```bash
python scripts/run.py -m module_list.txt --image-model gemini-2.0-flash-exp-image-generation
```

---

## 🎨 MODULE LIST 格式

MODULE LIST 是一个包含8个部分的结构化文本文档：

1. **图表目标和类型** - 图表说明
2. **主要主题/输入** - 关键视觉元素
3. **流程/方法/阶段** - 工作流步骤
4. **关系和流向** - 元素连接方式
5. **输出/读数/结果** - 预期可视化
6. **布局和视觉风格** - 设计规范
7. **文本标签和注释** - 标签和文本
8. **最终提示词** - 完整生成提示

查看 `examples/` 目录获取示例文件。

---

## 🔄 版本对比

| 特性 | v1.0 完整版 | v2.0 图像专用 | v2.1 双模式 |
|---------------|------------|--------------|-------------|
| 文本分析 | 内置 | OpenClaw Agent | OpenClaw Agent |
| 图像生成 | Gemini | Gemini | Gemini |
| 工作流 | 固定 | 模块化 | 双模式 |
| 使用场景 | 简单任务 | 精细控制 | 灵活选择 |

---

## 🛠️ 与 OpenClaw 集成

### 作为 OpenClaw Skill

```yaml
# 在代理配置中
skills:
  figforge:
    entry: scripts/run_complete.py
    env:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
```

### 独立使用

```python
from scripts.run import FigForgeGenerator

generator = FigForgeGenerator()
figure_path = generator.generate_figure(module_list_content)
```

---

## 📝 许可证

MIT License - 查看 [LICENSE](LICENSE)

## 🙏 致谢

- 原始项目: [FigForge](https://github.com/hengzzzhou/FigForge) by [@hengzzzhou](https://github.com/hengzzzhou)
- OpenClaw 适配: mathhyphen
- 技术支持: Google Gemini, OpenClaw

---

**祝您图表生成愉快！🎨✨**
