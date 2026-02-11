# FigForge - 科学图表生成器 | Scientific Figure Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenClaw Skill](https://img.shields.io/badge/OpenClaw-Skill-green.svg)](https://docs.openclaw.ai)

> **OpenClaw 优化的科学图表生成工具 (v2.1.0)**
> 
> 支持完整工作流：文本 → OpenClaw分析 → MODULE LIST → 图像
> 
> **OpenClaw-optimized scientific figure generator (v2.1.0)**
> 
> Supports full workflow: Text → OpenClaw analysis → MODULE LIST → Figure

---

## 🎯 快速开始 | Quick Start

### 方式1：完整工作流（推荐）| Full Workflow (Recommended)

```bash
# 分析文本并生成图像 | Analyze text and generate figure
python scripts/run_complete.py -i input.txt -o figure.png
```

### 方式2：仅图像生成 | Image Generation Only

```bash
# 从已有 MODULE LIST 生成图像 | Generate from existing MODULE LIST
python scripts/run.py -m module_list.txt -o figure.png
```

### OpenClaw 直接使用 | Direct OpenClaw Usage

```bash
openclaw run figforge -i my_paper.txt -o figure.png
```

---

## 📋 环境要求 | Requirements

- Python 3.8+
- `google-genai` 包 | package
- Gemini API Key（用于图像生成）| (for image generation)

```bash
pip install google-genai python-dotenv click
```

---

## 🔧 配置 | Configuration

### 环境变量 | Environment Variables

| 变量 | 必需 | 默认 | 说明 |
|------|------|------|------|
| `GEMINI_API_KEY` | ✅ 是/Yes | - | Google Gemini API 密钥 / API key |
| `IMAGE_MODEL` | ❌ 否/No | `models/gemini-3-pro-image-preview` | 图像生成模型 / Image generation model |
| `OUTPUT_DIR` | ❌ 否/No | `outputs` | 输出目录 / Output directory |

### 默认模型 | Default Model

**图像生成默认使用**: `models/gemini-3-pro-image-preview`

如需使用其他模型，可设置环境变量或命令行参数：
```bash
# 使用 Gemini 2.0 Flash
export IMAGE_MODEL="gemini-2.0-flash-exp-image-generation"

# 或在命令行指定 | Or specify in command line
python scripts/run.py -m module_list.txt --image-model gemini-2.0-flash-exp-image-generation
```

---

## 🎨 工作流程 | Workflow

### 完整工作流 (v2.1.0) | Full Workflow

```
输入文本 (Input Text)
    ↓
OpenClaw Agent 分析 (Analysis: Kimi/GLM/Qwen)
    ↓
MODULE LIST (结构化描述 / Structured description)
    ↓
Gemini 图像生成 (Image Generation)
    ↓
科学图表 (Scientific Figure)
```

### 图像专用模式 (v2.0) | Image-Only Mode

```
MODULE LIST (预生成 / Pre-generated)
    ↓
Gemini 图像生成 (Image Generation)
    ↓
科学图表 (Scientific Figure)
```

---

## 📖 使用示例 | Usage Examples

### 示例1：完整工作流 | Example 1: Full Workflow

```bash
# 设置 API 密钥 | Set API key
export GEMINI_API_KEY="your-gemini-api-key"

# 生成图像 | Generate figure
python scripts/run_complete.py -i methodology.txt -o results/figure1.png
```

### 示例2：使用预生成的 MODULE LIST | Example 2: Pre-generated MODULE LIST

```bash
python scripts/run.py -m module_list.txt -o figure.png
```

### 示例3：自定义模型 | Example 3: Custom Model

```bash
python scripts/run.py -m module_list.txt --image-model gemini-2.0-flash-exp-image-generation
```

---

## 🎨 MODULE LIST 格式 | MODULE LIST Format

MODULE LIST 是一个包含8个部分的结构化文本文档：

The MODULE LIST is a structured text document with 8 sections:

1. **图表目标和类型 / Figure Goal and Type** - 图表说明 / What the figure illustrates
2. **主要主题/输入 / Main Subjects/Inputs** - 关键视觉元素 / Key visual elements
3. **流程/方法/阶段 / Processes/Methods/Stages** - 工作流步骤 / Workflow steps
4. **关系和流向 / Relationships and Flow** - 元素连接方式 / How elements connect
5. **输出/读数/结果 / Outputs/Readouts/Results** - 预期可视化 / Expected visualizations
6. **布局和视觉风格 / Layout and Visual Style** - 设计规范 / Design specifications
7. **文本标签和注释 / Text Labels and Annotations** - 标签和文本 / Labels and text
8. **最终提示词 / Final Prompt** - 完整生成提示 / Complete generation prompt

查看 `examples/` 目录获取示例文件。
See `examples/` directory for sample files.

---

## 🔄 版本对比 | Version Comparison

| 特性 / Feature | v1.0 完整版 | v2.0 图像专用 | v2.1 双模式 |
|---------------|------------|--------------|-------------|
| 文本分析 / Text Analysis | 内置 / Built-in | OpenClaw Agent | OpenClaw Agent |
| 图像生成 / Image Generation | Gemini | Gemini | Gemini |
| 工作流 / Workflow | 固定 / Fixed | 模块化 / Modular | 双模式 / Dual-mode |
| 使用场景 / Use Case | 简单任务 | 精细控制 | 灵活选择 |

---

## 🛠️ 与 OpenClaw 集成 | OpenClaw Integration

### 作为 OpenClaw Skill | As OpenClaw Skill

```yaml
# 在代理配置中 | In agent config
skills:
  figforge:
    entry: scripts/run_complete.py
    env:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
```

### 独立使用 | Standalone Usage

```python
from scripts.run import FigForgeGenerator

generator = FigForgeGenerator()
figure_path = generator.generate_figure(module_list_content)
```

---

## 📝 许可证 | License

MIT License - 查看 [LICENSE](LICENSE)

## 🙏 致谢 | Credits

- 原始项目 / Original: [FigForge](https://github.com/hengzzzhou/FigForge) by [@hengzzzhou](https://github.com/hengzzzhou)
- OpenClaw 适配 / OpenClaw adaptation: mathhyphen
- 技术支持 / Powered by: Google Gemini, OpenClaw

---

**祝您图表生成愉快！| Happy Figure Generation! 🎨✨**
