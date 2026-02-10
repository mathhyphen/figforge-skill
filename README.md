# FigForge Skill - OpenClaw AI 科研绘图工具

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenClaw Skill](https://img.shields.io/badge/OpenClaw-Skill-green.svg)](https://docs.openclaw.ai)

> **本项目基于 [FigForge](https://github.com/hengzzzhou/FigForge) 原始仓库，已封装为 OpenClaw Skill，支持中文界面与 Gemini 3 Pro 系列模型。**

🎨 **基于 AI 的科研论文图表生成工具**

本工具使用两步工作流，将您的科研文本转化为符合 NeurIPS/ICLR 会议风格的发表级图表：
1. **Gemini-3-Pro-Preview** 分析科研文本，生成结构化 MODULE LIST
2. **Gemini-3-Pro-Image-Preview** 基于 MODULE LIST 创建专业图表

---

## 📋 目录

- [✨ 功能特点](#-功能特点)
- [🚀 快速开始](#-快速开始)
- [📸 示例](#-示例)
- [📖 工作原理](#-工作原理)
- [⚙️ 配置说明](#️-配置说明)
- [🛠️ 故障排除](#️-故障排除)
- [📝 许可证](#-许可证)

---

## ✨ 功能特点

- 🤖 **集成 AI 工作流**: Gemini 分析引导图像生成，效果更优
- 📊 **发表级质量**: 生成干净、符合会议风格的科研图表
- 🎯 **结构化方法**: 两步流程确保逻辑准确、可视化清晰
- 🔧 **开箱即用**: 已预配置 API Key 和模型参数
- 💾 **自动保存**: 同时保存 MODULE LIST 和生成图表
- 🌐 **双语支持**: 完整的中英文文档和界面

---

## 🚀 快速开始

### 方式一：作为 OpenClaw Skill 使用（推荐）

```bash
# 技能已安装到 OpenClaw，直接调用
openclaw run figforge -i input.txt
```

或在 OpenClaw 对话中直接请求：
> "帮我用 figforge 生成这张论文的架构图"

### 方式二：独立使用

```bash
# 1. 克隆仓库
git clone https://github.com/mathhyphen/figforge-skill.git
cd figforge-skill

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行（已预配置 Gemini API）
python scripts/run.py -i examples/sample_input.txt
```

### 命令行参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `--input FILE` | `-i` | 输入文件路径 |
| `--text TEXT` | `-t` | 直接输入科研文本 |
| `--output FILE` | `-o` | 自定义输出路径 |
| `--module-list-only` | | 仅生成 MODULE LIST |

---

## 📸 示例

### 示例 1 - 神经网络架构

![sample_input_1.png](output_case/sample_input_1.png)

### 示例 2 - 衰老小鼠大脑白质脆弱性研究 (Cell 2023)

基于 Hahn et al. 发表于 Cell 的研究论文，使用 FigForge 生成的科研图表：

> **论文引用**: Hahn O, Foltz A G, Atkins M, et al. Atlas of the aging mouse brain reveals white matter as vulnerable foci[J]. Cell, 2023, 186(19): 4117-4133. e22.

**图 1 - 白质脆弱性对比**
![hahn2023_white_matter_vulnerability.jpg](output_case/hahn2023_white_matter_vulnerability.jpg)
展示年轻 vs 衰老小鼠大脑中白质作为脆弱焦点的转录组学变化。

**图 2 - 空间成像-转录组学方法流程**
![hahn2023_workflow.jpg](output_case/hahn2023_workflow.jpg)
三步流程：(1) 数据采集（Micro-MRI + 空间转录组学）→ (2) 共配准 → (3) 分子解码与疾病对比。

### 更多示例

- [LiveSearchBench](https://arxiv.org/abs/2511.01409)
- [ReSo](https://arxiv.org/abs/2503.02390)
- [VIKI-R](https://arxiv.org/abs/2506.09049)

> 所有图表均采用扁平化会议风格设计，线条统一、配色专业。
> 
> **注**: Hahn et al. Cell 2023 示例图由 FigForge 基于论文方法描述自动生成，用于展示工具的科研图表生成能力。

---

## 📖 工作原理

### 第一步：MODULE LIST 生成（Gemini-3-Pro-Preview）

Gemini 模型分析科研文本，创建结构化 MODULE LIST，将架构分解为：

1. **输入层**: 数据源和预处理
2. **预处理/编码/嵌入**: 特征提取层
3. **核心架构/阶段/模块**: 主要模型组件
4. **特殊机制**: 注意力、记忆、路由等
5. **输出头**: 最终预测层

### 第二步：图表生成（Gemini-3-Pro-Image-Preview）

基于 MODULE LIST，生成专业图表，遵循以下设计原则：

- ✅ 扁平化、干净的会议风格（无渐变、阴影）
- ✅ 统一的细线宽
- ✅ 专业 pastel 配色
- ✅ 圆角矩形表示模块
- ✅ 清晰的箭头指示数据流
- ✅ 简洁标签（无长句）
- ✅ 纯白背景，间距合理

---

## ⚙️ 配置说明

### 默认配置（已预设）

```python
API_TYPE = "gemini"
GEMINI_API_KEY = "AIzaSyAgDSH-3h6reUvVJ9cxVn2_FQ_DA3LnSdg"
ANALYSIS_MODEL = "gemini-3-pro-preview"
IMAGE_MODEL = "gemini-3-pro-image-preview"
```

### 自定义配置（可选）

如需使用自己的 API Key 或切换模型，设置环境变量：

```bash
# Windows
set GEMINI_API_KEY="your-api-key"
set ANALYSIS_MODEL="gemini-3-pro-preview"
set IMAGE_MODEL="gemini-3-pro-image-preview"

# Linux/macOS
export GEMINI_API_KEY="your-api-key"
export ANALYSIS_MODEL="gemini-3-pro-preview"
export IMAGE_MODEL="gemini-3-pro-image-preview"
```

### 项目结构

```
figforge-skill/
├── scripts/
│   └── run.py                 # OpenClaw 标准入口
├── scientific_plotter.py      # 核心绘图逻辑
├── requirements.txt           # Python 依赖
├── SKILL.md                   # OpenClaw 技能文档
├── _meta.json                 # 技能元数据
├── package.json               # 包信息
├── prompts/                   # 提示词模板
├── examples/                  # 示例输入
└── outputs/                   # 生成图表（自动创建）
```

---

## 🛠️ 故障排除

**问题**: `ImportError: google-genai package is not installed`
- **解决**: `pip install google-genai`

**问题**: API 连接错误
- **解决**: 检查网络连接，或更换 API Key

**问题**: 图像生成失败
- **解决**: 确认 Gemini 3 Pro Image Preview 模型可用

---

## 📝 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

### 致谢

- 原始项目: [FigForge](https://github.com/hengzzzhou/FigForge) by [@hengzzzhou](https://github.com/hengzzzhou)
- OpenClaw 集成: [OpenClaw](https://github.com/openclaw/openclaw)
- 模型支持: Google Gemini 3 Pro 系列

---

**Happy Scientific Plotting! 🎨✨**

如有问题，请在 GitHub 提交 Issue。

---

# FigForge Skill - OpenClaw AI Scientific Figure Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenClaw Skill](https://img.shields.io/badge/OpenClaw-Skill-green.svg)](https://docs.openclaw.ai)

> **This project is based on the [FigForge](https://github.com/hengzzzhou/FigForge) original repository, packaged as an OpenClaw Skill with Chinese interface and Gemini 3 Pro series model support.**

🎨 **AI-powered scientific figure generation tool**

This tool uses a two-step workflow to transform your research text into publication-ready figures in NeurIPS/ICLR conference style:
1. **Gemini-3-Pro-Preview** analyzes research text and generates structured MODULE LIST
2. **Gemini-3-Pro-Image-Preview** creates professional figures based on the MODULE LIST

## ✨ Features

- 🤖 **Integrated AI Workflow**: Gemini analysis guides image generation for optimal results
- 📊 **Publication-Ready**: Generates clean, conference-style scientific figures
- 🎯 **Structured Approach**: Two-step process ensures logical and accurate visualizations
- 🔧 **Ready to Use**: Pre-configured API Key and model parameters
- 💾 **Automatic Saving**: Saves both MODULE LIST and generated figures
- 🌐 **Bilingual Support**: Complete Chinese and English documentation and interface

## 🚀 Quick Start

### Option 1: Use as OpenClaw Skill (Recommended)

```bash
# Skill is already installed in OpenClaw, call directly
openclaw run figforge -i input.txt
```

Or simply request in OpenClaw conversation:
> "Help me generate an architecture diagram for this paper using figforge"

### Option 2: Standalone Use

```bash
# 1. Clone repository
git clone https://github.com/mathhyphen/figforge-skill.git
cd figforge-skill

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (pre-configured with Gemini API)
python scripts/run.py -i examples/sample_input.txt
```

## ⚙️ Configuration

### Default Configuration (Pre-set)

```python
API_TYPE = "gemini"
GEMINI_API_KEY = "AIzaSyAgDSH-3h6reUvVJ9cxVn2_FQ_DA3LnSdg"
ANALYSIS_MODEL = "gemini-3-pro-preview"
IMAGE_MODEL = "gemini-3-pro-image-preview"
```

### Custom Configuration (Optional)

To use your own API Key or switch models, set environment variables:

```bash
# Windows
set GEMINI_API_KEY="your-api-key"
set ANALYSIS_MODEL="gemini-3-pro-preview"
set IMAGE_MODEL="gemini-3-pro-image-preview"

# Linux/macOS
export GEMINI_API_KEY="your-api-key"
export ANALYSIS_MODEL="gemini-3-pro-preview"
export IMAGE_MODEL="gemini-3-pro-image-preview"
```

## 📝 License

This project is open source under the [MIT License](LICENSE).

### Acknowledgments

- Original Project: [FigForge](https://github.com/hengzzzhou/FigForge) by [@hengzzzhou](https://github.com/hengzzzhou)
- OpenClaw Integration: [OpenClaw](https://github.com/openclaw/openclaw)
- Model Support: Google Gemini 3 Pro Series

---

**Happy Scientific Plotting! 🎨✨**

For questions, please submit an Issue on GitHub.
