---
name: figforge
description: |
  Scientific figure generator for OpenClaw.
  
  Two modes:
  1. Full workflow: Text → OpenClaw analysis → MODULE LIST → Figure
  2. Image only: MODULE LIST → Figure (uses Gemini image generation)
version: 2.0.0
author: mathhyphen (based on FigForge by hengzzzhou)
entry: scripts/run_complete.py
env:
  required:
    - GEMINI_API_KEY
  optional:
    - IMAGE_MODEL
    - OUTPUT_DIR
---

# FigForge - Scientific Figure Generator

Generate publication-quality scientific figures for OpenClaw.

## 🎯 Two Usage Modes

### Mode 1: Full Workflow (Recommended)

**Text → Analysis → Figure** (in one command)

```bash
# Analyzes text using OpenClaw agent, then generates figure
python scripts/run_complete.py -i input.txt -o figure.png
```

### Mode 2: Image Generation Only

**MODULE LIST → Figure** (if you already have MODULE LIST)

```bash
# Generate figure from pre-generated MODULE LIST
python scripts/run.py -m module_list.txt -o figure.png
```

---

## 🚀 Quick Start

### Direct Usage with OpenClaw

```bash
# Let OpenClaw handle everything
openclaw run figforge -i my_paper.txt -o figure.png
```

### Step-by-Step

1. **Prepare your text** (scientific description)
2. **Run FigForge**:
   ```bash
   export GEMINI_API_KEY="your-key"
   python scripts/run_complete.py -i my_text.txt
   ```
3. **Get your figure** in `outputs/` directory

---

## 📋 Usage Examples

### Example 1: Full Workflow

```bash
# Input: Raw scientific text
# Output: Generated figure
python scripts/run_complete.py -i methodology.txt -o results/figure1.png
```

### Example 2: With Custom Model

```bash
python scripts/run_complete.py -i input.txt --image-model gemini-2.0-flash-exp-image-generation
```

### Example 3: Using Pre-generated MODULE LIST

```bash
# If you already have MODULE LIST from previous analysis
python scripts/run.py -m module_list.txt -o figure.png
```

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | - | Google Gemini API key |
| `IMAGE_MODEL` | ❌ No | `gemini-2.0-flash-exp-image-generation` | Image model |
| `OUTPUT_DIR` | ❌ No | `outputs` | Output directory |

---

## 📝 How It Works

### Full Workflow (run_complete.py)

```
Your Text
    ↓
OpenClaw Agent (Kimi/GLM/Qwen) - Analysis
    ↓
MODULE LIST
    ↓
Gemini Image Generation
    ↓
Figure
```

### Image Only Mode (run.py)

```
MODULE LIST (pre-generated)
    ↓
Gemini Image Generation
    ↓
Figure
```

---

## 🎨 What is MODULE LIST?

A structured description with 8 sections:
1. Figure Goal and Type
2. Main Subjects/Inputs
3. Processes/Methods/Stages
4. Relationships and Flow
5. Outputs/Readouts/Results
6. Layout and Visual Style
7. Text Labels and Annotations
8. Final Prompt

See `examples/` for samples.

---

## 🔍 Troubleshooting

**"No input provided"**
→ Use `-i` for text file or `-m` for module list file

**"GEMINI_API_KEY not set"**
→ Set your API key: `export GEMINI_API_KEY="your-key"`

**"Input needs analysis"**
→ The tool detected raw text. It will show you a prompt to run in OpenClaw first.

---

## 📄 License

MIT License