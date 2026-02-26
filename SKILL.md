---
name: figforge
description: |
  Scientific figure generator with style transfer support.
  
  Features:
  1. Generate publication-quality figures from text or MODULE LIST
  2. Style transfer from reference images
  3. Two modes: strict style matching or blended (academic + reference)
version: 3.0.0
author: mathhyphen
entry: scripts/run_complete.py
env:
  required:
    - GEMINI_API_KEY
  optional:
    - IMAGE_MODEL
    - OUTPUT_DIR
---

# FigForge v3.0 - Scientific Figure Generator with Style Transfer

Generate publication-quality scientific figures with style consistency.

## ✨ Key Features

1. **Text-to-Figure**: Convert scientific descriptions to professional figures
2. **Style Transfer**: Match style from reference images
3. **Style Blending**: Combine reference style with academic standards
4. **Multi-modal**: Supports both text-only and image+text generation

## 🚀 Quick Start

### Basic Usage (No Style Transfer)

```bash
# From text
python scripts/run_complete.py -i my_paper.txt

# From MODULE LIST
python scripts/run.py -m module_list.txt
```

### With Style Transfer

```bash
# Strict mode - match reference style exactly
python scripts/run_complete.py -i input.txt -r reference_figure.png

# Blend mode - reference style + academic standards
python scripts/run_complete.py -i input.txt -r reference.png --blend
```

### As OpenClaw Skill

```bash
# Basic
openclaw run figforge -i my_paper.txt

# With style transfer
openclaw run figforge -i my_paper.txt -r reference.png
```

## 📖 Usage Guide

### Step 1: Prepare Your Input

**Option A: Raw Text**
Create a text file describing your scientific content:
```
We propose a novel method for brain MRI analysis consisting of 
three stages: preprocessing, feature extraction, and classification...
```

**Option B: MODULE LIST** (for precise control)
Create a structured MODULE LIST with 8 sections:
```
MODULE LIST

1. Figure Goal and Type: Flowchart showing proposed method
2. Main Subjects: Brain MRI data, CNN model
3. Processes: Preprocessing → Feature Extraction → Classification
4. Relationships: Sequential flow with feedback loops
5. Outputs: Classification labels, confidence scores
6. Layout: Vertical flow, left-aligned
7. Text Labels: Stage names, data dimensions
8. Final Prompt: [Detailed description for image generation]
```

### Step 2: (Optional) Prepare Reference Image

Choose a reference image that represents your desired style:
- Previous publication figures for consistency
- Journal examples (Nature, Cell, Science style)
- Any scientific illustration you like

**Supported formats**: PNG, JPG, JPEG

### Step 3: Generate Figure

```bash
# Without style reference
python scripts/run_complete.py -i input.txt -o output.png

# With style transfer (strict mode)
python scripts/run_complete.py -i input.txt -r reference.png -o output.png

# With style blending (conservative)
python scripts/run_complete.py -i input.txt -r reference.png --blend -o output.png
```

## 🎨 Style Transfer Modes

### Mode 1: Strict (`-r reference.png`)
- **Best for**: Matching existing figures exactly
- **Behavior**: Applies reference style as precisely as possible
- **Use case**: Series of figures, consistency across papers

### Mode 2: Blend (`-r reference.png --blend`)
- **Best for**: Adopting colors while maintaining academic clarity
- **Behavior**: Combines reference colors with academic layout standards
- **Use case**: Adapting interesting styles safely

## 📋 Command Reference

### scripts/run.py (Direct Generation)

```bash
python scripts/run.py \
  -m module_list.txt \          # Required: MODULE LIST file
  -o figure.png \               # Optional: Output path
  -r reference.png \            # Optional: Reference image for style
  --blend \                     # Optional: Blend mode (conservative)
  --image-model MODEL           # Optional: Custom Gemini model
```

### scripts/run_complete.py (Full Workflow)

```bash
python scripts/run_complete.py \
  -i input.txt \                # Input text (needs analysis)
  -m module_list.txt \          # Or: pre-generated MODULE LIST
  -o figure.png \               # Output path
  -r reference.png \            # Reference image
  --blend \                     # Blend mode
  --skip-analysis               # Skip if input is already MODULE LIST
```

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | - | Google Gemini API key |
| `IMAGE_MODEL` | ❌ No | `gemini-2.0-flash-exp-image-generation` | Image generation model |
| `OUTPUT_DIR` | ❌ No | `outputs` | Output directory |

## 📁 Directory Structure

```
figforge/
├── scripts/
│   ├── run.py                 # Direct generation
│   ├── run_complete.py        # Full workflow
│   └── prompts/
│       └── step2_figure_generation.txt
├── examples/                  # Example inputs
├── outputs/                   # Generated figures
├── SKILL.md                   # This file
└── README.md                  # Detailed documentation
```

## 🐛 Troubleshooting

**"GEMINI_API_KEY not set"**
```bash
export GEMINI_API_KEY="your-key-here"
```

**"No image generated"**
- Check that MODULE LIST is properly formatted
- Verify reference image is valid PNG/JPG

**"Style not applied correctly"**
- Try `--blend` mode for more conservative results
- Ensure reference image has clear, distinct style

## 📄 License

MIT License

## 🙏 Acknowledgments

Based on FigForge by hengzzzhou, enhanced with style transfer capabilities.
