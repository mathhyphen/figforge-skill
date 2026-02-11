# FigForge - Image Generation Skill for OpenClaw

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenClaw Skill](https://img.shields.io/badge/OpenClaw-Skill-green.svg)](https://docs.openclaw.ai)

> **OpenClaw-optimized scientific figure generator (v2.0.0)**
> 
> This version is redesigned to work with OpenClaw's analysis models (Kimi/GLM/Qwen).
> Text analysis → OpenClaw Agent | Image generation → This skill

---

## 🎯 New Architecture (v2.0.0)

```
Your Scientific Text
        ↓
┌─────────────────────────────────┐
│  OpenClaw Agent Analysis        │  ← Uses your default model (Kimi K2.5, GLM-4.7, etc.)
│  (Text → MODULE LIST)           │
└─────────────────────────────────┘
        ↓
MODULE LIST (structured description)
        ↓
┌─────────────────────────────────┐
│  FigForge Skill (v2.0.0)        │  ← This tool
│  (MODULE LIST → Figure)         │     Uses Gemini for image generation only
└─────────────────────────────────┘
        ↓
Publication-Quality Figure
```

**Why this design?**
- ✅ Use OpenClaw's best text analysis model for understanding scientific content
- ✅ Use Gemini only for what it excels at: high-quality image generation
- ✅ More cost-effective: text analysis can use cheaper models
- ✅ Iterative refinement: adjust MODULE LIST before generating image

---

## 🚀 Quick Start

### Step 1: Generate MODULE LIST with OpenClaw

In your OpenClaw chat:
```
@coder Please analyze this text and create a MODULE LIST for figure generation:

[Your scientific text here...]

Create a detailed MODULE LIST with all 8 sections: goal, subjects, processes, 
relationships, outputs, layout, labels, and final prompt.
```

Save the output to a file (e.g., `module_list.txt`).

### Step 2: Generate Figure

```bash
# Set Gemini API Key
export GEMINI_API_KEY="your-gemini-api-key"

# Generate figure from MODULE LIST
python scripts/run.py -m module_list.txt -o figure.png
```

---

## 📋 Requirements

- Python 3.8+
- `google-genai` package
- Gemini API Key (for image generation)

```bash
pip install google-genai python-dotenv click
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | - | Google Gemini API key |
| `IMAGE_MODEL` | ❌ No | `gemini-2.0-flash-exp-image-generation` | Image generation model |
| `OUTPUT_DIR` | ❌ No | `outputs` | Output directory |

---

## 📖 Usage Examples

### Basic Usage

```bash
python scripts/run.py -m module_list.txt
```

### Custom Output

```bash
python scripts/run.py -m module_list.txt -o my_paper/figure1.png
```

### Custom Image Model

```bash
python scripts/run.py -m module_list.txt --image-model gemini-2.0-flash-exp-image-generation
```

---

## 🎨 MODULE LIST Format

The MODULE LIST is a structured text document with 8 sections:

1. **Figure Goal and Type** - What the figure illustrates
2. **Main Subjects/Inputs** - Key visual elements
3. **Processes/Methods/Stages** - Workflow steps
4. **Relationships and Flow** - How elements connect
5. **Outputs/Readouts/Results** - Expected visualizations
6. **Layout and Visual Style** - Design specifications
7. **Text Labels and Annotations** - Labels and text
8. **Final Prompt** - Complete generation prompt

See `examples/` for sample MODULE LIST files.

---

## 🔄 Comparison: Original vs OpenClaw Version

| Feature | Original FigForge (v1) | OpenClaw Version (v2) |
|---------|-------------------|------------------|
| Text Analysis | Built-in (GPT-5) | OpenClaw Agent (Kimi/GLM/Qwen) |
| Image Generation | Gemini | Gemini |
| Flexibility | Fixed pipeline | Modular, use best model for each step |
| Cost | Higher (both steps use expensive models) | Lower (analysis can use cheaper models) |
| Refinement | Must regenerate both steps | Can refine MODULE LIST before image gen |

---

## 🛠️ Integration with OpenClaw

### As OpenClaw Skill

```yaml
# In agent config
skills:
  figforge:
    entry: scripts/run.py
    env:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
```

### Direct Usage

The skill can also be used standalone:

```python
from scripts.run import FigForgeGenerator

generator = FigForgeGenerator()
figure_path = generator.generate_figure(module_list_content)
```

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Credits

- Original project: [FigForge](https://github.com/hengzzzhou/FigForge) by [@hengzzzhou](https://github.com/hengzzzhou)
- OpenClaw adaptation: mathhyphen
- Powered by: Google Gemini, OpenClaw

---

**Happy Figure Generation! 🎨✨**
