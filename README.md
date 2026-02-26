# FigForge v3.0 - Scientific Figure Generator with Style Transfer

<div align="center">

**Generate publication-quality scientific figures with style consistency**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## ✨ Features

- 🎨 **Style Transfer**: Match style from reference images
- 📝 **Text-to-Figure**: Convert scientific descriptions to figures
- 🔄 **Style Blending**: Combine reference style with academic standards
- 🚀 **Easy to Use**: Simple CLI and OpenClaw integration

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/mathhyphen/figforge-skill.git
cd figforge-skill
pip install -r requirements.txt
```

### Set API Key

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Get your key from: https://makersuite.google.com/app/apikey

### Basic Usage

```bash
# Generate from text
python scripts/run_complete.py -i input.txt

# Generate from MODULE LIST
python scripts/run.py -m module_list.txt
```

---

## 🎨 Style Transfer (NEW in v3.0)

### What is Style Transfer?

Upload a reference image and FigForge will generate new figures matching that style - perfect for maintaining consistency across papers or matching journal aesthetics.

### Usage

```bash
# Strict mode - match reference style exactly
python scripts/run_complete.py -i input.txt -r reference.png

# Blend mode - reference colors + academic layout
python scripts/run_complete.py -i input.txt -r reference.png --blend
```

### Which Mode to Use?

| Mode | Best For | Behavior |
|------|---------|----------|
| **Default** (strict) | Matching existing figures | Applies reference style as precisely as possible |
| **Blend** | Adapting interesting styles | Combines reference colors with academic standards |

### Example

```bash
# Generate Figure 1 matching your lab's previous publication style
python scripts/run_complete.py -i paper_methods.txt -r previous_paper_fig1.png

# Generate with Nature journal style (from a Nature figure example)
python scripts/run_complete.py -i input.txt -r nature_example.png

# Generate with custom style but keep academic clarity
python scripts/run_complete.py -i input.txt -r custom_style.png --blend
```

---

## 📖 Detailed Usage

### Input Formats

#### Option 1: Raw Text

Create a text file describing your scientific content:

```
We propose a novel method for brain MRI analysis consisting of 
three stages: preprocessing, feature extraction, and classification.
The preprocessing step involves...
```

#### Option 2: MODULE LIST (Precise Control)

Create a structured specification:

```
MODULE LIST

1. Figure Goal and Type: Flowchart showing proposed method
2. Main Subjects: Brain MRI data, CNN model
3. Processes: Preprocessing → Feature Extraction → Classification
4. Relationships: Sequential flow with feedback loops
5. Outputs: Classification labels, confidence scores
6. Layout: Vertical flow, left-aligned, clean design
7. Text Labels: Stage names, data dimensions, arrows
8. Final Prompt: A professional flowchart showing...
```

### Command Reference

#### Full Workflow (Text → Analysis → Figure)

```bash
python scripts/run_complete.py -i input.txt [OPTIONS]

Options:
  -o, --output PATH          Output figure path
  -r, --reference PATH       Reference image for style transfer
  --blend                    Blend mode (conservative)
  --image-model MODEL        Custom Gemini model
```

#### Direct Generation (MODULE LIST → Figure)

```bash
python scripts/run.py -m module_list.txt [OPTIONS]

Options:
  -o, --output PATH          Output figure path
  -r, --reference PATH       Reference image for style transfer
  --blend                    Blend mode (conservative)
  --image-model MODEL        Custom Gemini model
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | - | Google Gemini API key |
| `IMAGE_MODEL` | ❌ No | `gemini-2.0-flash-exp-image-generation` | Image model |
| `OUTPUT_DIR` | ❌ No | `outputs` | Output directory |

### As OpenClaw Skill

Add to your OpenClaw skills directory:

```bash
# Install
openclaw skill install /path/to/figforge

# Use
openclaw run figforge -i input.txt
openclaw run figforge -i input.txt -r reference.png
```

---

## 📁 Project Structure

```
figforge/
├── scripts/
│   ├── run.py                 # Direct generation (MODULE LIST → Figure)
│   ├── run_complete.py        # Full workflow (Text → Figure)
│   └── prompts/
│       └── step2_figure_generation.txt
├── examples/                  # Example inputs
│   ├── optical_simulation_framework.txt
│   └── livesearchbench.txt
├── outputs/                   # Generated figures (created automatically)
├── SKILL.md                   # OpenClaw skill definition
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

---

## 🐛 Troubleshooting

### "GEMINI_API_KEY not set"

```bash
export GEMINI_API_KEY="your-key-here"
```

Or add to your `.env` file:
```
GEMINI_API_KEY=your-key-here
```

### "No image generated"

- Check MODULE LIST format (8 sections required)
- Verify Gemini API key is valid
- Check console output for errors

### "Style not applied correctly"

- Try `--blend` mode for more conservative results
- Ensure reference image has clear, distinct style
- Use high-quality reference images (PNG preferred)

### "google-genai not found"

```bash
pip install google-genai pillow
```

---

## 💡 Tips

### Best Practices for Style Transfer

1. **Choose good reference images**:
   - High resolution (≥1000px)
   - Clear, professional style
   - Similar content type (flowchart → flowchart)

2. **Match your use case**:
   - Previous paper figures → Strict mode
   - Interesting styles from other sources → Blend mode

3. **Maintain consistency**:
   - Use same reference for all figures in a paper
   - Create a style library for your lab

### MODULE LIST Best Practices

1. **Be specific in Section 6 (Layout)**:
   - Specify orientation (vertical/horizontal)
   - Define color scheme
   - Describe spacing and density

2. **Include all labels in Section 7**:
   - Arrow labels
   - Component names
   - Units and dimensions

3. **Section 8 should be comprehensive**:
   - Visual description
   - Style notes
   - Technical requirements

---

## 📚 Examples

### Example 1: Simple Flowchart

```bash
# Input: methods.txt
# "Our method has 3 stages: data collection, processing, and analysis..."

python scripts/run_complete.py -i methods.txt -o flowchart.png
```

### Example 2: Style-Matched Series

```bash
# Generate multiple figures with consistent style
python scripts/run_complete.py -i fig1_input.txt -r lab_style.png -o figure1.png
python scripts/run_complete.py -i fig2_input.txt -r lab_style.png -o figure2.png
python scripts/run_complete.py -i fig3_input.txt -r lab_style.png -o figure3.png
```

### Example 3: Journal-Specific Style

```bash
# Match Nature journal style
python scripts/run_complete.py -i input.txt -r nature_example.png -o nature_style_fig.png

# Match Cell journal style
python scripts/run_complete.py -i input.txt -r cell_example.png -o cell_style_fig.png
```

---

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Original FigForge concept by hengzzzhou
- Style transfer enhancement by mathhyphen
- Powered by Google Gemini

---

## 📮 Contact

- GitHub: [@mathhyphen](https://github.com/mathhyphen)
- Issues: [GitHub Issues](https://github.com/mathhyphen/figforge-skill/issues)

---

<div align="center">

**Made with ❤️ for researchers**

</div>
