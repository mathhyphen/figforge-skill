# FigForge OpenClaw Skill - Usage Summary

## Installation Status

✅ **Successfully installed** to: `C:\Users\Administrator\.openclaw\workspace-coder\skills\figforge`

## Skill Overview

**FigForge** is an AI-powered scientific figure generator that creates publication-quality NeurIPS/ICLR-style figures from research text.

### Core Features

- 🤖 **Two-step AI workflow**: GPT-5 analysis + Gemini image generation
- 📊 **Publication-ready**: Clean, professional scientific figures
- 🔧 **Flexible API support**: OpenAI-compatible and native Google Gemini APIs
- 💾 **Automatic output saving**: MODULE LIST + generated figures

## Quick Start

### 1. Set Up Environment Variables

**Required:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**Optional (but recommended):**
```bash
# For OpenAI-compatible endpoints (default)
export OPENAI_BASE_URL="https://your-endpoint.com/v1"
export API_TYPE="openai"

# For native Gemini API
export API_TYPE="gemini"
export GEMINI_API_KEY="your-gemini-api-key"

# Model settings
export ANALYSIS_MODEL="gpt-5"
export IMAGE_MODEL="gemini-2.5-flash-image"

# Output directory
export OUTPUT_DIR="outputs"
```

### 2. Run the Skill

**Note on Windows:** Set UTF-8 encoding for proper emoji display:
```powershell
$env:PYTHONIOENCODING="utf-8"
```

**Generate from a text file:**
```bash
python C:\Users\Administrator\.openclaw\workspace-coder\skills\figforge\scripts\run.py -i path/to/scientific-text.txt
```

**Generate from direct text:**
```bash
python C:\Users\Administrator\.openclaw\workspace-coder\skills\figforge\scripts\run.py -t "Your scientific text here..."
```

**Specify custom output:**
```bash
python C:\Users\Administrator\.openclaw\workspace-coder\skills\figforge\scripts\run.py -i input.txt -o my_figure.png
```

**Generate MODULE LIST only:**
```bash
python C:\Users\Administrator\.openclaw\workspace-coder\skills\figforge\scripts\run.py -i input.txt --module-list-only
```

## Usage Examples

### Example 1: Neural Network Architecture

```bash
python C:\Users\Administrator\.openclaw\workspace-coder\skills\figforge\scripts\run.py -t "We propose a transformer-based model with multi-head self-attention, layer normalization, and a feed-forward network. The model takes text embeddings as input and passes them through 6 encoder layers."
```

### Example 2: Computer Vision Model

```bash
python C:\Users\Administrator\.openclaw\workspace-coder\skills\figforge\scripts\run.py -i vision_model.txt -o architecture.png
```

### Example 3: Research Method Pipeline

```bash
python C:\Users\Administrator\.openclaw\workspace-coder\skills\figforge\scripts\run.py -t "Our method consists of three stages: data preprocessing with augmentation, feature extraction using a pre-trained CNN, and classification with a multi-layer perceptron."
```

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input FILE` | `-i` | Path to input text file |
| `--text TEXT` | `-t` | Scientific text as string |
| `--output FILE` | `-o` | Custom output path |
| `--module-list-only` | | Generate MODULE LIST only |
| `--help` | `-h` | Show help message |

## Output Files

The skill generates two files:

1. **Figure**: `{input_name}_{timestamp}.png` - The generated scientific figure
2. **MODULE LIST**: `{input_name}_module_list_{timestamp}.txt` - Structured architecture breakdown

Example output:
```
outputs/
├── sample_input_20250211_021522.png        # Generated figure
└── sample_input_module_list_20250211_021522.txt  # MODULE LIST
```

## Configuration Options

### API Configuration

**Option 1: OpenAI-Compatible API (Default)**
```bash
export API_TYPE="openai"
export OPENAI_BASE_URL="https://your-relay-endpoint.com/v1"
export OPENAI_API_KEY="your-api-key"
```

**Option 2: Native Google Gemini API**
```bash
export API_TYPE="gemini"
export OPENAI_BASE_URL="https://your-openai-endpoint.com/v1"
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | - | API key for OpenAI-compatible endpoint |
| `OPENAI_BASE_URL` | No | - | OpenAI-compatible API endpoint URL |
| `API_TYPE` | No | `openai` | `openai` or `gemini` |
| `GEMINI_API_KEY` | No | - | Google Gemini API key (when API_TYPE=gemini) |
| `ANALYSIS_MODEL` | No | `gpt-5` | Model for MODULE LIST generation |
| `IMAGE_MODEL` | No | `gemini-2.5-flash-image` | Model for figure generation |
| `OUTPUT_DIR` | No | `outputs` | Directory for output files |

## Dependencies

### Required
- Python 3.8+
- openai >= 1.0.0
- python-dotenv >= 1.0.0
- click >= 8.1.0
- pillow >= 10.0.0
- requests >= 2.31.0

### Optional (for native Gemini API)
- google-genai >= 0.2.0

**Install dependencies:**
```bash
pip install -r C:\Users\Administrator\.openclaw\workspace-coder\skills\figforge\requirements.txt
```

Or install optional dependencies:
```bash
pip install google-genai>=0.2.0
```

## How It Works

### Step 1: MODULE LIST Generation
GPT-5 analyzes your scientific text and creates a structured MODULE LIST:
- Input(s): Data sources and preprocessing
- Preprocessing/Encoding/Embedding: Feature extraction layers
- Core Architecture/Stages/Blocks: Main model components
- Special Mechanisms: Attention, memory, routing, etc.
- Output Head: Final prediction layers

### Step 2: Figure Generation
The image model generates a clean, professional figure following:
- Flat, clean conference style
- Consistent thin line weights
- Professional pastel color palette
- Rounded rectangles for module blocks
- Clear arrows indicating data flow
- Concise labels

## Tips for Best Results

1. **Provide Clear Text**: Detailed and structured input produces better MODULE LISTs
2. **Describe Flow**: Explicitly mention data flow and connections between components
3. **Specify Components**: Name specific layers, blocks, or mechanisms
4. **Review MODULE LIST**: Check the generated MODULE LIST before proceeding
5. **Iterate**: Regenerate figures with modified MODULE LIST for fine-tuning

## Troubleshooting

### Error: "OPENAI_API_KEY is required"
**Solution**: Set the `OPENAI_API_KEY` environment variable

### Error: "Prompt template not found"
**Solution**: Ensure the `prompts/` directory exists with template files

### Error: "google-genai package is not installed"
**Solution**: Install optional dependencies: `pip install google-genai`

### Image generation fails
**Solution**: Check that your API endpoint supports the specified image model

### Unicode/Emoji issues on Windows
**Solution**: Set `PYTHONIOENCODING=utf-8` before running:
```powershell
$env:PYTHONIOENCODING="utf-8"
python ...
```

## File Structure

```
figforge/
├── SKILL.md                    # Main skill documentation
├── _meta.json                  # Skill metadata
├── package.json                # Package information
├── requirements.txt            # Python dependencies
├── scientific_plotter.py      # Core implementation
├── scripts/
│   └── run.py                 # Entry point (standard OpenClaw format)
├── prompts/                   # Prompt templates
│   ├── step1_module_generation.txt
│   └── step2_figure_generation.txt
├── examples/                  # Example inputs
│   ├── sample_input.txt
│   └── ...
├── output_case/              # Example outputs
└── outputs/                  # Generated figures (auto-created)
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Credits

- Original repository: [FigForge](https://github.com/hengzzzhou/FigForge)
- Powered by OpenAI-compatible API endpoints and Google Gemini

---

**Happy Scientific Plotting! 🎨✨**
