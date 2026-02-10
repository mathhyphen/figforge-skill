---
name: figforge
description: |
  AI-powered scientific figure generator using GPT-5 and Gemini models.
  Generate publication-quality NeurIPS/ICLR-style scientific figures from research text.
metadata: {
  "openclaw": {
    "emoji": "🎨",
    "requires": {
      "bins": ["python3"],
      "python": ["openai>=1.0.0", "python-dotenv>=1.0.0", "click>=8.1.0", "pillow>=10.0.0", "requests>=2.31.0"],
      "optional_python": ["google-genai>=0.2.0"]
    },
    "env": ["OPENAI_API_KEY"],
    "optionalEnv": ["OPENAI_BASE_URL", "GEMINI_API_KEY", "API_TYPE", "ANALYSIS_MODEL", "IMAGE_MODEL", "OUTPUT_DIR"]
  }
}
---

# FigForge - AI Scientific Figure Generator 🎨

Generate publication-quality scientific figures using AI models with a sophisticated two-step workflow:

1. **GPT-5** analyzes your scientific text and generates a structured MODULE LIST
2. **Gemini-2.5-flash-image** creates a professional figure based on the MODULE LIST

Perfect for creating NeurIPS/ICLR-style scientific figures for research papers.

## Features

- 🤖 **Integrated AI Workflow**: GPT-5 analysis guides image generation for optimal results
- 📊 **Publication-Ready**: Generates clean, NeurIPS-style scientific figures
- 🎯 **Structured Approach**: Two-step process ensures logical, accurate visualizations
- 🔧 **Flexible Configuration**: Support for OpenAI-compatible and native Google Gemini APIs
- 💾 **Automatic Saving**: Saves both MODULE LIST and generated figures
- 🖥️ **Multiple Input Methods**: File-based or direct text input

## Quick Start

### 1. Configure Environment

Create a `.env` file or set environment variables:

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Optional (default: OpenAI-compatible endpoint)
export OPENAI_BASE_URL="https://your-endpoint.com/v1"
export API_TYPE="openai"  # or "gemini"

# Optional (for native Gemini API)
export GEMINI_API_KEY="your-gemini-api-key"

# Model settings
export ANALYSIS_MODEL="gpt-5"
export IMAGE_MODEL="gemini-2.5-flash-image"

# Output directory
export OUTPUT_DIR="outputs"
```

### 2. Generate a Figure

#### From a text file:
```bash
python {baseDir}/scripts/run.py -i path/to/your/scientific-text.txt
```

#### From direct text:
```bash
python {baseDir}/scripts/run.py -t "Your scientific text describing your model architecture..."
```

#### Specify custom output:
```bash
python {baseDir}/scripts/run.py -i input.txt -o my_figure.png
```

#### Generate MODULE LIST only:
```bash
python {baseDir}/scripts/run.py -i input.txt --module-list-only
```

## Usage Examples

### Example 1: Neural Network Architecture

```bash
python {baseDir}/scripts/run.py -t "We propose a transformer-based model with multi-head self-attention, layer normalization, and a feed-forward network. The model takes text embeddings as input and passes them through 6 encoder layers."
```

### Example 2: Vision Model

```bash
python {baseDir}/scripts/run.py -i vision_model_description.txt -o architecture_diagram.png
```

### Example 3: Research Method Pipeline

```bash
python {baseDir}/scripts/run.py -t "Our method consists of three stages: data preprocessing with augmentation, feature extraction using a pre-trained CNN, and classification with a multi-layer perceptron."
```

## API Configuration

### Option 1: OpenAI-Compatible API (Default)

```bash
export API_TYPE="openai"
export OPENAI_BASE_URL="https://your-relay-endpoint.com/v1"
export OPENAI_API_KEY="your-api-key"
```

This uses a unified OpenAI-compatible endpoint for both analysis (GPT-5) and image generation (Gemini-2.5-flash-image).

### Option 2: Native Google Gemini API

```bash
export API_TYPE="gemini"
export OPENAI_BASE_URL="https://your-openai-endpoint.com/v1"
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

When using `API_TYPE=gemini`:
- MODULE LIST generation uses the OpenAI-compatible API
- Image generation uses the native Google Gemini API

## Command-Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input FILE` | `-i` | Path to input text file |
| `--text TEXT` | `-t` | Scientific text as string |
| `--output FILE` | `-o` | Custom output path |
| `--module-list-only` | | Generate MODULE LIST without creating the figure |
| `--help` | `-h` | Show help message |

## Output Files

When generating a figure, the following files are created:

1. **Figure**: `{input_name}_{timestamp}.png` - The generated scientific figure
2. **MODULE LIST**: `{input_name}_module_list_{timestamp}.txt` - Structured breakdown of the architecture

Example:
```
outputs/
├── sample_input_20250211_021522.png        # Generated figure
└── sample_input_module_list_20250211_021522.txt  # MODULE LIST
```

## How It Works

### Step 1: MODULE LIST Generation

The GPT-5 model analyzes your scientific text and creates a structured MODULE LIST:

```
Input(s): Data sources and preprocessing
Preprocessing/Encoding/Embedding: Feature extraction layers
Core Architecture/Stages/Blocks: Main model components in sequence
Special Mechanisms: Attention, memory, routing, etc.
Output Head: Final prediction layers
```

### Step 2: Figure Generation

Using the MODULE LIST as a guide, the image model generates a clean, professional figure following:

- ✅ Flat, clean conference style (no gradients, shadows)
- ✅ Consistent thin line weights
- ✅ Professional pastel color palette
- ✅ Rounded rectangles for module blocks
- ✅ Clear arrows indicating data flow
- ✅ Concise labels (no long sentences)
- ✅ Pure white background with clean spacing

## Tips for Best Results

1. **Provide Clear Text**: The more detailed and structured your input text, the better the MODULE LIST
2. **Describe Flow**: Explicitly mention data flow and connections between components
3. **Specify Components**: Name specific layers, blocks, or mechanisms in your architecture
4. **Review MODULE LIST**: Check the generated MODULE LIST before proceeding to figure generation
5. **Iterate**: You can regenerate figures with modified MODULE LIST for fine-tuning

## Troubleshooting

### Error: "OPENAI_API_KEY is required"
**Solution**: Set the `OPENAI_API_KEY` environment variable

### Error: "Prompt template not found"
**Solution**: Ensure the `prompts/` directory exists with template files

### Error: "google-genai package is not installed"
**Solution**: Install optional dependencies: `pip install google-genai`

### Image generation fails
**Solution**: Check that your API endpoint supports the specified image model

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

## Advanced Configuration

All settings can be configured via environment variables:

```bash
# API Configuration
OPENAI_API_KEY="your-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
GEMINI_API_KEY="your-gemini-key"  # Only when API_TYPE=gemini
API_TYPE="openai"  # or "gemini"

# Model Selection
ANALYSIS_MODEL="gpt-5"
IMAGE_MODEL="gemini-2.5-flash-image"

# Output
OUTPUT_DIR="outputs"
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Credits

- Original repository: [FigForge](https://github.com/hengzzzhou/FigForge)
- Powered by OpenAI-compatible API endpoints and Google Gemini

---

**Happy Scientific Plotting! 🎨✨**
