---
name: figforge
description: |
  Scientific figure generator - Image generation component.
  Receives MODULE LIST from OpenClaw analysis and creates publication-quality figures.
  
  Workflow:
  1. OpenClaw Agent (Kimi/GLM/Qwen) analyzes text → generates MODULE LIST
  2. This skill receives MODULE LIST → generates figure using Gemini image model
version: 2.0.0
author: mathhyphen (based on FigForge by hengzzzhou)
entry: scripts/run.py
env:
  required:
    - GEMINI_API_KEY
  optional:
    - IMAGE_MODEL
    - OUTPUT_DIR
---

# FigForge - Image Generation Skill

Generate publication-quality scientific figures from pre-analyzed MODULE LIST.

## Workflow

Unlike the original FigForge, this OpenClaw Skill is designed to work with OpenClaw's 
default analysis models (Kimi K2.5, GLM-4.7, Qwen, etc.):

```
Your Scientific Text
        ↓
OpenClaw Agent Analysis (Kimi/GLM/Qwen)
        ↓
MODULE LIST (structured description)
        ↓
FigForge Skill (this tool)
        ↓
Publication-Quality Figure
```

## Prerequisites

1. **Generate MODULE LIST first** using OpenClaw:
   ```
   @coder Analyze this text and create a MODULE LIST for figure generation:
   [your scientific text here]
   ```

2. **Save the MODULE LIST** to a file (e.g., `module_list.txt`)

3. **Run this skill** to generate the figure

## Usage

### Basic Usage

```bash
# Set Gemini API Key (required for image generation)
export GEMINI_API_KEY="your-gemini-api-key"

# Generate figure from MODULE LIST
python scripts/run.py -m module_list.txt
```

### With Custom Output Path

```bash
python scripts/run.py -m module_list.txt -o my_figure.png
```

### Specify Image Model

```bash
python scripts/run.py -m module_list.txt --image-model gemini-2.0-flash-exp-image-generation
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | - | Google Gemini API key for image generation |
| `IMAGE_MODEL` | ❌ No | `gemini-2.0-flash-exp-image-generation` | Image generation model |
| `OUTPUT_DIR` | ❌ No | `outputs` | Output directory for generated figures |

## MODULE LIST Format

The MODULE LIST should be a structured text description containing:

1. **Figure Goal and Type**: What the figure illustrates
2. **Main Subjects/Inputs**: Key elements to visualize
3. **Processes/Methods/Stages**: Workflow steps
4. **Relationships and Flow**: How elements connect
5. **Outputs/Readouts/Results**: Expected visualizations
6. **Layout and Visual Style**: Design specifications
7. **Text Labels and Annotations**: Labels and descriptions
8. **Final Prompt**: Complete generation prompt

Example MODULE LIST structure:
```
MODULE LIST:

1. Figure Goal and Type:
   - Compare traditional vs proposed methods
   - Conceptual workflow diagram

2. Main Subjects / Inputs:
   - Input A: Description...
   - Input B: Description...

[... sections 3-7 ...]

8. Final Nano Banana Prompt:
   Create a scientific figure showing...
```

## Integration with OpenClaw

### As OpenClaw Skill

```yaml
# In your OpenClaw agent config
skills:
  figforge:
    entry: scripts/run.py
    env:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      OUTPUT_DIR: D:\vibe_coding\figures
```

### Manual Integration

1. Ask OpenClaw Agent to analyze your text:
   ```
   "Please analyze this methodology section and create a detailed 
    MODULE LIST for generating a scientific figure. Include all 8 
    sections: goal, subjects, processes, relationships, outputs, 
    layout, labels, and final prompt."
   ```

2. Save the output to `module_list.txt`

3. Run FigForge:
   ```bash
   python scripts/run.py -m module_list.txt
   ```

## Advantages of This Design

1. **Model Flexibility**: Use OpenClaw's best analysis model (Kimi/GLM/Qwen) for text understanding
2. **Cost Efficiency**: Text analysis can use cheaper/faster models
3. **Specialized Generation**: Only use Gemini for what it does best - image generation
4. **Iterative Refinement**: Can refine MODULE LIST before committing to image generation

## Troubleshooting

**Error: "GEMINI_API_KEY not set"**
- Set your Gemini API key: `export GEMINI_API_KEY="your-key"`

**Error: "Module list file not found"**
- Ensure the MODULE LIST file path is correct
- Generate MODULE LIST first using OpenClaw analysis

**Poor image quality**
- Refine the MODULE LIST with more detailed descriptions
- Ensure section 8 (Final Prompt) is comprehensive

## License

MIT License - See LICENSE for details.

## Credits

- Original FigForge: [hengzzzhou/FigForge](https://github.com/hengzzzhou/FigForge)
- OpenClaw Integration: mathhyphen
