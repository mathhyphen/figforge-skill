#!/usr/bin/env python3
"""
FigForge - Complete Workflow for OpenClaw

Two modes:
1. Direct text analysis + image generation (uses OpenClaw agent)
2. Module list only → image generation (original v2.0.0 behavior)

Usage:
    # Mode 1: Full workflow (text → analysis → figure)
    python run_complete.py -i input.txt -o figure.png
    
    # Mode 2: Image generation only (module list → figure)
    python run_complete.py -m module_list.txt -o figure.png
"""

import os
import sys
from pathlib import Path
from typing import Optional
import click

# Import the image generator from v2.0.0
try:
    from run import FigForgeGenerator
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from run import FigForgeGenerator


def analyze_with_openclaw(text: str) -> str:
    """
    Use OpenClaw's current agent to analyze text and generate MODULE LIST.
    This spawns a sub-agent to do the analysis.
    """
    # For now, we'll create a prompt that the user can run manually
    # In the future, this could use OpenClaw's agent API directly
    
    prompt = f"""Please analyze the following scientific text and create a detailed MODULE LIST for figure generation.

The MODULE LIST should have 8 sections:
1. Figure Goal and Type
2. Main Subjects / Inputs
3. Processes / Methods / Stages
4. Relationships and Flow
5. Outputs / Readouts / Results
6. Layout and Visual Style
7. Text Labels and Annotations
8. Final Nano Banana Prompt (single coherent paragraph)

Scientific Text:
{text}

Please output only the MODULE LIST in the specified format."""

    return prompt


@click.command()
@click.option(
    '--input', '-i',
    type=click.Path(exists=True, path_type=Path),
    help='Input text file (for full workflow: text → analysis → figure)'
)
@click.option(
    '--module-list', '-m',
    type=click.Path(exists=True, path_type=Path),
    help='Module list file (for image generation only)'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    help='Output path for generated figure'
)
@click.option(
    '--image-model',
    type=str,
    default=None,
    help='Image generation model'
)
@click.option(
    '--skip-analysis',
    is_flag=True,
    help='Skip analysis step (input is already a module list)'
)
def main(input: Optional[Path], module_list: Optional[Path], 
         output: Optional[Path], image_model: Optional[str],
         skip_analysis: bool):
    """
    FigForge Complete - Scientific Figure Generation
    
    Mode 1 (Full Workflow): -i input.txt
        Text → OpenClaw Analysis → MODULE LIST → Figure
    
    Mode 2 (Image Only): -m module_list.txt
        MODULE LIST → Figure
    """
    
    # Validate input
    if not input and not module_list:
        click.echo("Error: Please provide either --input (-i) or --module-list (-m)", err=True)
        sys.exit(1)
    
    if input and module_list:
        click.echo("Warning: Both --input and --module-list provided. Using --module-list.")
        input = None
    
    # Mode 2: Use provided module list directly
    if module_list or skip_analysis:
        file_to_use = module_list or input
        click.echo(f"Mode: Image generation from MODULE LIST")
        click.echo(f"Reading: {file_to_use}")
        
        module_content = file_to_use.read_text(encoding="utf-8")
        
    # Mode 1: Need to analyze text first
    else:
        click.echo(f"Mode: Full workflow (Text → Analysis → Figure)")
        click.echo(f"Reading input: {input}")
        
        text_content = input.read_text(encoding="utf-8")
        
        # Check if input looks like a module list already
        if "MODULE LIST" in text_content and "Figure Goal" in text_content:
            click.echo("Input appears to be a MODULE LIST. Using directly.")
            module_content = text_content
        else:
            # Generate analysis prompt
            click.echo("\n" + "="*80)
            click.echo("ANALYSIS STEP")
            click.echo("="*80)
            click.echo("The input text needs to be analyzed first.")
            click.echo("\nPlease run the following in OpenClaw to generate MODULE LIST:\n")
            
            prompt = analyze_with_openclaw(text_content)
            click.echo(prompt)
            
            click.echo("\n" + "="*80)
            click.echo("Save the above output to a file (e.g., module_list.txt)")
            click.echo("Then run: python run_complete.py -m module_list.txt")
            click.echo("="*80 + "\n")
            
            # Save the prompt for convenience
            prompt_file = input.with_suffix('.analysis_prompt.txt')
            prompt_file.write_text(prompt, encoding='utf-8')
            click.echo(f"Analysis prompt saved to: {prompt_file}")
            
            sys.exit(0)
    
    # Generate figure
    click.echo(f"\nGenerating figure...")
    try:
        generator = FigForgeGenerator(image_model=image_model)
        figure_path = generator.generate_figure(module_content, output, str(file_to_use))
        click.echo(f"\n✅ Figure generated: {figure_path}")
    except Exception as e:
        click.echo(f"\n❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
