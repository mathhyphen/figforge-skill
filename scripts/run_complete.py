#!/usr/bin/env python3
"""
FigForge - Complete Workflow with Style Transfer

Full pipeline: Text → Analysis → MODULE LIST → Figure (with optional style transfer)

Usage:
    # Full workflow without style
    python run_complete.py -i input.txt
    
    # Full workflow with style transfer
    python run_complete.py -i input.txt -r reference.png
    
    # With style blending (conservative)
    python run_complete.py -i input.txt -r reference.png --blend
"""

import os
import sys
from pathlib import Path
from typing import Optional
import click

# Import from run.py
try:
    from run import FigForgeGenerator
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from run import FigForgeGenerator


def analyze_with_openclaw(text: str) -> str:
    """Generate analysis prompt for OpenClaw agent"""
    
    prompt = f"""Please analyze the following scientific text and create a detailed MODULE LIST for figure generation.

The MODULE LIST should have 8 sections:
1. Figure Goal and Type
2. Main Subjects / Inputs
3. Processes / Methods / Stages
4. Relationships and Flow
5. Outputs / Readouts / Results
6. Layout and Visual Style
7. Text Labels and Annotations
8. Final Prompt (single coherent paragraph)

Scientific Text:
{text}

Output only the MODULE LIST in the specified format."""

    return prompt


@click.command()
@click.option(
    '--input', '-i',
    type=click.Path(exists=True, path_type=Path),
    help='Input text file (for full workflow)'
)
@click.option(
    '--module-list', '-m',
    type=click.Path(exists=True, path_type=Path),
    help='Module list file (skip analysis)'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    help='Output path for figure'
)
@click.option(
    '--reference', '-r',
    type=click.Path(exists=True, path_type=Path),
    help='Reference image for style transfer'
)
@click.option(
    '--blend',
    is_flag=True,
    help='Blend reference style with academic standards'
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
    help='Skip analysis (input is already module list)'
)
def main(
    input: Optional[Path],
    module_list: Optional[Path],
    output: Optional[Path],
    reference: Optional[Path],
    blend: bool,
    image_model: Optional[str],
    skip_analysis: bool
):
    """
    FigForge Complete - Scientific Figure Generation with Style Transfer
    
    Full workflow (recommended):
        python run_complete.py -i input.txt -r reference.png
    
    With style blending (conservative):
        python run_complete.py -i input.txt -r reference.png --blend
    
    Skip analysis (use pre-generated module list):
        python run_complete.py -m module_list.txt -r reference.png
    """
    
    print("="*60)
    print("FigForge Complete - Full Workflow")
    print("="*60)
    
    # Validate input
    if not input and not module_list:
        click.echo("Error: Provide --input (-i) or --module-list (-m)", err=True)
        sys.exit(1)
    
    if input and module_list:
        click.echo("Warning: Both inputs provided, using --module-list")
        input = None
    
    # Determine mode
    if module_list or skip_analysis:
        # Mode: Direct generation from MODULE LIST
        file_to_use = module_list or input
        print(f"\nMode: Direct generation from MODULE LIST")
        print(f"   File: {file_to_use}")
        module_content = file_to_use.read_text(encoding="utf-8")
        
    else:
        # Mode: Full workflow with analysis
        print(f"\nMode: Full workflow (Text -> Analysis -> Figure)")
        print(f"   Input: {input}")
        
        text_content = input.read_text(encoding="utf-8")
        
        # Check if already a module list
        if "MODULE LIST" in text_content and "Figure Goal" in text_content:
            print("   Input appears to be a MODULE LIST, using directly")
            module_content = text_content
        else:
            # Generate analysis prompt
            print("\n" + "="*60)
            print("ANALYSIS STEP REQUIRED")
            print("="*60)
            print("\nThis input needs to be analyzed first.")
            print("Run this prompt in OpenClaw to generate MODULE LIST:\n")
            
            prompt = analyze_with_openclaw(text_content)
            print(prompt)
            
            print("\n" + "="*60)
            print("Instructions:")
            print("   1. Copy the prompt above")
            print("   2. Run in OpenClaw to get MODULE LIST")
            print(f"   3. Save to file, then run:")
            print(f"      python run_complete.py -m your_module_list.txt")
            if reference:
                print(f"      python run_complete.py -m your_module_list.txt -r {reference}")
            print("="*60 + "\n")
            
            # Save prompt for convenience
            prompt_file = input.with_suffix('.analysis_prompt.txt')
            prompt_file.write_text(prompt, encoding='utf-8')
            print(f"Analysis prompt saved: {prompt_file}")
            
            sys.exit(0)
    
    # Show style transfer info
    if reference:
        print(f"\nStyle Transfer Enabled")
        print(f"   Reference: {reference}")
        mode_str = 'Blend (conservative)' if blend else 'Strict (exact match)'
        print(f"   Mode: {mode_str}")
    
    # Generate figure
    print(f"\nGenerating figure...")
    try:
        generator = FigForgeGenerator(image_model=image_model)
        figure_path = generator.generate_figure(
            module_content,
            output_path=output,
            input_filename=str(file_to_use if module_list or skip_analysis else input),
            reference_image=reference,
            blend_mode=blend
        )
        
        print(f"\n{'='*60}")
        print(f"SUCCESS!")
        print(f"   Figure: {figure_path}")
        if reference:
            print(f"   Style: Applied from {reference.name}")
        print(f"{'='*60}")
        
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
