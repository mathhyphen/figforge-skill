#!/usr/bin/env python3
"""
FigForge - AI Scientific Figure Generator
OpenClaw Skill Entry Point

This is the standard entry point for the FigForge skill.
It provides a clean interface to generate publication-quality scientific figures.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path to import ScientificPlotter
SKILL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SKILL_DIR))

try:
    from scientific_plotter import ScientificPlotter, main as cli_main
except ImportError as e:
    print(f"❌ Error importing ScientificPlotter: {e}")
    print(f"   SKILL_DIR: {SKILL_DIR}")
    print(f"   Python path: {sys.path}")
    sys.exit(1)


def main():
    """
    Main entry point for the FigForge skill.
    Delegates to the scientific_plotter CLI.
    """
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("   Please set it before running:")
        print("   export OPENAI_API_KEY='your-api-key'")
        print()
    
    # Ensure output directory exists
    output_dir = os.getenv("OUTPUT_DIR", "outputs")
    output_path = Path(output_dir)
    
    # Support relative paths from current directory
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_dir
    
    # Create output directory if it doesn't exist
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create output directory {output_path}: {e}")
        print(f"   Will attempt to use default location.")
    
    # Delegate to the main CLI
    try:
        return cli_main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
