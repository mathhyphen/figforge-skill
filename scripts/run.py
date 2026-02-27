#!/usr/bin/env python3
"""
FigForge - Scientific Figure Generator with Style Transfer

Features:
1. Generate figures from MODULE LIST
2. Style transfer from reference images
3. Multiple generation modes

Usage:
    # Basic generation
    python run.py -m module_list.txt
    
    # With reference image for style transfer
    python run.py -m module_list.txt -r reference.png
    
    # With style blending (conservative mode)
    python run.py -m module_list.txt -r reference.png --blend
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import click
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
import colorsys

# Google Gemini imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class StyleExtractor:
    """Extract style features from reference images"""
    
    def __init__(self, image_path: Path):
        self.image = Image.open(image_path).convert('RGB')
        self.width, self.height = self.image.size
    
    def extract_color_palette(self, n_colors: int = 5) -> List[str]:
        """Extract dominant colors from image"""
        # Resize for faster processing
        small = self.image.resize((150, 150))
        
        # Get colors
        pixels = list(small.getdata())
        
        # Simple color quantization
        color_counts = {}
        for pixel in pixels:
            # Round to reduce color space
            rounded = tuple((c // 10) * 10 for c in pixel)
            color_counts[rounded] = color_counts.get(rounded, 0) + 1
        
        # Get top colors
        top_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:n_colors]
        
        # Convert to hex
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(*color) for color, _ in top_colors]
        return hex_colors
    
    def analyze_brightness_contrast(self) -> Tuple[float, float]:
        """Analyze overall brightness and contrast"""
        pixels = list(self.image.getdata())
        
        # Convert to grayscale
        grays = [0.299*r + 0.587*g + 0.114*b for r, g, b in pixels]
        
        brightness = sum(grays) / len(grays) / 255.0
        contrast = max(grays) - min(grays) / 255.0
        
        return brightness, contrast
    
    def detect_style_characteristics(self) -> dict:
        """Detect overall style characteristics"""
        brightness, contrast = self.analyze_brightness_contrast()
        colors = self.extract_color_palette()
        
        # Determine if dark or light theme
        theme = "dark" if brightness < 0.5 else "light"
        
        # Determine color richness
        n_colors = len(set(colors))
        richness = "rich" if n_colors > 3 else "minimal"
        
        return {
            "theme": theme,
            "brightness": f"{brightness:.2f}",
            "contrast": f"{contrast:.2f}",
            "richness": richness,
            "color_palette": colors,
            "aspect_ratio": f"{self.width}:{self.height}"
        }


class CostEstimator:
    """Estimate API costs for image generation"""
    
    # Pricing data from Google AI (per 1M tokens)
    PRICING = {
        "gemini-3.1-flash-image-preview": {
            "input_per_1m": 0.25,      # $0.25 per 1M input tokens
            "output_per_1m": 60.00,    # $60 per 1M output tokens (images)
            "tokens_per_1k_image": 1120,  # 1K (1024x1024) image = 1120 tokens
            "tokens_per_2k_image": 1680,  # 2K image = 1680 tokens
            "tokens_per_4k_image": 2520,  # 4K image = 2520 tokens
        },
        "gemini-3-pro-image-preview": {
            "input_per_1m": 2.00,      # $2.00 per 1M input tokens
            "output_per_1m": 120.00,   # $120 per 1M output tokens (images)
            "tokens_per_1k_image": 1120,  # 1K/2K image = 1120 tokens
            "tokens_per_4k_image": 2000,  # 4K image = 2000 tokens
        },
        "gemini-2.0-flash-exp-image-generation": {
            "input_per_1m": 0.10,
            "output_per_1m": 30.00,
            "tokens_per_image": 1290,  # ~$0.039 per image
        },
    }
    
    DEFAULT_RESOLUTION = "1K"  # Default estimate resolution
    
    @classmethod
    def estimate_cost(cls, model: str, resolution: str = "1K", num_images: int = 1) -> dict:
        """
        Estimate cost for image generation
        
        Args:
            model: Model name
            resolution: Image resolution (512, 1K, 2K, 4K)
            num_images: Number of images to generate
        
        Returns:
            dict with cost breakdown
        """
        pricing = cls.PRICING.get(model, cls.PRICING["gemini-3.1-flash-image-preview"])
        
        # Get tokens per image based on resolution
        if resolution in ["512", "0.5K"]:
            tokens_per_image = pricing.get("tokens_per_512_image", 747)
        elif resolution in ["1K", "1024", "1k"]:
            tokens_per_image = pricing.get("tokens_per_1k_image", 1120)
        elif resolution in ["2K", "2048", "2k"]:
            tokens_per_image = pricing.get("tokens_per_2k_image", 1680)
        elif resolution in ["4K", "4096", "4k"]:
            tokens_per_image = pricing.get("tokens_per_4k_image", 2520)
        else:
            tokens_per_image = pricing.get("tokens_per_image", 1120)
        
        # Calculate costs (assume input is negligible for image generation)
        output_cost_per_image = (tokens_per_image / 1_000_000) * pricing["output_per_1m"]
        total_cost = output_cost_per_image * num_images
        
        return {
            "model": model,
            "resolution": resolution,
            "num_images": num_images,
            "tokens_per_image": tokens_per_image,
            "cost_per_image": output_cost_per_image,
            "total_cost": total_cost,
            "currency": "USD"
        }
    
    @classmethod
    def format_cost(cls, cost_info: dict) -> str:
        """Format cost information for display"""
        lines = [
            f"  Model: {cost_info['model']}",
            f"  Resolution: {cost_info['resolution']}",
            f"  Estimated cost per image: ${cost_info['cost_per_image']:.4f}",
            f"  Total estimated cost: ${cost_info['total_cost']:.4f}",
        ]
        return "\n".join(lines)


class FigForgeGenerator:
    """Enhanced figure generator with style transfer support"""
    
    def __init__(self, image_model: Optional[str] = None):
        load_dotenv()
        
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package required. Install: pip install google-genai")
        
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.image_model = image_model or os.getenv(
            "IMAGE_MODEL", 
            "gemini-3.1-flash-image-preview"
        )
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Load templates
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.base_template = self._load_template("step2_figure_generation.txt")
        
        # Initialize cost estimator
        self.cost_estimator = CostEstimator()
    
    def estimate_generation_cost(self, resolution: str = "1K", num_images: int = 1) -> dict:
        """Estimate cost for current model"""
        return self.cost_estimator.estimate_cost(self.image_model, resolution, num_images)
    
    def _load_template(self, filename: str) -> str:
        """Load prompt template"""
        template_path = self.prompts_dir / filename
        if not template_path.exists():
            # Fallback to default prompt
            return """You are an expert scientific illustrator specializing in publication-quality figures for top-tier journals.

Create a professional scientific figure based on:
{module_list}

Guidelines:
1. Use clean, professional vector-style graphics
2. Follow layout and color specifications
3. Include all technical components and annotations
4. Ensure clear typography and hierarchy
5. Maintain professional academic aesthetic

Generate the figure now."""
        return template_path.read_text(encoding="utf-8")
    
    def _build_strict_style_prompt(self, module_list: str, style_features: dict) -> str:
        """Build prompt with strict style replacement"""
        colors_str = ", ".join(style_features['color_palette'][:5])
        
        return f"""You are an expert scientific illustrator.

Create a scientific figure matching the EXACT style of the reference image provided.

STYLE SPECIFICATIONS (follow precisely):
- Color Palette: Use these exact colors: {colors_str}
- Theme: {style_features['theme'].upper()} background theme
- Visual Density: {style_features['richness']} color scheme
- Aspect Ratio: {style_features['aspect_ratio']}
- Overall Aesthetic: Match the reference image's visual style exactly

CONTENT TO ILLUSTRATE:
{module_list}

CRITICAL REQUIREMENTS:
1. Match the reference image's color palette EXACTLY
2. Use similar visual weight and element spacing
3. Maintain the reference's overall aesthetic and mood
4. Ensure ALL text remains clearly readable
5. Keep scientific notation accurate and clear

Generate the figure now."""
    
    def _build_blend_style_prompt(self, module_list: str, style_features: dict) -> str:
        """Build prompt with blended style (conservative)"""
        colors_str = ", ".join(style_features['color_palette'][:3])
        
        return f"""You are an expert scientific illustrator for Nature, Cell, and Science journals.

Create a publication-quality figure with the following specifications:

BASE ACADEMIC STYLE (maintain these standards):
- Clean, professional layout suitable for high-impact journals
- Clear hierarchical typography with generous white space
- Professional academic aesthetic
- Suitable for both color and black-and-white printing

STYLE ELEMENTS TO ADOPT from reference image:
- Primary Color Palette: Incorporate these colors: {colors_str}
- Overall Theme: {style_features['theme']} theme influence
- Visual Mood: Match the reference's aesthetic feel

CONTENT TO ILLUSTRATE:
{module_list}

BALANCE INSTRUCTIONS:
- Apply the reference color palette while maintaining academic clarity
- Use the reference's visual style but ensure generous white space
- Match aesthetic mood but keep layout clean and professional
- Ensure text remains highly legible (minimum 8pt equivalent)

Generate the figure now."""
    
    def generate_figure(
        self,
        module_list: str,
        output_path: Optional[Path] = None,
        input_filename: Optional[str] = None,
        reference_image: Optional[Path] = None,
        blend_mode: bool = False,
        show_cost: bool = True
    ) -> Path:
        """
        Generate figure with optional style transfer
        
        Args:
            module_list: The MODULE LIST specification
            output_path: Custom output path
            input_filename: Original input filename
            reference_image: Path to reference image for style transfer
            blend_mode: If True, blend reference style with academic standards
            show_cost: If True, display cost estimate
        """
        # Show cost estimate
        if show_cost:
            print("\n" + "-"*60)
            print("💰 COST ESTIMATE")
            print("-"*60)
            cost_info = self.estimate_generation_cost(resolution="1K", num_images=1)
            print(self.cost_estimator.format_cost(cost_info))
            print("-"*60 + "\n")
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if input_filename:
                base_name = Path(input_filename).stem
                output_path = self.output_dir / f"{base_name}_{timestamp}.png"
            else:
                output_path = self.output_dir / f"scientific_figure_{timestamp}.png"
        
        # Build prompt based on mode
        if reference_image:
            print(f"Extracting style from: {reference_image}")
            extractor = StyleExtractor(reference_image)
            style_features = extractor.detect_style_characteristics()
            
            print(f"  Detected theme: {style_features['theme']}")
            print(f"  Color palette: {', '.join(style_features['color_palette'][:3])}")
            
            if blend_mode:
                print("  Mode: Blended (academic standards + reference style)")
                prompt = self._build_blend_style_prompt(module_list, style_features)
            else:
                print("  Mode: Strict (match reference style exactly)")
                prompt = self._build_strict_style_prompt(module_list, style_features)
            
            # Load reference image for multimodal input
            ref_img = Image.open(reference_image)
            
            # Generate with multimodal input
            print(f"\nGenerating figure with style transfer...")
            return self._generate_with_image(prompt, ref_img, output_path)
        
        else:
            # Standard generation without style transfer
            print(f"Generating figure using standard academic style...")
            prompt = self.base_template.format(module_list=module_list)
            return self._generate_text_only(prompt, output_path)
    
    def _generate_with_image(
        self,
        prompt: str,
        reference_image: Image.Image,
        output_path: Path
    ) -> Path:
        """Generate with multimodal input (text + image)"""
        try:
            chat = self.gemini_client.chats.create(
                model=self.image_model,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    temperature=0.7
                )
            )
            
            # Send multimodal message
            response = chat.send_message([prompt, reference_image])
            
            return self._process_response(response, output_path)
            
        except Exception as e:
            print(f"Error in multimodal generation: {e}")
            raise
    
    def _generate_text_only(self, prompt: str, output_path: Path) -> Path:
        """Generate with text-only input"""
        try:
            chat = self.gemini_client.chats.create(
                model=self.image_model,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE'],
                    temperature=0.7
                )
            )
            
            response = chat.send_message(prompt)
            
            return self._process_response(response, output_path)
            
        except Exception as e:
            print(f"Error in text-only generation: {e}")
            raise
    
    def _process_response(self, response, output_path: Path) -> Path:
        """Process generation response and save image"""
        image_saved = False
        
        for part in response.parts:
            if part.text is not None:
                print(f"  Model: {part.text[:150]}...")
            elif hasattr(part, 'as_image') and (image := part.as_image()):
                image.save(str(output_path))
                print(f"\nFigure saved: {output_path}")
                image_saved = True
                break
        
        if not image_saved:
            raise ValueError("No image generated in response")
        
        return output_path


@click.command()
@click.option(
    '--module-list', '-m',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to MODULE LIST file'
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
    help='Blend reference style with academic standards (conservative mode)'
)
@click.option(
    '--image-model',
    type=str,
    default=None,
    help='Image generation model'
)
def main(
    module_list: Path,
    output: Optional[Path],
    reference: Optional[Path],
    blend: bool,
    image_model: Optional[str]
):
    """
    FigForge - Generate scientific figures with style transfer support
    
    Examples:
        # Basic usage
        python run.py -m module_list.txt
        
        # With style transfer (strict mode - match reference exactly)
        python run.py -m module_list.txt -r reference.png
        
        # With style blending (conservative - academic + reference style)
        python run.py -m module_list.txt -r reference.png --blend
    """
    print("="*60)
    print("FigForge - Scientific Figure Generator")
    print("="*60)
    
    # Read MODULE LIST
    print(f"\nReading: {module_list}")
    module_content = module_list.read_text(encoding="utf-8")
    
    # Initialize generator
    try:
        generator = FigForgeGenerator(image_model=image_model)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    # Generate
    try:
        figure_path = generator.generate_figure(
            module_content,
            output_path=output,
            input_filename=str(module_list),
            reference_image=reference,
            blend_mode=blend
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Success! Figure generated: {figure_path}")
        
        # Show final cost
        cost_info = generator.estimate_generation_cost(resolution="1K", num_images=1)
        print(f"💰 Actual cost: ~${cost_info['cost_per_image']:.4f}")
        print(f"{'='*60}")
        
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
