import os
from google import genai
from google.genai import types

# 鐩存帴鐢熸垚鍥剧墖
# 从环境变量读取 API key
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

prompt = """Create a professional scientific research framework diagram for a GBM (glioblastoma) study.

Layout: Three vertical sections connected by flowing arrows from left to right.

LEFT SECTION (Red/Pink theme):
Title: "Current Challenges"
Three stacked boxes:
1. Cross-scale Semantic Gap - MRI brain scan icon disconnected from DNA/molecular icon
2. Heterogeneous Missing Data - Incomplete data matrix with missing cells
3. Black Box AI - Neural network with lock symbol

CENTER SECTION (Multi-color theme):
Title: "Four Research Modules"
Four connected boxes in pipeline flow:
1. Semantic Alignment (Blue) - Medical LLM fusing multimodal data
2. Missing Data Completion (Green) - Rectified Flow generating data
3. Explainable Prediction (Purple) - Cross-modal Transformer with attention
4. Clinical Validation (Orange) - Multi-center testing

RIGHT SECTION (Green theme):
Title: "Research Outcomes"
Three stacked boxes:
1. Precision Prognosis - Brain MRI with survival curve
2. Clinical Decision Support - Doctor using AI interface
3. Paradigm Evolution - Arrow showing progress

BOTTOM: Purple banner with text "From Experience-Driven to Semantic-Aware AI-Powered Oncology"

Style: Clean modern scientific illustration, flat design, distinct colors for each section, professional medical research aesthetic, white background, publication quality for academic paper.
"""

print("Generating AI framework image...")
response = client.models.generate_content(
    model='gemini-2.0-flash-exp-image-generation',
    contents=[prompt],
    config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
)

# 淇濆瓨鍥剧墖
for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        with open('D:/apps/figforge/outputs/gbm_framework_direct.png', 'wb') as f:
            f.write(part.inline_data.data)
        print("鉁?Image saved: gbm_framework_direct.png")
        break
else:
    print("No image generated")
    print(response.text)

