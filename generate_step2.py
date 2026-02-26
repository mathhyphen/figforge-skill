import os
from google import genai
from google.genai import types

# 璇诲彇 MODULE LIST
with open('D:/apps/figforge/outputs/gbm_framework_v2_module_list_20260220_150835.txt', 'r', encoding='utf-8') as f:
    module_list = f.read()

# 璇诲彇鐢熸垚妯℃澘
with open('D:/apps/figforge/prompts/step2_figure_generation.txt', 'r', encoding='utf-8') as f:
    template = f.read()

# 鏋勫缓瀹屾暣 prompt
full_prompt = template.replace('{module_list}', module_list)

print("Step 2: Generating figure using gemini-3-pro-image-preview...")

# 从环境变量读取 API key
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model='gemini-3-pro-image-preview',
    contents=[full_prompt],
    config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
)

# 淇濆瓨鍥剧墖
for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        with open('D:/apps/figforge/outputs/gbm_framework_final_ai.png', 'wb') as f:
            f.write(part.inline_data.data)
        print("Figure saved: gbm_framework_final_ai.png")
        break
else:
    print("No image in response")
    print(response.text[:500] if response.text else "Empty response")

