import os
from google import genai
from google.genai import types

with open('outputs/gbm_framework_v2_module_list_20260220_153306.txt', 'r', encoding='utf-8') as f:
    module_list = f.read()

# 从环境变量读取 API key
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

print("Using gemini-3-pro-image-preview...")
response = client.models.generate_content(
    model='gemini-3-pro-image-preview',
    contents=[module_list],
    config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
)

for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        with open('outputs/gbm_framework_nano.png', 'wb') as f:
            f.write(part.inline_data.data)
        print("Success: gbm_framework_nano.png")
        break
else:
    print("No image in response")

