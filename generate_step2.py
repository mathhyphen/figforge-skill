import os
from google import genai
from google.genai import types

# 读取 MODULE LIST
with open('D:/apps/figforge/outputs/gbm_framework_v2_module_list_20260220_150835.txt', 'r', encoding='utf-8') as f:
    module_list = f.read()

# 读取生成模板
with open('D:/apps/figforge/prompts/step2_figure_generation.txt', 'r', encoding='utf-8') as f:
    template = f.read()

# 构建完整 prompt
full_prompt = template.replace('{module_list}', module_list)

print("Step 2: Generating figure using gemini-3-pro-image-preview...")

client = genai.Client(api_key='AIzaSyAgDSH-3h6reUvVJ9cxVn2_FQ_DA3LnSdg')

response = client.models.generate_content(
    model='gemini-3-pro-image-preview',
    contents=[full_prompt],
    config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
)

# 保存图片
for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        with open('D:/apps/figforge/outputs/gbm_framework_final_ai.png', 'wb') as f:
            f.write(part.inline_data.data)
        print("Figure saved: gbm_framework_final_ai.png")
        break
else:
    print("No image in response")
    print(response.text[:500] if response.text else "Empty response")
