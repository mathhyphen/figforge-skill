from google import genai
from google.genai import types
import sys

try:
    with open('outputs/gbm_framework_v2_module_list_20260220_153306.txt', 'r', encoding='utf-8') as f:
        module_list = f.read()

    client = genai.Client(api_key='AIzaSyAgDSH-3h6reUvVJ9cxVn2_FQ_DA3LnSdg')

    print("Attempting to generate with gemini-3-pro-image-preview (Nano Banana Pro)...")
    response = client.models.generate_content(
        model='gemini-3-pro-image-preview',
        contents=[module_list],
        config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            with open('outputs/gbm_framework_nanobanana.png', 'wb') as f:
                f.write(part.inline_data.data)
            print("SUCCESS: gbm_framework_nanobanana.png generated!")
            sys.exit(0)
    
    print("WARNING: No image in response")
    if response.text:
        print(f"Response text: {response.text[:200]}")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(1)
