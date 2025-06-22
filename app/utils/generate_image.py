

import requests
from PIL import Image
from io import BytesIO
import os
import base64

STABILITY_API_KEY = ""

def generate_image_from_prompt_old(prompt: str):
    ENGINE_ID = "stable-diffusion-xl-1024-v1-0"
    API_URL = f"https://api.stability.ai/v1/generation/{ENGINE_ID}/text-to-image"

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json"
    }

    json_data = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30
    }

    response = requests.post(API_URL, headers=headers, json=json_data)

    if response.status_code != 200:
        return f"[Error: {response.status_code}] {response.text}"

    data = response.json()
    base64_img = data["artifacts"][0]["base64"]
    image_bytes = base64.b64decode(base64_img)
    image = Image.open(BytesIO(image_bytes))

    return image

# Dummy function for UI testing: always returns the last output.png
def dummy_generate_image_from_prompt(prompt: str):
    from PIL import Image
    return Image.open("output.png")


# Gemini-based image generation function

# Gemini-based image generation function with same signature and output as generate_image_from_prompt
def generate_image_from_prompt(prompt: str):
    """
    Generate an image using Google Gemini models.
    Args:
        prompt (str): The text prompt for image generation.
    Returns:
        Image.Image: PIL Image object if successful, else error string.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "[Error] google-generativeai package not installed. Please install it with: pip install google-generativeai"

    api_key = ""
    if not api_key:
        return "[Error] Gemini API key not provided. Set GEMINI_API_KEY environment variable."

    try:
        client = genai.Client(api_key=api_key)
        contents = prompt
        response = client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        for part in response.candidates[0].content.parts:
            if getattr(part, 'inline_data', None) is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                return image
        return "[Error] No image data returned from Gemini API."
    except Exception as e:
        return f"[Error: Gemini API] {str(e)}"



