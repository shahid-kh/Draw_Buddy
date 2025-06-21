
import requests
from PIL import Image
from io import BytesIO
import os
import base64

STABILITY_API_KEY = ""

def generate_image_from_prompt(prompt: str):
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
