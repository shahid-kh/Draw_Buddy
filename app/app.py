
# New endpoint for typed prompt
from fastapi import Body
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import vosk
from utils.generate_image import generate_image_from_prompt
import uuid
import json
import os

# Initialize FastAPI app
app = FastAPI()

# Mount static directory to serve frontend
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Redirect root path to the frontend page
@app.get("/")
def root():
    return RedirectResponse(url="/static")

# Allow all CORS (helpful for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Vosk model once at startup
model = vosk.Model("models/vosk-model-small-en-us-0.15")


@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    audio_id = str(uuid.uuid4())
    webm_path = f"{audio_id}.webm"
    wav_path = f"{audio_id}.wav"

    try:
        # Save uploaded .webm file
        with open(webm_path, "wb") as f:
            f.write(await file.read())

        # Convert to .wav using pydub
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav")

        # Read waveform
        waveform, samplerate = sf.read(wav_path)
        if waveform.ndim > 1:
            waveform = waveform[:, 0]  # convert to mono
        waveform = (waveform * 32767).astype(np.int16)

        # Transcribe
        recognizer = vosk.KaldiRecognizer(model, samplerate)
        recognizer.AcceptWaveform(waveform.tobytes())
        result = json.loads(recognizer.Result())
        prompt = result.get("text", "").strip()

        if not prompt:
            return JSONResponse(status_code=400, content={"error": "No speech detected."})

        # Generate image
        image = generate_image_from_prompt(prompt)
        if isinstance(image, str):  # Error string
            return JSONResponse(status_code=500, content={"error": image})

        # Save as output.png (always overwrites)
        image.save("output.png")

        return {
            "text": prompt,
            "image_url": "/output.png"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # Always delete temporary files
        for path in [webm_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)

@app.post("/process-text/")
async def process_text(data: dict = Body(...)):
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return JSONResponse(status_code=400, content={"error": "No prompt provided."})
    try:
        image = generate_image_from_prompt(prompt)
        if isinstance(image, str):  # Error string
            return JSONResponse(status_code=500, content={"error": image})
        image.save("output.png")
        return {
            "text": prompt,
            "image_url": "/output.png"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/output.png")
def serve_image():
    return FileResponse("output.png", media_type="image/png", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })
