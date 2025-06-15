# Use slim Python base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio + vosk
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port that FastAPI will run on
EXPOSE 7860

# Start FastAPI using Uvicorn (instead of python app.py)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

