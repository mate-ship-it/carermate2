FROM python:3.11-slim

# Install ffmpeg system package
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set up a virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy & install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload the Somali ASR model into the HF cache
RUN python - <<EOF
from transformers import pipeline
pipeline("automatic-speech-recognition", model="Mustafaa4a/ASR-Somali")
EOF

# Copy your bot code
COPY . /app

# Launch
CMD ["python", "bot.py"]
