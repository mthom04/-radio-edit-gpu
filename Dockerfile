# CUDA + cuDNN + PyTorch prebuilt, so both Demucs (needs torch+CUDA) and
# faster-whisper (needs cuDNN via ctranslate2) work without extra setup.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# ffmpeg is required by both normalize_audio() and Demucs itself.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY clean_song.py .
COPY rp_handler.py .

CMD ["python3", "-u", "rp_handler.py"]
