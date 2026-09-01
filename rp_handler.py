#!/usr/bin/env python3
"""
rp_handler.py -- RunPod serverless handler for the Radio Edit Engine GPU worker

Runs INSIDE the RunPod container. Receives one whole song per job (no
chunking -- the GPU container isn't subject to the shared-hosting resource
governor that made chunking necessary in worker.py), runs:

    normalize -> separate_stems (Demucs, GPU) -> transcribe_vocals
    (faster-whisper, GPU)

and returns the vocals stem, the instrumental stem, and the raw word list
(same shape as clean_song.py's transcribe_vocals output) so worker.py can
run detect_explicit_words() and everything downstream exactly as it does
today.

Input (job["input"]):
    {
        "audio_b64": "<base64-encoded audio file, any format ffmpeg reads>",
        "audio_ext": "mp3",            # optional, default "mp3"
        "whisper_model": "medium",     # optional, default "medium"
        "demucs_model": "htdemucs"     # optional, default "htdemucs"
    }

Output:
    {
        "vocals_b64": "<base64 WAV>",
        "instrumental_b64": "<base64 WAV>",
        "words": [{"word": str, "start": float, "end": float}, ...]
    }

    On failure:
    {"error": "<message>"}
"""

import base64
import shutil
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import runpod

from clean_song import (
    validate_audio,
    normalize_audio,
    separate_stems,
    transcribe_vocals,
)


def _decode_audio_input(audio_b64: str, audio_ext: str, work_dir: Path) -> Path:
    raw = base64.b64decode(audio_b64)
    in_path = work_dir / f"input.{audio_ext}"
    in_path.write_bytes(raw)
    return in_path


def _encode_audio_output(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def handler(job):
    job_input = job.get("input", {}) or {}

    audio_b64 = job_input.get("audio_b64")
    if not audio_b64:
        return {"error": "Missing required input field: audio_b64"}

    audio_ext = job_input.get("audio_ext", "mp3").lstrip(".")
    whisper_model = job_input.get("whisper_model", "medium")
    demucs_model = job_input.get("demucs_model", "htdemucs")

    with TemporaryDirectory(dir="/tmp") as tmp:
        work_dir = Path(tmp)
        try:
            input_path = _decode_audio_input(audio_b64, audio_ext, work_dir)
            validate_audio(input_path)

            normalized = normalize_audio(input_path, work_dir)

            vocals_path, instrumental_path = separate_stems(
                normalized, work_dir, model=demucs_model
            )

            words = transcribe_vocals(vocals_path, model_size=whisper_model)

            return {
                "vocals_b64": _encode_audio_output(vocals_path),
                "instrumental_b64": _encode_audio_output(instrumental_path),
                "words": words,
            }

        except Exception as e:
            return {
                "error": f"{e}",
                "traceback": traceback.format_exc(),
            }


runpod.serverless.start({"handler": handler})
