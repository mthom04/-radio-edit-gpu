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
        "vocals_url": "<short-lived download link, WAV>",
        "instrumental_url": "<short-lived download link, WAV>",
        "words": [{"word": str, "start": float, "end": float}, ...]
    }

    On failure:
    {"error": "<message>"}

Why URLs instead of raw audio: RunPod caps a job's response at 10 MB for
/run. A full-quality vocals+instrumental WAV pair for even a short song
blows past that easily, and RunPod just silently drops the oversized
response (job shows COMPLETED with no output). So instead, both stems get
uploaded to a RunPod Network Volume (accessed here via its S3-compatible
API) and we hand back short-lived presigned download links -- tiny JSON
response, no size limit problem, regardless of song length.

Required environment variables (set on the endpoint, not in this file):
    RUNPOD_S3_ENDPOINT_URL   e.g. https://s3api-us-nc-2.runpod.io
    RUNPOD_S3_REGION         e.g. us-nc-2
    RUNPOD_S3_BUCKET         the network volume ID, e.g. gkxo0dk9dg
    RUNPOD_S3_ACCESS_KEY     from RunPod Settings -> S3 API Keys
    RUNPOD_S3_SECRET_KEY     from RunPod Settings -> S3 API Keys
"""

import base64
import os
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
import runpod

from clean_song import (
    validate_audio,
    normalize_audio,
    separate_stems,
    transcribe_vocals,
)

PRESIGNED_URL_TTL_SECONDS = 3600  # 1 hour -- plenty of time for worker.py to download


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
        region_name=os.environ["RUNPOD_S3_REGION"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_KEY"],
    )


def _upload_and_get_url(s3, local_path: Path, key: str) -> str:
    bucket = os.environ["RUNPOD_S3_BUCKET"]
    s3.upload_file(str(local_path), bucket, key)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=PRESIGNED_URL_TTL_SECONDS,
    )


def _decode_audio_input(audio_b64: str, audio_ext: str, work_dir: Path) -> Path:
    raw = base64.b64decode(audio_b64)
    in_path = work_dir / f"input.{audio_ext}"
    in_path.write_bytes(raw)
    return in_path


def handler(job):
    job_input = job.get("input", {}) or {}
    job_id = job.get("id", "unknown_job")

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

            s3 = _s3_client()
            vocals_url = _upload_and_get_url(s3, vocals_path, f"{job_id}/vocals.wav")
            instrumental_url = _upload_and_get_url(
                s3, instrumental_path, f"{job_id}/instrumental.wav"
            )

            return {
                "vocals_url": vocals_url,
                "instrumental_url": instrumental_url,
                "words": words,
            }

        except Exception as e:
            return {
                "error": f"{e}",
                "traceback": traceback.format_exc(),
            }


runpod.serverless.start({"handler": handler})
