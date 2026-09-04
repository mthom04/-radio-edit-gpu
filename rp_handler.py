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
        "input_key": "<key of the source audio, already uploaded to S3>",
        "audio_ext": "mp3",            # optional, default "mp3"
        "whisper_model": "medium",     # optional, default "medium"
        "demucs_model": "htdemucs",    # optional, default "htdemucs" (ignored for voice_only)
        "job_type": "song"             # optional, "song" (default) or "voice_only"
    }

Output (job_type == "song"):
    {
        "vocals_key": "<job_id>/vocals.wav",
        "instrumental_key": "<job_id>/instrumental.wav",
        "words": [{"word": str, "start": float, "end": float}, ...]
    }

Output (job_type == "voice_only"):
    {
        "audio_key": "<job_id>/audio.wav",
        "words": [{"word": str, "start": float, "end": float}, ...]
    }

    On failure:
    {"error": "<message>"}

Why S3 on both sides: RunPod caps a job's request/response bodies at
10 MB each. A base64-encoded input file blows past that for anything
longer than a couple minutes (this is what broke on a 20-min interview
upload), and a full-quality vocals+instrumental WAV pair blows past it
on the way back out too. So both directions go through a RunPod Network
Volume (via its S3-compatible API): worker.py uploads the source audio
and hands us a key instead of raw bytes, and we hand back storage keys
for our output instead of raw bytes. worker.py uses the same S3
credentials to fetch results directly, rather than a presigned "guest
link" -- RunPod's S3-compatible storage doesn't reliably honor presigned
URLs.

voice_only mode (interviews / spoken-word content with no music bed):
skips separate_stems entirely and transcribes the normalized input
directly, since there's nothing to isolate vocals from. Saves the
Demucs GPU pass and returns a single audio key instead of a stem pair.

Required environment variables (set on the endpoint, not in this file):
    RUNPOD_S3_ENDPOINT_URL   e.g. https://s3api-us-nc-2.runpod.io
    RUNPOD_S3_REGION         e.g. us-nc-2
    RUNPOD_S3_BUCKET         the network volume ID, e.g. gkxo0dk9dg
    RUNPOD_S3_ACCESS_KEY     from RunPod Settings -> S3 API Keys
    RUNPOD_S3_SECRET_KEY     from RunPod Settings -> S3 API Keys
"""

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


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
        region_name=os.environ["RUNPOD_S3_REGION"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_KEY"],
    )


def _upload(s3, local_path: Path, key: str) -> str:
    bucket = os.environ["RUNPOD_S3_BUCKET"]
    s3.upload_file(str(local_path), bucket, key)
    return key


def _download_audio_input(s3, input_key: str, audio_ext: str, work_dir: Path) -> Path:
    bucket = os.environ["RUNPOD_S3_BUCKET"]
    in_path = work_dir / f"input.{audio_ext}"
    s3.download_file(bucket, input_key, str(in_path))
    return in_path


def handler(job):
    job_input = job.get("input", {}) or {}
    job_id = job.get("id", "unknown_job")

    input_key = job_input.get("input_key")
    if not input_key:
        return {"error": "Missing required input field: input_key"}

    audio_ext = job_input.get("audio_ext", "mp3").lstrip(".")
    whisper_model = job_input.get("whisper_model", "medium")
    demucs_model = job_input.get("demucs_model", "htdemucs")
    job_type = job_input.get("job_type", "song")

    with TemporaryDirectory(dir="/tmp") as tmp:
        work_dir = Path(tmp)
        try:
            s3 = _s3_client()
            input_path = _download_audio_input(s3, input_key, audio_ext, work_dir)
            validate_audio(input_path)

            normalized = normalize_audio(input_path, work_dir)

            if job_type == "voice_only":
                # No music bed to separate from -- transcribe the
                # normalized input directly and skip Demucs entirely.
                words = transcribe_vocals(normalized, model_size=whisper_model)
                audio_key = _upload(s3, normalized, f"{job_id}/audio.wav")
                return {
                    "audio_key": audio_key,
                    "words": words,
                }

            vocals_path, instrumental_path = separate_stems(
                normalized, work_dir, model=demucs_model
            )

            words = transcribe_vocals(vocals_path, model_size=whisper_model)

            vocals_key = _upload(s3, vocals_path, f"{job_id}/vocals.wav")
            instrumental_key = _upload(
                s3, instrumental_path, f"{job_id}/instrumental.wav"
            )

            return {
                "vocals_key": vocals_key,
                "instrumental_key": instrumental_key,
                "words": words,
            }

        except Exception as e:
            return {
                "error": f"{e}",
                "traceback": traceback.format_exc(),
            }


runpod.serverless.start({"handler": handler})
