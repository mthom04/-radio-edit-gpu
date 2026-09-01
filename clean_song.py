#!/usr/bin/env python3
"""
Radio Edit Engine — "Clean My Song" style pipeline
====================================================

Pipeline:
  upload -> validate -> normalize -> separate stems (Demucs)
  -> transcribe vocals w/ word timestamps (faster-whisper)
  -> flag explicit words (level-based word lists + categories)
  -> producer review (optional, --review-only)
  -> build mute regions -> mute vocal (with fades)
  -> recombine with instrumental -> export mp3/wav

Install (on the Python worker box, NOT the PHP host):
    pip install -r requirements.txt
    # Demucs also needs ffmpeg on PATH:
    #   sudo apt-get install ffmpeg

Usage:
    # Full run, radio-safe level, exports clean_song.mp3
    python clean_song.py input.mp3 --level radio --out clean_song.mp3

    # Producer review only: outputs a JSON report of flagged words,
    # does NOT touch the audio. This is what your PHP front end should
    # call first so a human approves the word list before muting.
    python clean_song.py input.mp3 --level radio --review-only --report report.json

    # Apply a producer-edited report (KEEP/REMOVE decisions already made)
    python clean_song.py input.mp3 --apply-report report.json --out clean_song.mp3
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from pydub import AudioSegment

# --------------------------------------------------------------------------
# 1. CONFIG — clean levels & word categories
# --------------------------------------------------------------------------

# Each level maps category -> action ("mute" or "keep").
# Extend CATEGORY_WORDS below with your station's real word lists;
# these are intentionally left as small illustrative placeholders.
CLEAN_LEVELS = {
    "light":  {"profanity_strong": "mute", "profanity_mild": "keep",
               "slur": "mute", "sexual": "keep", "drug": "keep", "violence": "keep"},
    "radio":  {"profanity_strong": "mute", "profanity_mild": "mute",
               "slur": "mute", "sexual": "mute", "drug": "keep", "violence": "keep"},
    "family": {"profanity_strong": "mute", "profanity_mild": "mute",
               "slur": "mute", "sexual": "mute", "drug": "mute", "violence": "mute"},
    "strict": {"profanity_strong": "mute", "profanity_mild": "mute",
               "slur": "mute", "sexual": "mute", "drug": "mute", "violence": "mute"},
}

# Word -> category. Load your real, station-specific list from a file in
# production (see load_word_categories()) instead of hardcoding it here.
CATEGORY_WORDS = {
    # populate via --wordlist wordlist.json, see load_word_categories()
}

# Words that should NEVER be auto-flagged regardless of level, even if they
# appear in the word list below (guards against false positives like "dope"
# meaning music, not drugs). Keep this list short and deliberate.
ALWAYS_KEEP = {"dope"}


def load_word_categories(path: str | None) -> dict:
    """Load a {word: category} JSON file. Falls back to CATEGORY_WORDS."""
    if not path:
        return CATEGORY_WORDS
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# 2. UPLOAD / VALIDATE / NORMALIZE
# --------------------------------------------------------------------------

SUPPORTED_EXTS = {".mp3", ".wav", ".m4a", ".flac"}


def validate_audio(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported format {path.suffix}. Use one of {SUPPORTED_EXTS}")


def normalize_audio(path: Path, work_dir: Path) -> Path:
    """
    Convert to a consistent WAV for processing. Calls ffmpeg directly
    (instead of via pydub's internal call) with -nostdin, so it can never
    hang waiting for interactive input when run from cron/SSH with no
    real terminal attached.
    """
    out = work_dir / "normalized.wav"
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-i", str(path),
        "-ar", "44100", "-ac", "2",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=180)
    return out


# --------------------------------------------------------------------------
# 3. SEPARATE STEMS (Demucs)
# --------------------------------------------------------------------------

def separate_stems(input_wav: Path, work_dir: Path, model: str = "htdemucs") -> tuple[Path, Path]:
    """
    Runs Demucs in two-stem mode: vocals vs everything else.
    Requires: pip install demucs, and ffmpeg on PATH.
    Returns (vocals_path, instrumental_path).
    """
    out_root = work_dir / "separated"
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", model,
        "-o", str(out_root),
        str(input_wav),
    ]
    # Generous timeout: normal separation is ~6-7 min/chunk even under load.
    # 20 min means "genuinely hung", not "just slow" -- worker.py's retry
    # logic catches this like any other transient failure.
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL, timeout=1200)

    stem_dir = out_root / model / input_wav.stem
    vocals = stem_dir / "vocals.wav"
    instrumental = stem_dir / "no_vocals.wav"
    if not vocals.exists() or not instrumental.exists():
        raise RuntimeError("Demucs did not produce expected stem files.")
    return vocals, instrumental


# --------------------------------------------------------------------------
# 3b. CHUNKED SEPARATION (avoids one long-lived Demucs process — each chunk
# is a short subprocess call that starts and exits quickly, so no single
# process stays alive/heavy long enough to hit shared-hosting resource caps)
# --------------------------------------------------------------------------

def split_into_chunks(normalized_wav: Path, work_dir: Path, chunk_seconds: int = 60) -> list[Path]:
    """
    Splits a normalized WAV into fixed-length chunks using ffmpeg's segment
    muxer. Returns the list of chunk file paths, in order.
    """
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pattern = chunks_dir / "chunk_%03d.wav"

    cmd = [
        "ffmpeg", "-y", "-nostdin", "-i", str(normalized_wav),
        "-f", "segment", "-segment_time", str(chunk_seconds),
        "-c", "pcm_s16le",
        str(pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=180)

    return sorted(chunks_dir.glob("chunk_*.wav"))


def separate_chunk(chunk_path: Path, work_dir: Path, model: str = "htdemucs") -> tuple[Path, Path]:
    """
    Same as separate_stems(), but for a single short chunk file. Each call
    is a short-lived subprocess -- that's the point: many short bursts
    instead of one long-lived process.
    """
    return separate_stems(chunk_path, work_dir, model=model)


def stitch_chunks(chunk_paths: list[Path], output_path: Path) -> Path:
    """Concatenates chunk audio files in order into a single output file."""
    combined = AudioSegment.empty()
    for p in chunk_paths:
        combined += AudioSegment.from_file(p)
    combined.export(output_path, format="wav")
    return output_path


def extract_audio_window(source_path: Path, start_sec: float, end_sec: float, out_path: Path) -> Path:
    """
    Cuts a time window out of a longer audio file. Calls ffmpeg directly
    with -nostdin -- same reason as normalize_audio(): pydub's internal
    ffmpeg call can hang indefinitely without it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end_sec - start_sec)
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-ss", str(start_sec), "-t", str(duration),
        "-i", str(source_path),
        str(out_path),
    ]
    subprocess.run(
        cmd, check=True, capture_output=True,
        stdin=subprocess.DEVNULL, timeout=120,
    )
    return out_path


# --------------------------------------------------------------------------
# 4b. TRANSCRIBE VIA OPENAI'S WHISPER API (alternative to local faster-whisper
# -- moves the heaviest remaining compute off this server entirely, at a
# small per-song API cost)
# --------------------------------------------------------------------------

def transcribe_vocals_openai(vocals_path: Path, api_key: str) -> list[dict]:
    """
    Returns a list of {"word": str, "start": float, "end": float}, same
    shape as transcribe_vocals(), but computed by OpenAI's API instead of
    a local model. Requires: pip install openai, and an OPENAI_API_KEY.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    with open(vocals_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    words = []
    for w in getattr(transcript, "words", []) or []:
        words.append({
            "word": w["word"].strip().lower().strip(".,!?\"'") if isinstance(w, dict) else w.word.strip().lower().strip(".,!?\"'"),
            "start": round(w["start"] if isinstance(w, dict) else w.start, 3),
            "end": round(w["end"] if isinstance(w, dict) else w.end, 3),
        })
    return words


# --------------------------------------------------------------------------
# 4. TRANSCRIBE WITH WORD-LEVEL TIMESTAMPS (faster-whisper)
# --------------------------------------------------------------------------

PUNCT_STRIP = ".,!?\"'\u2018\u2019\u201c\u201d-*\u2026~"


def transcribe_vocals(vocals_path: Path, model_size: str = "medium") -> list[dict]:
    """
    Returns a list of {"word": str, "start": float, "end": float}.
    Requires: pip install faster-whisper

    Settings tuned for music vocals (not clean speech):
    - vad_filter=True: skips truly silent stretches instead of letting the
      model invent giant multi-second "words" spanning instrumental gaps.
    - condition_on_previous_text=False: stops the model from getting stuck
      repeating a bad guess over and over (the "-h -h -h" hallucination loop).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, compute_type="auto")
    segments, _info = model.transcribe(
        str(vocals_path),
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    words = []
    for seg in segments:
        for w in seg.words:
            cleaned = w.word.strip().lower().strip(PUNCT_STRIP)
            if not cleaned:
                continue
            words.append({
                "word": cleaned,
                "start": round(w.start, 3),
                "end": round(w.end, 3),
            })
    return words


# --------------------------------------------------------------------------
# 5. FLAG EXPLICIT WORDS
# --------------------------------------------------------------------------

def detect_explicit_words(words: list[dict], level: str, word_categories: dict) -> list[dict]:
    """
    Tags each transcribed word with a category + KEEP/REMOVE decision
    based on the selected clean level. Matches common word variants
    (plurals, -ing/-in' forms) against the word list, not just exact
    matches -- e.g. "niggas" matches an entry for "nigga".
    """
    level_rules = CLEAN_LEVELS[level]
    flagged = []

    def lookup_category(token):
        if token in word_categories:
            return word_categories[token]
        for suffix in ("ing", "in'", "es", "s", "n", "'"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                stripped = token[: -len(suffix)]
                if stripped in word_categories:
                    return word_categories[stripped]
        return None

    for w in words:
        token = w["word"]
        if token in ALWAYS_KEEP:
            continue

        category = lookup_category(token)
        if category is None:
            continue

        action = level_rules.get(category, "keep")
        flagged.append({
            **w,
            "category": category,
            "decision": "REMOVE" if action == "mute" else "KEEP",
        })

    return flagged


# --------------------------------------------------------------------------
# 6. BUILD MUTE REGIONS
# --------------------------------------------------------------------------

def create_mute_regions(flagged_words: list[dict], pad_sec: float = 0.03, min_duration_sec: float = 0.25) -> list[tuple[float, float]]:
    """
    Only words with decision == REMOVE become mute regions, padded slightly.
    Enforces a minimum region length -- transcription timestamps sometimes
    collapse to zero duration (start == end) for a word, which would
    otherwise produce a mute window too narrow to actually cover the word.
    """
    regions = []
    for w in flagged_words:
        if w["decision"] != "REMOVE":
            continue
        start = max(0.0, w["start"] - pad_sec)
        end = w["end"] + pad_sec
        if end - start < min_duration_sec:
            end = start + min_duration_sec
        regions.append((start, end))
    # merge overlapping/adjacent regions
    regions.sort()
    merged = []
    for start, end in regions:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


# --------------------------------------------------------------------------
# 7. MUTE VOCAL REGIONS (with short crossfades so it doesn't click)
# --------------------------------------------------------------------------

def mute_vocal_regions(vocals_path: Path, regions: list[tuple[float, float]], fade_ms: int = 30) -> AudioSegment:
    audio = AudioSegment.from_file(vocals_path)

    for start_sec, end_sec in regions:
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        end_ms = min(end_ms, len(audio))
        if start_ms >= end_ms:
            continue

        before = audio[:start_ms]
        muted_chunk = AudioSegment.silent(duration=end_ms - start_ms)
        after = audio[end_ms:]

        # crossfade the boundaries so silence doesn't click in/out
        before = before.fade_out(min(fade_ms, len(before)))
        after = after.fade_in(min(fade_ms, len(after)))

        audio = before + muted_chunk + after

    return audio


# --------------------------------------------------------------------------
# 7b. BLEED-THROUGH DETECTION + SECONDARY DUCK
#
# Muting the isolated vocal track assumes the word ONLY lives there. On
# songs where AI separation does not fully isolate loud/aggressive vocals,
# some of the word can bleed into the instrumental track too, so muting
# the vocal alone does not make it inaudible. This checks for that
# specific situation per-word and, ONLY when bleed is detected, briefly
# ducks the instrumental at that exact spot too -- everywhere else stays
# untouched, matching the standard vocal-only-mute approach.
# --------------------------------------------------------------------------

def _segment_dbfs(audio, start_sec, end_sec):
    start_ms = max(0, int(start_sec * 1000))
    end_ms = min(len(audio), int(end_sec * 1000))
    if start_ms >= end_ms:
        return float("-inf")
    return audio[start_ms:end_ms].dBFS


def detect_bleed_regions(original_vocals_path, instrumental_path, regions, threshold_db=8.0):
    """
    Compares original vocal loudness against instrumental loudness at each
    mute region. If the instrumental is within threshold_db of the original
    vocal, that indicates bleed-through. Returns only regions needing a duck.
    """
    original_vocals = AudioSegment.from_file(original_vocals_path)
    instrumental = AudioSegment.from_file(instrumental_path)

    bleeding = []
    for start_sec, end_sec in regions:
        vocal_level = _segment_dbfs(original_vocals, start_sec, end_sec)
        instrumental_level = _segment_dbfs(instrumental, start_sec, end_sec)
        if vocal_level == float("-inf"):
            continue
        if instrumental_level >= vocal_level - threshold_db:
            bleeding.append((start_sec, end_sec))

    return bleeding


def duck_instrumental_regions(instrumental_path, regions, duck_db=-18.0, fade_ms=60):
    """
    Reduces (not fully mutes) the instrumental volume during given regions --
    a brief targeted dip, not a full cut. Only called on regions
    detect_bleed_regions() flagged as actually needed.
    """
    audio = AudioSegment.from_file(instrumental_path)

    for start_sec, end_sec in regions:
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        end_ms = min(end_ms, len(audio))
        if start_ms >= end_ms:
            continue

        before = audio[:start_ms]
        ducked_chunk = audio[start_ms:end_ms] + duck_db
        after = audio[end_ms:]

        before = before.fade_out(min(fade_ms, len(before)))
        ducked_chunk = ducked_chunk.fade_in(min(fade_ms, len(ducked_chunk))).fade_out(min(fade_ms, len(ducked_chunk)))
        after = after.fade_in(min(fade_ms, len(after)))

        audio = before + ducked_chunk + after

    return audio


# --------------------------------------------------------------------------
# 8. RECOMBINE + EXPORT
# --------------------------------------------------------------------------

def recombine_stems(clean_vocals: AudioSegment, instrumental) -> AudioSegment:
    """
    Combines the (muted) vocal track with the instrumental. instrumental
    can be either a Path (loaded fresh, untouched) or an already-processed
    AudioSegment (e.g. one with a secondary duck applied).
    """
    if isinstance(instrumental, (str, Path)):
        instrumental = AudioSegment.from_file(instrumental)
    length = max(len(clean_vocals), len(instrumental))
    clean_vocals = clean_vocals + AudioSegment.silent(duration=max(0, length - len(clean_vocals)))
    instrumental = instrumental + AudioSegment.silent(duration=max(0, length - len(instrumental)))
    return clean_vocals.overlay(instrumental)


def export_song(mix: AudioSegment, out_path: Path, fmt: str = "mp3", bitrate: str = "320k") -> None:
    if fmt == "mp3":
        mix.export(out_path, format="mp3", bitrate=bitrate)
    else:
        mix.export(out_path, format="wav")


# --------------------------------------------------------------------------
# ORCHESTRATION
# --------------------------------------------------------------------------

def run_pipeline(input_path: Path, level: str, out_path: Path, work_dir: Path,
                  word_categories: dict, whisper_model: str, demucs_model: str,
                  review_only: bool, report_path: Path | None,
                  apply_report_path: Path | None) -> None:

    work_dir.mkdir(parents=True, exist_ok=True)

    if apply_report_path:
        # Producer already reviewed a report — skip straight to muting.
        report = json.loads(apply_report_path.read_text())
        vocals_path = Path(report["vocals_path"])
        instrumental_path = Path(report["instrumental_path"])
        flagged_words = report["words"]
    else:
        validate_audio(input_path)
        normalized = normalize_audio(input_path, work_dir)

        print("Separating vocals / instrumental...", file=sys.stderr)
        vocals_path, instrumental_path = separate_stems(normalized, work_dir, model=demucs_model)

        print("Transcribing vocals...", file=sys.stderr)
        words = transcribe_vocals(vocals_path, model_size=whisper_model)

        print("Scanning for explicit content...", file=sys.stderr)
        flagged_words = detect_explicit_words(words, level, word_categories)

        if review_only:
            report = {
                "input": str(input_path),
                "level": level,
                "vocals_path": str(vocals_path),
                "instrumental_path": str(instrumental_path),
                "words": flagged_words,
            }
            dest = report_path or Path("report.json")
            dest.write_text(json.dumps(report, indent=2))
            print(f"Review report written to {dest} ({len(flagged_words)} flagged words). "
                  f"No audio was modified.", file=sys.stderr)
            return

    regions = create_mute_regions(flagged_words)
    print(f"Muting {len(regions)} region(s)...", file=sys.stderr)
    clean_vocals = mute_vocal_regions(vocals_path, regions)

    print("Recombining with instrumental...", file=sys.stderr)
    mix = recombine_stems(clean_vocals, instrumental_path)

    fmt = out_path.suffix.lower().lstrip(".") or "mp3"
    export_song(mix, out_path, fmt=fmt)
    print(f"Done -> {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Radio Edit Engine — clean explicit songs")
    ap.add_argument("input", nargs="?", type=Path, help="Input song file")
    ap.add_argument("--level", choices=CLEAN_LEVELS.keys(), default="radio")
    ap.add_argument("--out", type=Path, default=Path("clean_song.mp3"))
    ap.add_argument("--work-dir", type=Path, default=Path("./_work"))
    ap.add_argument("--wordlist", type=str, default=None, help="Path to word->category JSON")
    ap.add_argument("--whisper-model", default="medium",
                     help="tiny/base/small/medium/large-v3 (bigger = more accurate, slower)")
    ap.add_argument("--demucs-model", default="htdemucs")
    ap.add_argument("--review-only", action="store_true",
                     help="Only produce a JSON report of flagged words; don't touch audio")
    ap.add_argument("--report", type=Path, default=None, help="Where to write --review-only report")
    ap.add_argument("--apply-report", type=Path, default=None,
                     help="Skip transcription; apply a producer-approved report JSON")
    args = ap.parse_args()

    if not args.input and not args.apply_report:
        ap.error("input song is required unless using --apply-report")

    word_categories = load_word_categories(args.wordlist)

    run_pipeline(
        input_path=args.input,
        level=args.level,
        out_path=args.out,
        work_dir=args.work_dir,
        word_categories=word_categories,
        whisper_model=args.whisper_model,
        demucs_model=args.demucs_model,
        review_only=args.review_only,
        report_path=args.report,
        apply_report_path=args.apply_report,
    )


if __name__ == "__main__":
    main()
