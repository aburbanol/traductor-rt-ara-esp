"""
Real-time speech translation using faster-whisper with GPU acceleration.

Features
- Captures microphone audio in real time
- Transcribes a primary language (Spanish or Arabic)
- Produces two translations (e.g., Spanish + English or Arabic + English)
- Lists microphone devices to help pick the correct input

Usage examples
--------------
# List microphones
python real_time_translation.py --list-devices

# Arabic -> Spanish + English
python real_time_translation.py --primary-language ar --model medium

# Spanish -> Arabic + English with larger model
python real_time_translation.py --primary-language es --model large-v3 --segment-seconds 4
"""
from __future__ import annotations

import argparse
import queue
import sys
import time
from typing import Dict, Tuple

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from transformers import pipeline

SAMPLE_RATE = 16000
DEFAULT_SEGMENT_SECONDS = 5

TRANSLATION_MODELS: Dict[Tuple[str, str], str] = {
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("es", "ar"): "Helsinki-NLP/opus-mt-es-ar",
    ("ar", "es"): "Helsinki-NLP/opus-mt-ar-es",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
}


def list_microphones() -> None:
    """Print available audio devices with indexes."""
    print("Available audio input devices:\n")
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            print(f"[{idx}] {device['name']} (inputs: {device['max_input_channels']})")
    print("\nPass --input-device with the desired index.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time Whisper translation")
    parser.add_argument(
        "--model",
        default="medium",
        help='faster-whisper model size (e.g., "medium", "large-v3")',
    )
    parser.add_argument(
        "--primary-language",
        choices=["es", "ar"],
        default="ar",
        help="Language you are speaking (es=Spanish, ar=Arabic)",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=DEFAULT_SEGMENT_SECONDS,
        help="Seconds of audio per transcription chunk",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Microphone device index (use --list-devices to view)",
    )
    parser.add_argument(
        "--targets",
        nargs=2,
        metavar=("LANG1", "LANG2"),
        default=None,
        help="Two target languages for translation (choices: es, ar, en)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available microphone devices and exit",
    )
    return parser


def load_translation_pipeline(source: str, target: str):
    key = (source, target)
    if key not in TRANSLATION_MODELS:
        raise ValueError(f"Unsupported translation pair: {source}->{target}")

    model_name = TRANSLATION_MODELS[key]
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading translation model {model_name} on {'cuda' if device == 0 else 'cpu'}...")
    return pipeline("translation", model=model_name, device=device)


def init_translators(source: str, targets):
    translators = {}
    for target in targets:
        if target == source:
            continue
        translators[(source, target)] = load_translation_pipeline(source, target)
    return translators


def translate_text(text: str, source: str, target: str, translators) -> str:
    if target == source:
        return text
    translator = translators.get((source, target))
    if translator is None:
        translator = load_translation_pipeline(source, target)
        translators[(source, target)] = translator
    result = translator(text, max_length=400)
    return result[0]["translation_text"].strip()


def transcribe_chunk(model: WhisperModel, audio: np.ndarray, language: str) -> str:
    segments, _ = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
        task="transcribe",
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text


def stream_loop(args: argparse.Namespace):
    primary = args.primary_language
    targets = args.targets or (["es", "en"] if primary == "ar" else ["ar", "en"])
    translators = init_translators(primary, targets)

    print(
        f"Primary language: {primary} | Targets: {', '.join(targets)} | Model: {args.model} | Segment: {args.segment_seconds}s"
    )
    print("Press Ctrl+C to stop.\n")

    model = WhisperModel(
        args.model,
        device="cuda",
        compute_type="float16",
    )

    audio_queue: queue.Queue[bytes] = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # RawInputStream entrega un buffer tipo cffi -> lo convertimos a bytes
        audio_queue.put(bytes(indata))

    buffer = []
    accumulated_samples = 0
    chunk_samples = args.segment_seconds * SAMPLE_RATE

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=0,
        device=args.input_device,
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        print("Listening...")
        while True:
            data = audio_queue.get()
            buffer.append(np.frombuffer(data, dtype=np.float32))
            accumulated_samples += len(buffer[-1])

            if accumulated_samples >= chunk_samples:
                audio_chunk = np.concatenate(buffer)[:chunk_samples]
                buffer = []
                accumulated_samples = 0

                start_time = time.time()
                transcript = transcribe_chunk(model, audio_chunk, primary)
                if not transcript:
                    continue

                translations = {}
                for target in targets:
                    translations[target] = translate_text(
                        transcript, primary, target, translators
                    )

                elapsed = time.time() - start_time
                print("\n---- New Segment ----")
                print(f"Original ({primary}): {transcript}")
                for target, text in translations.items():
                    print(f"Translation -> {target}: {text}")
                print(f"Processing time: {elapsed:.2f}s\n")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_devices:
        list_microphones()
        return

    try:
        stream_loop(args)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
