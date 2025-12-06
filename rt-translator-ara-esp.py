"""
Real-time speech translation using faster-whisper with GPU acceleration.

Features
- Captures microphone audio in real time
- Transcribes a primary language (Spanish or Arabic)
- Produces two translations (e.g., Spanish + English or Arabic + English)
- Lists microphone devices to help pick the correct input
- Shows two GUI windows:
    * English Translation Monitor (last 3–4 English lines, minimal)
    * Realtime AR/ES window (semi-transparent, big centered text for OBS)

Usage examples
--------------
# List microphones
python rt-translator-ara-esp.py --list-devices

# Arabic -> Spanish + English (default model: small)
python rt-translator-ara-esp.py --primary-language ar --model small

# Spanish -> Arabic + English
python rt-translator-ara-esp.py --primary-language es --model small --segment-seconds 4
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
import tkinter as tk
from tkinter import ttk

# ---------- Audio / Model config ----------

SAMPLE_RATE = 16000
DEFAULT_SEGMENT_SECONDS = 5  # seconds per chunk

TRANSLATION_MODELS: Dict[Tuple[str, str], str] = {
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("es", "ar"): "Helsinki-NLP/opus-mt-es-ar",
    ("ar", "es"): "Helsinki-NLP/opus-mt-ar-es",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
}

# ---------- GUI: English monitor window (minimalist) ----------

english_window = None
english_label = None
english_history = []
MAX_HISTORY_LINES = 4  # last N lines to keep


def create_english_window():
    """Create a simple window to monitor the last few English translations."""
    global english_window, english_label, english_history

    english_history = []

    english_window = tk.Tk()
    english_window.title("English Translation Monitor")
    english_window.geometry("800x300")
    english_window.configure(bg="white")

    label = ttk.Label(
        english_window,
        text="Waiting for translations...",
        font=("Segoe UI", 16),
        background="white",
        foreground="black",
        anchor="nw",
        justify="left",
        wraplength=760,
    )
    label.pack(fill="both", expand=True, padx=10, pady=10)

    english_window.label = label
    global english_label
    english_label = label


def update_english_window(new_line: str):
    """Append a new English line to the history and refresh the window."""
    global english_history
    if not new_line:
        return

    english_history.append(new_line.strip())
    if len(english_history) > MAX_HISTORY_LINES:
        english_history = english_history[-MAX_HISTORY_LINES:]

    text_to_show = "\n\n".join(english_history)
    if english_label is not None:
        english_label.config(text=text_to_show)
        english_window.update_idletasks()
        english_window.update()


# ---------- GUI: Realtime AR/ES window (semi-transparent, OBS-friendly) ----------

realtime_window = None
realtime_label = None


def create_realtime_window():
    """Create semi-transparent window showing the latest AR + ES line."""
    global realtime_window, realtime_label

    realtime_window = tk.Toplevel()
    realtime_window.title("Realtime AR/ES")
    realtime_window.geometry("900x200")
    realtime_window.configure(bg="#1a1a1a")  # dark gray
    realtime_window.attributes("-alpha", 0.78)  # ~80% opacity

    label = tk.Label(
        realtime_window,
        text="Waiting...",
        font=("Segoe UI", 30, "bold"),
        bg="#1a1a1a",
        fg="white",
        anchor="center",
        justify="center",
        wraplength=850,
    )
    label.pack(fill="both", expand=True, padx=20, pady=20)

    realtime_window.label = label
    realtime_label = label


def update_realtime_window(ar_text: str, es_text: str):
    """Show the latest AR + ES text in a semi-transparent centered window."""
    global realtime_label
    if realtime_label is None:
        return

    display = f"AR: {ar_text}\nES: {es_text}"
    realtime_label.config(text=display)
    realtime_window.update_idletasks()
    realtime_window.update()


# ---------- Utility functions ----------


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
        default="small",  # default to small for faster first tests
        help='faster-whisper model size (e.g., "small", "medium", "large-v3")',
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
        "--overlap-seconds",
        type=int,
        default=1,
        help="Seconds of overlap between chunks for smoother context",
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
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index to use (0 = first GPU).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available microphone devices and exit",
    )
    parser.add_argument(
        "--condition-on-prev",
        action="store_true",
        help="Let Whisper condition on previous text for better coherence",
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


def transcribe_chunk(
    model: WhisperModel,
    audio: np.ndarray,
    language: str,
    condition_on_previous_text: bool = True,
    initial_prompt: str | None = None,
) -> str:
    segments, _ = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        task="transcribe",
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text


# ---------- Main streaming loop ----------


def stream_loop(args: argparse.Namespace):
    primary = args.primary_language
    targets = args.targets or (["es", "en"] if primary == "ar" else ["ar", "en"])
    translators = init_translators(primary, targets)

    print(
        f"Primary language: {primary} | Targets: {', '.join(targets)} | "
        f"Model: {args.model} | Segment: {args.segment_seconds}s"
    )
    print("Press Ctrl+C to stop.\n")

    # Load Whisper model on GPU
    model = WhisperModel(
        args.model,
        device="cuda",
        device_index=args.device_index,
        compute_type="float16",
    )
    # model = WhisperModel(
    #     args.model,
    #     device="cpu",
    #     compute_type="int8",  # más rápido en CPU
    # )


    # Create GUI windows
    create_english_window()
    create_realtime_window()

    audio_queue: queue.Queue[bytes] = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        # RawInputStream gives a cffi buffer; convert to bytes
        audio_queue.put(bytes(indata))

    buffer_samples: list[np.ndarray] = []
    accumulated_samples = 0
    chunk_samples = args.segment_seconds * SAMPLE_RATE
    overlap_samples = max(0, args.overlap_seconds * SAMPLE_RATE)
    hop_samples = max(1, chunk_samples - overlap_samples)
    transcript_history: list[str] = []

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
            chunk_array = np.frombuffer(data, dtype=np.float32)
            buffer_samples.append(chunk_array)
            accumulated_samples += len(chunk_array)

            # Process as long as we have enough audio for a window
            while accumulated_samples >= chunk_samples:
                merged = np.concatenate(buffer_samples)
                audio_chunk = merged[:chunk_samples]

                # Keep overlap by trimming only the hop size
                remaining = merged[hop_samples:]
                buffer_samples = [remaining]
                accumulated_samples = len(remaining)

                start_time = time.time()
                prev_prompt = " ".join(transcript_history[-3:]) if transcript_history else None
                transcript = transcribe_chunk(
                    model,
                    audio_chunk,
                    primary,
                    condition_on_previous_text=args.condition_on_prev,
                    initial_prompt=prev_prompt,
                )
                if not transcript:
                    continue

                transcript_history.append(transcript)

                translations = {}
                for target in targets:
                    translations[target] = translate_text(
                        transcript, primary, target, translators
                    )

                elapsed = time.time() - start_time

                # ----- Console output -----
                print("\n---- New Window ----")
                print(
                    f"Original ({primary}): {transcript} "
                    f"[window {args.segment_seconds}s, overlap {args.overlap_seconds}s]"
                )
                for target, text in translations.items():
                    print(f"Translation -> {target}: {text}")
                print(f"Processing time: {elapsed:.2f}s\n")

                # ----- Update English monitor window -----
                english_text = translations.get("en")
                if english_text:
                    update_english_window(english_text)

                # ----- Update realtime AR/ES window -----
                if primary == "ar":
                    ar_text = transcript
                    es_text = translations.get("es", "")
                else:
                    es_text = transcript
                    ar_text = translations.get("ar", "")

                update_realtime_window(ar_text or "", es_text or "")


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
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()
