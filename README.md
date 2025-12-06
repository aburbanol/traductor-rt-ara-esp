# traductor-rt-ara-esp

Real-time Arabic/Spanish speech translator using faster-whisper (GPU by default) plus Hugging Face translation pipelines.

## Setup
```bash
# (optional) venv
python -m venv .venv
.\.venv\Scripts\activate

# GPU build of PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other deps
pip install faster-whisper transformers sounddevice numpy sentencepiece
```

If you must run on CPU, edit the `WhisperModel` call in `rt-translator-ara-esp.py` to `device="cpu", compute_type="int8"` (already shown as a commented option).

## Usage
List microphones and pick the input index:
```bash
python rt-translator-ara-esp.py --list-devices
```

Arabic -> Spanish + English (default model: `small`):
```bash
python rt-translator-ara-esp.py --primary-language ar --model small --input-device <index>
```

Spanish -> Arabic + English (tighter segments):
```bash
python rt-translator-ara-esp.py --primary-language es --model small --segment-seconds 4 --input-device <index>
```

Custom targets (choices: `es`, `ar`, `en`), e.g., Spanish -> English + Arabic:
```bash
python rt-translator-ara-esp.py --primary-language es --targets en ar --input-device <index>
```

GPU choice (if you have multiple GPUs):
```bash
python rt-translator-ara-esp.py --primary-language ar --model medium --device-index 0 --input-device <index>
```

Smoother, overlapping windows with coherence:
```bash
python rt-translator-ara-esp.py --primary-language es --model large-v3 --segment-seconds 6 --overlap-seconds 2 --condition-on-prev --input-device <index>
```

Press `Ctrl+C` to stop. Two windows will show translations: a minimal English monitor and a semi-transparent AR/ES overlay for OBS.
