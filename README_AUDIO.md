# audio_test.py

This script loads an audio emotion classification model saved in HDF5 format (default: `emotion_audio_m3b.h5`) and predicts the emotion of a WAV file or a short microphone recording.

Quick start

1. Create and activate a Python environment (recommended):

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install dependencies:

   pip install -r requirements.txt

3. Run the script on a file:

   python audio_test.py --file path\to\your_audio.wav

   or record from the microphone (default 3 seconds):

   python audio_test.py

Options

- `--model` : path to the HDF5 model (default `emotion_audio_m3b.h5`)
- `--file`  : path to a WAV file to analyze
- `--record-duration` : seconds to record when `--file` is not provided (default 3)
- `--labels` : optional comma-separated label list overriding defaults

Notes

- The script uses a simple feature extractor (mean MFCC, chroma, mel). The model must have been trained with compatible features.
- If the model's output size doesn't match the default label list, pass `--labels` to match the correct order.
