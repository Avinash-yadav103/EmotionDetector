#!/usr/bin/env python3
"""
audio_test.py

Load an audio emotion classification model (HDF5) and predict emotion from
an input WAV file or by recording a short clip from the microphone.

Usage examples:
  python audio_test.py --file example.wav
  python audio_test.py            # records default 3 seconds from mic

Dependencies: librosa, sounddevice, soundfile, numpy, tensorflow
"""

import argparse
import os
import tempfile
import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except Exception:
    sd = None
    sf = None

import librosa
from tensorflow.keras.models import load_model


def extract_feature(file_name, sr=16000, duration=3, n_mels=64):
    """Extract mel spectrogram features to match the training pipeline.
    
    Returns a mel spectrogram with shape (n_mels, time_frames, 1) matching your model's expected input.
    """
    # Load audio with same parameters as training
    audio, _ = librosa.load(file_name, sr=sr)
    
    # Pad or trim to fixed length (same as training)
    max_len = int(sr * duration)  # Ensure max_len is integer
    if len(audio) > max_len:
        audio = audio[:max_len]
    else:
        pad_length = max_len - len(audio)
        audio = np.pad(audio, (0, int(pad_length)))  # Ensure padding is integer
    
    # Convert to mel spectrogram (same as training)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize between 0 and 1 (same as training)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    # Add channel dimension to match model input: (n_mels, time_frames, 1)
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
    
    return mel_spec_db.astype(np.float32)


def record_temp_wav(duration=3, sr=16000):
    """Record from the default microphone for `duration` seconds and return the temp file path.

    Requires sounddevice and soundfile. If they are not available, this raises RuntimeError.
    """
    if sd is None or sf is None:
        raise RuntimeError("Recording requires sounddevice and soundfile packages. Install them or pass --file <path>")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmpname = tmp.name

    print(f"Recording {duration} seconds from microphone (sample rate {sr})...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    # soundfile expects shape (n_samples, channels)
    sf.write(tmpname, recording, sr)
    print(f"Saved recording to {tmpname}")
    return tmpname


def load_labels(labels_arg):
    if labels_arg:
        labels = [s.strip() for s in labels_arg.split(',') if s.strip()]
        return labels
    # Updated label order to match your notebook's class_names
    return ('neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'pleasant_surprise')


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from an audio file or microphone recording using a Keras .h5 model")
    parser.add_argument('--file', '-f', help='Path to WAV file to analyze')
    parser.add_argument('--record-duration', '-r', type=float, default=3.0, help='Seconds to record when --file is not provided')
    parser.add_argument('--model', '-m', default='emotion_audio_m3b.h5', help='Path to the Keras HDF5 model (default: emotion_audio_m3b.h5)')
    parser.add_argument('--labels', '-l', help='Comma-separated labels overriding the default label order')
    parser.add_argument('--sr', type=int, default=16000, help='Sampling rate used for feature extraction (default 16000)')
    parser.add_argument('--n-mels', type=int, default=64, help='Number of mel frequency bands (default 64)')
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    input_file = args.file
    temp_file = None
    try:
        if input_file is None:
            # record from mic
            input_file = record_temp_wav(duration=args.record_duration, sr=args.sr)
            temp_file = input_file

        print(f"Extracting features from: {input_file}")
        features = extract_feature(input_file, sr=args.sr, duration=args.record_duration, n_mels=args.n_mels)
        print(f"Feature shape: {features.shape}")
        
        # Add batch dimension
        x = np.expand_dims(features, axis=0)
        print(f"Input shape for model: {x.shape}")

        print(f"Loading model: {model_path}")
        model = load_model(model_path)

        # predict
        preds = model.predict(x)
        if preds.ndim == 2:
            probs = preds[0]
        else:
            probs = np.array(preds).ravel()

        labels = load_labels(args.labels)
        if len(labels) != probs.shape[0]:
            print(f"Warning: number of labels ({len(labels)}) does not match model output size ({probs.shape[0]}).")

        max_index = int(np.argmax(probs))
        predicted = labels[max_index] if max_index < len(labels) else str(max_index)

        print("Prediction:\n-----------")
        print(f"Predicted emotion: {predicted}")
        print("Probabilities:")
        for i, p in enumerate(probs):
            lab = labels[i] if i < len(labels) else str(i)
            print(f"  {lab:15s}: {p:.4f}")

    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
    finally:
        if temp_file:
            try:
                os.remove(temp_file)
            except Exception:
                pass


if __name__ == '__main__':
    main()
