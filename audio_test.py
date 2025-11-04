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


def extract_feature(file_name, sr=22050, n_mfcc=40):
	"""Extract features (mfcc, chroma, mel) from an audio file and return a 1D vector.

	The vector is a concatenation of mean MFCCs, mean chroma, and mean mel bands.
	"""
	y, sr = librosa.load(file_name, sr=sr)
	if y.size == 0:
		raise ValueError("Loaded audio is empty")

	# Short-time Fourier transform for chroma
	stft = np.abs(librosa.stft(y))

	mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
	mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

	features = np.hstack([mfccs, chroma, mel])
	return features


def record_temp_wav(duration=3, sr=22050):
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
	# default label order similar to webcam_test.py
	return ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def main():
	parser = argparse.ArgumentParser(description="Predict emotion from an audio file or microphone recording using a Keras .h5 model")
	parser.add_argument('--file', '-f', help='Path to WAV file to analyze')
	parser.add_argument('--record-duration', '-r', type=float, default=3.0, help='Seconds to record when --file is not provided')
	parser.add_argument('--model', '-m', default='emotion_audio_m3b.h5', help='Path to the Keras HDF5 model (default: emotion_audio_m3b.h5)')
	parser.add_argument('--labels', '-l', help='Comma-separated labels overriding the default label order')
	parser.add_argument('--sr', type=int, default=22050, help='Sampling rate used for feature extraction (default 22050)')
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
		features = extract_feature(input_file, sr=args.sr)
		x = np.expand_dims(features, axis=0)

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
			print("Warning: number of labels does not match model output size.")

		max_index = int(np.argmax(probs))
		predicted = labels[max_index] if max_index < len(labels) else str(max_index)

		print("Prediction:\n-----------")
		print(f"Predicted emotion: {predicted}")
		print("Probabilities:")
		for i, p in enumerate(probs):
			lab = labels[i] if i < len(labels) else str(i)
			print(f"  {lab:10s}: {p:.4f}")

	except Exception as e:
		print("Error:", e)
	finally:
		if temp_file:
			try:
				os.remove(temp_file)
			except Exception:
				pass


if __name__ == '__main__':
	main()
