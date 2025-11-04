from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import cv2
import numpy as np
import librosa
import tempfile
import base64
from werkzeug.utils import secure_filename
from keras.models import model_from_json, load_model
from tensorflow.keras.preprocessing import image
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
face_model = None
audio_model = None
face_cascade = None

def load_models():
    global face_model, audio_model, face_cascade
    
    try:
        # Load face emotion model
        if os.path.exists("Facial Expression Recognition.json") and os.path.exists("fer.h5"):
            face_model = model_from_json(open("Facial Expression Recognition.json", "r").read())
            face_model.load_weights('fer.h5')
            print("✅ Face emotion model loaded successfully")
        
        # Load audio emotion model
        if os.path.exists("emotion_audio_m3b.h5"):
            audio_model = load_model("emotion_audio_m3b.h5")
            print("✅ Audio emotion model loaded successfully")
        
        # Load face cascade
        if os.path.exists("haarcascade_frontalface_default.xml"):
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            print("✅ Face cascade loaded successfully")
            
    except Exception as e:
        print(f"Error loading models: {e}")

def extract_audio_features(file_path, sr=16000, duration=3, n_mels=64):
    """Extract mel spectrogram features from audio file"""
    try:
        # Load audio
        audio, _ = librosa.load(file_path, sr=sr)
        
        # Pad or trim to fixed length
        max_len = int(sr * duration)
        if len(audio) > max_len:
            audio = audio[:max_len]
        else:
            pad_length = max_len - len(audio)
            audio = np.pad(audio, (0, int(pad_length)))
        
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize between 0 and 1
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Add channel dimension
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        
        return mel_spec_db.astype(np.float32)
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face-emotion')
def face_emotion():
    return render_template('face_emotion.html')

@app.route('/audio-emotion')
def audio_emotion():
    return render_template('audio_emotion.html')

@app.route('/predict-face', methods=['POST'])
def predict_face():
    try:
        if face_model is None or face_cascade is None:
            return jsonify({'error': 'Face emotion model not loaded'})
        
        # Get image data from request
        image_data = request.json['image']
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, part
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces_detected = face_cascade.detectMultiScale(gray_img, 1.32, 5)
        
        results = []
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        
        for (x, y, w, h) in faces_detected:
            # Extract face region
            roi_gray = gray_img[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Preprocess for model
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            
            # Predict emotion
            predictions = face_model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotions[max_index]
            confidence = float(predictions[0][max_index])
            
            results.append({
                'emotion': predicted_emotion,
                'confidence': confidence,
                'bbox': [int(x), int(y), int(w), int(h)],
                'probabilities': {emotions[i]: float(predictions[0][i]) for i in range(len(emotions))}
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    try:
        if audio_model is None:
            return jsonify({'error': 'Audio emotion model not loaded'})
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'})
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Save uploaded file
        filename = secure_filename(audio_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Extract features
        features = extract_audio_features(filepath)
        
        if features is None:
            return jsonify({'error': 'Failed to extract audio features'})
        
        # Add batch dimension
        x = np.expand_dims(features, axis=0)
        
        # Predict emotion
        predictions = audio_model.predict(x)
        probs = predictions[0]
        
        # Audio emotion labels
        labels = ('neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'pleasant_surprise')
        
        max_index = np.argmax(probs)
        predicted_emotion = labels[max_index] if max_index < len(labels) else str(max_index)
        confidence = float(probs[max_index])
        
        # Clean up uploaded file
        os.remove(filepath)
        
        result = {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': {labels[i]: float(probs[i]) for i in range(min(len(labels), len(probs)))}
        }
        
        return jsonify({'result': result})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    status = {
        'face_model': face_model is not None,
        'audio_model': audio_model is not None,
        'face_cascade': face_cascade is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)