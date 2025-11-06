import os
import cv2
import numpy as np
import tempfile
import threading
import time
import base64
from flask import Flask, render_template, request, jsonify, Response
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import librosa
import sounddevice as sd
import soundfile as sf
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Global variables
face_model = None
audio_model = None
face_cascade = None

def initialize_models():
    """Initialize both face and audio models"""
    global face_model, audio_model, face_cascade
    
    try:
        # Load face model
        if os.path.exists("Facial Expression Recognition.json") and os.path.exists("fer.h5"):
            with open("Facial Expression Recognition.json", "r") as json_file:
                face_model = model_from_json(json_file.read())
            face_model.load_weights('fer.h5')
            print("✅ Face model loaded successfully")
        else:
            print("❌ Face model files not found")
            
        # Load audio model
        if os.path.exists('emotion_audio_m3b.h5'):
            audio_model = load_model('emotion_audio_m3b.h5')
            print("✅ Audio model loaded successfully")
        else:
            print("❌ Audio model file not found")
            
        # Load face cascade
        if os.path.exists('haarcascade_frontalface_default.xml'):
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            print("✅ Face cascade loaded successfully")
        else:
            print("❌ Face cascade file not found")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")

def extract_audio_features(file_path, sr=16000, duration=3, n_mels=64):
    """Extract mel spectrogram features for audio emotion prediction"""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Ensure fixed length
        max_len = int(sr * duration)
        if len(audio) > max_len:
            audio = audio[:max_len]
        else:
            pad_length = max_len - len(audio)
            audio = np.pad(audio, (0, pad_length))
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=n_mels, 
            fmax=8000,
            hop_length=512,
            n_fft=2048
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize between 0 and 1
        if mel_spec_db.max() != mel_spec_db.min():
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        else:
            mel_spec_db = np.zeros_like(mel_spec_db)
        
        # Add channel dimension
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        return mel_spec_db.astype(np.float32)
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face_emotion')
def face_emotion():
    return render_template('face_emotion.html')

@app.route('/audio_emotion')
def audio_emotion():
    return render_template('audio_emotion.html')

@app.route('/health')
def health_check():
    """Check if models are loaded"""
    return jsonify({
        'face_model': face_model is not None,
        'audio_model': audio_model is not None,
        'face_cascade': face_cascade is not None
    })

@app.route('/predict-face', methods=['POST'])
def predict_face():
    """Predict emotion from face image"""
    try:
        if face_model is None or face_cascade is None:
            return jsonify({'error': 'Face detection model not loaded'})
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'})
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image then to OpenCV format
        pil_image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with better parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        results = []
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Prepare for prediction
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels = img_pixels / 255.0
            
            # Predict emotion
            predictions = face_model.predict(img_pixels, verbose=0)
            emotion_probs = predictions[0]
            
            # Get predicted emotion
            max_index = np.argmax(emotion_probs)
            predicted_emotion = emotions[max_index]
            confidence = emotion_probs[max_index]
            
            # Create probabilities dictionary
            probabilities = {emotions[i]: float(emotion_probs[i]) for i in range(len(emotions))}
            
            results.append({
                'emotion': predicted_emotion,
                'confidence': float(confidence),
                'bbox': [int(x), int(y), int(w), int(h)],
                'probabilities': probabilities
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Error in face prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    """Record and predict emotion from audio"""
    try:
        if audio_model is None:
            return jsonify({'error': 'Audio model not loaded'})
        
        # Check if it's a JSON request (for recording) or form data (file upload)
        if request.is_json:
            # Recording from microphone
            data = request.get_json()
            duration = float(data.get('duration', 3))
            sr = 16000
            
            print(f"Recording {duration} seconds from microphone...")
            
            # Record audio
            recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
            sd.wait()  # Wait for recording to complete
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, recording, sr)
                temp_path = tmp.name
            
        else:
            # File upload
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'})
            
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No audio file selected'})
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_file.save(tmp.name)
                temp_path = tmp.name
                duration = 3  # Default duration for file analysis
        
        try:
            # Extract features
            features = extract_audio_features(temp_path, sr=16000, duration=duration)
            if features is None:
                return jsonify({'error': 'Failed to extract audio features'})
            
            # Add batch dimension
            x = np.expand_dims(features, axis=0)
            
            # Predict emotion
            predictions = audio_model.predict(x, verbose=0)
            probs = predictions[0]
            
            # Audio emotion labels (matching your model)
            labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'pleasant_surprise']
            
            # Get predicted emotion
            max_index = int(np.argmax(probs))
            predicted_emotion = labels[max_index]
            confidence = float(probs[max_index])
            
            # Create probabilities dictionary
            probabilities = {labels[i]: float(probs[i]) for i in range(len(labels))}
            
            # Create results for different response formats
            if request.is_json:
                # Recording response format
                results = []
                for i, prob in enumerate(probs):
                    results.append({
                        'emotion': labels[i],
                        'probability': float(prob)
                    })
                
                results.sort(key=lambda x: x['probability'], reverse=True)
                
                return jsonify({
                    'predicted_emotion': predicted_emotion,
                    'confidence': confidence,
                    'all_predictions': results
                })
            else:
                # File upload response format
                return jsonify({
                    'result': {
                        'emotion': predicted_emotion,
                        'confidence': confidence,
                        'probabilities': probabilities
                    }
                })
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
    except Exception as e:
        print(f"Error in audio prediction: {e}")
        return jsonify({'error': f'Audio prediction failed: {str(e)}'})

# Initialize models when starting the app
initialize_models()

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)