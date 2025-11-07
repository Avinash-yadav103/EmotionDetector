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
import scipy.signal
from scipy.io import wavfile
import subprocess
import shutil
import wave

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

def convert_webm_to_wav_ffmpeg(input_path, output_path):
    """Convert WebM to WAV using FFmpeg"""
    try:
        cmd = [
            'ffmpeg', '-i', input_path, 
            '-ar', '22050', '-ac', '1', 
            '-acodec', 'pcm_s16le', 
            '-y', output_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed: {e}")
        return False
    except Exception as e:
        print(f"FFmpeg not available: {e}")
        return False

def convert_webm_with_pydub(input_path, output_path):
    """Convert WebM to WAV using pydub"""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path, format="webm")
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Pydub conversion failed: {e}")
        return False

def extract_audio_from_webm(webm_path):
    """Extract raw audio data from WebM file using multiple methods"""
    try:
        # Method 1: Try direct librosa loading with different parameters
        methods = [
            lambda: librosa.load(webm_path, sr=22050, duration=10),
            lambda: librosa.load(webm_path, sr=16000, duration=10),
            lambda: librosa.load(webm_path, sr=None, duration=10),
        ]
        
        for i, method in enumerate(methods, 1):
            try:
                print(f"Trying direct librosa method {i}...")
                audio_data, sr = method()
                if audio_data is not None and len(audio_data) > 0:
                    print(f"✅ Direct librosa method {i} succeeded")
                    # Resample to 22050 if needed
                    if sr != 22050:
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=22050)
                    return audio_data, 22050
            except Exception as e:
                print(f"Direct librosa method {i} failed: {e}")
                continue
        
        # Method 2: Try conversion then loading
        wav_path = webm_path.replace('.webm', '_converted.wav')
        
        # Try FFmpeg first
        if convert_webm_to_wav_ffmpeg(webm_path, wav_path):
            try:
                audio_data, sr = librosa.load(wav_path, sr=22050, duration=10)
                os.unlink(wav_path)  # Clean up
                return audio_data, sr
            except Exception as e:
                print(f"Loading converted FFmpeg file failed: {e}")
        
        # Try pydub
        if convert_webm_with_pydub(webm_path, wav_path):
            try:
                audio_data, sr = librosa.load(wav_path, sr=22050, duration=10)
                os.unlink(wav_path)  # Clean up
                return audio_data, sr
            except Exception as e:
                print(f"Loading converted pydub file failed: {e}")
        
        return None, None
        
    except Exception as e:
        print(f"All audio extraction methods failed: {e}")
        return None, None

def extract_audio_features_v2(audio_data, sr=22050, duration=3):
    """Enhanced audio feature extraction with multiple feature types"""
    try:
        # Ensure audio is the right length
        target_length = int(sr * duration)
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        else:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        
        # Remove silence and normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
        
        # 4. Mel spectrogram (for CNN model) - simplified approach
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sr, 
            n_mels=128,
            fmax=8000,
            hop_length=512,
            n_fft=2048
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        if np.std(mel_spec_db) > 0:
            mel_spec_normalized = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-6)
        else:
            mel_spec_normalized = mel_spec_db
        
        # Resize to expected input shape
        target_width = 130
        if mel_spec_normalized.shape[1] > target_width:
            mel_spec_normalized = mel_spec_normalized[:, :target_width]
        elif mel_spec_normalized.shape[1] < target_width:
            pad_width = target_width - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
        
        # Add channel dimension for CNN
        mel_spec_normalized = np.expand_dims(mel_spec_normalized, axis=-1)
        
        return mel_spec_normalized.astype(np.float32)
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

def load_audio_file(file_path):
    """Try multiple methods to load audio file"""
    audio_data = None
    sr = 22050
    
    methods = [
        # Method 1: Direct librosa load
        lambda: librosa.load(file_path, sr=sr, duration=10),
        # Method 2: Try with different backends
        lambda: librosa.load(file_path, sr=sr, duration=10, res_type='kaiser_fast'),
        # Method 3: Try soundfile first then librosa
        lambda: load_with_soundfile(file_path, sr),
    ]
    
    for i, method in enumerate(methods, 1):
        try:
            print(f"Trying audio loading method {i}...")
            audio_data, loaded_sr = method()
            if audio_data is not None and len(audio_data) > 0:
                print(f"✅ Audio loaded successfully with method {i}")
                return audio_data, loaded_sr
        except Exception as e:
            print(f"Method {i} failed: {e}")
            continue
    
    return None, None

def load_with_soundfile(file_path, target_sr):
    """Load audio using soundfile then resample"""
    try:
        import soundfile as sf
        audio_data, sr = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
            
        return audio_data, target_sr
    except Exception as e:
        print(f"Soundfile loading failed: {e}")
        raise e

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
    """Enhanced audio emotion prediction with WebM support"""
    try:
        if audio_model is None:
            return jsonify({'error': 'Audio model not loaded'})
        
        temp_paths = []
        
        try:
            # Check if it's a file upload or JSON request
            if request.files and 'audio' in request.files:
                # Handle file upload from MediaRecorder
                audio_file = request.files['audio']
                
                if audio_file.filename == '':
                    return jsonify({'error': 'No audio file selected'})
                
                # Read file content to check size
                file_content = audio_file.read()
                audio_file.seek(0)  # Reset file pointer
                file_size = len(file_content)
                
                print(f"🎤 Received audio file: {audio_file.filename}, size: {file_size} bytes")
                
                if file_size < 100:
                    return jsonify({'error': 'Audio file is too small. Please record a longer audio.'})
                
                # Save the file
                file_ext = '.webm'  # Default for MediaRecorder
                if audio_file.content_type:
                    if 'wav' in audio_file.content_type:
                        file_ext = '.wav'
                    elif 'mp4' in audio_file.content_type:
                        file_ext = '.mp4'
                    elif 'ogg' in audio_file.content_type:
                        file_ext = '.ogg'
                
                with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                    tmp.write(file_content)
                    temp_path = tmp.name
                    temp_paths.append(temp_path)
                
                print(f"📁 Saved audio file to: {temp_path}")
                
                # Extract audio data using enhanced method
                if file_ext == '.webm':
                    audio_data, sr = extract_audio_from_webm(temp_path)
                else:
                    # Try direct loading for other formats
                    try:
                        audio_data, sr = librosa.load(temp_path, sr=22050, duration=10)
                    except Exception as e:
                        print(f"Direct loading failed: {e}")
                        audio_data, sr = None, None
                
                if audio_data is None or len(audio_data) == 0:
                    return jsonify({'error': 'Could not extract audio from the uploaded file. Please try recording again or use a different browser.'})
                    
            else:
                # Handle JSON request (server-side recording - not recommended for WebM issues)
                return jsonify({'error': 'Server-side recording not supported. Please use the web interface for recording.'})
            
            # Validate audio data
            if len(audio_data) == 0:
                return jsonify({'error': 'No audio data found. Please try recording again.'})
            
            if np.max(np.abs(audio_data)) < 0.001:
                return jsonify({'error': 'Audio is too quiet. Please speak louder and try again.'})
            
            print(f"📊 Audio stats: length={len(audio_data)}, max_amplitude={np.max(np.abs(audio_data)):.4f}, sample_rate={sr}")
            
            # Extract features using the enhanced method
            features = extract_audio_features_v2(audio_data, sr=sr, duration=3)
            
            if features is None:
                return jsonify({'error': 'Failed to extract audio features. Please try again with a different recording.'})
            
            print(f"🔧 Feature shape: {features.shape}")
            
            # Add batch dimension
            x = np.expand_dims(features, axis=0)
            
            # Make prediction
            predictions = audio_model.predict(x, verbose=0)
            probs = predictions[0]
            
            print(f"🎯 Raw predictions: {probs}")
            
            # Audio emotion labels
            labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'pleasant_surprise']
            
            # Get predicted emotion
            max_index = int(np.argmax(probs))
            predicted_emotion = labels[max_index]
            confidence = float(probs[max_index])
            
            print(f"🏆 Predicted: {predicted_emotion} with confidence: {confidence:.4f}")
            
            # File upload response format
            probabilities = {labels[i]: float(probs[i]) for i in range(len(labels))}
            return jsonify({
                'result': {
                    'emotion': predicted_emotion,
                    'confidence': confidence,
                    'probabilities': probabilities
                },
                'debug_info': {
                    'audio_length': len(audio_data),
                    'sample_rate': sr,
                    'max_amplitude': float(np.max(np.abs(audio_data))),
                    'feature_shape': list(features.shape),
                    'file_size': file_size
                }
            })
            
        finally:
            # Clean up all temporary files
            for temp_path in temp_paths:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        print(f"🗑️ Cleaned up temp file: {temp_path}")
                except Exception as cleanup_error:
                    print(f"⚠️ Could not clean up {temp_path}: {cleanup_error}")
            
    except Exception as e:
        print(f"❌ Error in audio prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Audio processing failed: {str(e)}'})

# Initialize models when starting the app
initialize_models()

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)