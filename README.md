# Multimodal Emotion Detection System

A comprehensive emotion detection system that combines **Real-Time Facial Expression Recognition** using Convolutional Neural Networks (CNN) and **Audio Emotion Classification** using deep learning techniques. This project provides both standalone implementations and a unified web application for multimodal emotion analysis.

## 🌟 Features

- **Real-Time Facial Expression Detection**: Live webcam-based emotion recognition using CNN
- **Audio Emotion Classification**: Speech emotion recognition from audio files or microphone input
- **Web Application**: Flask-based web interface for both modalities
- **Multiple Model Architectures**: Various CNN and RNN architectures for optimal performance
- **Professional Grade**: Achieves 81.31% accuracy on facial emotion recognition

## 🛠️ System Architecture

### Facial Expression Recognition
- **Model**: Convolutional Neural Network with 5 Conv2D layers
- **Input**: 48x48 grayscale facial images
- **Output**: 7 emotion classes (anger, disgust, fear, happiness, sadness, surprise, neutral)
- **Accuracy**: 81.31% on test data

### Audio Emotion Classification
- **Models**: Multiple architectures (CNN-GRU, CNN-LSTM, hybrid models)
- **Input**: Mel-spectrograms from audio signals
- **Output**: 9 emotion classes (neutral, calm, happy, sad, angry, fearful, disgust, surprised, pleasant_surprise)
- **Features**: MFCC, Chroma, Mel-frequency features

## 📁 Project Structure

```
├── README.md                          # Main project documentation
├── README_AUDIO.md                    # Audio emotion detection documentation
├── README_FACIAL.md                   # Facial emotion detection documentation
├── app.py                            # Flask web application
├── requirements.txt                  # Python dependencies
├── requirements_web.txt              # Web app specific dependencies
│
├── Models & Notebooks/
│   ├── Real-Time Facial Expression Detection & Recognition using CNN.ipynb
│   ├── Copy_of_Audio.ipynb          # Audio emotion detection notebook
│   ├── fer.h5                       # Trained facial emotion model
│   ├── emotion_audio_m3b.h5         # Trained audio emotion model
│   └── Facial Expression Recognition.json
│
├── Standalone Scripts/
│   ├── webcam_test.py               # Facial emotion detection script
│   └── audio_test.py                # Audio emotion detection script
│
├── Web Application/
│   ├── templates/                   # HTML templates
│   ├── static/                      # CSS, JS, and assets
│   └── uploads/                     # File upload directory
│
└── Assets/
    └── haarcascade_frontalface_default.xml  # Face detection cascade
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for facial emotion detection)
- Microphone (for audio emotion detection)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Avinash-yadav103/EmotionDetector.git
cd EmotionDetector
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
# For standalone scripts
pip install -r requirements.txt

# For web application
pip install -r requirements_web.txt
```

### Running the Applications

#### Web Application
```bash
python app.py
```
Navigate to `http://localhost:5000` in your browser.

#### Standalone Scripts
```bash
# Facial emotion detection
python webcam_test.py

# Audio emotion detection from file
python audio_test.py --file path/to/audio.wav

# Audio emotion detection from microphone
python audio_test.py --record-duration 5
```

## 📊 Model Performance

### Facial Expression Recognition
- **Dataset**: FER-2013 (35,887 grayscale images)
- **Test Accuracy**: 81.31%
- **Test Loss**: 0.605
- **Classes**: 7 emotions (anger, disgust, fear, happiness, sadness, surprise, neutral)

### Audio Emotion Classification
- **Datasets**: RAVDESS + TESS (Toronto Emotional Speech Set)
- **Multiple Models**: Various architectures achieving different performance levels
- **Classes**: 9 emotions including pleasant surprise
- **Features**: Mel-spectrograms, MFCC, Chroma features

## 🔬 Technical Details

### Facial Expression Model Architecture
```
Sequential Model:
├── Conv2D (32 filters, 3x3) + BatchNorm + ReLU + Dropout
├── Conv2D (64 filters, 3x3) + BatchNorm + ReLU + MaxPool2D
├── Conv2D (64 filters, 3x3) + BatchNorm + ReLU + Dropout
├── Conv2D (128 filters, 3x3) + BatchNorm + ReLU + MaxPool2D
├── Conv2D (128 filters, 3x3) + BatchNorm + ReLU + MaxPool2D
├── Flatten + Dense(250) + ReLU + Dropout
└── Dense(7) + Softmax
```

### Audio Emotion Model Variants
- **Model 1**: CNN-GRU hybrid architecture
- **Model 2**: CNN-LSTM hybrid architecture
- **Model 3**: Optimized CNN-LSTM with GlobalAveragePooling
- **Model 4**: Enhanced CNN-LSTM with regularization

## 🎯 Use Cases

- **Healthcare**: Mental health monitoring and assessment
- **Education**: Student engagement analysis
- **Security**: Behavioral analysis and threat detection
- **Entertainment**: Interactive gaming and content personalization
- **Research**: Psychological and behavioral studies
- **Customer Service**: Sentiment analysis and satisfaction monitoring

## 📈 Future Enhancements

- [ ] Multimodal fusion for combined audio-visual emotion recognition
- [ ] Real-time processing optimization
- [ ] Mobile application development
- [ ] Additional emotion categories
- [ ] Improved model architectures (Vision Transformers, etc.)
- [ ] Cloud deployment and API services

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Avinash Yadav**
- GitHub: [@Avinash-yadav103](https://github.com/Avinash-yadav103)
- LinkedIn: [Avinash Yadav](https://www.linkedin.com/in/avinash-yadav103/)

## 🙏 Acknowledgments

- **FER-2013 Dataset**: Facial expression recognition dataset
- **RAVDESS Dataset**: Ryerson Audio-Visual Database of Emotional Speech and Song
- **TESS Dataset**: Toronto Emotional Speech Set from Kaggle
- **OpenCV**: Computer vision library for face detection
- **TensorFlow/Keras**: Deep learning framework
- **Librosa**: Audio analysis library

---

*This project demonstrates the power of deep learning in emotion recognition and provides a foundation for building more sophisticated emotion AI systems.*
