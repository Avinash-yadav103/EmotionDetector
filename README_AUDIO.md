# Audio Emotion Detection System

A sophisticated deep learning system for classifying emotions from speech audio using advanced neural network architectures and comprehensive audio feature extraction.

## 🎯 Overview

This audio emotion detection system leverages mel-spectrograms and multiple deep learning architectures to classify emotions from speech audio. The system combines Convolutional Neural Networks (CNNs) with Recurrent Neural Networks (RNNs) to achieve robust emotion recognition from audio signals.

## 📊 Datasets Used

### Primary Datasets

1. **RAVDESS (Ryerson Audio-Visual Database)**
   - **Description**: Professional actors speaking and singing with emotional expressions
   - **Size**: 7,356 files from 24 professional actors (12 female, 12 male)
   - **Emotions**: 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
   - **Format**: WAV files, 48 kHz, 16-bit
   - **Language**: English (North American accent)

2. **TESS (Toronto Emotional Speech Set)**
   - **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
   - **Description**: Emotional speech recordings from two actresses
   - **Size**: 2,800 files
   - **Emotions**: 7 emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, neutral)
   - **Format**: WAV files
   - **Speakers**: Two actresses aged 26 and 64 years

### Combined Dataset Statistics
- **Total Files**: ~10,156 audio files after augmentation
- **Emotion Classes**: 9 classes (merged and standardized)
- **Data Augmentation**: Applied to balance class distribution
- **Train/Validation Split**: 80/20 stratified split

## 🏗️ Model Architectures

### Model 1: CNN-GRU Hybrid
```python
Sequential([
    Conv2D(32, (3,3)) + BatchNorm + MaxPool + Dropout
    Conv2D(64, (3,3)) + BatchNorm + MaxPool + Dropout
    Reshape for RNN
    GRU(128, return_sequences=False)
    Dense(128) + Dropout
    Dense(9, activation='softmax')
])
```

### Model 2: CNN-LSTM Hybrid
```python
Sequential([
    Conv2D(32, (3,3)) + BatchNorm + MaxPool + Dropout
    Conv2D(64, (3,3)) + BatchNorm + MaxPool + Dropout
    Reshape for RNN
    LSTM(128, return_sequences=False)
    Dense(128) + Dropout
    Dense(9, activation='softmax')
])
```

### Model 3: Optimized CNN-LSTM
```python
Sequential([
    Conv2D(16, (3,3)) + BatchNorm + MaxPool(3,3) + Dropout
    Conv2D(32, (3,3)) + BatchNorm + MaxPool(3,3) + Dropout
    Reshape(70, 32)
    LSTM(64, return_sequences=True)
    GlobalAveragePooling1D()
    Dense(32) + Dropout
    Dense(9, activation='softmax')
])
```

### Model 4: Enhanced CNN-LSTM with Regularization
```python
Sequential([
    Conv2D(32, (3,3)) + L2 Regularization + BatchNorm + MaxPool + Dropout
    Conv2D(64, (3,3)) + L2 Regularization + BatchNorm + MaxPool + Dropout
    Reshape for RNN
    LSTM(128, dropout=0.3)
    Dense(64) + L2 Regularization + Dropout
    Dense(9, activation='softmax')
])
```

## 🔧 Feature Extraction

### Audio Preprocessing
- **Sampling Rate**: 16 kHz
- **Duration**: Fixed 3-second segments
- **Padding/Trimming**: Applied for consistent length

### Feature Types
1. **Mel-Spectrograms**
   - **n_mels**: 64 mel frequency bands
   - **fmax**: 8000 Hz
   - **Normalization**: Min-Max scaling to [0,1]

2. **MFCC Features** (for simple classifier)
   - 13 MFCC coefficients
   - Delta and delta-delta features

3. **Chroma Features**
   - 12-dimensional chroma vectors
   - Pitch class profiles

### Data Augmentation Techniques
- **Pitch Shifting**: ±2 semitones
- **Time Stretching**: 0.9x to 1.1x speed
- **Noise Addition**: Gaussian noise (σ=0.005)
- **Time Shifting**: ±0.1 second shifts

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.8+
- Librosa 0.9+
- NumPy, Pandas, Matplotlib, Seaborn
- Microphone for real-time recording

### Installation

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Additional audio processing libraries**
```bash
pip install librosa soundfile tensorflow-io
```

### Usage

#### Analyze Audio File
```bash
python audio_test.py --file path/to/your_audio.wav
```

#### Real-time Microphone Recording
```bash
python audio_test.py --record-duration 5
```

#### Custom Model and Labels
```bash
python audio_test.py --model custom_model.h5 --labels "neutral,happy,sad,angry"
```

### Command Line Options
- `--model`: Path to HDF5 model file (default: `emotion_audio_m3b.h5`)
- `--file`: Path to WAV file for analysis
- `--record-duration`: Recording duration in seconds (default: 3)
- `--labels`: Comma-separated emotion labels (optional)

## 📈 Model Performance

### Training Configuration
- **Optimizer**: Adam (learning_rate=1e-3)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: Up to 100 (with early stopping)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

### Performance Metrics
- **Validation Accuracy**: Varies by model (60-75%)
- **Class Balancing**: Applied using RandomOverSampler
- **Regularization**: L2 regularization and Dropout layers
- **Cross-validation**: Stratified train-test split

### Emotion Classes
1. **Neutral** (0)
2. **Calm** (1)
3. **Happy** (2)
4. **Sad** (3)
5. **Angry** (4)
6. **Fearful** (5)
7. **Disgust** (6)
8. **Surprised** (7)
9. **Pleasant Surprise** (8)

## 🔬 Technical Implementation

### Audio Processing Pipeline
1. **Audio Loading**: Librosa with 16kHz sampling
2. **Preprocessing**: Length normalization and amplitude scaling
3. **Feature Extraction**: Mel-spectrogram computation
4. **Normalization**: Min-max scaling to [0,1] range
5. **Model Inference**: TensorFlow/Keras prediction
6. **Post-processing**: Argmax for class prediction

### Model Training Process
1. **Data Loading**: RAVDESS and TESS datasets
2. **Label Standardization**: Unified emotion mapping
3. **Data Augmentation**: Class balancing and audio augmentation
4. **Feature Generation**: Mel-spectrogram conversion
5. **Model Training**: Multiple architecture experiments
6. **Validation**: Performance evaluation and model selection

## 🎛️ Configuration

### Audio Parameters
```python
SR = 16000          # Sampling rate
DURATION = 3        # Audio duration in seconds
N_MELS = 64         # Number of mel frequency bands
BATCH_SIZE = 32     # Training batch size
```

### Model Hyperparameters
```python
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
L2_REGULARIZATION = 0.001
PATIENCE = 12       # Early stopping patience
```

## 📁 File Structure

```
Audio Emotion Detection/
├── Copy_of_Audio.ipynb           # Main training notebook
├── audio_test.py                 # Inference script
├── emotion_audio_m3b.h5         # Best trained model
├── Audio_model_files/           # Additional model variants
│   └── Audio/
│       ├── emotion_audio_m1.h5
│       ├── emotion_audio_m2.h5
│       └── emotion_audio_m3.h5
└── requirements.txt             # Python dependencies
```

## 🚧 Troubleshooting

### Common Issues

1. **Audio Format Issues**
   - Ensure WAV format with compatible sampling rate
   - Convert using: `librosa.load(file, sr=16000)`

2. **Model Loading Errors**
   - Verify TensorFlow version compatibility
   - Check model file integrity

3. **Microphone Recording Issues**
   - Install: `pip install sounddevice`
   - Check microphone permissions

4. **Memory Issues with Large Models**
   - Reduce batch size
   - Use model variants with fewer parameters

## 🔮 Future Improvements

- [ ] **Real-time Processing**: Optimize for streaming audio
- [ ] **Multi-language Support**: Extend to non-English languages
- [ ] **Transformer Models**: Implement attention-based architectures
- [ ] **Model Compression**: Quantization for mobile deployment
- [ ] **Ensemble Methods**: Combine multiple model predictions
- [ ] **Cross-dataset Validation**: Evaluate generalization across datasets

## 📚 References

1. **RAVDESS Dataset**: Livingstone & Russo (2018)
2. **TESS Dataset**: Dupuis & Pichora-Fuller (2010)
3. **Mel-spectrograms**: Stevens & Volkmann (1940)
4. **Deep Learning for Audio**: Goodfellow et al. (2016)

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- New model architectures
- Additional datasets integration
- Performance optimizations
- Real-time processing improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Avinash Yadav**
- GitHub: [@Avinash-yadav103](https://github.com/Avinash-yadav103)
- Email: avinash.yadav103@example.com

---

*This audio emotion detection system represents state-of-the-art techniques in speech emotion recognition, combining robust feature extraction with advanced deep learning architectures for reliable emotion classification.*
