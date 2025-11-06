let isRecording = false;
let recordingProgress = null;
let currentAudioBlob = null;

document.addEventListener('DOMContentLoaded', function() {
    // Duration slider
    const durationSlider = document.getElementById('duration-slider');
    const durationValue = document.getElementById('duration-value');
    const startButton = document.getElementById('start-recording');
    
    if (durationSlider && durationValue) {
        durationSlider.addEventListener('input', function() {
            durationValue.textContent = this.value;
        });
    }

    // Start Recording Button
    if (startButton) {
        startButton.addEventListener('click', function() {
            if (isRecording) return;
            startRecording();
        });
    }
    
    // Check microphone permissions
    checkMicrophonePermissions();
});

async function checkMicrophonePermissions() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => track.stop());
        console.log('✅ Microphone permission granted');
    } catch (error) {
        console.error('❌ Microphone permission denied:', error);
        showError('Microphone access is required for audio emotion detection. Please allow microphone permissions.');
    }
}

async function startRecording() {
    if (isRecording) return;
    
    const durationSlider = document.getElementById('duration-slider');
    const duration = durationSlider ? parseInt(durationSlider.value) : 3;
    
    try {
        isRecording = true;
        updateRecordingStatus('Recording', 'danger');
        
        // Show recording UI
        showElement('recording-progress');
        hideElement('duration-controls');
        hideElement('recording-controls');
        
        // Add recording class to microphone icon
        const micIcon = document.getElementById('mic-icon');
        if (micIcon) {
            micIcon.classList.add('recording');
        }
        
        // Start progress animation
        animateProgress(duration);
        
        // Send recording request to backend
        const response = await fetch('/predict-audio', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                duration: duration
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayAudioResults(data);
        }
        
    } catch (error) {
        console.error('Error during recording:', error);
        let errorMsg = 'Failed to record audio. ';
        
        if (error.message.includes('HTTP error')) {
            errorMsg += 'Server connection failed. Please check your internet connection.';
        } else {
            errorMsg += 'Please check your microphone and try again.';
        }
        
        showError(errorMsg);
    } finally {
        resetRecordingUI();
    }
}

function animateProgress(duration) {
    let progress = 0;
    const progressBar = document.getElementById('progress-bar');
    const increment = 100 / (duration * 10); // Update every 100ms
    
    recordingProgress = setInterval(() => {
        progress += increment;
        if (progressBar) {
            progressBar.style.width = progress + '%';
        }
        
        if (progress >= 100) {
            clearInterval(recordingProgress);
            showProcessingStatus();
        }
    }, 100);
}

function showProcessingStatus() {
    hideElement('recording-progress');
    showElement('processing-status');
    updateRecordingStatus('Processing', 'primary');
}

function displayAudioResults(data) {
    // Handle different response formats
    let predicted_emotion, confidence, all_predictions;
    
    if (data.predicted_emotion) {
        // Recording format
        predicted_emotion = data.predicted_emotion;
        confidence = data.confidence;
        all_predictions = data.all_predictions;
    } else if (data.result) {
        // File upload format
        predicted_emotion = data.result.emotion;
        confidence = data.result.confidence;
        // Convert probabilities object to array format
        all_predictions = Object.entries(data.result.probabilities).map(([emotion, probability]) => ({
            emotion: emotion,
            probability: probability
        }));
    } else {
        showError('Invalid response format from server');
        return;
    }
    
    // Sort predictions by probability
    const sortedPredictions = all_predictions.sort((a, b) => b.probability - a.probability);
    
    const resultsHtml = `
        <div class="audio-result-card">
            <div class="predicted-emotion">${predicted_emotion.replace('_', ' ')}</div>
            <div class="confidence-score">Confidence: ${(confidence * 100).toFixed(1)}%</div>
            
            <div class="emotion-probabilities">
                <h6 class="mb-3"><i class="fas fa-chart-bar me-2"></i>All Emotions</h6>
                ${sortedPredictions.map(pred => {
                    const percentage = (pred.probability * 100).toFixed(1);
                    const color = getEmotionColor(pred.emotion);
                    
                    return `
                        <div class="probability-row">
                            <span class="emotion-name">${pred.emotion.replace('_', ' ')}</span>
                            <div class="probability-bar-container">
                                <div class="probability-bar-fill" 
                                     style="width: ${percentage}%; background-color: ${color};">
                                </div>
                            </div>
                            <span class="probability-percentage">${percentage}%</span>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
    `;
    
    const audioResults = document.getElementById('audio-results');
    if (audioResults) {
        audioResults.innerHTML = resultsHtml;
    }
    
    // Show success notification
    showNotification('Audio analysis completed successfully!', 'success');
}

function getEmotionColor(emotion) {
    const colors = {
        'angry': '#dc3545',
        'disgust': '#6f42c1',
        'fear': '#fd7e14',
        'fearful': '#fd7e14',
        'happy': '#ffc107',
        'sad': '#17a2b8',
        'surprise': '#e83e8c',
        'surprised': '#e83e8c',
        'neutral': '#6c757d',
        'calm': '#20c997',
        'pleasant_surprise': '#28a745'
    };
    return colors[emotion] || '#007bff';
}

function resetRecordingUI() {
    isRecording = false;
    
    // Reset UI elements
    showElement('duration-controls');
    showElement('recording-controls');
    hideElement('recording-progress');
    hideElement('processing-status');
    
    // Reset progress bar
    const progressBar = document.getElementById('progress-bar');
    if (progressBar) {
        progressBar.style.width = '0%';
    }
    
    // Remove recording class from microphone icon
    const micIcon = document.getElementById('mic-icon');
    if (micIcon) {
        micIcon.classList.remove('recording');
    }
    
    // Clear progress interval
    if (recordingProgress) {
        clearInterval(recordingProgress);
        recordingProgress = null;
    }
    
    updateRecordingStatus('Ready', 'secondary');
}

function updateRecordingStatus(text, type) {
    const statusBadge = document.getElementById('recording-status-badge');
    if (statusBadge) {
        statusBadge.textContent = text;
        statusBadge.className = `badge bg-${type} ms-2`;
    }
}

function showError(message) {
    const audioResults = document.getElementById('audio-results');
    if (audioResults) {
        audioResults.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }
    
    showNotification(message, 'danger');
}

function showNotification(message, type) {
    // Create notification
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        <i class="fas fa-${getIconForType(type)} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification && notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function getIconForType(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Utility functions
function showElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove('d-none');
        element.style.display = 'block';
    }
}

function hideElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('d-none');
        element.style.display = 'none';
    }
}

// Handle page unload
window.addEventListener('beforeunload', function() {
    if (isRecording && recordingProgress) {
        clearInterval(recordingProgress);
    }
});