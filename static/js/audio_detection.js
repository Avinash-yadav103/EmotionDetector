let isRecording = false;
let recordingProgress = null;
let mediaRecorder = null;
let audioChunks = [];
let currentStream = null;
let audioContext = null;
let isEdge = false;

document.addEventListener('DOMContentLoaded', function() {
    // Detect Microsoft Edge
    isEdge = /Edg/i.test(navigator.userAgent);
    console.log('Browser detected:', isEdge ? 'Microsoft Edge' : 'Other');
    
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
    
    // Check microphone permissions and quality
    checkMicrophoneSetup();
});

async function checkMicrophoneSetup() {
    try {
        // Edge-optimized audio constraints
        const audioConstraints = isEdge ? {
            audio: {
                echoCancellation: true,   // Keep enabled for Edge
                noiseSuppression: true,   // Keep enabled for Edge
                autoGainControl: true,    // Keep enabled for Edge
                sampleRate: 48000,        // Edge prefers 48kHz
                channelCount: 1,
                latency: 0.1
            }
        } : {
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
                sampleRate: 44100,
                channelCount: 1
            }
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(audioConstraints);
        
        console.log('✅ Microphone access granted for', isEdge ? 'Edge' : 'other browser');
        
        // Test audio level with Edge compatibility
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: isEdge ? 48000 : 44100
        });
        
        const source = audioContext.createMediaStreamSource(stream);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        analyser.smoothingTimeConstant = 0.8;
        source.connect(analyser);
        
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        // Test for 2 seconds
        let testCount = 0;
        let maxLevel = 0;
        const testInterval = setInterval(() => {
            analyser.getByteTimeDomainData(dataArray);
            
            // Calculate RMS level
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) {
                const sample = (dataArray[i] - 128) / 128.0;
                sum += sample * sample;
            }
            const rms = Math.sqrt(sum / dataArray.length);
            const level = rms * 100;
            
            maxLevel = Math.max(maxLevel, level);
            testCount++;
            
            if (testCount >= 20) {
                clearInterval(testInterval);
                
                console.log('Max audio level detected:', maxLevel);
                
                if (maxLevel < 0.5) {
                    showNotification('Microphone level is very low. Please check your microphone settings in Windows and speak louder.', 'warning');
                } else if (maxLevel < 2) {
                    showNotification('Microphone level is low. Please speak louder during recording for better results.', 'warning');
                } else {
                    console.log('✅ Microphone setup looks good');
                    showNotification(`Microphone test completed successfully! (${isEdge ? 'Edge optimized' : 'Standard mode'})`, 'success');
                }
                
                // Cleanup
                stream.getTracks().forEach(track => track.stop());
                if (audioContext && audioContext.state !== 'closed') {
                    audioContext.close();
                }
            }
        }, 100);
        
    } catch (error) {
        console.error('❌ Microphone setup issue:', error);
        
        let errorMsg = 'Microphone access failed. ';
        if (error.name === 'NotAllowedError') {
            errorMsg += 'Please allow microphone permissions in Edge. Go to Settings > Site permissions > Microphone.';
        } else if (error.name === 'NotFoundError') {
            errorMsg += 'No microphone found. Please check your audio devices in Windows Settings.';
        } else {
            errorMsg += 'Please check your microphone settings in Windows and Edge.';
        }
        
        showError(errorMsg);
    }
}

async function startRecording() {
    if (isRecording) return;
    
    const durationSlider = document.getElementById('duration-slider');
    const duration = durationSlider ? parseInt(durationSlider.value) : 3;
    
    try {
        isRecording = true;
        audioChunks = [];
        
        updateRecordingStatus('Starting...', 'warning');
        
        // Edge-optimized audio constraints
        const audioConstraints = isEdge ? {
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 48000,
                channelCount: 1,
                latency: 0.1,
                volume: 1.0
            }
        } : {
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
                sampleRate: 44100,
                channelCount: 1
            }
        };
        
        currentStream = await navigator.mediaDevices.getUserMedia(audioConstraints);
        
        // Edge-specific MediaRecorder options
        let options = {};
        
        if (isEdge) {
            // Edge preferred MIME types in order
            const edgeMimeTypes = [
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/mp4;codecs=mp4a.40.2',
                'audio/mp4',
                'audio/ogg;codecs=opus'
            ];
            
            for (const mimeType of edgeMimeTypes) {
                if (MediaRecorder.isTypeSupported(mimeType)) {
                    options.mimeType = mimeType;
                    options.audioBitsPerSecond = 128000; // Higher bitrate for Edge
                    console.log(`Edge: Using MIME type: ${mimeType}`);
                    break;
                }
            }
        } else {
            // Other browsers
            const standardMimeTypes = [
                'audio/wav',
                'audio/webm;codecs=pcm',
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/mp4'
            ];
            
            for (const mimeType of standardMimeTypes) {
                if (MediaRecorder.isTypeSupported(mimeType)) {
                    options.mimeType = mimeType;
                    console.log(`Standard: Using MIME type: ${mimeType}`);
                    break;
                }
            }
        }
        
        console.log('MediaRecorder options:', options);
        
        mediaRecorder = new MediaRecorder(currentStream, options);
        
        // Handle recorded data with Edge-specific handling
        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
                console.log(`Audio chunk received: ${event.data.size} bytes (total chunks: ${audioChunks.length})`);
            }
        };
        
        // Handle recording completion
        mediaRecorder.onstop = async function() {
            console.log('🎤 Recording stopped, processing audio...');
            await processRecordedAudio();
        };
        
        // Handle errors
        mediaRecorder.onerror = function(event) {
            console.error('MediaRecorder error:', event.error);
            showError(`Recording error: ${event.error.message}. ${isEdge ? 'Try updating Edge or restarting the browser.' : 'Please try again.'}`);
            resetRecordingUI();
        };
        
        // Start recording with Edge-optimized interval
        const timeslice = isEdge ? 500 : 250; // Larger chunks for Edge
        mediaRecorder.start(timeslice);
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

        // Show recording instructions
        showNotification(`🎤 Recording started! ${isEdge ? 'Edge optimized mode active.' : ''} Please speak clearly into your microphone.`, 'info');
        
        // Start progress animation
        animateProgress(duration);
        
        // Stop recording after duration
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                console.log('Stopping recording after', duration, 'seconds');
                mediaRecorder.stop();
                if (currentStream) {
                    currentStream.getTracks().forEach(track => track.stop());
                }
            }
        }, duration * 1000);
        
    } catch (error) {
        console.error('❌ Error starting recording:', error);
        
        let errorMsg = 'Failed to start recording. ';
        if (error.name === 'NotAllowedError') {
            errorMsg += isEdge ? 'Please allow microphone permissions in Edge settings.' : 'Please allow microphone permissions.';
        } else if (error.name === 'NotFoundError') {
            errorMsg += 'No microphone found. Please check your audio devices in Windows Settings.';
        } else {
            errorMsg += isEdge ? 'Edge-specific error. Try restarting the browser or updating Edge.' : 'Please check your microphone and try again.';
        }
        
        showError(errorMsg);
        resetRecordingUI();
    }
}

function animateProgress(duration) {
    let progress = 0;
    const progressBar = document.getElementById('progress-bar');
    const increment = 100 / (duration * 20); // Update every 50ms
    
    recordingProgress = setInterval(() => {
        progress += increment;
        if (progressBar) {
            progressBar.style.width = Math.min(progress, 100) + '%';
        }
        
        if (progress >= 100) {
            clearInterval(recordingProgress);
            recordingProgress = null;
        }
    }, 50);
}

function showProcessingStatus() {
    hideElement('recording-progress');
    showElement('processing-status');
    updateRecordingStatus('Processing', 'primary');
}

async function processRecordedAudio() {
    try {
        showProcessingStatus();
        
        if (audioChunks.length === 0) {
            throw new Error('No audio data recorded. Please try again.');
        }
        
        // Create audio blob with Edge-specific handling
        let mimeType = 'audio/webm'; // Default
        if (mediaRecorder && mediaRecorder.mimeType) {
            mimeType = mediaRecorder.mimeType;
        }
        
        const audioBlob = new Blob(audioChunks, { type: mimeType });
        
        console.log('📊 Audio blob details:', {
            size: audioBlob.size,
            type: audioBlob.type,
            chunks: audioChunks.length,
            browser: isEdge ? 'Edge' : 'Other'
        });
        
        if (audioBlob.size < 500) { // Lower threshold for Edge
            throw new Error('Recording is too short or empty. Please try again and speak louder.');
        }
        
        // Create FormData with Edge-optimized filename
        const formData = new FormData();
        
        let filename = 'recording.webm';
        if (mimeType.includes('wav')) {
            filename = 'recording.wav';
        } else if (mimeType.includes('mp4')) {
            filename = 'recording.mp4';
        } else if (mimeType.includes('ogg')) {
            filename = 'recording.ogg';
        }
        
        // Add browser info for server-side processing
        formData.append('audio', audioBlob, filename);
        formData.append('browser', isEdge ? 'edge' : 'other');
        formData.append('mime_type', mimeType);
        
        console.log('📤 Sending audio to server...', {filename, mimeType});
        
        // Send with longer timeout for Edge
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), isEdge ? 45000 : 30000); // 45s for Edge
        
        const response = await fetch('/predict-audio', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error (${response.status}): ${errorText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayAudioResults(data);
        }
        
    } catch (error) {
        console.error('❌ Error processing audio:', error);
        
        if (error.name === 'AbortError') {
            showError(`Request timed out. ${isEdge ? 'Edge may need more time to process. Try with a shorter recording.' : 'Please try again with a shorter recording.'}`);
        } else {
            showError(`Failed to process audio: ${error.message} ${isEdge ? '(Edge browser detected - this may be a compatibility issue)' : ''}`);
        }
    } finally {
        resetRecordingUI();
    }
}

function displayAudioResults(data) {
    try {
        // Handle response format
        let predicted_emotion, confidence, all_predictions;
        
        if (data.result) {
            predicted_emotion = data.result.emotion;
            confidence = data.result.confidence;
            all_predictions = Object.entries(data.result.probabilities).map(([emotion, probability]) => ({
                emotion: emotion,
                probability: probability
            }));
        } else {
            showError('Invalid response format from server');
            return;
        }
        
        if (!predicted_emotion || !all_predictions) {
            showError('Incomplete response from server');
            return;
        }
        
        // Sort predictions by probability
        const sortedPredictions = all_predictions.sort((a, b) => b.probability - a.probability);
        
        // Create results HTML with Edge-specific info
        const resultsHtml = `
            <div class="audio-result-card">
                <div class="text-center mb-3">
                    <i class="fas fa-check-circle text-success fa-2x mb-2"></i>
                    <h5 class="text-success">Analysis Complete! ${isEdge ? '(Edge Optimized)' : ''}</h5>
                </div>
                
                <div class="predicted-emotion">${predicted_emotion.replace('_', ' ')}</div>
                <div class="confidence-score">Confidence: ${(confidence * 100).toFixed(1)}%</div>
                
                <div class="emotion-probabilities">
                    <h6 class="mb-3"><i class="fas fa-chart-bar me-2"></i>All Emotions</h6>
                    ${sortedPredictions.map((pred, index) => {
                        const percentage = (pred.probability * 100).toFixed(1);
                        const color = getEmotionColor(pred.emotion);
                        const isTop3 = index < 3;
                        
                        return `
                            <div class="probability-row ${isTop3 ? 'top-prediction' : 'low-prediction'}">
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
                
                ${data.debug_info ? `
                    <div class="debug-info mt-3">
                        <small class="text-muted">
                            <i class="fas fa-info-circle me-1"></i>
                            Audio: ${data.debug_info.audio_length} samples, 
                            Level: ${(data.debug_info.max_amplitude * 100).toFixed(1)}%,
                            File: ${(data.debug_info.file_size / 1024).toFixed(1)}KB
                            ${isEdge ? ', Browser: Edge' : ''}
                        </small>
                    </div>
                ` : ''}
            </div>
        `;
        
        const audioResults = document.getElementById('audio-results');
        if (audioResults) {
            audioResults.innerHTML = resultsHtml;
        }
        
        // Show success notification
        showNotification(`🎉 Detected emotion: ${predicted_emotion.replace('_', ' ')} (${(confidence * 100).toFixed(1)}% confidence) ${isEdge ? '- Edge optimized!' : ''}`, 'success');
        
    } catch (error) {
        console.error('Error displaying results:', error);
        showError('Failed to display results. Please try again.');
    }
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
    
    // Clean up media recorder and stream
    if (mediaRecorder) {
        mediaRecorder = null;
    }
    
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    
    // Clear audio chunks
    audioChunks = [];
    
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
                <strong>Error:</strong> ${message}
                <hr>
                <small>
                    <strong>Edge-specific troubleshooting:</strong><br>
                    ${isEdge ? `
                    • Make sure Edge is updated to the latest version<br>
                    • Go to Edge Settings > Site permissions > Microphone and allow this site<br>
                    • Check Windows Privacy Settings > Microphone permissions<br>
                    • Try restarting Edge or your computer<br>
                    • Clear Edge's cache and cookies for this site<br>
                    ` : `
                    • Ensure your microphone is working and enabled<br>
                    • Check browser microphone permissions<br>
                    • Try refreshing the page<br>
                    `}
                    • Speak clearly and loudly during recording<br>
                    • Try using Chrome or Firefox as an alternative
                </small>
            </div>
        `;
    }
    
    showNotification(message, 'danger');
}

function showNotification(message, type) {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.position-fixed.alert');
    existingNotifications.forEach(notification => {
        if (notification.parentNode) {
            notification.remove();
        }
    });
    
    // Create new notification
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 350px; max-width: 500px;';
    
    notification.innerHTML = `
        <i class="fas fa-${getIconForType(type)} me-2"></i>
        <strong>${getTypeLabel(type)}:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" onclick="this.parentElement.remove()"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after appropriate time
    const autoRemoveTime = type === 'success' ? 8000 : type === 'danger' ? 15000 : 10000;
    setTimeout(() => {
        if (notification && notification.parentNode) {
            notification.remove();
        }
    }, autoRemoveTime);
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

function getTypeLabel(type) {
    const labels = {
        'success': 'Success',
        'danger': 'Error',
        'warning': 'Warning',
        'info': 'Info'
    };
    return labels[type] || 'Notice';
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
    if (isRecording) {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
    }
    
    if (recordingProgress) {
        clearInterval(recordingProgress);
    }
});