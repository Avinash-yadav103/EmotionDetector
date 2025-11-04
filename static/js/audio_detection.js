let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let recordingTimer = null;
let recordingStartTime = null;
let currentAudioBlob = null;

document.addEventListener('DOMContentLoaded', function() {
    const uploadZone = document.getElementById('uploadZone');
    const audioFile = document.getElementById('audioFile');
    const recordBtn = document.getElementById('recordBtn');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    
    // File upload handling
    uploadZone.addEventListener('click', () => audioFile.click());
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('drop', handleFileDrop);
    audioFile.addEventListener('change', handleFileSelect);
    
    // Recording handling
    recordBtn.addEventListener('click', toggleRecording);
    
    // Control buttons
    analyzeBtn.addEventListener('click', analyzeAudio);
    clearBtn.addEventListener('click', clearAudio);
});

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleFileDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleAudioFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleAudioFile(file);
    }
}

function handleAudioFile(file) {
    // Validate file type
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/m4a'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid audio file (WAV, MP3, M4A, OGG)', 'audioResults');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size too large. Please upload a file smaller than 10MB.', 'audioResults');
        return;
    }
    
    currentAudioBlob = file;
    
    // Show audio player
    const audioPlayback = document.getElementById('audioPlayback');
    audioPlayback.src = URL.createObjectURL(file);
    audioPlayback.classList.remove('hidden');
    
    // Enable analyze button
    document.getElementById('analyzeBtn').disabled = false;
    
    // Update UI
    document.getElementById('uploadZone').innerHTML = `
        <i class="fas fa-check-circle" style="color: #4caf50;"></i>
        <h3>Audio File Ready</h3>
        <p>${file.name}</p>
    `;
    
    // Clear any previous results
    document.getElementById('audioResults').innerHTML = '<p class="no-results">Click "Analyze Audio" to see results</p>';
}

async function toggleRecording() {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        recordedChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: 'audio/wav' });
            currentAudioBlob = blob;
            
            // Show audio player
            const audioPlayback = document.getElementById('audioPlayback');
            audioPlayback.src = URL.createObjectURL(blob);
            audioPlayback.classList.remove('hidden');
            
            // Enable analyze button
            document.getElementById('analyzeBtn').disabled = false;
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        isRecording = true;
        recordingStartTime = Date.now();
        
        // Update UI
        const recordBtn = document.getElementById('recordBtn');
        recordBtn.classList.add('recording');
        recordBtn.innerHTML = '<i class="fas fa-stop"></i> <span>Stop Recording</span>';
        
        // Show recording status
        const recordingStatus = document.getElementById('recordingStatus');
        recordingStatus.classList.remove('hidden');
        
        // Start timer
        updateRecordingTime();
        recordingTimer = setInterval(updateRecordingTime, 100);
        
    } catch (error) {
        console.error('Error starting recording:', error);
        showError('Could not access microphone. Please check permissions.', 'audioResults');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    isRecording = false;
    
    // Clear timer
    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }
    
    // Update UI
    const recordBtn = document.getElementById('recordBtn');
    recordBtn.classList.remove('recording');
    recordBtn.innerHTML = '<i class="fas fa-microphone"></i> <span>Start Recording</span>';
    
    // Hide recording status
    document.getElementById('recordingStatus').classList.add('hidden');
    
    // Clear results
    document.getElementById('audioResults').innerHTML = '<p class="no-results">Click "Analyze Audio" to see results</p>';
}

function updateRecordingTime() {
    if (recordingStartTime) {
        const elapsed = Date.now() - recordingStartTime;
        const seconds = Math.floor(elapsed / 1000);
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        
        document.getElementById('recordingTime').textContent = 
            `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
}

async function analyzeAudio() {
    if (!currentAudioBlob) {
        showError('No audio file to analyze', 'audioResults');
        return;
    }
    
    showLoading('audioLoading');
    
    try {
        const formData = new FormData();
        formData.append('audio', currentAudioBlob, 'audio.wav');
        
        const response = await fetch('/predict-audio', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error, 'audioResults');
        } else {
            displayAudioResults(data.result);
        }
        
    } catch (error) {
        console.error('Error analyzing audio:', error);
        showError('Failed to analyze audio. Please try again.', 'audioResults');
    } finally {
        hideLoading('audioLoading');
    }
}

function displayAudioResults(result) {
    const resultsContainer = document.getElementById('audioResults');
    
    const html = `
        <div class="result-card">
            <div class="result-emotion">${result.emotion}</div>
            <div class="result-confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
            
            <div class="probabilities">
                <h4>All Emotions:</h4>
                ${Object.entries(result.probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([emotion, prob]) => `
                        <div class="probability-item">
                            <span>${emotion}</span>
                            <div class="probability-bar">
                                <div class="probability-fill" style="width: ${prob * 100}%"></div>
                            </div>
                            <span>${(prob * 100).toFixed(1)}%</span>
                        </div>
                    `).join('')}
            </div>
        </div>
    `;
    
    resultsContainer.innerHTML = html;
}

function clearAudio() {
    // Reset file input
    document.getElementById('audioFile').value = '';
    
    // Clear current audio
    currentAudioBlob = null;
    
    // Hide audio player
    document.getElementById('audioPlayback').classList.add('hidden');
    
    // Disable analyze button
    document.getElementById('analyzeBtn').disabled = true;
    
    // Reset upload zone
    document.getElementById('uploadZone').innerHTML = `
        <i class="fas fa-cloud-upload-alt"></i>
        <h3>Upload Audio File</h3>
        <p>Drop your audio file here or click to browse</p>
    `;
    
    // Clear results
    document.getElementById('audioResults').innerHTML = '<p class="no-results">Upload or record audio to see analysis</p>';
    
    // Stop recording if active
    if (isRecording) {
        stopRecording();
    }
}