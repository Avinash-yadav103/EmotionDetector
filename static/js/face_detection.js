let video = null;
let canvas = null;
let ctx = null;
let stream = null;
let isCapturing = false;

document.addEventListener('DOMContentLoaded', function() {
    video = document.getElementById('video');
    canvas = document.getElementById('overlay');
    
    if (canvas) {
        ctx = canvas.getContext('2d');
    }
    
    const startBtn = document.getElementById('startCamera');
    const stopBtn = document.getElementById('stopCamera');
    const captureBtn = document.getElementById('capturePhoto');
    
    if (startBtn) startBtn.addEventListener('click', startCamera);
    if (stopBtn) stopBtn.addEventListener('click', stopCamera);
    if (captureBtn) captureBtn.addEventListener('click', captureAndAnalyze);
    
    // Check if models are loaded
    checkModelStatus();
});

async function checkModelStatus() {
    try {
        const response = await fetch('/health');
        const status = await response.json();
        
        if (!status.face_model) {
            showError('Face detection model is not loaded. Please check if model files exist.', 'results');
            return;
        }
        
        if (!status.face_cascade) {
            showError('Face cascade is not loaded. Please check if haarcascade file exists.', 'results');
            return;
        }
        
        if (status.face_model && status.face_cascade) {
            updateStatus('Models Loaded', 'success');
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        showError('Cannot connect to server. Please refresh the page.', 'results');
    }
}

async function startCamera() {
    try {
        updateStatus('Connecting...', 'warning');
        
        const constraints = {
            video: { 
                width: { ideal: 640, max: 1280 },
                height: { ideal: 480, max: 720 },
                facingMode: 'user'
            }
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            // Set canvas size to match video
            if (canvas) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                
                // Position canvas to overlay video
                const videoRect = video.getBoundingClientRect();
                canvas.style.position = 'absolute';
                canvas.style.top = '0';
                canvas.style.left = '0';
                canvas.style.width = '100%';
                canvas.style.height = '100%';
                canvas.style.pointerEvents = 'none';
            }
            
            // Update button states
            document.getElementById('startCamera').disabled = true;
            document.getElementById('stopCamera').disabled = false;
            document.getElementById('capturePhoto').disabled = false;
            
            updateStatus('Connected', 'success');
            
            // Clear any previous error messages
            const results = document.getElementById('results');
            if (results && (results.innerHTML.includes('error') || results.innerHTML.includes('Camera stopped'))) {
                results.innerHTML = '<p class="no-results">Camera started. Click "Capture & Analyze" to detect emotions.</p>';
            }
        };
        
        await video.play();
        
    } catch (err) {
        console.error('Error accessing camera:', err);
        updateStatus('Error', 'danger');
        
        let errorMsg = 'Could not access camera. ';
        if (err.name === 'NotAllowedError') {
            errorMsg += 'Please allow camera permissions and try again.';
        } else if (err.name === 'NotFoundError') {
            errorMsg += 'No camera found on this device.';
        } else if (err.name === 'NotReadableError') {
            errorMsg += 'Camera is already in use by another application.';
        } else {
            errorMsg += 'Please check your camera and try again.';
        }
        showError(errorMsg, 'results');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }
    
    // Clear canvas
    if (ctx && canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    // Update button states
    document.getElementById('startCamera').disabled = false;
    document.getElementById('stopCamera').disabled = true;
    document.getElementById('capturePhoto').disabled = true;
    
    updateStatus('Disconnected', 'secondary');
    
    // Clear results
    const results = document.getElementById('results');
    if (results) {
        results.innerHTML = '<p class="no-results">Camera stopped. Click "Start Camera" to begin detection.</p>';
    }
}

async function captureAndAnalyze() {
    if (isCapturing || !video || !video.videoWidth) return;
    
    isCapturing = true;
    showLoading('loading');
    
    try {
        // Create a temporary canvas to capture the current frame
        const captureCanvas = document.createElement('canvas');
        const captureCtx = captureCanvas.getContext('2d');
        
        // Set canvas size to match video
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        
        // Draw the current video frame
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
        
        // Convert to base64
        const imageData = captureCanvas.toDataURL('image/jpeg', 0.8);
        
        // Send to backend for analysis
        const response = await fetch('/predict-face', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error, 'results');
        } else {
            displayFaceResults(data.results);
            drawBoundingBoxes(data.results);
        }
        
    } catch (error) {
        console.error('Error analyzing image:', error);
        showError('Failed to analyze image. Please check your connection and try again.', 'results');
    } finally {
        hideLoading('loading');
        isCapturing = false;
    }
}

function displayFaceResults(results) {
    const resultsContainer = document.getElementById('results');
    if (!resultsContainer) return;
    
    if (results.length === 0) {
        resultsContainer.innerHTML = '<p class="no-results">No faces detected in the image. Please ensure your face is clearly visible and try again.</p>';
        return;
    }
    
    let html = '';
    results.forEach((result, index) => {
        html += `
            <div class="result-card">
                <div class="result-emotion">${result.emotion}</div>
                <div class="result-confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
                
                <div class="probabilities">
                    <h4>All Emotions:</h4>
                    ${Object.entries(result.probabilities)
                        .sort((a, b) => b[1] - a[1])
                        .map(([emotion, prob]) => `
                            <div class="probability-item">
                                <span style="text-transform: capitalize;">${emotion}</span>
                                <div class="probability-bar">
                                    <div class="probability-fill" style="width: ${prob * 100}%"></div>
                                </div>
                                <span>${(prob * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                </div>
            </div>
        `;
    });
    
    resultsContainer.innerHTML = html;
}

function drawBoundingBoxes(results) {
    if (!ctx || !canvas || !video) return;
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate scale factors
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    
    results.forEach(result => {
        const [x, y, w, h] = result.bbox;
        
        // Scale coordinates to canvas size
        const scaledX = x * scaleX;
        const scaledY = y * scaleY;
        const scaledW = w * scaleX;
        const scaledH = h * scaleY;
        
        // Draw bounding box
        ctx.strokeStyle = '#28a745';
        ctx.lineWidth = 3;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
        
        // Prepare label text
        const text = `${result.emotion} (${(result.confidence * 100).toFixed(0)}%)`;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        
        // Measure text width
        const textMetrics = ctx.measureText(text);
        const textWidth = textMetrics.width;
        const textHeight = 20;
        
        // Draw label background
        ctx.fillStyle = '#28a745';
        ctx.fillRect(scaledX, scaledY - textHeight - 5, textWidth + 10, textHeight + 5);
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(text, scaledX + 5, scaledY - 8);
    });
}

function updateStatus(text, type) {
    const statusBadge = document.getElementById('status-indicator');
    if (statusBadge) {
        statusBadge.textContent = text;
        statusBadge.className = `badge bg-${type} ms-2`;
    }
}

function showError(message, containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
    }
}

function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.remove('hidden');
    }
}

function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.add('hidden');
    }
}

// Handle page unload - clean up camera
window.addEventListener('beforeunload', function() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});