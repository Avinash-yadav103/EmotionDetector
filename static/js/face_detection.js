let video = null;
let canvas = null;
let ctx = null;
let stream = null;
let isCapturing = false;

document.addEventListener('DOMContentLoaded', function() {
    video = document.getElementById('video');
    canvas = document.getElementById('overlay');
    ctx = canvas.getContext('2d');
    
    const startBtn = document.getElementById('startCamera');
    const stopBtn = document.getElementById('stopCamera');
    const captureBtn = document.getElementById('capturePhoto');
    
    startBtn.addEventListener('click', startCamera);
    stopBtn.addEventListener('click', stopCamera);
    captureBtn.addEventListener('click', captureAndAnalyze);
});

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        
        video.srcObject = stream;
        video.play();
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };
        
        document.getElementById('startCamera').disabled = true;
        document.getElementById('stopCamera').disabled = false;
        document.getElementById('capturePhoto').disabled = false;
        
    } catch (err) {
        console.error('Error accessing camera:', err);
        showError('Could not access camera. Please check permissions.', 'results');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    document.getElementById('startCamera').disabled = false;
    document.getElementById('stopCamera').disabled = true;
    document.getElementById('capturePhoto').disabled = true;
    
    // Clear results
    document.getElementById('results').innerHTML = '<p class="no-results">Camera stopped</p>';
}

async function captureAndAnalyze() {
    if (isCapturing) return;
    
    isCapturing = true;
    showLoading('loading');
    
    try {
        // Create a temporary canvas to capture the current frame
        const captureCanvas = document.createElement('canvas');
        const captureCtx = captureCanvas.getContext('2d');
        
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        
        // Draw the current video frame
        captureCtx.drawImage(video, 0, 0);
        
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
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error, 'results');
        } else {
            displayFaceResults(data.results);
            drawBoundingBoxes(data.results);
        }
        
    } catch (error) {
        console.error('Error analyzing image:', error);
        showError('Failed to analyze image. Please try again.', 'results');
    } finally {
        hideLoading('loading');
        isCapturing = false;
    }
}

function displayFaceResults(results) {
    const resultsContainer = document.getElementById('results');
    
    if (results.length === 0) {
        resultsContainer.innerHTML = '<p class="no-results">No faces detected in the image</p>';
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
    });
    
    resultsContainer.innerHTML = html;
}

function drawBoundingBoxes(results) {
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    results.forEach(result => {
        const [x, y, w, h] = result.bbox;
        
        // Draw bounding box
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        
        // Draw label background
        ctx.fillStyle = '#ff6b6b';
        ctx.fillRect(x, y - 30, w, 30);
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.font = '16px Inter';
        ctx.fillText(`${result.emotion} (${(result.confidence * 100).toFixed(0)}%)`, x + 5, y - 10);
    });
}

// Clear bounding boxes when camera stops
document.getElementById('stopCamera').addEventListener('click', () => {
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
});