document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const connectBtn = document.getElementById('connect-btn');
    const recordBtn = document.querySelector('.record-btn');
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    const loadingOverlay = document.getElementById('loading-overlay');
    const videoFeed = document.getElementById('live-feed');
    
    const camFps = document.getElementById('cam-fps');
    const camLatency = document.getElementById('cam-latency');
    
    // State
    let isConnected = false;
    let isRecording = false;
    let telemetryInterval;
    
    // Initialize modes
    const modeBadges = document.querySelectorAll('.mode-badge');
    modeBadges.forEach(badge => {
        badge.addEventListener('click', () => {
            modeBadges.forEach(b => b.classList.remove('active'));
            badge.classList.add('active');
        });
    });

    // Connect Button Handling
    connectBtn.addEventListener('click', () => {
        if (!isConnected) {
            // Start connection
            connectBtn.classList.add('playing');
            connectBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"/></svg>';
            
            loadingOverlay.classList.remove('hidden');
            
            // Simulate connection delay
            setTimeout(() => {
                loadingOverlay.classList.add('hidden');
                isConnected = true;
                
                // Update status UI
                statusDot.classList.remove('disconnected');
                statusDot.classList.add('connected');
                statusText.classList.remove('disconnected');
                statusText.classList.add('connected');
                statusText.textContent = 'LIVE STREAM CONNECTED';
                
                // Start tracking video via getUserMedia if possible
                if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                    navigator.mediaDevices.getUserMedia({ video: true })
                        .then(stream => {
                            videoFeed.srcObject = stream;
                        })
                        .catch(err => {
                            console.warn("Camera access denied or unavailable, using stub.", err);
                            // Fallback dummy styling since it's just black
                        });
                }
                
                startTelemetry();
            }, 1500);
            
        } else {
            // Disconnect
            isConnected = false;
            connectBtn.classList.remove('playing');
            connectBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M5 3l14 9-14 9V3z"/></svg>';
            
            // Update status UI
            statusDot.classList.add('disconnected');
            statusDot.classList.remove('connected');
            statusText.classList.add('disconnected');
            statusText.classList.remove('connected');
            statusText.textContent = 'LIVE STREAM DISCONNECTED';
            
            if (videoFeed.srcObject) {
                videoFeed.srcObject.getTracks().forEach(track => track.stop());
                videoFeed.srcObject = null;
            }
            
            stopTelemetry();
            
            if(isRecording) {
                toggleRecording();
            }
        }
    });

    // Record Button Handling
    recordBtn.addEventListener('click', toggleRecording);
    
    function toggleRecording() {
        if (!isConnected) return; // Only record if stream is connected
        
        isRecording = !isRecording;
        if (isRecording) {
            recordBtn.classList.add('recording');
        } else {
            recordBtn.classList.remove('recording');
        }
    }

    // Telemetry Simulation
    function startTelemetry() {
        telemetryInterval = setInterval(() => {
            // Simulate FPS around 30
            const fps = (29 + Math.random() * 2).toFixed(1);
            camFps.textContent = fps;
            
            // Simulate Latency around 90ms
            const lat = Math.floor(80 + Math.random() * 25);
            camLatency.textContent = lat;
            
            // Simulate pose fluctuations
            const pitch = (2.4 + (Math.random() - 0.5) * 0.2).toFixed(1);
            const roll = (-0.1 + (Math.random() - 0.5) * 0.1).toFixed(1);
            const yaw = (184.2 + (Math.random() - 0.5) * 0.5).toFixed(1);
            
            const poseVals = document.querySelectorAll('.pose-value');
            if (poseVals.length >= 3) {
                poseVals[0].textContent = (pitch >= 0 ? '+' : '') + pitch + '°';
                poseVals[1].textContent = (roll >= 0 ? '+' : '') + roll + '°';
                poseVals[2].textContent = yaw + '°';
            }
        }, 1000);
    }
    
    function stopTelemetry() {
        clearInterval(telemetryInterval);
        camFps.textContent = '--';
        camLatency.textContent = '--';
        
        // Reset pose values
        const poseVals = document.querySelectorAll('.pose-value');
        if (poseVals.length >= 3) {
            poseVals[0].textContent = '+2.4°';
            poseVals[1].textContent = '-0.1°';
            poseVals[2].textContent = '184.2°';
        }
    }
});
