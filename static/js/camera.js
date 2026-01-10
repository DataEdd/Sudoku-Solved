// camera.js - WebRTC camera management

const Camera = {
    stream: null,
    videoElement: null,
    canvasElement: null,
    flashOn: false,

    // Initialize camera
    async initialize(videoId, canvasId) {
        this.videoElement = document.getElementById(videoId);
        this.canvasElement = document.getElementById(canvasId);

        if (!this.videoElement || !this.canvasElement) {
            console.error('Camera elements not found');
            return false;
        }

        try {
            // Stop any existing stream
            this.stop();

            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });

            this.videoElement.srcObject = this.stream;
            await this.videoElement.play();
            return true;
        } catch (err) {
            console.error('Camera error:', err);
            return false;
        }
    },

    // Capture current frame
    capture() {
        if (!this.stream || !this.videoElement) {
            return null;
        }

        const ctx = this.canvasElement.getContext('2d');
        this.canvasElement.width = this.videoElement.videoWidth;
        this.canvasElement.height = this.videoElement.videoHeight;
        ctx.drawImage(this.videoElement, 0, 0);

        return new Promise(resolve => {
            this.canvasElement.toBlob(
                blob => resolve(blob),
                'image/jpeg',
                0.9
            );
        });
    },

    // Get image as data URL
    getImageDataUrl() {
        if (!this.canvasElement) return null;
        return this.canvasElement.toDataURL('image/jpeg', 0.9);
    },

    // Stop camera stream
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.videoElement) {
            this.videoElement.srcObject = null;
        }
        this.flashOn = false;
    },

    // Toggle flash (torch) if supported
    async toggleFlash() {
        if (!this.stream) return false;

        const track = this.stream.getVideoTracks()[0];
        if (!track) return false;

        try {
            const capabilities = track.getCapabilities();

            if (capabilities.torch) {
                this.flashOn = !this.flashOn;
                await track.applyConstraints({
                    advanced: [{ torch: this.flashOn }]
                });
                return this.flashOn;
            }
        } catch (err) {
            console.error('Flash toggle error:', err);
        }

        return false;
    },

    // Check if camera is available
    isAvailable() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    },

    // Check if flash is supported
    async isFlashSupported() {
        if (!this.stream) return false;

        const track = this.stream.getVideoTracks()[0];
        if (!track) return false;

        try {
            const capabilities = track.getCapabilities();
            return !!capabilities.torch;
        } catch {
            return false;
        }
    }
};
