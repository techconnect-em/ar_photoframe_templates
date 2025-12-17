// OneEuroFilter Class
class OneEuroFilter {
    constructor(fps, minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
        this.fps = fps;
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;
        this.x = null;
        this.dx = null;
        this.lastTime = null;
    }

    filter(value, timestamp) {
        if (this.lastTime && timestamp) {
            this.fps = 1.0 / ((timestamp - this.lastTime) / 1000.0);
        }
        this.lastTime = timestamp;

        if (this.x === null) {
            this.x = value;
            this.dx = 0;
            return value;
        }

        const dx = (value - this.x) * this.fps;
        const edx = this.alpha(this.dCutoff) * dx + (1 - this.alpha(this.dCutoff)) * this.dx;
        this.dx = edx;

        const cutoff = this.minCutoff + this.beta * Math.abs(edx);
        const result = this.alpha(cutoff) * value + (1 - this.alpha(cutoff)) * this.x;
        this.x = result;
        return result;
    }

    alpha(cutoff) {
        const te = 1.0 / this.fps;
        const tau = 1.0 / (2 * Math.PI * cutoff);
        return 1.0 / (1.0 + tau / te);
    }

    reset() {
        this.x = null;
        this.dx = null;
        this.lastTime = null;
    }
}

// Fade Logic Class
class SmoothedLandmark {
    constructor(config = {}) {
        this.filters = {
            x: new OneEuroFilter(30, config.minCutoff || 0.01, config.beta || 10.0), // High beta for responsiveness
            y: new OneEuroFilter(30, config.minCutoff || 0.01, config.beta || 10.0),
            width: new OneEuroFilter(30, config.minCutoff || 0.1, config.beta || 5.0),
            height: new OneEuroFilter(30, config.minCutoff || 0.1, config.beta || 5.0),
            angle: new OneEuroFilter(30, config.minCutoff || 0.1, config.beta || 5.0),
        };
        this.lastTransform = null;
        this.lastSeen = 0;
        this.fadeDuration = 200; // ms
        this.opacity = 0;
    }

    update(currTransform, timestamp) {
        if (currTransform) {
            this.lastSeen = timestamp;
            this.opacity = 1;

            this.lastTransform = {
                x: this.filters.x.filter(currTransform.x, timestamp),
                y: this.filters.y.filter(currTransform.y, timestamp),
                width: this.filters.width.filter(currTransform.width, timestamp),
                height: this.filters.height.filter(currTransform.height, timestamp),
                angle: this.filters.angle.filter(currTransform.angle, timestamp)
            };
            return { ...this.lastTransform, opacity: 1 };
        } else {
            // Logic for fading out
            if (!this.lastTransform) return null;

            const timeSince = timestamp - this.lastSeen;
            if (timeSince > this.fadeDuration) {
                this.opacity = 0;
                this.filters.x.reset(); // Reset filters if lost for too long
                this.lastTransform = null;
                return null;
            }

            this.opacity = 1 - (timeSince / this.fadeDuration);
            return { ...this.lastTransform, opacity: this.opacity };
        }
    }
}
