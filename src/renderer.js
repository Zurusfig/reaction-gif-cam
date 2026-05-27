// Canvas rendering: webcam feed + landmark overlay + GIF overlay
// Outputs to an offscreen canvas that can later be piped to a virtual camera.

const LANDMARK_COLOR = 'rgba(0, 220, 120, 0.7)';
const LANDMARK_RADIUS = 1.5;

// Key landmark indices to draw (subset for clarity — full mesh = 478 pts)
// Using a sparse subset for the overlay: contours only
const FACE_CONTOUR_INDICES = [
  // Jawline
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
  // Lips outer
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
  375, 321, 405, 314, 17, 84, 181, 91, 146,
  // Left eye
  33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
  // Right eye
  362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
  // Nose bridge
  168, 6, 197, 195, 5, 4, 1, 19, 94,
  // Brows
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
];

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    // Offscreen canvas for future virtual-camera piping
    this.offscreen = new OffscreenCanvas(canvas.width, canvas.height);
    this.offCtx = this.offscreen.getContext('2d');
    this.showLandmarks = true;
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.offscreen.width = w;
    this.offscreen.height = h;
  }

  // Draw mirrored webcam frame + optional landmark overlay
  drawFrame(videoEl, landmarks) {
    const { ctx, canvas } = this;
    const w = canvas.width, h = canvas.height;

    ctx.save();
    // Mirror (selfie view)
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(videoEl, 0, 0, w, h);
    ctx.restore();

    if (this.showLandmarks && landmarks?.length) {
      this._drawLandmarks(landmarks[0], w, h);
    }

    // Sync to offscreen for future virtual camera output
    this.offCtx.drawImage(this.canvas, 0, 0);
  }

  _drawLandmarks(lm, w, h) {
    const { ctx } = this;
    ctx.fillStyle = LANDMARK_COLOR;
    for (const idx of FACE_CONTOUR_INDICES) {
      const pt = lm[idx];
      if (!pt) continue;
      // x is mirrored to match the flipped video
      const x = (1 - pt.x) * w;
      const y = pt.y * h;
      ctx.beginPath();
      ctx.arc(x, y, LANDMARK_RADIUS, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Freeze current canvas into an ImageBitmap for transition use
  async freezeFrame() {
    return createImageBitmap(this.canvas);
  }

  // Returns the offscreen canvas (future: pipe to virtual camera)
  getOffscreenCanvas() {
    return this.offscreen;
  }
}
