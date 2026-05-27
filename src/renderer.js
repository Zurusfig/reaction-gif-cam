// Canvas rendering: mirrored webcam feed + face landmark + pose skeleton overlay.
// Outputs to an offscreen canvas that can later be piped to a virtual camera.

const FACE_COLOR = 'rgba(0, 220, 120, 0.7)';
const FACE_RADIUS = 1.5;

const POSE_COLOR = 'rgba(255, 80, 180, 0.9)';
const POSE_JOINT_COLOR = 'rgba(255, 220, 0, 0.95)';
const POSE_LINE_WIDTH = 3;
const POSE_JOINT_RADIUS = 5;

// Sparse face contour subset of the 478-pt mesh
const FACE_CONTOUR_INDICES = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
  375, 321, 405, 314, 17, 84, 181, 91, 146,
  33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
  362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
  168, 6, 197, 195, 5, 4, 1, 19, 94,
  70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
];

// Upper-body pose connections (MediaPipe 33-pt model)
const POSE_CONNECTIONS = [
  [11, 12],          // shoulders
  [11, 13], [13, 15], // left arm: shoulder→elbow→wrist
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24],          // hips
];
const POSE_VIS_THRESH = 0.5;

// Hand connections (MediaPipe 21-pt model)
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],        // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],        // index
  [5, 9], [9, 10], [10, 11], [11, 12],   // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20],// pinky
  [0, 17],                               // palm base
];
const HAND_COLOR = 'rgba(80, 200, 255, 0.9)';
const HAND_JOINT_COLOR = 'rgba(255, 255, 255, 0.95)';

export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.offscreen = new OffscreenCanvas(canvas.width, canvas.height);
    this.offCtx = this.offscreen.getContext('2d');
    this.showLandmarks = true;
    this.showSkeleton = true;
  }

  resize(w, h) {
    this.canvas.width = w;
    this.canvas.height = h;
    this.offscreen.width = w;
    this.offscreen.height = h;
  }

  // Draw mirrored webcam + optional face landmarks + pose skeleton + hands
  drawFrame(videoEl, faceLandmarks, poseLandmarks, handLandmarks) {
    const { ctx, canvas } = this;
    const w = canvas.width, h = canvas.height;

    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(videoEl, 0, 0, w, h);
    ctx.restore();

    if (this.showSkeleton && poseLandmarks?.length) {
      this._drawSkeleton(poseLandmarks[0], w, h);
    }
    if (this.showSkeleton && handLandmarks?.length) {
      for (const hand of handLandmarks) this._drawHand(hand, w, h);
    }
    if (this.showLandmarks && faceLandmarks?.length) {
      this._drawFaceLandmarks(faceLandmarks[0], w, h);
    }

    this.offCtx.drawImage(this.canvas, 0, 0);
  }

  _drawHand(lm, w, h) {
    const { ctx } = this;
    const px = pt => (1 - pt.x) * w;  // mirror x
    const py = pt => pt.y * h;

    ctx.strokeStyle = HAND_COLOR;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = lm[a], pb = lm[b];
      if (!pa || !pb) continue;
      ctx.beginPath();
      ctx.moveTo(px(pa), py(pa));
      ctx.lineTo(px(pb), py(pb));
      ctx.stroke();
    }

    ctx.fillStyle = HAND_JOINT_COLOR;
    for (const pt of lm) {
      ctx.beginPath();
      ctx.arc(px(pt), py(pt), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _drawFaceLandmarks(lm, w, h) {
    const { ctx } = this;
    ctx.fillStyle = FACE_COLOR;
    for (const idx of FACE_CONTOUR_INDICES) {
      const pt = lm[idx];
      if (!pt) continue;
      const x = (1 - pt.x) * w;  // mirror to match flipped video
      const y = pt.y * h;
      ctx.beginPath();
      ctx.arc(x, y, FACE_RADIUS, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  _drawSkeleton(lm, w, h) {
    const { ctx } = this;
    const px = pt => (1 - pt.x) * w;  // mirror x
    const py = pt => pt.y * h;

    // Bones
    ctx.strokeStyle = POSE_COLOR;
    ctx.lineWidth = POSE_LINE_WIDTH;
    ctx.lineCap = 'round';
    for (const [a, b] of POSE_CONNECTIONS) {
      const pa = lm[a], pb = lm[b];
      if (!pa || !pb) continue;
      if ((pa.visibility ?? 1) < POSE_VIS_THRESH || (pb.visibility ?? 1) < POSE_VIS_THRESH) continue;
      ctx.beginPath();
      ctx.moveTo(px(pa), py(pa));
      ctx.lineTo(px(pb), py(pb));
      ctx.stroke();
    }

    // Joints (shoulders, elbows, wrists)
    ctx.fillStyle = POSE_JOINT_COLOR;
    for (const idx of [11, 12, 13, 14, 15, 16]) {
      const pt = lm[idx];
      if (!pt || (pt.visibility ?? 1) < POSE_VIS_THRESH) continue;
      ctx.beginPath();
      ctx.arc(px(pt), py(pt), POSE_JOINT_RADIUS, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  async freezeFrame() {
    return createImageBitmap(this.canvas);
  }

  getOffscreenCanvas() {
    return this.offscreen;
  }
}
