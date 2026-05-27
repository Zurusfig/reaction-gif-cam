import { initDetector, detectFrame, classifyExpression, classifyMotion, updateMotionBuffer } from './detector.js';
import { Renderer } from './renderer.js';
import './style.css';

// ─── DOM ─────────────────────────────────────────────────────────────────────

const video = document.getElementById('webcam');
const mainCanvas = document.getElementById('main-canvas');
const overlayCanvas = document.getElementById('overlay-canvas');
const debugExpression = document.getElementById('debug-expression');
const debugMotion = document.getElementById('debug-motion');
const debugTrigger = document.getElementById('debug-trigger');
const debugFps = document.getElementById('debug-fps');
const statusEl = document.getElementById('status');

// ─── State ───────────────────────────────────────────────────────────────────

const renderer = new Renderer(mainCanvas);

let frameCount = 0;
let fpsTimer = 0;

// ─── Webcam Setup ─────────────────────────────────────────────────────────────

async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise(res => { video.onloadedmetadata = res; });
  await video.play();

  const { videoWidth: w, videoHeight: h } = video;
  renderer.resize(w, h);
  overlayCanvas.width = w;
  overlayCanvas.height = h;
}

// ─── Main Loop ────────────────────────────────────────────────────────────────

function mainLoop(timestampMs) {
  if (video.readyState < 2) { requestAnimationFrame(mainLoop); return; }

  frameCount++;
  if (timestampMs - fpsTimer >= 1000) {
    debugFps.textContent = frameCount;
    frameCount = 0;
    fpsTimer = timestampMs;
  }

  const result = detectFrame(video, timestampMs);

  let expression = 'neutral';
  let motion = 'still';

  if (result?.faceLandmarks?.length) {
    updateMotionBuffer(result.faceLandmarks);
    expression = classifyExpression(result.faceBlendshapes);
    motion = classifyMotion(result.faceLandmarks);
    renderer.drawFrame(video, result.faceLandmarks);
  } else {
    renderer.drawFrame(video, null);
    expression = 'no-face';
  }

  debugExpression.textContent = expression;
  debugMotion.textContent = motion;

  requestAnimationFrame(mainLoop);
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

async function boot() {
  statusEl.textContent = 'Requesting camera…';
  await startWebcam();

  statusEl.textContent = 'Loading face model…';
  await initDetector();

  statusEl.textContent = '';
  document.getElementById('status-bar').style.display = 'none';

  requestAnimationFrame(mainLoop);
}

boot().catch(err => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message}`;
});
