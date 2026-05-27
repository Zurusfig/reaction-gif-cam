import { initDetector, detectFrame, classifyExpression, classifyMotion, updateMotionBuffer, getBlendshapeScores } from './detector.js';
import { Renderer } from './renderer.js';
import { loadGifs, matchGif } from './database.js';
import { preloadGif, crossfadeIn, crossfadeOut } from './transition.js';
import './style.css';

// ─── DOM ─────────────────────────────────────────────────────────────────────

const video = document.getElementById('webcam');
const mainCanvas = document.getElementById('main-canvas');
const overlayCanvas = document.getElementById('overlay-canvas');
const debugExpression = document.getElementById('debug-expression');
const debugMotion = document.getElementById('debug-motion');
const debugInfo = document.getElementById('debug-hands');
const debugTrigger = document.getElementById('debug-trigger');
const debugFps = document.getElementById('debug-fps');
const statusEl = document.getElementById('status');

// ─── App state ───────────────────────────────────────────────────────────────

const STATE = Object.freeze({
  LIVE: 'live',
  HELD: 'held',
  PLAYING: 'playing',
  COOLDOWN: 'cooldown',
});

const HOLD_MS     = 500;
const PLAY_MS     = 3000;
const COOLDOWN_MS = 3000;

const renderer = new Renderer(mainCanvas);

let appState = STATE.LIVE;
let heldSince = 0;
let heldExpr = '';
let heldMotion = '';
let cooldownUntil = 0;

let frameCount = 0;
let fpsTimer = 0;

// ─── Webcam ───────────────────────────────────────────────────────────────────

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

// ─── Trigger + playback ───────────────────────────────────────────────────────

async function triggerPlayback(expr, motion) {
  try {
    const match = matchGif({ expression: expr, motion });
    if (!match) {
      appState = STATE.LIVE;
      return;
    }

    debugTrigger.textContent = `${expr} + ${motion} → ${match.id}`;

    const img = await preloadGif(match.url);
    const frozen = await renderer.freezeFrame();

    const gifStartedAt = performance.now();
    await crossfadeIn(frozen, img, overlayCanvas);

    const elapsed = performance.now() - gifStartedAt;
    const remaining = Math.max(0, PLAY_MS - elapsed);
    await new Promise(res => setTimeout(res, remaining));

    await crossfadeOut(img, overlayCanvas);

    appState = STATE.COOLDOWN;
    cooldownUntil = performance.now() + COOLDOWN_MS;
  } catch (err) {
    console.error('GIF playback failed:', err);
    debugTrigger.textContent = `Error: ${err.message}`;
    // Always reset so the app doesn't get stuck in PLAYING
    appState = STATE.LIVE;
  }
}

// ─── Main loop ────────────────────────────────────────────────────────────────

function mainLoop(ts) {
  if (video.readyState < 2) { requestAnimationFrame(mainLoop); return; }

  frameCount++;
  if (ts - fpsTimer >= 1000) {
    debugFps.textContent = frameCount;
    frameCount = 0;
    fpsTimer = ts;
  }

  if (appState === STATE.COOLDOWN) {
    renderer.drawFrame(video, null);
    if (ts >= cooldownUntil) appState = STATE.LIVE;
    requestAnimationFrame(mainLoop);
    return;
  }

  if (appState === STATE.PLAYING) {
    renderer.drawFrame(video, null);
    requestAnimationFrame(mainLoop);
    return;
  }

  // ── LIVE / HELD: run detection ─────────────────────────────────────────────
  const result = detectFrame(video, ts);
  const faceLandmarks = result?.face?.faceLandmarks;
  const faceBlendshapes = result?.face?.faceBlendshapes;
  const poseLandmarks = result?.pose?.landmarks;

  let expression = 'neutral';
  let motion = 'still';

  if (faceLandmarks?.length) {
    updateMotionBuffer(faceLandmarks);
    expression = classifyExpression(faceBlendshapes);
    motion = classifyMotion(faceLandmarks, poseLandmarks);
    renderer.drawFrame(video, faceLandmarks);
  } else {
    renderer.drawFrame(video, null);
    expression = 'no-face';
  }

  debugExpression.textContent = expression;
  debugMotion.textContent = motion;

  // Show key raw scores in debug for tuning
  const scores = getBlendshapeScores(faceBlendshapes);
  debugInfo.textContent = `jaw:${scores.jawOpen ?? '—'}  smile:${scores.mouthSmileLeft ?? '—'}  brow:${scores.browOuterUpLeft ?? '—'}  lookUp:${scores.eyeLookUpLeft ?? '—'}  pose:${poseLandmarks?.length ? '✓' : '✗'}`;

  // ── Hold / trigger logic ──────────────────────────────────────────────────
  if (expression === 'no-face' || expression === 'neutral') {
    appState = STATE.LIVE;
    heldExpr = '';
    heldMotion = '';
  } else if (appState === STATE.LIVE) {
    appState = STATE.HELD;
    heldSince = ts;
    heldExpr = expression;
    heldMotion = motion;
  } else if (appState === STATE.HELD) {
    if (expression !== heldExpr || motion !== heldMotion) {
      heldSince = ts;
      heldExpr = expression;
      heldMotion = motion;
    }
    if (ts - heldSince >= HOLD_MS) {
      appState = STATE.PLAYING;
      triggerPlayback(heldExpr, heldMotion);
    }
  }

  requestAnimationFrame(mainLoop);
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

async function boot() {
  statusEl.textContent = 'Requesting camera…';
  await startWebcam();

  statusEl.textContent = 'Loading models…';
  await initDetector();
  await loadGifs();

  statusEl.textContent = '';
  document.getElementById('status-bar').style.display = 'none';

  requestAnimationFrame(mainLoop);
}

boot().catch(err => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message}`;
});
