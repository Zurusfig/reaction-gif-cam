import { initDetector, detectFrame, classifyExpression, classifyMotion, updateMotionBuffer } from './detector.js';
import { Renderer } from './renderer.js';
import { loadGifs, matchGif } from './database.js';
import { preloadGif, crossfadeIn, drawGifFrame, crossfadeOut } from './transition.js';
import './style.css';

// ─── DOM ─────────────────────────────────────────────────────────────────────

const video = document.getElementById('webcam');
const mainCanvas = document.getElementById('main-canvas');
const overlayCanvas = document.getElementById('overlay-canvas');
const debugExpression = document.getElementById('debug-expression');
const debugMotion = document.getElementById('debug-motion');
const debugHands = document.getElementById('debug-hands');
const debugTrigger = document.getElementById('debug-trigger');
const debugFps = document.getElementById('debug-fps');
const statusEl = document.getElementById('status');

// ─── App state ───────────────────────────────────────────────────────────────

const STATE = Object.freeze({
  LIVE: 'live',
  HELD: 'held',       // waiting for 500ms stable hold
  PLAYING: 'playing', // GIF active (fade-in, show, fade-out handled async)
  COOLDOWN: 'cooldown',
});

const HOLD_MS = 500;
const PLAY_MS = 3000;
const COOLDOWN_MS = 3000;

const renderer = new Renderer(mainCanvas);

let appState = STATE.LIVE;
let heldSince = 0;
let heldExpr = '';
let heldMotion = '';
let cooldownUntil = 0;

// GIF playback — managed by triggerPlayback(), read by main loop
let gifImg = null;          // loaded Image element
let gifStartedAt = 0;       // when PLAYING state began

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
  const match = matchGif({ expression: expr, motion });
  if (!match) {
    // No match — silently return to LIVE
    appState = STATE.LIVE;
    return;
  }

  debugTrigger.textContent = `${expr} + ${motion} → ${match.id}`;

  // Load GIF (fast if cached by browser)
  const img = await preloadGif(match.url);
  const frozen = await renderer.freezeFrame();

  appState = STATE.PLAYING;
  gifImg = img;
  gifStartedAt = performance.now();

  // Transition: frozen webcam → GIF
  await crossfadeIn(frozen, img, overlayCanvas);

  // Hold GIF for remaining play time after fade-in (fade-in counts toward PLAY_MS)
  const elapsed = performance.now() - gifStartedAt;
  const remaining = Math.max(0, PLAY_MS - elapsed);
  await new Promise(res => setTimeout(res, remaining));

  // Transition: GIF → transparent (live webcam shows through)
  await crossfadeOut(img, overlayCanvas);

  gifImg = null;
  appState = STATE.COOLDOWN;
  cooldownUntil = performance.now() + COOLDOWN_MS;
}

// ─── Main loop ────────────────────────────────────────────────────────────────

function mainLoop(ts) {
  if (video.readyState < 2) { requestAnimationFrame(mainLoop); return; }

  // FPS
  frameCount++;
  if (ts - fpsTimer >= 1000) {
    debugFps.textContent = frameCount;
    frameCount = 0;
    fpsTimer = ts;
  }

  // ── COOLDOWN: just keep webcam running, no detection ──────────────────────
  if (appState === STATE.COOLDOWN) {
    renderer.drawFrame(video, null);
    if (ts >= cooldownUntil) appState = STATE.LIVE;
    requestAnimationFrame(mainLoop);
    return;
  }

  // ── PLAYING: overlay handles the GIF; keep webcam rendering underneath ────
  if (appState === STATE.PLAYING) {
    renderer.drawFrame(video, null);
    // drawGifFrame is called by the async transition functions via their own rAF;
    // we don't draw here to avoid fighting with the transition's rAF ticks.
    requestAnimationFrame(mainLoop);
    return;
  }

  // ── LIVE / HELD: run detection ─────────────────────────────────────────────
  const result = detectFrame(video, ts);
  const faceLandmarks = result?.face?.faceLandmarks;
  const faceBlendshapes = result?.face?.faceBlendshapes;
  const handLandmarks = result?.hand?.landmarks;

  let expression = 'neutral';
  let motion = 'still';

  if (faceLandmarks?.length) {
    updateMotionBuffer(faceLandmarks);
    expression = classifyExpression(faceBlendshapes);
    motion = classifyMotion(faceLandmarks, handLandmarks);
    renderer.drawFrame(video, faceLandmarks);
  } else {
    renderer.drawFrame(video, null);
    expression = 'no-face';
  }

  debugExpression.textContent = expression;
  debugMotion.textContent = motion;
  debugHands.textContent = handLandmarks?.length
    ? `${handLandmarks.length} hand${handLandmarks.length > 1 ? 's' : ''}`
    : 'none';

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
    // Reset hold if state changed
    if (expression !== heldExpr || motion !== heldMotion) {
      heldSince = ts;
      heldExpr = expression;
      heldMotion = motion;
    }
    if (ts - heldSince >= HOLD_MS) {
      appState = STATE.PLAYING; // set early so loop doesn't re-trigger
      triggerPlayback(heldExpr, heldMotion);
    }
  }

  requestAnimationFrame(mainLoop);
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

async function boot() {
  statusEl.textContent = 'Requesting camera…';
  await startWebcam();

  statusEl.textContent = 'Loading face model…';
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
