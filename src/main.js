import { initDetector, detectFrame, classifyExpression, classifyMotion, updateMotionBuffer, getBlendshapeScores } from './detector.js';
import { Renderer } from './renderer.js';
import { loadGifs, matchGif } from './database.js';
import { preloadGif, crossfadeIn, crossfadeOut } from './transition.js';
import './style.css';

// ─── DOM ─────────────────────────────────────────────────────────────────────

const video       = document.getElementById('webcam');
const mainCanvas  = document.getElementById('main-canvas');
const overlayCanvas = document.getElementById('overlay-canvas');
const gifOverlay  = document.getElementById('gif-overlay');   // native <img> for GIF
const debugExpression = document.getElementById('debug-expression');
const debugMotion     = document.getElementById('debug-motion');
const debugInfo       = document.getElementById('debug-hands');
const debugTrigger    = document.getElementById('debug-trigger');
const debugFps        = document.getElementById('debug-fps');
const statusEl        = document.getElementById('status');

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
  overlayCanvas.width  = w;
  overlayCanvas.height = h;
}

// ─── Preload all GIF images at boot ──────────────────────────────────────────
// Stores loaded HTMLImageElement on each db entry so playback is instant.

async function preloadAllGifs(gifs) {
  await Promise.all(
    gifs.map(async g => {
      try {
        g._img = await preloadGif(g.url);
      } catch {
        console.warn(`Could not preload ${g.url}`);
      }
    })
  );
}

// ─── Trigger + playback ───────────────────────────────────────────────────────

async function triggerPlayback(expr, motion) {
  try {
    const match = matchGif({ expression: expr, motion });
    if (!match) { appState = STATE.LIVE; return; }
    if (!match._img) { appState = STATE.LIVE; return; } // failed to preload

    debugTrigger.textContent = `${expr} + ${motion} → ${match.id}`;

    // Set the GIF src on the overlay img element BEFORE freezing
    gifOverlay.src = match._img.src;

    const frozen = await renderer.freezeFrame();

    // Crossfade: frozen canvas → GIF img element
    await crossfadeIn(frozen, gifOverlay, overlayCanvas);

    // GIF img is now fully visible and animating natively — just wait
    await new Promise(res => setTimeout(res, Math.max(0, PLAY_MS - 500)));

    // Fade out GIF img
    await crossfadeOut(gifOverlay);

    appState = STATE.COOLDOWN;
    cooldownUntil = performance.now() + COOLDOWN_MS;
  } catch (err) {
    console.error('GIF playback failed:', err);
    debugTrigger.textContent = `ERR: ${err.message}`;
    gifOverlay.style.display = 'none';
    overlayCanvas.getContext('2d').clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
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

  // ── LIVE / HELD: detection ─────────────────────────────────────────────────
  const result        = detectFrame(video, ts);
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

  const sc = getBlendshapeScores(faceBlendshapes);
  debugInfo.textContent = `jaw:${sc.jawOpen ?? '—'} smile:${sc.mouthSmileLeft ?? '—'} brow:${sc.browOuterUpLeft ?? '—'} blink:${sc.eyeBlinkLeft ?? '—'} lookUp:${sc.eyeLookUpLeft ?? '—'} pose:${poseLandmarks?.length ? '✓' : '✗'}`;

  // ── Hold / trigger ─────────────────────────────────────────────────────────
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

  statusEl.textContent = 'Loading GIFs…';
  const gifs = await loadGifs();
  await preloadAllGifs(gifs);

  statusEl.textContent = '';
  document.getElementById('status-bar').style.display = 'none';

  requestAnimationFrame(mainLoop);
}

boot().catch(err => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message}`;
});
