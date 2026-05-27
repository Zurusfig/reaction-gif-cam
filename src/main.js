import { initDetector, detectFrame, classifyExpression, classifyMotion, classifyHandGesture, updateMotionBuffer, getBlendshapeScores } from './detector.js';
import { Renderer } from './renderer.js';
import { loadGifs, matchGif } from './database.js';
import { crossfadeIn, crossfadeOut } from './transition.js';
import './style.css';

// ─── DOM ─────────────────────────────────────────────────────────────────────

const video         = document.getElementById('webcam');
const mainCanvas    = document.getElementById('main-canvas');
const overlayCanvas = document.getElementById('overlay-canvas');
const gifOverlay    = document.getElementById('gif-overlay');
const debugExpression = document.getElementById('debug-expression');
const debugMotion     = document.getElementById('debug-motion');
const debugInfo       = document.getElementById('debug-hands');
const debugTrigger    = document.getElementById('debug-trigger');
const debugFps        = document.getElementById('debug-fps');
const statusEl        = document.getElementById('status');
const logEl           = document.getElementById('event-log');

// ─── On-screen logger (so we can debug without the console) ───────────────────

const logLines = [];
function log(msg) {
  const line = `${(performance.now() / 1000).toFixed(1)}s  ${msg}`;
  logLines.unshift(line);
  if (logLines.length > 8) logLines.pop();
  if (logEl) logEl.textContent = logLines.join('\n');
  console.log('[reaction]', msg);
}

// ─── App state ───────────────────────────────────────────────────────────────

const STATE = Object.freeze({ LIVE: 'live', HELD: 'held', PLAYING: 'playing', COOLDOWN: 'cooldown' });

const HOLD_MS     = 500;
const PLAY_MS     = 3000;
const COOLDOWN_MS = 3000;
const FADE_MS     = 500;

const renderer = new Renderer(mainCanvas);

let gifDb = [];
let appState = STATE.LIVE;
let heldSince = 0;
let heldExpr = '';
let heldMotion = '';
let heldGesture = '';
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
  log(`webcam ${w}×${h}`);
}

// ─── Playback ──────────────────────────────────────────────────────────────────
// Plays a GIF directly by URL — no dependency on preload succeeding.

async function playMatch(match, reason) {
  try {
    log(`PLAY ${match.id} (${reason})`);
    debugTrigger.textContent = `${match.id} (${reason})`;

    // Point the overlay img at the GIF and ensure it's loaded enough to show
    gifOverlay.src = match.url;
    if (!gifOverlay.complete || gifOverlay.naturalWidth === 0) {
      await new Promise((res, rej) => {
        gifOverlay.onload = res;
        gifOverlay.onerror = () => rej(new Error(`load failed: ${match.url}`));
      });
    }
    log(`  img ready ${gifOverlay.naturalWidth}×${gifOverlay.naturalHeight}`);

    const frozen = await renderer.freezeFrame();
    await crossfadeIn(frozen, gifOverlay, overlayCanvas);
    log('  faded in, holding');

    await wait(Math.max(0, PLAY_MS - FADE_MS));
    await crossfadeOut(gifOverlay);
    log('  done');

    appState = STATE.COOLDOWN;
    cooldownUntil = performance.now() + COOLDOWN_MS;
  } catch (err) {
    log(`  ERROR: ${err.message}`);
    gifOverlay.style.display = 'none';
    overlayCanvas.getContext('2d').clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    appState = STATE.LIVE;
  }
}

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

async function triggerFromDetection(expr, motion, gesture, jawOpen) {
  const match = matchGif({ expression: expr, motion, gesture, jawOpen });
  if (!match) {
    log(`no match for ${expr}+${motion}+${gesture}`);
    appState = STATE.LIVE;
    return;
  }
  await playMatch(match, `${expr}+${motion}+${gesture}`);
}

// ─── Manual keyboard trigger (bypasses detection) ─────────────────────────────
// Press 1-9 to force-play the Nth GIF. Useful for testing playback in isolation.

window.addEventListener('keydown', e => {
  const n = parseInt(e.key, 10);
  if (Number.isNaN(n) || n < 1 || n > gifDb.length) return;
  if (appState === STATE.PLAYING) return;
  appState = STATE.PLAYING;
  playMatch(gifDb[n - 1], `key ${n}`);
});

// ─── Main loop ────────────────────────────────────────────────────────────────

function mainLoop(ts) {
  if (video.readyState < 2) { requestAnimationFrame(mainLoop); return; }

  frameCount++;
  if (ts - fpsTimer >= 1000) {
    debugFps.textContent = frameCount;
    frameCount = 0;
    fpsTimer = ts;
  }

  // Always run detection so the skeleton stays live even during playback
  const result        = detectFrame(video, ts);
  const faceLandmarks = result?.face?.faceLandmarks;
  const faceBlend     = result?.face?.faceBlendshapes;
  const poseLandmarks = result?.pose?.landmarks;
  const handLandmarks = result?.hand?.landmarks;

  renderer.drawFrame(video, faceLandmarks, poseLandmarks, handLandmarks);

  // While a GIF is playing or cooling down, skip the trigger logic
  if (appState === STATE.PLAYING) { requestAnimationFrame(mainLoop); return; }
  if (appState === STATE.COOLDOWN) {
    if (ts >= cooldownUntil) { appState = STATE.LIVE; log('cooldown end'); }
    requestAnimationFrame(mainLoop);
    return;
  }

  let expression = 'no-face';
  let motion = 'still';
  let gesture = 'none';
  let jawOpenScore = 0;

  if (faceLandmarks?.length) {
    updateMotionBuffer(faceLandmarks);
    expression = classifyExpression(faceBlend);
    motion = classifyMotion(faceLandmarks, poseLandmarks);
    gesture = classifyHandGesture(handLandmarks, faceLandmarks);
  }

  const sc = getBlendshapeScores(faceBlend);
  jawOpenScore = parseFloat(sc.jawOpen ?? 0);

  debugExpression.textContent = expression;
  debugMotion.textContent = motion;
  debugInfo.textContent = `jaw:${sc.jawOpen ?? '—'} smile:${sc.mouthSmileLeft ?? '—'} brow:${sc.browOuterUpLeft ?? '—'} blink:${sc.eyeBlinkLeft ?? '—'} lookUp:${sc.eyeLookUpLeft ?? '—'} pose:${poseLandmarks?.length ? '✓' : '✗'} hand:${gesture}`;

  // ── Hold / trigger ─────────────────────────────────────────────────────────
  // Idle only when nothing is happening: no face, OR neutral face AND no motion AND no gesture.
  const idle = expression === 'no-face' || (expression === 'neutral' && motion === 'still' && gesture === 'none');
  if (idle) {
    appState = STATE.LIVE;
    heldExpr = '';
    heldMotion = '';
    heldGesture = '';
  } else if (appState === STATE.LIVE) {
    appState = STATE.HELD;
    heldSince = ts;
    heldExpr = expression;
    heldMotion = motion;
    heldGesture = gesture;
  } else if (appState === STATE.HELD) {
    if (expression !== heldExpr || motion !== heldMotion || gesture !== heldGesture) {
      heldSince = ts;
      heldExpr = expression;
      heldMotion = motion;
      heldGesture = gesture;
    }
    if (ts - heldSince >= HOLD_MS) {
      appState = STATE.PLAYING;
      triggerFromDetection(heldExpr, heldMotion, heldGesture, jawOpenScore);
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
  log('models loaded');

  gifDb = await loadGifs();
  log(`${gifDb.length} gifs in db (press 1-${gifDb.length} to test playback)`);

  statusEl.textContent = '';
  document.getElementById('status-bar').style.display = 'none';

  requestAnimationFrame(mainLoop);
}

boot().catch(err => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message}`;
  log(`BOOT ERROR: ${err.message}`);
});
