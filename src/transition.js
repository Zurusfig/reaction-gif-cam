// Transition module — swappable strategy: crossfade (now) → Delaunay morph (later)
// Public API: preloadGif, crossfadeIn, crossfadeOut

const FADE_IN_MS = 500;
const FADE_OUT_MS = 500;

export async function preloadGif(url) {
  const img = new Image();
  img.src = url;
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
  return img;
}

// Crossfade from a frozen webcam frame into the GIF over FADE_IN_MS.
// Returns when the fade is complete; the gif img is left at full opacity.
export async function crossfadeIn(frozenFrame, gifImg, overlayCanvas) {
  const ctx = overlayCanvas.getContext('2d');
  const w = overlayCanvas.width, h = overlayCanvas.height;
  const start = performance.now();

  return new Promise(resolve => {
    function tick() {
      const t = Math.min((performance.now() - start) / FADE_IN_MS, 1);
      ctx.clearRect(0, 0, w, h);
      ctx.globalAlpha = 1 - t;
      ctx.drawImage(frozenFrame, 0, 0, w, h);
      ctx.globalAlpha = t;
      ctx.drawImage(gifImg, 0, 0, w, h);
      ctx.globalAlpha = 1;
      if (t < 1) requestAnimationFrame(tick);
      else resolve();
    }
    requestAnimationFrame(tick);
  });
}

// Hold the GIF at full opacity for `durationMs`, redrawing every rAF tick
// so the browser advances the GIF's animation frames.
export function holdGif(gifImg, overlayCanvas, durationMs) {
  const ctx = overlayCanvas.getContext('2d');
  const w = overlayCanvas.width, h = overlayCanvas.height;
  const end = performance.now() + durationMs;

  return new Promise(resolve => {
    function tick() {
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(gifImg, 0, 0, w, h);
      if (performance.now() < end) requestAnimationFrame(tick);
      else resolve();
    }
    requestAnimationFrame(tick);
  });
}

// Crossfade from GIF back to transparent (live webcam underneath shows through).
export async function crossfadeOut(gifImg, overlayCanvas) {
  const ctx = overlayCanvas.getContext('2d');
  const w = overlayCanvas.width, h = overlayCanvas.height;
  const start = performance.now();

  return new Promise(resolve => {
    function tick() {
      const t = Math.min((performance.now() - start) / FADE_OUT_MS, 1);
      ctx.clearRect(0, 0, w, h);
      ctx.globalAlpha = 1 - t;
      ctx.drawImage(gifImg, 0, 0, w, h);
      ctx.globalAlpha = 1;
      if (t < 1) requestAnimationFrame(tick);
      else { ctx.clearRect(0, 0, w, h); resolve(); }
    }
    requestAnimationFrame(tick);
  });
}

// Future: Delaunay morph strategy replaces crossfadeIn.
// async function delaunayMorphIn(frozenFrame, gifImg, overlayCanvas, landmarks) { ... }
