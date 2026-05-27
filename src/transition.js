// Transition module — swappable strategy (crossfade now, Delaunay morph later).
//
// GIF is displayed via a native <img> element so the browser advances animation
// frames automatically — no rAF redraw loop required.
// The overlay canvas handles only the frozen-webcam portion of the crossfade.

const FADE_MS = 500;

export async function preloadGif(url) {
  const img = new Image();
  img.src = url;
  if (!img.complete) {
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });
  }
  return img;
}

// Crossfade: frozen webcam (canvas) fades out while GIF img fades in.
export async function crossfadeIn(frozenFrame, gifEl, overlayCanvas) {
  const ctx = overlayCanvas.getContext('2d');
  const w = overlayCanvas.width, h = overlayCanvas.height;

  // Draw the frozen frame as the starting state
  ctx.clearRect(0, 0, w, h);
  ctx.drawImage(frozenFrame, 0, 0, w, h);

  // Show GIF img but invisible yet
  gifEl.style.display = 'block';
  gifEl.style.opacity = '0';

  const start = performance.now();

  await new Promise(resolve => {
    function tick() {
      const t = Math.min((performance.now() - start) / FADE_MS, 1);
      // Frozen frame fades out on overlay canvas
      ctx.clearRect(0, 0, w, h);
      ctx.globalAlpha = 1 - t;
      ctx.drawImage(frozenFrame, 0, 0, w, h);
      ctx.globalAlpha = 1;
      // GIF img fades in via style (browser animates GIF natively)
      gifEl.style.opacity = String(t);

      if (t < 1) requestAnimationFrame(tick);
      else {
        ctx.clearRect(0, 0, w, h); // overlay no longer needed
        gifEl.style.opacity = '1';
        resolve();
      }
    }
    requestAnimationFrame(tick);
  });
}

// Fade GIF img back to transparent, then hide it.
export async function crossfadeOut(gifEl) {
  const start = performance.now();
  await new Promise(resolve => {
    function tick() {
      const t = Math.min((performance.now() - start) / FADE_MS, 1);
      gifEl.style.opacity = String(1 - t);
      if (t < 1) requestAnimationFrame(tick);
      else { gifEl.style.display = 'none'; resolve(); }
    }
    requestAnimationFrame(tick);
  });
}

// Future: Delaunay morph replaces crossfadeIn.
// async function delaunayMorphIn(frozenFrame, gifEl, overlayCanvas, landmarks) { ... }
