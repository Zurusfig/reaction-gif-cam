// Transition module — swappable: crossfade (now) → Delaunay morph (later)
// The active strategy is selected at the bottom; caller uses runTransition().

const FADE_DURATION_MS = 500;

// overlayCanvas: the canvas drawn on top of the webcam feed
async function crossfade(frozenFrame, gifUrl, overlayCanvas) {
  const ctx = overlayCanvas.getContext('2d');
  const w = overlayCanvas.width, h = overlayCanvas.height;

  const img = new Image();
  img.src = gifUrl;
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });

  const start = performance.now();

  return new Promise(resolve => {
    function tick() {
      const elapsed = performance.now() - start;
      const t = Math.min(elapsed / FADE_DURATION_MS, 1);

      ctx.clearRect(0, 0, w, h);
      // frozen webcam frame fades out
      ctx.globalAlpha = 1 - t;
      ctx.drawImage(frozenFrame, 0, 0, w, h);
      // GIF fades in
      ctx.globalAlpha = t;
      ctx.drawImage(img, 0, 0, w, h);
      ctx.globalAlpha = 1;

      if (t < 1) requestAnimationFrame(tick);
      else resolve(img);
    }
    requestAnimationFrame(tick);
  });
}

// Future: Delaunay morph strategy goes here.
// async function delaunayMorph(frozenFrame, gifUrl, overlayCanvas, landmarks) { ... }

export const runTransition = crossfade;
