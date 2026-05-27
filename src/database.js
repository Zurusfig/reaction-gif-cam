// GIF database loader and matcher.
// Abstracted so future sources (user uploads, remote API) can plug in
// by replacing loadGifs() without touching matching logic.

let gifDb = [];

export async function loadGifs() {
  const res = await fetch('/gifs.json');
  gifDb = await res.json();
  return gifDb;  // caller (main.js) preloads images and stores them as _img on each entry
}

// Returns a random matching GIF entry or null.
// null tag field = any value accepted
// gesture: optional hand gesture required
// mouth_closed: if true, jawOpen must be below threshold
export function matchGif({ expression, motion, gesture = 'none', jawOpen = 0 }) {
  const matches = gifDb.filter(g => {
    const exprMatch   = g.tags.expression   === null || g.tags.expression   === undefined || g.tags.expression   === expression;
    const motionMatch = g.tags.motion       === null || g.tags.motion       === undefined || g.tags.motion       === motion;
    const gestureMatch = !g.tags.gesture    || g.tags.gesture === gesture;
    const mouthOk     = !g.tags.mouth_closed || jawOpen < 0.20;
    return exprMatch && motionMatch && gestureMatch && mouthOk;
  });
  if (!matches.length) return null;
  return matches[Math.floor(Math.random() * matches.length)];
}
