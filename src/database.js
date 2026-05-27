// GIF database loader and matcher.
// Abstracted so future sources (user uploads, remote API) can plug in
// by replacing loadGifs() without touching matching logic.

let gifDb = [];

export async function loadGifs() {
  const res = await fetch('/gifs.json');
  gifDb = await res.json();
  return gifDb;
}

// Returns a random matching GIF entry or null.
// expression: required exact match
// motion: null in the db entry means any motion accepted
export function matchGif({ expression, motion }) {
  const matches = gifDb.filter(g => {
    const exprMatch = g.tags.expression === expression;
    const motionMatch = g.tags.motion === null || g.tags.motion === motion;
    return exprMatch && motionMatch;
  });
  if (!matches.length) return null;
  return matches[Math.floor(Math.random() * matches.length)];
}
