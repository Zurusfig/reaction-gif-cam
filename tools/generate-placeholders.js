#!/usr/bin/env node
// Generates 3 tiny (64x64) animated placeholder GIFs for testing.
// Each cycles between 2 colors with a text-like pattern to be visually distinct.

import { writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT = join(__dirname, '../public/gifs');
mkdirSync(OUT, { recursive: true });

// Minimal GIF89a encoder — supports indexed-color frames with a delay.
function encodeGif(width, height, frames, { loop = 0 } = {}) {
  const parts = [];

  // ── Header ────────────────────────────────────────────────────────────────
  parts.push(Buffer.from('GIF89a'));

  // Find max colors across all frames
  const numColors = frames[0].palette.length;
  const ctSize = Math.max(2, Math.ceil(Math.log2(numColors)));
  const ctLen = 1 << ctSize; // power-of-2 that fits numColors

  // Logical Screen Descriptor
  const lsd = Buffer.alloc(7);
  lsd.writeUInt16LE(width, 0);
  lsd.writeUInt16LE(height, 2);
  lsd.writeUInt8(0x80 | 0x70 | (ctSize - 1), 4); // GCT present, 8-bit color, ctSize
  lsd.writeUInt8(0, 5); // background index
  lsd.writeUInt8(0, 6); // pixel aspect
  parts.push(lsd);

  // Global Color Table (use first frame's palette, pad to ctLen)
  const palette = frames[0].palette;
  const gct = Buffer.alloc(ctLen * 3, 0);
  for (let i = 0; i < palette.length; i++) {
    gct[i * 3] = palette[i][0];
    gct[i * 3 + 1] = palette[i][1];
    gct[i * 3 + 2] = palette[i][2];
  }
  parts.push(gct);

  // Netscape loop extension
  if (loop >= 0) {
    const ns = Buffer.alloc(19);
    ns[0] = 0x21; ns[1] = 0xFF; ns[2] = 0x0B;
    Buffer.from('NETSCAPE2.0').copy(ns, 3);
    ns[14] = 0x03; ns[15] = 0x01;
    ns.writeUInt16LE(loop, 16);
    ns[18] = 0x00;
    parts.push(ns);
  }

  // ── Frames ────────────────────────────────────────────────────────────────
  for (const frame of frames) {
    const delayCs = Math.round(frame.delayMs / 10); // centiseconds

    // Graphic Control Extension
    const gce = Buffer.alloc(8);
    gce[0] = 0x21; gce[1] = 0xF9; gce[2] = 0x04;
    gce[3] = 0x00; // disposal = do not dispose, no user input, no transparency
    gce.writeUInt16LE(delayCs, 4);
    gce[6] = 0x00; // transparent index (unused)
    gce[7] = 0x00;
    parts.push(gce);

    // Image Descriptor
    const id = Buffer.alloc(10);
    id[0] = 0x2C;
    id.writeUInt16LE(0, 1); id.writeUInt16LE(0, 3); // left, top
    id.writeUInt16LE(width, 5); id.writeUInt16LE(height, 7);
    id[9] = 0x00; // no local CT, not interlaced
    parts.push(id);

    // Image Data — LZW compressed
    const minCodeSize = Math.max(2, ctSize);
    parts.push(lzwCompress(frame.pixels, minCodeSize));
  }

  // Trailer
  parts.push(Buffer.from([0x3B]));
  return Buffer.concat(parts);
}

// LZW encoder for GIF image data
function lzwCompress(pixels, minCodeSize) {
  const clearCode = 1 << minCodeSize;
  const eoiCode = clearCode + 1;

  // Bit stream writer (LSB-first per GIF spec)
  const bytes = [];
  let bitBuf = 0, bitPos = 0;
  let codeSize = minCodeSize + 1;

  function emit(code) {
    bitBuf |= code << bitPos;
    bitPos += codeSize;
    while (bitPos >= 8) {
      bytes.push(bitBuf & 0xFF);
      bitBuf >>= 8;
      bitPos -= 8;
    }
  }

  function flush() {
    if (bitPos > 0) { bytes.push(bitBuf & 0xFF); bitPos = 0; bitBuf = 0; }
  }

  // LZW encode
  let table = new Map();
  function resetTable() {
    table.clear();
    for (let i = 0; i < clearCode; i++) table.set(String(i), i);
    codeSize = minCodeSize + 1;
  }

  resetTable();
  emit(clearCode);

  let prefix = String(pixels[0]);
  for (let i = 1; i < pixels.length; i++) {
    const next = String(pixels[i]);
    const key = prefix + ',' + next;
    if (table.has(key)) {
      prefix = key;
    } else {
      emit(table.get(prefix));
      const newCode = table.size;
      table.set(key, newCode);
      // Grow code size when table exceeds current range
      if (newCode >= (1 << codeSize) && codeSize < 12) codeSize++;
      if (newCode >= 4096) { emit(clearCode); resetTable(); }
      prefix = next;
    }
  }
  emit(table.get(prefix));
  emit(eoiCode);
  flush();

  // Pack into sub-blocks (max 255 bytes each)
  const out = [Buffer.from([minCodeSize])];
  for (let i = 0; i < bytes.length; i += 255) {
    const chunk = bytes.slice(i, i + 255);
    out.push(Buffer.from([chunk.length, ...chunk]));
  }
  out.push(Buffer.from([0x00])); // block terminator
  return Buffer.concat(out);
}

// ── Frame builders ──────────────────────────────────────────────────────────

function solidFrame(width, height, colorIndex, palette, delayMs) {
  return { pixels: new Uint8Array(width * height).fill(colorIndex), palette, delayMs };
}

function checkerFrame(width, height, a, b, offset, palette, delayMs) {
  const pixels = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      pixels[y * width + x] = ((x >> 3) + (y >> 3) + offset) % 2 === 0 ? a : b;
    }
  }
  return { pixels, palette, delayMs };
}

function stripesFrame(width, height, a, b, offset, palette, delayMs) {
  const pixels = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      pixels[y * width + x] = ((x + offset) >> 3) % 2 === 0 ? a : b;
    }
  }
  return { pixels, palette, delayMs };
}

// ── Generate 3 placeholder GIFs ────────────────────────────────────────────

const W = 64, H = 64, DELAY = 250;

// smile-nod.gif — green/yellow checker that shifts
const greenPalette = [[0,0,0],[34,197,94],[250,204,21]]; // black, green, yellow
const smileFrames = [
  checkerFrame(W, H, 1, 2, 0, greenPalette, DELAY),
  checkerFrame(W, H, 1, 2, 1, greenPalette, DELAY),
  checkerFrame(W, H, 1, 2, 0, greenPalette, DELAY),
  checkerFrame(W, H, 2, 1, 1, greenPalette, DELAY),
];
writeFileSync(join(OUT, 'smile-nod.gif'), encodeGif(W, H, smileFrames));
console.log('✓ smile-nod.gif');

// surprise.gif — red/orange flash
const redPalette = [[0,0,0],[239,68,68],[251,146,60]]; // black, red, orange
const surpriseFrames = [
  solidFrame(W, H, 1, redPalette, DELAY),
  solidFrame(W, H, 2, redPalette, DELAY),
  solidFrame(W, H, 1, redPalette, DELAY),
  solidFrame(W, H, 2, redPalette, DELAY),
];
writeFileSync(join(OUT, 'surprise.gif'), encodeGif(W, H, surpriseFrames));
console.log('✓ surprise.gif');

// smile-shake.gif — blue/cyan horizontal stripes that scroll
const bluePalette = [[0,0,0],[59,130,246],[34,211,238]]; // black, blue, cyan
const shakeFrames = [
  stripesFrame(W, H, 1, 2, 0, bluePalette, DELAY),
  stripesFrame(W, H, 1, 2, 4, bluePalette, DELAY),
  stripesFrame(W, H, 1, 2, 8, bluePalette, DELAY),
  stripesFrame(W, H, 1, 2, 4, bluePalette, DELAY),
];
writeFileSync(join(OUT, 'smile-shake.gif'), encodeGif(W, H, shakeFrames));
console.log('✓ smile-shake.gif');

console.log('All placeholder GIFs written to public/gifs/');
