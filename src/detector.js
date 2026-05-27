import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// Blendshape names we care about for expression detection
const SMILE_SHAPES = ['mouthSmileLeft', 'mouthSmileRight'];
const SURPRISE_SHAPES = ['browInnerUp', 'jawOpen'];
const FROWN_SHAPES = ['mouthFrownLeft', 'mouthFrownRight'];
const BROW_RAISE_SHAPES = ['browOuterUpLeft', 'browOuterUpRight'];

// Thresholds
const SMILE_THRESH = 0.4;
const SURPRISE_JAW_THRESH = 0.35;
const SURPRISE_BROW_THRESH = 0.25;
const FROWN_THRESH = 0.35;

// Nose tip landmark index in MediaPipe 478-point model
const NOSE_TIP_IDX = 4;

// Motion detection
const MOTION_BUFFER_SIZE = 30;
const NOD_Y_THRESH = 0.018;   // normalized units
const SHAKE_X_THRESH = 0.018;
const TILT_THRESH = 0.02;     // ear-to-ear Y difference

let faceLandmarker = null;

export async function initDetector() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU',
    },
    outputFaceBlendshapes: true,
    runningMode: 'VIDEO',
    numFaces: 1,
  });

  return faceLandmarker;
}

export function detectFrame(videoEl, timestampMs) {
  if (!faceLandmarker) return null;
  return faceLandmarker.detectForVideo(videoEl, timestampMs);
}

// ─── Expression ─────────────────────────────────────────────────────────────

function avg(shapes, names) {
  let sum = 0, count = 0;
  for (const s of shapes) {
    if (names.includes(s.categoryName)) { sum += s.score; count++; }
  }
  return count ? sum / count : 0;
}

export function classifyExpression(blendshapes) {
  if (!blendshapes?.length) return 'neutral';
  const shapes = blendshapes[0].categories;

  const smileScore = avg(shapes, SMILE_SHAPES);
  const jawOpen = shapes.find(s => s.categoryName === 'jawOpen')?.score ?? 0;
  const browInner = shapes.find(s => s.categoryName === 'browInnerUp')?.score ?? 0;
  const frownScore = avg(shapes, FROWN_SHAPES);
  const browRaise = avg(shapes, BROW_RAISE_SHAPES);

  if (jawOpen > SURPRISE_JAW_THRESH && browInner > SURPRISE_BROW_THRESH) return 'surprise';
  if (smileScore > SMILE_THRESH) return 'smile';
  if (frownScore > FROWN_THRESH) return 'frown';
  if (browRaise > 0.35) return 'raised_brows';
  return 'neutral';
}

// ─── Motion ─────────────────────────────────────────────────────────────────

const noseBuf = []; // { x, y }[]

export function updateMotionBuffer(landmarks) {
  if (!landmarks?.length) return;
  const nose = landmarks[0][NOSE_TIP_IDX];
  noseBuf.push({ x: nose.x, y: nose.y });
  if (noseBuf.length > MOTION_BUFFER_SIZE) noseBuf.shift();
}

function oscillationAmplitude(values) {
  if (values.length < 6) return 0;
  // Count direction reversals and measure peak-to-peak in the last N frames
  let maxVal = -Infinity, minVal = Infinity;
  let reversals = 0;
  let prevDir = 0;
  for (let i = 1; i < values.length; i++) {
    const d = values[i] - values[i - 1];
    const dir = d > 0.002 ? 1 : d < -0.002 ? -1 : 0;
    if (dir !== 0 && dir !== prevDir && prevDir !== 0) reversals++;
    if (dir !== 0) prevDir = dir;
    maxVal = Math.max(maxVal, values[i]);
    minVal = Math.min(minVal, values[i]);
  }
  return { amplitude: maxVal - minVal, reversals };
}

// Left ear ~234, right ear ~454 in 478-pt model
const LEFT_EAR_IDX = 234;
const RIGHT_EAR_IDX = 454;

export function classifyMotion(landmarks) {
  if (noseBuf.length < 10 || !landmarks?.length) return 'still';

  const xs = noseBuf.map(p => p.x);
  const ys = noseBuf.map(p => p.y);
  const { amplitude: yAmp, reversals: yRev } = oscillationAmplitude(ys);
  const { amplitude: xAmp, reversals: xRev } = oscillationAmplitude(xs);

  // Tilt: compare Y position of left vs right ear
  const lm = landmarks[0];
  const earDiff = Math.abs(lm[LEFT_EAR_IDX].y - lm[RIGHT_EAR_IDX].y);

  if (yAmp > NOD_Y_THRESH && yRev >= 1) return 'nod';
  if (xAmp > SHAKE_X_THRESH && xRev >= 1) return 'shake';
  if (earDiff > TILT_THRESH) return 'tilt';
  return 'still';
}

export function clearMotionBuffer() {
  noseBuf.length = 0;
}
