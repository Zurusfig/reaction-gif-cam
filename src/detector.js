import { FaceLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ─── Blendshape thresholds ───────────────────────────────────────────────────

const SMILE_SHAPES = ['mouthSmileLeft', 'mouthSmileRight'];
const FROWN_SHAPES = ['mouthFrownLeft', 'mouthFrownRight'];
const BROW_RAISE_SHAPES = ['browOuterUpLeft', 'browOuterUpRight'];

const TONGUE_OUT_THRESH   = 0.5;
const SMILE_THRESH        = 0.4;
const SURPRISE_JAW_THRESH = 0.35;
const SURPRISE_BROW_THRESH = 0.25;
const FROWN_THRESH        = 0.35;
const BROW_RAISE_THRESH   = 0.35;

// ─── Landmark indices ────────────────────────────────────────────────────────

const NOSE_TIP_IDX   = 4;
const LEFT_EAR_IDX   = 234;
const RIGHT_EAR_IDX  = 454;
// Face zone reference points for gesture detection
const EYEBROW_TOP_IDX = 10;   // forehead center / top of head
const BROW_L_IDX      = 70;   // left brow inner corner
const CHIN_IDX        = 152;  // chin bottom
// Hand landmark indices
const WRIST_IDX       = 0;
const INDEX_TIP_IDX   = 8;
const MIDDLE_TIP_IDX  = 12;

// ─── Motion buffer ───────────────────────────────────────────────────────────

const MOTION_BUFFER_SIZE = 30;
const NOD_Y_THRESH   = 0.018;
const SHAKE_X_THRESH = 0.018;
const TILT_THRESH    = 0.02;

const noseBuf = [];

// ─── Model instances ─────────────────────────────────────────────────────────

let faceLandmarker = null;
let handLandmarker = null;

export async function initDetector() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
  );

  [faceLandmarker, handLandmarker] = await Promise.all([
    FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      outputFaceBlendshapes: true,
      runningMode: 'VIDEO',
      numFaces: 1,
    }),
    HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
    }),
  ]);
}

// Returns { face, hand } results for the frame
export function detectFrame(videoEl, timestampMs) {
  if (!faceLandmarker || !handLandmarker) return null;
  return {
    face: faceLandmarker.detectForVideo(videoEl, timestampMs),
    hand: handLandmarker.detectForVideo(videoEl, timestampMs),
  };
}

// ─── Expression classifier ───────────────────────────────────────────────────

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

  const tongueOut = shapes.find(s => s.categoryName === 'tongueOut')?.score ?? 0;
  const jawOpen   = shapes.find(s => s.categoryName === 'jawOpen')?.score ?? 0;
  const browInner = shapes.find(s => s.categoryName === 'browInnerUp')?.score ?? 0;
  const smileScore = avg(shapes, SMILE_SHAPES);
  const frownScore = avg(shapes, FROWN_SHAPES);
  const browRaise  = avg(shapes, BROW_RAISE_SHAPES);

  // Priority: most distinctive first
  if (tongueOut > TONGUE_OUT_THRESH) return 'tongue_out';
  if (jawOpen > SURPRISE_JAW_THRESH && browInner > SURPRISE_BROW_THRESH) return 'surprise';
  if (smileScore > SMILE_THRESH) return 'smile';
  if (frownScore > FROWN_THRESH) return 'frown';
  if (browRaise > BROW_RAISE_THRESH) return 'raised_brows';
  return 'neutral';
}

// ─── Head motion classifier ──────────────────────────────────────────────────

export function updateMotionBuffer(landmarks) {
  if (!landmarks?.length) return;
  const nose = landmarks[0][NOSE_TIP_IDX];
  noseBuf.push({ x: nose.x, y: nose.y });
  if (noseBuf.length > MOTION_BUFFER_SIZE) noseBuf.shift();
}

function oscillationAmplitude(values) {
  if (values.length < 6) return { amplitude: 0, reversals: 0 };
  let maxVal = -Infinity, minVal = Infinity;
  let reversals = 0, prevDir = 0;
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

function classifyHeadMotion(landmarks) {
  if (noseBuf.length < 10 || !landmarks?.length) return 'still';
  const lm = landmarks[0];

  const xs = noseBuf.map(p => p.x);
  const ys = noseBuf.map(p => p.y);
  const { amplitude: yAmp, reversals: yRev } = oscillationAmplitude(ys);
  const { amplitude: xAmp, reversals: xRev } = oscillationAmplitude(xs);
  const earDiff = Math.abs(lm[LEFT_EAR_IDX].y - lm[RIGHT_EAR_IDX].y);

  if (yAmp > NOD_Y_THRESH && yRev >= 1) return 'nod';
  if (xAmp > SHAKE_X_THRESH && xRev >= 1) return 'shake';
  if (earDiff > TILT_THRESH) return 'tilt';
  return 'still';
}

// ─── Hand gesture classifier ─────────────────────────────────────────────────

function classifyGesture(handLandmarks, faceLandmarks) {
  if (!handLandmarks?.length || !faceLandmarks?.length) return 'none';

  const face = faceLandmarks[0];
  // Reference Y positions from face (normalized 0=top, 1=bottom)
  const browY = face[BROW_L_IDX].y;
  const noseY = face[NOSE_TIP_IDX].y;
  const chinY = face[CHIN_IDX].y;
  const faceLeft  = Math.min(face[LEFT_EAR_IDX].x, face[RIGHT_EAR_IDX].x);
  const faceRight = Math.max(face[LEFT_EAR_IDX].x, face[RIGHT_EAR_IDX].x);

  // Hands above brows = on head (mirror-flip x for display, don't flip for detection)
  const handsAboveBrow = handLandmarks.filter(h => h[WRIST_IDX].y < browY);
  if (handsAboveBrow.length >= 1) return 'hands_on_head';

  // Fingertips in the lower-face zone (between nose and just below chin) = hand_on_chin
  for (const hand of handLandmarks) {
    const tip = hand[INDEX_TIP_IDX];
    const mid = hand[MIDDLE_TIP_IDX];
    const inFaceX = (tip.x > faceLeft - 0.1 && tip.x < faceRight + 0.1);
    const inChinY = (tip.y > noseY && tip.y < chinY + 0.15);
    if (inFaceX && inChinY) return 'hand_on_chin';
    const inFaceX2 = (mid.x > faceLeft - 0.1 && mid.x < faceRight + 0.1);
    const inChinY2 = (mid.y > noseY && mid.y < chinY + 0.15);
    if (inFaceX2 && inChinY2) return 'hand_on_chin';
  }

  return 'none';
}

// ─── Combined motion output ──────────────────────────────────────────────────
// Head motion takes priority (more intentional); gestures fill in when head is still.

export function classifyMotion(faceLandmarks, handLandmarks) {
  const headMotion = classifyHeadMotion(faceLandmarks);
  if (headMotion !== 'still') return headMotion;

  const gesture = classifyGesture(handLandmarks, faceLandmarks);
  return gesture !== 'none' ? gesture : 'still';
}

export function clearMotionBuffer() {
  noseBuf.length = 0;
}
