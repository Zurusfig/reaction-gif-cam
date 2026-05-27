import { FaceLandmarker, PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ─── Blendshape thresholds ───────────────────────────────────────────────────

const SMILE_SHAPES = ['mouthSmileLeft', 'mouthSmileRight'];
const FROWN_SHAPES = ['mouthFrownLeft', 'mouthFrownRight'];
const BROW_RAISE_SHAPES = ['browOuterUpLeft', 'browOuterUpRight'];

const TONGUE_OUT_THRESH    = 0.4;
const SMILE_THRESH         = 0.4;
const SURPRISE_JAW_THRESH  = 0.35;
const SURPRISE_BROW_THRESH = 0.25;
const FROWN_THRESH         = 0.35;
const BROW_RAISE_THRESH    = 0.35;

// ─── Face landmark indices ───────────────────────────────────────────────────

const NOSE_TIP_IDX  = 4;
const LEFT_EAR_IDX  = 234;
const RIGHT_EAR_IDX = 454;
const BROW_L_IDX    = 70;   // left brow — used as vertical threshold for "above brow"
const CHIN_IDX      = 152;  // chin bottom

// ─── Pose landmark indices (MediaPipe 33-point body model) ───────────────────

const POSE_NOSE_IDX          = 0;
const POSE_LEFT_WRIST_IDX    = 15;
const POSE_RIGHT_WRIST_IDX   = 16;
const POSE_VISIBILITY_THRESH = 0.5;

// ─── Motion buffer ───────────────────────────────────────────────────────────

const MOTION_BUFFER_SIZE = 30;
const NOD_Y_THRESH   = 0.018;
const SHAKE_X_THRESH = 0.018;
const TILT_THRESH    = 0.02;

const noseBuf = [];

// ─── Model instances ─────────────────────────────────────────────────────────

let faceLandmarker = null;
let poseLandmarker = null;

export async function initDetector() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
  );

  [faceLandmarker, poseLandmarker] = await Promise.all([
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
    PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    }),
  ]);
}

// Returns { face, pose } results for the frame
export function detectFrame(videoEl, timestampMs) {
  if (!faceLandmarker || !poseLandmarker) return null;
  return {
    face: faceLandmarker.detectForVideo(videoEl, timestampMs),
    pose: poseLandmarker.detectForVideo(videoEl, timestampMs),
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

  const tongueOut  = shapes.find(s => s.categoryName === 'tongueOut')?.score ?? 0;
  const jawOpen    = shapes.find(s => s.categoryName === 'jawOpen')?.score ?? 0;
  const browInner  = shapes.find(s => s.categoryName === 'browInnerUp')?.score ?? 0;
  const smileScore = avg(shapes, SMILE_SHAPES);
  const frownScore = avg(shapes, FROWN_SHAPES);
  const browRaise  = avg(shapes, BROW_RAISE_SHAPES);

  if (tongueOut > TONGUE_OUT_THRESH) return 'tongue_out';
  if (jawOpen > SURPRISE_JAW_THRESH && browInner > SURPRISE_BROW_THRESH) return 'surprise';
  if (smileScore > SMILE_THRESH) return 'smile';
  if (frownScore > FROWN_THRESH) return 'frown';
  if (browRaise > BROW_RAISE_THRESH) return 'raised_brows';
  return 'neutral';
}

// Raw tongue score — exposed for debug display
export function getTongueScore(blendshapes) {
  if (!blendshapes?.length) return 0;
  return blendshapes[0].categories.find(s => s.categoryName === 'tongueOut')?.score ?? 0;
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

// ─── Pose-based gesture classifier ───────────────────────────────────────────
// Uses body wrist positions relative to face landmarks for reliable arm tracking.

function classifyGesture(poseLandmarks, faceLandmarks) {
  if (!poseLandmarks?.length || !faceLandmarks?.length) return 'none';

  const pose = poseLandmarks[0];
  const face = faceLandmarks[0];

  const browY = face[BROW_L_IDX].y;     // upper boundary: above = on head
  const noseY = face[NOSE_TIP_IDX].y;   // lower boundary for chin zone start
  const chinY = face[CHIN_IDX].y;       // lower boundary for chin zone end

  const lw = pose[POSE_LEFT_WRIST_IDX];
  const rw = pose[POSE_RIGHT_WRIST_IDX];

  const lVisible = lw.visibility >= POSE_VISIBILITY_THRESH;
  const rVisible = rw.visibility >= POSE_VISIBILITY_THRESH;

  // Wrist above brow line → hands on head
  const leftOnHead  = lVisible && lw.y < browY;
  const rightOnHead = rVisible && rw.y < browY;
  if (leftOnHead || rightOnHead) return 'hands_on_head';

  // Wrist between nose and chin → hand on chin/face
  const leftOnChin  = lVisible && lw.y > noseY && lw.y < chinY + 0.12;
  const rightOnChin = rVisible && rw.y > noseY && rw.y < chinY + 0.12;
  if (leftOnChin || rightOnChin) return 'hand_on_chin';

  return 'none';
}

// ─── Combined motion output ──────────────────────────────────────────────────
// Head motion wins when active; pose gestures fill in when head is still.

export function classifyMotion(faceLandmarks, poseLandmarks) {
  const headMotion = classifyHeadMotion(faceLandmarks);
  if (headMotion !== 'still') return headMotion;

  const gesture = classifyGesture(poseLandmarks, faceLandmarks);
  return gesture !== 'none' ? gesture : 'still';
}

export function clearMotionBuffer() {
  noseBuf.length = 0;
}
