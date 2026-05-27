import { FaceLandmarker, PoseLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// ─── Blendshape thresholds ───────────────────────────────────────────────────

const SMILE_SHAPES     = ['mouthSmileLeft', 'mouthSmileRight'];
const FROWN_SHAPES     = ['mouthFrownLeft', 'mouthFrownRight'];
const BROW_RAISE_SHAPES = ['browOuterUpLeft', 'browOuterUpRight'];
const EYE_LOOK_UP      = ['eyeLookUpLeft', 'eyeLookUpRight'];
const EYE_WIDE         = ['eyeWideLeft', 'eyeWideRight'];

const SMILE_THRESH          = 0.42;  // each side — accepts small smiles
const EVIL_SMILE_THRESH     = 0.58;  // both sides — big deliberate smile
const SURPRISE_JAW_THRESH   = 0.35;
const SURPRISE_BROW_THRESH  = 0.25;
const FROWN_THRESH          = 0.35;
const BROW_RAISE_THRESH     = 0.35;
const LOOKING_UP_THRESH     = 0.45;
const EVIL_LOOK_UP_THRESH   = 0.35;  // looking up required for evil_smile
const EYE_WIDE_THRESH       = 0.5;
const EYE_CLOSED_THRESH     = 0.5;   // both eyes must exceed this

// ─── Face landmark indices ───────────────────────────────────────────────────

const NOSE_TIP_IDX  = 4;
const BROW_L_IDX    = 70;
const CHIN_IDX      = 152;
const TOP_HEAD_IDX  = 10;

// ─── Pose landmark indices ───────────────────────────────────────────────────

const POSE_LEFT_SHOULDER_IDX  = 11;
const POSE_RIGHT_SHOULDER_IDX = 12;
const POSE_LEFT_ELBOW_IDX     = 13;
const POSE_RIGHT_ELBOW_IDX    = 14;
const POSE_LEFT_WRIST_IDX      = 15;
const POSE_RIGHT_WRIST_IDX     = 16;
const POSE_VISIBILITY_THRESH   = 0.5;
const PRAYING_DIST_THRESH      = 0.14;  // normalized wrist distance for clasped hands
const ELBOW_SHOULDER_Y_THRESH  = 0.12;  // elbow must be at roughly shoulder height for cinema

// ─── Motion buffer ───────────────────────────────────────────────────────────

const MOTION_BUFFER_SIZE = 60;
const NOD_Y_THRESH   = 0.030;   // larger amplitude required — deliberate nod only
const SHAKE_X_THRESH = 0.016;
const AXIS_DOMINANCE = 1.8;     // winning axis must be 1.8× the other

const noseBuf = [];

// ─── Model instances ─────────────────────────────────────────────────────────

let faceLandmarker = null;
let poseLandmarker = null;
let handLandmarker = null;

export async function initDetector() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
  );

  [faceLandmarker, poseLandmarker, handLandmarker] = await Promise.all([
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

export function detectFrame(videoEl, timestampMs) {
  if (!faceLandmarker || !poseLandmarker || !handLandmarker) return null;
  return {
    face: faceLandmarker.detectForVideo(videoEl, timestampMs),
    pose: poseLandmarker.detectForVideo(videoEl, timestampMs),
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

  const jawOpen     = shapes.find(s => s.categoryName === 'jawOpen')?.score ?? 0;
  const browInner   = shapes.find(s => s.categoryName === 'browInnerUp')?.score ?? 0;
  const blinkL      = shapes.find(s => s.categoryName === 'eyeBlinkLeft')?.score ?? 0;
  const blinkR      = shapes.find(s => s.categoryName === 'eyeBlinkRight')?.score ?? 0;
  const smileL      = shapes.find(s => s.categoryName === 'mouthSmileLeft')?.score ?? 0;
  const smileR      = shapes.find(s => s.categoryName === 'mouthSmileRight')?.score ?? 0;
  const frownScore  = avg(shapes, FROWN_SHAPES);
  const browRaise   = avg(shapes, BROW_RAISE_SHAPES);
  const lookUpScore = avg(shapes, EYE_LOOK_UP);
  const eyeWideScore = avg(shapes, EYE_WIDE);

  // Priority: most distinctive / intentional first
  if (blinkL > EYE_CLOSED_THRESH && blinkR > EYE_CLOSED_THRESH) return 'eye_closed';
  if (jawOpen > SURPRISE_JAW_THRESH && browInner > SURPRISE_BROW_THRESH) return 'surprise';
  // Evil smile: big deliberate grin while eyes roll up
  if (smileL > EVIL_SMILE_THRESH && smileR > EVIL_SMILE_THRESH && lookUpScore > EVIL_LOOK_UP_THRESH) return 'evil_smile';
  if (smileL > SMILE_THRESH && smileR > SMILE_THRESH) return 'smile';
  if (frownScore > FROWN_THRESH) return 'frown';
  if (browRaise > BROW_RAISE_THRESH) return 'raised_brows';
  if (eyeWideScore > EYE_WIDE_THRESH) return 'eye_wide';
  return 'neutral';
}

// Debug helpers
export function getBlendshapeScores(blendshapes) {
  if (!blendshapes?.length) return {};
  const shapes = blendshapes[0].categories;
  const pick = ['jawOpen', 'browInnerUp', 'mouthSmileLeft', 'mouthSmileRight',
                 'eyeBlinkLeft', 'eyeBlinkRight',
                 'eyeLookUpLeft', 'eyeLookUpRight', 'browOuterUpLeft', 'browOuterUpRight'];
  const out = {};
  for (const s of shapes) if (pick.includes(s.categoryName)) out[s.categoryName] = s.score.toFixed(2);
  return out;
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
    const dir = d > 0.001 ? 1 : d < -0.001 ? -1 : 0;
    if (dir !== 0 && dir !== prevDir && prevDir !== 0) reversals++;
    if (dir !== 0) prevDir = dir;
    maxVal = Math.max(maxVal, values[i]);
    minVal = Math.min(minVal, values[i]);
  }
  return { amplitude: maxVal - minVal, reversals };
}

function classifyHeadMotion(landmarks) {
  if (noseBuf.length < 10 || !landmarks?.length) return 'still';

  const xs = noseBuf.map(p => p.x);
  const ys = noseBuf.map(p => p.y);
  const { amplitude: yAmp, reversals: yRev } = oscillationAmplitude(ys);
  const { amplitude: xAmp, reversals: xRev } = oscillationAmplitude(xs);

  // Require the active axis to dominate — prevents head shakes from registering as nods
  if (yAmp > NOD_Y_THRESH && yRev >= 3 && yAmp > xAmp * AXIS_DOMINANCE) return 'nod';
  if (xAmp > SHAKE_X_THRESH && xRev >= 3 && xAmp > yAmp * AXIS_DOMINANCE) return 'shake';
  // Tilt removed — no GIF uses it and the low threshold caused constant flicker
  // that reset the hold timer and blocked other triggers from firing.
  return 'still';
}

// ─── Pose-based gesture classifier ───────────────────────────────────────────

function classifyGesture(poseLandmarks, faceLandmarks) {
  if (!poseLandmarks?.length || !faceLandmarks?.length) return 'none';

  const pose = poseLandmarks[0];
  const face = faceLandmarks[0];

  const browY = face[BROW_L_IDX].y;
  const noseY = face[NOSE_TIP_IDX].y;
  const chinY = face[CHIN_IDX].y;

  const lw = pose[POSE_LEFT_WRIST_IDX];
  const rw = pose[POSE_RIGHT_WRIST_IDX];
  const le = pose[POSE_LEFT_ELBOW_IDX];
  const re = pose[POSE_RIGHT_ELBOW_IDX];
  const ls = pose[POSE_LEFT_SHOULDER_IDX];
  const rs = pose[POSE_RIGHT_SHOULDER_IDX];
  const lVis = lw.visibility >= POSE_VISIBILITY_THRESH;
  const rVis = rw.visibility >= POSE_VISIBILITY_THRESH;
  const leVis = le?.visibility >= POSE_VISIBILITY_THRESH;
  const reVis = re?.visibility >= POSE_VISIBILITY_THRESH;

  // ABSOLUTE CINEMA: elbows at shoulder height (perpendicular), wrists raised above elbows.
  // Checked first so the cinema pose doesn't bleed into hands_on_head.
  if (lVis && rVis && leVis && reVis) {
    const lElbowAtShoulder = Math.abs(le.y - ls.y) < ELBOW_SHOULDER_Y_THRESH;
    const rElbowAtShoulder = Math.abs(re.y - rs.y) < ELBOW_SHOULDER_Y_THRESH;
    const lWristAboveElbow = lw.y < le.y;
    const rWristAboveElbow = rw.y < re.y;
    if (lElbowAtShoulder && rElbowAtShoulder && lWristAboveElbow && rWristAboveElbow) {
      return 'hands_up';
    }
  }

  // Hands literally on top of head: wrists at/above the top-of-head landmark
  const topHeadY = face[TOP_HEAD_IDX].y;
  if ((lVis && lw.y < topHeadY + 0.04) || (rVis && rw.y < topHeadY + 0.04)) return 'hands_on_head';

  // Praying with hands at mouth level (covering mouth) — Higuruma gesture
  if (lVis && rVis) {
    const dist = Math.hypot(lw.x - rw.x, lw.y - rw.y);
    const avgWristY = (lw.y + rw.y) / 2;
    if (dist < PRAYING_DIST_THRESH && avgWristY > noseY && avgWristY < chinY) {
      return 'praying_mouth';
    }
  }

  // Praying: both wrists visible, close together, at body/face level
  if (lVis && rVis) {
    const dist = Math.hypot(lw.x - rw.x, lw.y - rw.y);
    const avgWristY = (lw.y + rw.y) / 2;
    if (dist < PRAYING_DIST_THRESH && avgWristY > browY && avgWristY < chinY + 0.4) {
      return 'praying';
    }
  }

  const lOnChin = lVis && lw.y > noseY && lw.y < chinY + 0.12;
  const rOnChin = rVis && rw.y > noseY && rw.y < chinY + 0.12;
  if (lOnChin || rOnChin) return 'hand_on_chin';

  return 'none';
}

// ─── Hand gesture classifier (finger-level, using HandLandmarker) ─────────────

function isThumbsUp(hand) {
  const thumbTip = hand[4], thumbIP = hand[3];
  const indexTip = hand[8], indexPIP = hand[6];
  const midTip   = hand[12], midPIP  = hand[10];
  const ringTip  = hand[16], ringPIP = hand[14];
  if (!thumbTip || !thumbIP || !indexTip || !indexPIP || !midTip || !ringTip) return false;
  // Thumb clearly pointing up, other fingers curled below their PIP joints
  return thumbTip.y < thumbIP.y - 0.03
    && indexTip.y > indexPIP.y
    && midTip.y   > midPIP.y
    && ringTip.y  > ringPIP.y;
}

export function classifyHandGesture(handLandmarks, faceLandmarks) {
  if (!handLandmarks?.length || !faceLandmarks?.length) return 'none';
  const face = faceLandmarks[0];
  const noseY = face[NOSE_TIP_IDX].y;
  const chinY = face[CHIN_IDX].y;
  const browY = face[BROW_L_IDX].y;
  const faceXCenter = face[NOSE_TIP_IDX].x;
  const faceHalfW = 0.15;

  for (const hand of handLandmarks) {
    const tip   = hand[8];   // index fingertip
    const wrist = hand[0];
    const midTip = hand[12]; // middle fingertip
    if (!tip || !wrist) continue;

    // Index finger tip near lip zone (relaxed bounds — no eye-roll required)
    const inFaceX = tip.x > faceXCenter - faceHalfW && tip.x < faceXCenter + faceHalfW;
    const inLipY  = tip.y > noseY && tip.y < chinY;
    if (inFaceX && inLipY) return 'index_on_lip';

    // Thumbs up: thumb extended upward, other fingers curled
    if (isThumbsUp(hand)) return 'thumbs_up';

    // Face palm: wrist at chin level while fingers reach up past nose
    const palmX = wrist.x > faceXCenter - faceHalfW && wrist.x < faceXCenter + faceHalfW;
    const palmY = wrist.y > noseY && wrist.y < chinY + 0.1;
    const fingersOverFace = midTip && midTip.y < noseY;
    if (palmX && palmY && fingersOverFace) return 'face_palm';

    // Wrist near chin (hand stroking beard)
    const wristOnChin = wrist.y > noseY && wrist.y < chinY + 0.10
      && wrist.x > faceXCenter - faceHalfW - 0.1 && wrist.x < faceXCenter + faceHalfW + 0.1;
    if (wristOnChin) return 'hand_on_chin';
  }
  return 'none';
}

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
