// Removed top-level import to prevent blocking on load error
// import { FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker } from '@mediapipe/tasks-vision';

const TARGET_CAPTURE_SHORT_EDGE = 1440;
const TARGET_CAPTURE_LONG_EDGE = 1920;
const CAMERA_CROP_PADDING_RATIO = 0.08;
const MEDIAPIPE_WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';

// Models
const FACE_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const POSE_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const HAND_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// Assets
const CROWN_IMAGE_PATH = 'assets/crown.png';
const MEDAL_IMAGE_PATH = 'assets/medal.png';
const TROPHY_IMAGE_PATH = 'assets/trophy.png';

// FPS Limits (to reduce load)
const FACE_DETECTION_FPS = 15;
const POSE_DETECTION_FPS = 10;
const HAND_DETECTION_FPS = 10;

// Transform Constants
// Crown
const CROWN_SIZE_RATIO = 1.9; // Adjusted for eye-distance based calculation
const CROWN_Y_OFFSET_RATIO = 1.5; // Adjusted for eye-based center
// Medal
const MEDAL_SIZE_RATIO = 0.9;
const MEDAL_Y_OFFSET_RATIO = 0.5;
// Trophy
const TROPHY_SIZE_RATIO = 5.0; // Adjusted for wrist-finger base calculation
const TROPHY_Y_OFFSET_RATIO = -2.5;

const CAMERA_CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: 'environment',
    width: { ideal: TARGET_CAPTURE_SHORT_EDGE },
    height: { ideal: TARGET_CAPTURE_LONG_EDGE },
  },
};

const FALLBACK_CAMERA_CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: 'environment',
    width: { ideal: 720 },
    height: { ideal: 1280 },
  },
};

const FRAMES = [
  { id: 'animated-frame', type: 'video', webmSrc: 'assets/frame_alpha.webm', mp4Src: 'assets/frame_alpha.mp4', label: 'animated' },
  { id: 'fallback-image', type: 'image', imageSrc: 'assets/ios_frame_001.png', label: 'fallback' },
];
const FALLBACK_FRAME_INDEX = FRAMES.findIndex(f => f.type === 'image');
const IOS_FRAME_IMAGES = Array.from({ length: 64 }, (_, i) => `assets/ios_frame_${String(i + 1).padStart(3, '0')}.png`);
const IOS_FRAME_INTERVAL_MS = 100;

// Global State
let activeFrameIndex = 0;
let mediaStream = null;
let isCapturingScreenshot = false;
let currentCapture = null;
let iosFrameIndex = 0;
let iosFrameTimer = null;
let shouldResumeCamera = false;

// DOM Elements
let cameraEl, frameImageEl, frameVideoEl, shutterButton, captureCanvas, previewEl;
let crownCanvas, medalCanvas, trophyCanvas;
let crownCtx, medalCtx, trophyCtx;
let toggleFace, togglePose, toggleHand;
let shareModal, capturedImage;

// Landmarkers & Images
let faceLandmarker, poseLandmarker, handLandmarker;
let crownImage, medalImage, trophyImage;

// Detection State (Objects to handle tracking & smoothing)
let faceTracker, poseTracker, handTracker;
let faceLoopId, poseLoopId, handLoopId;

function $(selector) { return document.querySelector(selector); }

function clampResolutionValue(preferred, range) {
  if (!range) return preferred;
  let value = preferred;
  if (typeof range.max === 'number') value = Math.min(value, range.max);
  if (typeof range.min === 'number') value = Math.max(value, range.min);
  return Math.round(value);
}

function createConstraints(baseConstraints, facingMode) {
  return { ...baseConstraints, video: { ...baseConstraints.video, facingMode } };
}

function isSafari() {
  const ua = navigator.userAgent;
  return /Safari/i.test(ua) && !/Chrome|Chromium|CriOS|Edg|OPR/i.test(ua);
}

// ==================== Initialization & Loading ====================

async function initLandmarkers() {
  try {
    const { FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker } = await import('@mediapipe/tasks-vision');

    const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_PATH);

    // Initialize in parallel
    const pFace = FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO', numFaces: 1
    });
    const pPose = PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: POSE_LANDMARKER_MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO', numPoses: 1
    });
    const pHand = HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_LANDMARKER_MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO', numHands: 1
    });

    [faceLandmarker, poseLandmarker, handLandmarker] = await Promise.all([pFace, pPose, pHand]);
    console.log('All Landmarkers initialized');
    updateDetectionStatus();
  } catch (error) {
    console.warn('Landmarker initialization failed:', error);
    alert('Failed to load AR models. Please check your internet connection.');
  }
}

function loadImage(src) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => { console.warn('Failed to load:', src); resolve(null); };
    img.src = src;
  });
}

function updateDetectionStatus() {
  // Start/Stop based on toggles and camera/model readiness
  if (!cameraEl || cameraEl.readyState < 2) return;

  // Face
  if (toggleFace.checked && faceLandmarker && crownImage) {
    if (!faceLoopId) runFaceDetectionLoop();
    crownCanvas.hidden = false;
  } else {
    cancelAnimationFrame(faceLoopId);
    faceLoopId = null;
    crownCanvas.hidden = true;
    faceTracker.opacity = 0;
    crownCtx && crownCtx.clearRect(0, 0, crownCanvas.width, crownCanvas.height);
  }

  // Pose
  if (togglePose.checked && poseLandmarker && medalImage) {
    if (!poseLoopId) runPoseDetectionLoop();
    medalCanvas.hidden = false;
  } else {
    cancelAnimationFrame(poseLoopId);
    poseLoopId = null;
    medalCanvas.hidden = true;
    poseTracker.opacity = 0;
    medalCtx && medalCtx.clearRect(0, 0, medalCanvas.width, medalCanvas.height);
  }

  // Hand
  if (toggleHand.checked && handLandmarker && trophyImage) {
    if (!handLoopId) runHandDetectionLoop();
    trophyCanvas.hidden = false;
  } else {
    cancelAnimationFrame(handLoopId);
    handLoopId = null;
    trophyCanvas.hidden = true;
    handTracker.opacity = 0;
    trophyCtx && trophyCtx.clearRect(0, 0, trophyCanvas.width, trophyCanvas.height);
  }
}

// ==================== Coordinate Mapping ====================

// Maps normalized video coordinates (0-1) to Preview Canvas pixels (handling object-fit: cover)
function mapVideoToPreview(normX, normY, previewW, previewH, videoW, videoH) {
  const videoAspect = videoW / videoH;
  const previewAspect = previewW / previewH;

  let scale, displayedW, displayedH, offsetX, offsetY;

  if (videoAspect > previewAspect) {
    // Video is wider: crop sides
    scale = previewH / videoH;
    displayedW = videoW * scale;
    displayedH = previewH;
    offsetX = (previewW - displayedW) / 2;
    offsetY = 0;
  } else {
    // Video is taller: crop top/bottom
    scale = previewW / videoWidth; // typo fixed below
    scale = previewW / videoW;
    displayedW = previewW;
    displayedH = videoH * scale;
    offsetX = 0;
    offsetY = (previewH - displayedH) / 2;
  }

  return {
    x: (normX * displayedW) + offsetX,
    y: (normY * displayedH) + offsetY,
    scale: scale
  };
}

// ==================== Detection Loops ====================

let lastFaceTime = 0;
function runFaceDetectionLoop() {
  faceLoopId = requestAnimationFrame(runFaceDetectionLoop);
  const now = performance.now();
  if (now - lastFaceTime >= 1000 / FACE_DETECTION_FPS) {
    lastFaceTime = now;
    detectFace(now);
  }
}

let lastPoseTime = 0;
function runPoseDetectionLoop() {
  poseLoopId = requestAnimationFrame(runPoseDetectionLoop);
  const now = performance.now();
  if (now - lastPoseTime >= 1000 / POSE_DETECTION_FPS) {
    lastPoseTime = now;
    detectPose(now);
  }
}

let lastHandTime = 0;
function runHandDetectionLoop() {
  handLoopId = requestAnimationFrame(runHandDetectionLoop);
  const now = performance.now();
  if (now - lastHandTime >= 1000 / HAND_DETECTION_FPS) {
    lastHandTime = now;
    detectHand(now);
  }
}

// ==================== Detection Logic ====================

function detectFace(now) {
  let rawTransform = null;
  try {
    const results = faceLandmarker.detectForVideo(cameraEl, now);
    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      rawTransform = computeCrownTransform(results.faceLandmarks[0]);
    }
  } catch (e) { console.warn(e); }

  const smoothed = faceTracker.update(rawTransform, now);
  drawOverlay(crownCtx, crownImage, smoothed, crownCanvas);
}

function detectPose(now) {
  let rawTransform = null;
  try {
    const results = poseLandmarker.detectForVideo(cameraEl, now);
    if (results.landmarks && results.landmarks.length > 0) {
      rawTransform = computeMedalTransform(results.landmarks[0]);
    }
  } catch (e) { console.warn(e); }

  const smoothed = poseTracker.update(rawTransform, now);
  drawOverlay(medalCtx, medalImage, smoothed, medalCanvas);
}

function detectHand(now) {
  let rawTransform = null;
  try {
    const results = handLandmarker.detectForVideo(cameraEl, now);
    if (results.landmarks && results.landmarks.length > 0) {
      rawTransform = computeTrophyTransform(results.landmarks[0]);
    }
  } catch (e) { console.warn(e); }

  const smoothed = handTracker.update(rawTransform, now);
  drawOverlay(trophyCtx, trophyImage, smoothed, trophyCanvas);
}

// ==================== Transform Calculations ====================

function computeCrownTransform(landmarks) {
  // Use eyes for better stability
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];

  if (!leftEye || !rightEye) return null;

  const w = previewEl.clientWidth;
  const h = previewEl.clientHeight;
  const vw = cameraEl.videoWidth;
  const vh = cameraEl.videoHeight;

  // Map normalized coords to preview coords directly
  const pLeft = mapVideoToPreview(leftEye.x, leftEye.y, w, h, vw, vh);
  const pRight = mapVideoToPreview(rightEye.x, rightEye.y, w, h, vw, vh);

  const eyeDist = Math.hypot(pRight.x - pLeft.x, pRight.y - pLeft.y);

  // Center between eyes
  const centerX = (pLeft.x + pRight.x) / 2;
  const centerY = (pLeft.y + pRight.y) / 2;
  const eyeAngle = Math.atan2(pRight.y - pLeft.y, pRight.x - pLeft.x);

  const crownWidth = eyeDist * CROWN_SIZE_RATIO;
  const crownHeight = crownWidth * (crownImage.height / crownImage.width);

  return {
    x: centerX,
    y: centerY - (crownHeight * CROWN_Y_OFFSET_RATIO),
    width: crownWidth,
    height: crownHeight,
    angle: eyeAngle
  };
}

function computeMedalTransform(landmarks) {
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];

  if (!leftShoulder || !rightShoulder) return null;
  if ((leftShoulder.visibility ?? 1) < 0.5 || (rightShoulder.visibility ?? 1) < 0.5) return null;

  const w = previewEl.clientWidth;
  const h = previewEl.clientHeight;
  const vw = cameraEl.videoWidth;
  const vh = cameraEl.videoHeight;

  const pLeft = mapVideoToPreview(leftShoulder.x, leftShoulder.y, w, h, vw, vh);
  const pRight = mapVideoToPreview(rightShoulder.x, rightShoulder.y, w, h, vw, vh);

  const shoulderWidth = Math.hypot(pRight.x - pLeft.x, pRight.y - pLeft.y);
  const shoulderAngle = Math.atan2(pRight.y - pLeft.y, pRight.x - pLeft.x);

  const centerX = (pLeft.x + pRight.x) / 2;
  const centerY = (pLeft.y + pRight.y) / 2;

  const medalWidth = shoulderWidth * MEDAL_SIZE_RATIO;
  const medalHeight = medalWidth * (medalImage.height / medalImage.width);

  return {
    x: centerX,
    y: centerY + (shoulderWidth * MEDAL_Y_OFFSET_RATIO),
    width: medalWidth,
    height: medalHeight,
    angle: shoulderAngle // Use shoulder rotation
  };
}

function computeTrophyTransform(landmarks) {
  const wrist = landmarks[0];
  const indexBase = landmarks[5]; // Index finger MCP
  const pinkyBase = landmarks[17]; // Pinky MCP

  if (!wrist || !indexBase || !pinkyBase) return null;

  const w = previewEl.clientWidth;
  const h = previewEl.clientHeight;
  const vw = cameraEl.videoWidth;
  const vh = cameraEl.videoHeight;

  const pWrist = mapVideoToPreview(wrist.x, wrist.y, w, h, vw, vh);
  const pIndex = mapVideoToPreview(indexBase.x, indexBase.y, w, h, vw, vh);
  const pPinky = mapVideoToPreview(pinkyBase.x, pinkyBase.y, w, h, vw, vh);

  // Calculate palm size and center
  // Palm center roughly between middle of index/pinky base and wrist
  const palmCenterX = (pWrist.x + pIndex.x + pPinky.x) / 3;
  const palmCenterY = (pWrist.y + pIndex.y + pPinky.y) / 3;

  // Hand scale based on Wrist to index base (stable bone)
  const handSize = Math.hypot(pIndex.x - pWrist.x, pIndex.y - pWrist.y);

  const trophyWidth = handSize * TROPHY_SIZE_RATIO;
  const trophyHeight = trophyWidth * (trophyImage.height / trophyImage.width);

  return {
    x: palmCenterX,
    y: palmCenterY + (handSize * TROPHY_Y_OFFSET_RATIO),
    width: trophyWidth,
    height: trophyHeight,
    angle: 0 // Keep trophy upright or could align with hand
  };
}

// ==================== Utilities ====================

function drawOverlay(ctx, img, t, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!t || t.opacity <= 0.01) return;

  ctx.save();
  ctx.globalAlpha = t.opacity;
  ctx.translate(t.x, t.y + t.height / 2);
  ctx.rotate(t.angle);
  ctx.drawImage(img, -t.width / 2, -t.height / 2, t.width, t.height);
  ctx.restore();
}

function updateCanvasSizes() {
  if (!previewEl) return;
  const rect = previewEl.getBoundingClientRect();
  [crownCanvas, medalCanvas, trophyCanvas].forEach(c => {
    if (c && (c.width !== rect.width || c.height !== rect.height)) {
      c.width = rect.width;
      c.height = rect.height;
    }
  });
}

// ==================== Camera & Capture ====================

async function startCamera(facingMode = 'environment') {
  if (!navigator.mediaDevices?.getUserMedia) return alert('Camera not supported');
  try {
    if (mediaStream) stopCamera();
    mediaStream = await getCameraStream(facingMode);
    await applyBestAvailableResolution(mediaStream);
    cameraEl.srcObject = mediaStream;
    await cameraEl.play();
    updateDetectionStatus();
  } catch (e) {
    console.error(e);
    alert('Camera access failed');
  }
}

function stopCamera() {
  mediaStream?.getTracks().forEach(t => t.stop());
  cameraEl.srcObject = null;
  mediaStream = null;
  cancelAnimationFrame(faceLoopId);
  cancelAnimationFrame(poseLoopId);
  cancelAnimationFrame(handLoopId);
}

async function getCameraStream(facingMode) {
  try {
    return await navigator.mediaDevices.getUserMedia(createConstraints(CAMERA_CONSTRAINTS, facingMode));
  } catch (e) {
    if (e.name === 'OverconstrainedError') {
      return navigator.mediaDevices.getUserMedia(createConstraints(FALLBACK_CAMERA_CONSTRAINTS, facingMode));
    }
    throw e;
  }
}

async function applyBestAvailableResolution(stream) {
  const [track] = stream?.getVideoTracks() ?? [];
  if (!track?.getCapabilities) return;
  const caps = track.getCapabilities();
  const settings = track.getSettings();
  const w = settings.width ?? caps.width?.max ?? 0;
  const h = settings.height ?? caps.height?.max ?? 0;
  const landscape = w >= h;
  const tLong = clampResolutionValue(TARGET_CAPTURE_LONG_EDGE, landscape ? caps.width : caps.height);
  const tShort = clampResolutionValue(TARGET_CAPTURE_SHORT_EDGE, landscape ? caps.height : caps.width);
  try {
    await track.applyConstraints(landscape ? { width: tLong, height: tShort } : { width: tShort, height: tLong });
  } catch (e) { console.warn(e); }
}

async function handleScreenshotCapture() {
  if (isCapturingScreenshot || !cameraEl || cameraEl.readyState < 2) return;
  isCapturingScreenshot = true;
  shutterButton.classList.add('capturing');
  try {
    const canvas = composeScreenshot();
    const blob = await new Promise((res, rej) => canvas.toBlob(b => b ? res(b) : rej(), 'image/png', 0.95));
    openShareModal(URL.createObjectURL(blob), blob, `arframe-${Date.now()}.png`);
  } catch (e) {
    alert('Capture failed');
    console.error(e);
  } finally {
    setTimeout(() => shutterButton.classList.remove('capturing'), 350);
    isCapturingScreenshot = false;
  }
}

function composeScreenshot() {
  const w = cameraEl.videoWidth || 720;
  const h = cameraEl.videoHeight || 1280;
  const pw = previewEl?.clientWidth;
  const ph = previewEl?.clientHeight;
  const tAspect = (pw && ph) ? pw / ph : w / h;
  const vAspect = w / h;

  // 1. Draw Camera (Background) with Crop
  // Logic here effectively replicates "object-fit: cover" for the Canvas
  let sx = 0, sy = 0, sw = w, sh = h;
  if (vAspect > tAspect) {
    sw = h * tAspect;
    sw = Math.min(w, sw * (1 + CAMERA_CROP_PADDING_RATIO));
    sx = (w - sw) / 2;
  } else {
    sh = w / tAspect;
    sh = Math.min(h, sh * (1 + CAMERA_CROP_PADDING_RATIO));
    sy = (h - sh) / 2;
  }

  captureCanvas.width = Math.round(sw);
  captureCanvas.height = Math.round(sh);
  const ctx = captureCanvas.getContext('2d');
  ctx.drawImage(cameraEl, sx, sy, sw, sh, 0, 0, captureCanvas.width, captureCanvas.height);

  // 2. Draw Overlays
  // Since our trackers are already in "Preview Pixels" (displayed coordinates),
  // we just need to scale them to "Capture Pixels"

  const scaleX = captureCanvas.width / pw;
  const scaleY = captureCanvas.height / ph;

  // Face
  if (toggleFace.checked && faceTracker && faceTracker.lastTransform && faceTracker.opacity > 0) {
    drawOverlayOnCapture(ctx, crownImage, faceTracker.lastTransform, scaleX, scaleY, faceTracker.opacity);
  }
  // Pose
  if (togglePose.checked && poseTracker && poseTracker.lastTransform && poseTracker.opacity > 0) {
    drawOverlayOnCapture(ctx, medalImage, poseTracker.lastTransform, scaleX, scaleY, poseTracker.opacity);
  }
  // Hand
  if (toggleHand.checked && handTracker && handTracker.lastTransform && handTracker.opacity > 0) {
    drawOverlayOnCapture(ctx, trophyImage, handTracker.lastTransform, scaleX, scaleY, handTracker.opacity);
  }

  // 3. Draw Frame
  const fEl = FRAMES[activeFrameIndex].type === 'video' ? frameVideoEl : frameImageEl;
  if (fEl && !fEl.hidden) ctx.drawImage(fEl, 0, 0, captureCanvas.width, captureCanvas.height);

  return captureCanvas;
}

function drawOverlayOnCapture(ctx, img, t, scaleX, scaleY, opacity) {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.translate(t.x * scaleX, (t.y * scaleY) + (t.height * scaleY) / 2);
  ctx.rotate(t.angle);
  ctx.drawImage(img, -(t.width * scaleX) / 2, -(t.height * scaleY) / 2, t.width * scaleX, t.height * scaleY);
  ctx.restore();
}

function openShareModal(url, blob, name) {
  if (currentCapture?.objectUrl) URL.revokeObjectURL(currentCapture.objectUrl);
  currentCapture = { objectUrl: url, blob, filename: name };
  capturedImage.src = url;
  shareModal.classList.remove('hidden');
}

function init() {
  cameraEl = $('#camera');
  frameImageEl = $('#frameImage');
  frameVideoEl = $('#frameVideo');
  shutterButton = $('#shutterButton');
  captureCanvas = $('#captureCanvas');
  previewEl = $('.preview');
  shareModal = $('#share-modal');
  capturedImage = $('#captured-image');

  crownCanvas = $('#crownCanvas');
  medalCanvas = $('#medalCanvas');
  trophyCanvas = $('#trophyCanvas');
  crownCtx = crownCanvas.getContext('2d');
  medalCtx = medalCanvas.getContext('2d');
  trophyCtx = trophyCanvas.getContext('2d');

  toggleFace = $('#toggleFace');
  togglePose = $('#togglePose');
  toggleHand = $('#toggleHand');

  // Initialize Trackers with Smoothing
  faceTracker = new SmoothedLandmark({ minCutoff: 0.05, beta: 5.0 });
  poseTracker = new SmoothedLandmark({ minCutoff: 0.1, beta: 2.0 });
  handTracker = new SmoothedLandmark({ minCutoff: 0.5, beta: 10.0 }); // Fast response for hand

  updateCanvasSizes();
  window.addEventListener('resize', updateCanvasSizes);

  // Frame animation
  const safari = isSafari();
  if (safari) {
    frameVideoEl.hidden = true;
    frameImageEl.hidden = false;
    iosFrameTimer = setInterval(() => {
      iosFrameIndex = (iosFrameIndex + 1) % 64;
      frameImageEl.src = IOS_FRAME_IMAGES[iosFrameIndex];
    }, IOS_FRAME_INTERVAL_MS);
  } else {
    frameImageEl.hidden = true;
    frameVideoEl.hidden = false;
    frameVideoEl.play().catch(() => { });
  }

  // Toggles
  [toggleFace, togglePose, toggleHand].forEach(t => t.addEventListener('change', updateDetectionStatus));

  // Screenshot & Share
  shutterButton.addEventListener('click', handleScreenshotCapture);
  $('#close-modal').addEventListener('click', () => shareModal.classList.add('hidden'));
  $('#share-button').addEventListener('click', async () => {
    if (navigator.share) {
      try { await navigator.share({ files: [new File([currentCapture.blob], currentCapture.filename, { type: 'image/png' })] }); }
      catch (e) { console.warn(e); }
    } else {
      const a = document.createElement('a');
      a.href = currentCapture.objectUrl;
      a.download = currentCapture.filename;
      a.click();
    }
  });

  // Start Camera Immediately
  startCamera();

  // Load assets & Landmarkers in background
  Promise.all([
    loadImage(CROWN_IMAGE_PATH).then(img => crownImage = img),
    loadImage(MEDAL_IMAGE_PATH).then(img => medalImage = img),
    loadImage(TROPHY_IMAGE_PATH).then(img => trophyImage = img),
    initLandmarkers()
  ]).then(() => {
    console.log('Assets and Landmarkers loaded');
    updateDetectionStatus();
  });
}

document.addEventListener('DOMContentLoaded', init);
const TARGET_CAPTURE_LONG_EDGE = 1920;
const CAMERA_CROP_PADDING_RATIO = 0.08;
const MEDIAPIPE_WASM_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';

// Models
const FACE_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const POSE_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const HAND_LANDMARKER_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// Assets
const CROWN_IMAGE_PATH = 'assets/crown.png';
const MEDAL_IMAGE_PATH = 'assets/medal.png';
const TROPHY_IMAGE_PATH = 'assets/trophy.png';

// FPS Limits (to reduce load)
const FACE_DETECTION_FPS = 15;
const POSE_DETECTION_FPS = 10;
const HAND_DETECTION_FPS = 10;
const SMOOTHING_FACTOR = 0.3;

// Transform Constants
// Crown
const CROWN_SIZE_RATIO = 1.6;
const CROWN_Y_OFFSET_RATIO = 0.8;
// Medal
const MEDAL_SIZE_RATIO = 0.6;
const MEDAL_Y_OFFSET_RATIO = 0.25;
// Trophy
const TROPHY_SIZE_RATIO = 3.8;
const TROPHY_Y_OFFSET_RATIO = -2.0;

const CAMERA_CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: 'environment',
    width: { ideal: TARGET_CAPTURE_SHORT_EDGE },
    height: { ideal: TARGET_CAPTURE_LONG_EDGE },
  },
};

const FALLBACK_CAMERA_CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: 'environment',
    width: { ideal: 720 },
    height: { ideal: 1280 },
  },
};

const FRAMES = [
  { id: 'animated-frame', type: 'video', webmSrc: 'assets/frame_alpha.webm', mp4Src: 'assets/frame_alpha.mp4', label: 'animated' },
  { id: 'fallback-image', type: 'image', imageSrc: 'assets/ios_frame_001.png', label: 'fallback' },
];
const FALLBACK_FRAME_INDEX = FRAMES.findIndex(f => f.type === 'image');
const IOS_FRAME_IMAGES = Array.from({ length: 64 }, (_, i) => `assets/ios_frame_${String(i + 1).padStart(3, '0')}.png`);
const IOS_FRAME_INTERVAL_MS = 100;

// Global State
let activeFrameIndex = 0;
let mediaStream = null;
let isCapturingScreenshot = false;
let currentCapture = null;
let iosFrameIndex = 0;
let iosFrameTimer = null;
let shouldResumeCamera = false;

// DOM Elements
let cameraEl, frameImageEl, frameVideoEl, shutterButton, captureCanvas, previewEl;
let crownCanvas, medalCanvas, trophyCanvas;
let crownCtx, medalCtx, trophyCtx;
let toggleFace, togglePose, toggleHand;
let shareModal, capturedImage;

// Landmarkers & Images
let faceLandmarker, poseLandmarker, handLandmarker;
let crownImage, medalImage, trophyImage;

// Detection State (Running, Last Time, Last Transform)
let faceState = { running: false, lastTime: 0, lastTransform: null };
let poseState = { running: false, lastTime: 0, lastTransform: null };
let handState = { running: false, lastTime: 0, lastTransform: null };

function $(selector) { return document.querySelector(selector); }

function clampResolutionValue(preferred, range) {
  if (!range) return preferred;
  let value = preferred;
  if (typeof range.max === 'number') value = Math.min(value, range.max);
  if (typeof range.min === 'number') value = Math.max(value, range.min);
  return Math.round(value);
}

function createConstraints(baseConstraints, facingMode) {
  return { ...baseConstraints, video: { ...baseConstraints.video, facingMode } };
}

function isSafari() {
  const ua = navigator.userAgent;
  return /Safari/i.test(ua) && !/Chrome|Chromium|CriOS|Edg|OPR/i.test(ua);
}

function canPlayVp9WebM() {
  const video = document.createElement('video');
  return video && typeof video.canPlayType === 'function' &&
    (video.canPlayType('video/webm; codecs="vp9"') === 'probably' || video.canPlayType('video/webm; codecs="vp9"') === 'maybe');
}

// ==================== Initialization & Loading ====================

async function initLandmarkers() {
  try {
    const { FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker } = await import('@mediapipe/tasks-vision');

    const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_PATH);

    // Initialize in parallel
    const pFace = FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_LANDMARKER_MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO', numFaces: 1
    });
    const pPose = PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: POSE_LANDMARKER_MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO', numPoses: 1
    });
    const pHand = HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_LANDMARKER_MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO', numHands: 1
    });

    [faceLandmarker, poseLandmarker, handLandmarker] = await Promise.all([pFace, pPose, pHand]);
    console.log('All Landmarkers initialized');
    updateDetectionStatus();
  } catch (error) {
    console.warn('Landmarker initialization failed:', error);
    alert('Failed to load AR models. Please check your internet connection.');
  }
}

function loadImage(src) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => { console.warn('Failed to load:', src); resolve(null); };
    img.src = src;
  });
}

function updateDetectionStatus() {
  // Start/Stop based on toggles and camera/model readiness
  if (!cameraEl || cameraEl.readyState < 2) return;

  // Face
  if (toggleFace.checked && faceLandmarker && crownImage) {
    if (!faceState.running) { faceState.running = true; runFaceDetectionLoop(); }
    crownCanvas.hidden = false;
  } else {
    faceState.running = false;
    crownCanvas.hidden = true;
    faceState.lastTransform = null;
    crownCtx && crownCtx.clearRect(0, 0, crownCanvas.width, crownCanvas.height);
  }

  // Pose
  if (togglePose.checked && poseLandmarker && medalImage) {
    if (!poseState.running) { poseState.running = true; runPoseDetectionLoop(); }
    medalCanvas.hidden = false;
  } else {
    poseState.running = false;
    medalCanvas.hidden = true;
    poseState.lastTransform = null;
    medalCtx && medalCtx.clearRect(0, 0, medalCanvas.width, medalCanvas.height);
  }

  // Hand
  if (toggleHand.checked && handLandmarker && trophyImage) {
    if (!handState.running) { handState.running = true; runHandDetectionLoop(); }
    trophyCanvas.hidden = false;
  } else {
    handState.running = false;
    trophyCanvas.hidden = true;
    handState.lastTransform = null;
    trophyCtx && trophyCtx.clearRect(0, 0, trophyCanvas.width, trophyCanvas.height);
  }
}

// ==================== Detection Loops ====================

function runFaceDetectionLoop() {
  if (!faceState.running) return;
  const now = Date.now();
  if (now - faceState.lastTime >= 1000 / FACE_DETECTION_FPS) {
    faceState.lastTime = now;
    detectFace();
  }
  requestAnimationFrame(runFaceDetectionLoop);
}

function runPoseDetectionLoop() {
  if (!poseState.running) return;
  const now = Date.now();
  if (now - poseState.lastTime >= 1000 / POSE_DETECTION_FPS) {
    poseState.lastTime = now;
    detectPose();
  }
  requestAnimationFrame(runPoseDetectionLoop);
}

function runHandDetectionLoop() {
  if (!handState.running) return;
  const now = Date.now();
  if (now - handState.lastTime >= 1000 / HAND_DETECTION_FPS) {
    handState.lastTime = now;
    detectHand();
  }
  requestAnimationFrame(runHandDetectionLoop);
}

// ==================== Detection Logic ====================

function detectFace() {
  try {
    const results = faceLandmarker.detectForVideo(cameraEl, performance.now());
    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      const transform = computeCrownTransform(results.faceLandmarks[0]);
      if (transform) {
        const smoothed = smoothTransform(faceState.lastTransform, transform);
        drawOverlay(crownCtx, crownImage, smoothed, crownCanvas);
        faceState.lastTransform = smoothed;
        return;
      }
    }
  } catch (e) { console.warn(e); }
  crownCtx.clearRect(0, 0, crownCanvas.width, crownCanvas.height);
  faceState.lastTransform = null;
}

function detectPose() {
  try {
    const results = poseLandmarker.detectForVideo(cameraEl, performance.now());
    if (results.landmarks && results.landmarks.length > 0) {
      const transform = computeMedalTransform(results.landmarks[0]);
      if (transform) {
        const smoothed = smoothTransform(poseState.lastTransform, transform);
        drawOverlay(medalCtx, medalImage, smoothed, medalCanvas);
        poseState.lastTransform = smoothed;
        return;
      }
    }
  } catch (e) { console.warn(e); }
  medalCtx.clearRect(0, 0, medalCanvas.width, medalCanvas.height);
  poseState.lastTransform = null;
}

function detectHand() {
  try {
    const results = handLandmarker.detectForVideo(cameraEl, performance.now());
    if (results.landmarks && results.landmarks.length > 0) {
      const transform = computeTrophyTransform(results.landmarks[0]);
      if (transform) {
        const smoothed = smoothTransform(handState.lastTransform, transform);
        drawOverlay(trophyCtx, trophyImage, smoothed, trophyCanvas);
        handState.lastTransform = smoothed;
        return;
      }
    }
  } catch (e) { console.warn(e); }
  trophyCtx.clearRect(0, 0, trophyCanvas.width, trophyCanvas.height);
  handState.lastTransform = null;
}

// ==================== Transform Calculations ====================

function computeCrownTransform(landmarks) {
  const forehead = landmarks[10];
  const leftTemple = landmarks[234];
  const rightTemple = landmarks[454];
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];

  if (!forehead || !leftTemple || !rightTemple || !leftEye || !rightEye) return null;

  const w = crownCanvas.width;
  const h = crownCanvas.height;
  const foreheadX = forehead.x * w;
  const foreheadY = forehead.y * h;
  const leftX = leftTemple.x * w;
  const rightX = rightTemple.x * w;
  const leftY = leftTemple.y * h;
  const rightY = rightTemple.y * h;

  const faceWidth = Math.sqrt(Math.pow(rightX - leftX, 2) + Math.pow(rightY - leftY, 2));
  const crownWidth = faceWidth * CROWN_SIZE_RATIO;
  const crownHeight = crownWidth * (crownImage.height / crownImage.width);

  const eyeAngle = Math.atan2(
    (rightEye.y * h) - (leftEye.y * h),
    (rightEye.x * w) - (leftEye.x * w)
  );

  return {
    x: foreheadX,
    y: foreheadY - (crownHeight * CROWN_Y_OFFSET_RATIO),
    width: crownWidth,
    height: crownHeight,
    angle: eyeAngle
  };
}

function computeMedalTransform(landmarks) {
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  if (!leftShoulder || !rightShoulder) return null;
  if ((leftShoulder.visibility ?? 0) < 0.5 || (rightShoulder.visibility ?? 0) < 0.5) return null;

  const w = medalCanvas.width;
  const h = medalCanvas.height;
  const lx = leftShoulder.x * w;
  const ly = leftShoulder.y * h;
  const rx = rightShoulder.x * w;
  const ry = rightShoulder.y * h;

  const shoulderWidth = Math.sqrt(Math.pow(rx - lx, 2) + Math.pow(ry - ly, 2));
  const centerX = (lx + rx) / 2;
  const centerY = (ly + ry) / 2;
  const medalWidth = shoulderWidth * MEDAL_SIZE_RATIO;
  const medalHeight = medalWidth * (medalImage.height / medalImage.width);

  return {
    x: centerX,
    y: centerY + (shoulderWidth * MEDAL_Y_OFFSET_RATIO),
    width: medalWidth,
    height: medalHeight,
    angle: 0 // No rotation for medal per latest fix
  };
}

function computeTrophyTransform(landmarks) {
  const wrist = landmarks[0];
  const middleBase = landmarks[9];
  if (!wrist || !middleBase) return null;

  const w = trophyCanvas.width;
  const h = trophyCanvas.height;
  const wx = wrist.x * w;
  const wy = wrist.y * h;
  const mx = middleBase.x * w;
  const my = middleBase.y * h;

  const handSize = Math.sqrt(Math.pow(mx - wx, 2) + Math.pow(my - wy, 2));

  // Palm center
  let pcx = wx, pcy = wy;
  const indices = [0, 5, 9, 13, 17]; // wrist + fingers
  let validPoints = 0;
  let sumX = 0, sumY = 0;
  indices.forEach(idx => {
    if (landmarks[idx]) {
      sumX += landmarks[idx].x * w;
      sumY += landmarks[idx].y * h;
      validPoints++;
    }
  });
  if (validPoints > 0) { pcx = sumX / validPoints; pcy = sumY / validPoints; }

  const trophyWidth = handSize * TROPHY_SIZE_RATIO;
  const trophyHeight = trophyWidth * (trophyImage.height / trophyImage.width);

  return {
    x: pcx,
    y: pcy + (handSize * TROPHY_Y_OFFSET_RATIO),
    width: trophyWidth,
    height: trophyHeight,
    angle: 0 // No rotation for trophy per latest fix
  };
}

// ==================== Utilities ====================

function smoothTransform(last, current) {
  if (!last) return current;
  return {
    x: lerp(last.x, current.x, SMOOTHING_FACTOR),
    y: lerp(last.y, current.y, SMOOTHING_FACTOR),
    width: lerp(last.width, current.width, SMOOTHING_FACTOR),
    height: lerp(last.height, current.height, SMOOTHING_FACTOR),
    angle: lerpAngle(last.angle, current.angle, SMOOTHING_FACTOR)
  };
}

function lerp(a, b, t) { return a + (b - a) * t; }
function lerpAngle(a, b, t) {
  let diff = b - a;
  while (diff > Math.PI) diff -= Math.PI * 2;
  while (diff < -Math.PI) diff += Math.PI * 2;
  return a + diff * t;
}

function drawOverlay(ctx, img, t, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.translate(t.x, t.y + t.height / 2);
  ctx.rotate(t.angle);
  ctx.drawImage(img, -t.width / 2, -t.height / 2, t.width, t.height);
  ctx.restore();
}

function updateCanvasSizes() {
  if (!previewEl) return;
  const rect = previewEl.getBoundingClientRect();
  [crownCanvas, medalCanvas, trophyCanvas].forEach(c => {
    if (c && (c.width !== rect.width || c.height !== rect.height)) {
      c.width = rect.width;
      c.height = rect.height;
    }
  });
}

// ==================== Camera & Capture ====================

async function startCamera(facingMode = 'environment') {
  if (!navigator.mediaDevices?.getUserMedia) return alert('Camera not supported');
  try {
    if (mediaStream) stopCamera();
    mediaStream = await getCameraStream(facingMode);
    await applyBestAvailableResolution(mediaStream);
    cameraEl.srcObject = mediaStream;
    await cameraEl.play();
    updateDetectionStatus();
  } catch (e) {
    console.error(e);
    alert('Camera access failed');
  }
}

function stopCamera() {
  mediaStream?.getTracks().forEach(t => t.stop());
  cameraEl.srcObject = null;
  mediaStream = null;
  [faceState, poseState, handState].forEach(s => s.running = false);
}

async function getCameraStream(facingMode) {
  try {
    return await navigator.mediaDevices.getUserMedia(createConstraints(CAMERA_CONSTRAINTS, facingMode));
  } catch (e) {
    if (e.name === 'OverconstrainedError') {
      return navigator.mediaDevices.getUserMedia(createConstraints(FALLBACK_CAMERA_CONSTRAINTS, facingMode));
    }
    throw e;
  }
}

async function applyBestAvailableResolution(stream) {
  const [track] = stream?.getVideoTracks() ?? [];
  if (!track?.getCapabilities) return;
  const caps = track.getCapabilities();
  const settings = track.getSettings();
  const w = settings.width ?? caps.width?.max ?? 0;
  const h = settings.height ?? caps.height?.max ?? 0;
  const landscape = w >= h;
  const tLong = clampResolutionValue(TARGET_CAPTURE_LONG_EDGE, landscape ? caps.width : caps.height);
  const tShort = clampResolutionValue(TARGET_CAPTURE_SHORT_EDGE, landscape ? caps.height : caps.width);
  try {
    await track.applyConstraints(landscape ? { width: tLong, height: tShort } : { width: tShort, height: tLong });
  } catch (e) { console.warn(e); }
}

async function handleScreenshotCapture() {
  if (isCapturingScreenshot || !cameraEl || cameraEl.readyState < 2) return;
  isCapturingScreenshot = true;
  shutterButton.classList.add('capturing');
  try {
    const canvas = composeScreenshot();
    const blob = await new Promise((res, rej) => canvas.toBlob(b => b ? res(b) : rej(), 'image/png', 0.95));
    openShareModal(URL.createObjectURL(blob), blob, `arframe-${Date.now()}.png`);
  } catch (e) {
    alert('Capture failed');
    console.error(e);
  } finally {
    setTimeout(() => shutterButton.classList.remove('capturing'), 350);
    isCapturingScreenshot = false;
  }
}

function composeScreenshot() {
  const w = cameraEl.videoWidth || 720;
  const h = cameraEl.videoHeight || 1280;
  const pw = previewEl?.clientWidth;
  const ph = previewEl?.clientHeight;
  const tAspect = (pw && ph) ? pw / ph : w / h;
  const vAspect = w / h;

  let sx = 0, sy = 0, sw = w, sh = h;
  if (vAspect > tAspect) {
    sw = h * tAspect;
    sw = Math.min(w, sw * (1 + CAMERA_CROP_PADDING_RATIO));
    sx = (w - sw) / 2;
  } else {
    sh = w / tAspect;
    sh = Math.min(h, sh * (1 + CAMERA_CROP_PADDING_RATIO));
    sy = (h - sh) / 2;
  }

  captureCanvas.width = Math.round(sw);
  captureCanvas.height = Math.round(sh);
  const ctx = captureCanvas.getContext('2d');
  ctx.drawImage(cameraEl, sx, sy, sw, sh, 0, 0, captureCanvas.width, captureCanvas.height);

  // Draw Overlays if enabled
  const outputW = captureCanvas.width;
  const outputH = captureCanvas.height;

  // Face
  if (toggleFace.checked && faceState.lastTransform && crownImage) {
    drawOverlayOnCapture(ctx, crownImage, faceState.lastTransform, crownCanvas, outputW, outputH);
  }
  // Pose
  if (togglePose.checked && poseState.lastTransform && medalImage) {
    drawOverlayOnCapture(ctx, medalImage, poseState.lastTransform, medalCanvas, outputW, outputH);
  }
  // Hand
  if (toggleHand.checked && handState.lastTransform && trophyImage) {
    drawOverlayOnCapture(ctx, trophyImage, handState.lastTransform, trophyCanvas, outputW, outputH);
  }

  // Frame
  const fEl = FRAMES[activeFrameIndex].type === 'video' ? frameVideoEl : frameImageEl;
  if (fEl && !fEl.hidden) ctx.drawImage(fEl, 0, 0, outputW, outputH);

  return captureCanvas;
}

function drawOverlayOnCapture(ctx, img, t, cvs, outW, outH) {
  // Scale transform from preview coords to capture coords
  const scaleX = outW / cvs.width;
  const scaleY = outH / cvs.height;

  ctx.save();
  ctx.translate(t.x * scaleX, (t.y * scaleY) + (t.height * scaleY) / 2);
  ctx.rotate(t.angle);
  ctx.drawImage(img, -(t.width * scaleX) / 2, -(t.height * scaleY) / 2, t.width * scaleX, t.height * scaleY);
  ctx.restore();
}

function openShareModal(url, blob, name) {
  if (currentCapture?.objectUrl) URL.revokeObjectURL(currentCapture.objectUrl);
  currentCapture = { objectUrl: url, blob, filename: name };
  capturedImage.src = url;
  shareModal.classList.remove('hidden');
}

function init() {
  cameraEl = $('#camera');
  frameImageEl = $('#frameImage');
  frameVideoEl = $('#frameVideo');
  shutterButton = $('#shutterButton');
  captureCanvas = $('#captureCanvas');
  previewEl = $('.preview');
  shareModal = $('#share-modal');
  capturedImage = $('#captured-image');

  crownCanvas = $('#crownCanvas');
  medalCanvas = $('#medalCanvas');
  trophyCanvas = $('#trophyCanvas');
  crownCtx = crownCanvas.getContext('2d');
  medalCtx = medalCanvas.getContext('2d');
  trophyCtx = trophyCanvas.getContext('2d');

  toggleFace = $('#toggleFace');
  togglePose = $('#togglePose');
  toggleHand = $('#toggleHand');

  updateCanvasSizes();
  window.addEventListener('resize', updateCanvasSizes);

  // Frame animation
  const safari = isSafari();
  if (safari) {
    frameVideoEl.hidden = true;
    frameImageEl.hidden = false;
    iosFrameTimer = setInterval(() => {
      iosFrameIndex = (iosFrameIndex + 1) % 64;
      frameImageEl.src = IOS_FRAME_IMAGES[iosFrameIndex];
    }, IOS_FRAME_INTERVAL_MS);
  } else {
    frameImageEl.hidden = true;
    frameVideoEl.hidden = false;
    frameVideoEl.play().catch(() => { });
  }

  // Toggles
  [toggleFace, togglePose, toggleHand].forEach(t => t.addEventListener('change', updateDetectionStatus));

  // Screenshot & Share
  shutterButton.addEventListener('click', handleScreenshotCapture);
  $('#close-modal').addEventListener('click', () => shareModal.classList.add('hidden'));
  $('#share-button').addEventListener('click', async () => {
    if (navigator.share) {
      try { await navigator.share({ files: [new File([currentCapture.blob], currentCapture.filename, { type: 'image/png' })] }); }
      catch (e) { console.warn(e); }
    } else {
      const a = document.createElement('a');
      a.href = currentCapture.objectUrl;
      a.download = currentCapture.filename;
      a.click();
    }
  });

  // Start Camera Immediately
  startCamera();

  // Load assets & Landmarkers in background
  Promise.all([
    loadImage(CROWN_IMAGE_PATH).then(img => crownImage = img),
    loadImage(MEDAL_IMAGE_PATH).then(img => medalImage = img),
    loadImage(TROPHY_IMAGE_PATH).then(img => trophyImage = img),
    initLandmarkers()
  ]).then(() => {
    console.log('Assets and Landmarkers loaded');
    updateDetectionStatus();
  });
}

document.addEventListener('DOMContentLoaded', init);
