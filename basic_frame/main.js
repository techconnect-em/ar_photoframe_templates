const TARGET_CAPTURE_SHORT_EDGE = 1440;
const TARGET_CAPTURE_LONG_EDGE = 1920;
const CAMERA_CROP_PADDING_RATIO = 0.08;

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
  {
    id: 'animated-frame',
    type: 'video',
    webmSrc: 'assets/frame_alpha.webm',
    mp4Src: 'assets/frame_alpha.mp4',
    label: 'animated',
  },
  {
    id: 'fallback-image',
    type: 'image',
    imageSrc: 'assets/ios_frame_001.png',
    label: 'fallback',
  },
];
const FALLBACK_FRAME_INDEX = FRAMES.findIndex(
  (frame) => frame.type === 'image',
);

const IOS_FRAME_IMAGES = [
  'assets/ios_frame_001.png',
  'assets/ios_frame_002.png',
  'assets/ios_frame_003.png',
  'assets/ios_frame_004.png',
  'assets/ios_frame_005.png',
  'assets/ios_frame_006.png',
  'assets/ios_frame_007.png',
  'assets/ios_frame_008.png',
  'assets/ios_frame_009.png',
  'assets/ios_frame_010.png',
  'assets/ios_frame_011.png',
  'assets/ios_frame_012.png',
  'assets/ios_frame_013.png',
  'assets/ios_frame_014.png',
  'assets/ios_frame_015.png',
  'assets/ios_frame_016.png',
  'assets/ios_frame_017.png',
  'assets/ios_frame_018.png',
  'assets/ios_frame_019.png',
  'assets/ios_frame_020.png',
  'assets/ios_frame_021.png',
  'assets/ios_frame_022.png',
  'assets/ios_frame_023.png',
  'assets/ios_frame_024.png',
  'assets/ios_frame_025.png',
  'assets/ios_frame_026.png',
  'assets/ios_frame_027.png',
  'assets/ios_frame_028.png',
  'assets/ios_frame_029.png',
  'assets/ios_frame_030.png',
  'assets/ios_frame_031.png',
  'assets/ios_frame_032.png',
  'assets/ios_frame_033.png',
  'assets/ios_frame_034.png',
  'assets/ios_frame_035.png',
  'assets/ios_frame_036.png',
  'assets/ios_frame_037.png',
  'assets/ios_frame_038.png',
  'assets/ios_frame_039.png',
  'assets/ios_frame_040.png',
  'assets/ios_frame_041.png',
  'assets/ios_frame_042.png',
  'assets/ios_frame_043.png',
  'assets/ios_frame_044.png',
  'assets/ios_frame_045.png',
  'assets/ios_frame_046.png',
  'assets/ios_frame_047.png',
  'assets/ios_frame_048.png',
  'assets/ios_frame_049.png',
  'assets/ios_frame_050.png',
  'assets/ios_frame_051.png',
  'assets/ios_frame_052.png',
  'assets/ios_frame_053.png',
  'assets/ios_frame_054.png',
  'assets/ios_frame_055.png',
  'assets/ios_frame_056.png',
  'assets/ios_frame_057.png',
  'assets/ios_frame_058.png',
  'assets/ios_frame_059.png',
  'assets/ios_frame_060.png',
  'assets/ios_frame_061.png',
  'assets/ios_frame_062.png',
  'assets/ios_frame_063.png',
  'assets/ios_frame_064.png',
];
const IOS_FRAME_INTERVAL_MS = 100;

let activeFrameIndex = 0;
let mediaStream = null;
let isCapturingScreenshot = false;
let currentCapture = null;

let cameraEl;
let frameImageEl;
let frameVideoEl;
let shutterButton;
let downloadLink;
let captureCanvas;
let shareModal;
let capturedImage;
let shareModalShareButton;
let shareModalCloseButton;
let cameraStatusText;
let previewEl;
let shouldResumeCamera = false;
let iosFrameIndex = 0;
let iosFrameTimer = null;

function $(selector) {
  return document.querySelector(selector);
}

function clampResolutionValue(preferred, range) {
  if (!range) {
    return preferred;
  }
  let value = preferred;
  if (typeof range.max === 'number') {
    value = Math.min(value, range.max);
  }
  if (typeof range.min === 'number') {
    value = Math.max(value, range.min);
  }
  return Math.round(value);
}

function createConstraints(baseConstraints, facingMode) {
  return {
    ...baseConstraints,
    video: { ...baseConstraints.video, facingMode },
  };
}

function isSafari() {
  const ua = navigator.userAgent;
  const isSafariLike =
    /Safari/i.test(ua) && !/Chrome|Chromium|CriOS|Edg|OPR/i.test(ua);
  return isSafariLike;
}

function canPlayVp9WebM() {
  const video = document.createElement('video');
  if (!video || typeof video.canPlayType !== 'function') {
    return false;
  }
  const result = video.canPlayType('video/webm; codecs="vp9"');
  return result === 'probably' || result === 'maybe';
}

function startIosFrameAnimation() {
  if (!frameImageEl || IOS_FRAME_IMAGES.length === 0) {
    return;
  }
  stopIosFrameAnimation();
  frameImageEl.hidden = false;
  iosFrameIndex = 0;
  frameImageEl.src = IOS_FRAME_IMAGES[iosFrameIndex];
  iosFrameTimer = window.setInterval(() => {
    iosFrameIndex = (iosFrameIndex + 1) % IOS_FRAME_IMAGES.length;
    frameImageEl.src = IOS_FRAME_IMAGES[iosFrameIndex];
  }, IOS_FRAME_INTERVAL_MS);
}

function stopIosFrameAnimation() {
  if (iosFrameTimer) {
    clearInterval(iosFrameTimer);
    iosFrameTimer = null;
  }
}

async function getCameraStream(facingMode) {
  try {
    const constraints = createConstraints(CAMERA_CONSTRAINTS, facingMode);
    return await navigator.mediaDevices.getUserMedia(constraints);
  } catch (error) {
    if (error.name === 'OverconstrainedError') {
      console.warn(
        '高解像度のカメラ取得に失敗したため、HDにフォールバックします。',
        error,
      );
      const fallbackConstraints = createConstraints(
        FALLBACK_CAMERA_CONSTRAINTS,
        facingMode,
      );
      return navigator.mediaDevices.getUserMedia(fallbackConstraints);
    }
    throw error;
  }
}

async function applyBestAvailableResolution(stream) {
  const [videoTrack] = stream?.getVideoTracks() ?? [];
  if (
    !videoTrack ||
    typeof videoTrack.getCapabilities !== 'function' ||
    typeof videoTrack.applyConstraints !== 'function'
  ) {
    return;
  }
  const capabilities = videoTrack.getCapabilities();
  if (!capabilities?.width || !capabilities?.height) {
    return;
  }
  const settings =
    typeof videoTrack.getSettings === 'function'
      ? videoTrack.getSettings()
      : {};
  const widthReference = settings.width ?? capabilities.width.max ?? 0;
  const heightReference = settings.height ?? capabilities.height.max ?? 0;
  const prefersLandscape = widthReference >= heightReference;

  const longRange = prefersLandscape ? capabilities.width : capabilities.height;
  const shortRange = prefersLandscape
    ? capabilities.height
    : capabilities.width;

  const targetLong = clampResolutionValue(TARGET_CAPTURE_LONG_EDGE, longRange);
  const targetShort = clampResolutionValue(
    TARGET_CAPTURE_SHORT_EDGE,
    shortRange,
  );

  const constraints = prefersLandscape
    ? { width: targetLong, height: targetShort }
    : { width: targetShort, height: targetLong };

  try {
    await videoTrack.applyConstraints(constraints);
  } catch (error) {
    console.warn('高解像度の制約適用に失敗しました:', error);
  }
}

async function startCamera(facingMode = 'environment') {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert('このブラウザはカメラに対応していません。');
    return;
  }

  try {
    if (mediaStream) {
      stopCamera();
    }

    mediaStream = await getCameraStream(facingMode);
    await applyBestAvailableResolution(mediaStream);
    cameraEl.srcObject = mediaStream;
    await cameraEl.play();
    updateCameraStatus(true);
  } catch (error) {
    console.error('カメラの起動に失敗しました:', error);
    alert('カメラへのアクセスが拒否されたか、利用できませんでした。');
    updateCameraStatus(false, true);
  }
}

function stopCamera() {
  if (!mediaStream) {
    return;
  }
  mediaStream.getTracks().forEach((track) => track.stop());
  cameraEl.srcObject = null;
  mediaStream = null;
  updateCameraStatus(false);
}

function updateCameraStatus(isActive, isError = false) {
  if (!shutterButton) {
    return;
  }
  if (cameraStatusText) {
    if (isError) {
      cameraStatusText.textContent = 'カメラを利用できません';
    } else if (isActive) {
      cameraStatusText.textContent = '';
    } else {
      cameraStatusText.textContent = 'カメラを起動しています...';
    }
  }
  // shutterButton.textContent = isActive ? '撮影する' : '撮影開始';
}

async function handleShutterClick() {
  handleScreenshotCapture();
}

async function handleScreenshotCapture() {
  if (isCapturingScreenshot) {
    return;
  }
  if (!cameraEl || cameraEl.readyState < 2) {
    alert('カメラ映像の準備ができていません。');
    return;
  }
  isCapturingScreenshot = true;
  shutterButton.classList.add('capturing');

  try {
    const canvas = composeScreenshot();
    const blob = await canvasToBlob(canvas);
    const filename = createCaptureFilename();
    const objectUrl = URL.createObjectURL(blob);
    openShareModal(objectUrl, blob, filename);
  } catch (error) {
    console.error('撮影に失敗しました:', error);
    alert('撮影に失敗しました。もう一度お試しください。');
  } finally {
    setTimeout(() => {
      shutterButton.classList.remove('capturing');
    }, 350);
    isCapturingScreenshot = false;
  }
}

function composeScreenshot() {
  const rawWidth = cameraEl.videoWidth || 720;
  const rawHeight = cameraEl.videoHeight || 1280;
  if (!rawWidth || !rawHeight) {
    throw new Error('カメラのサイズを取得できませんでした');
  }

  const previewWidth = previewEl?.clientWidth;
  const previewHeight = previewEl?.clientHeight;
  const targetAspect =
    previewWidth && previewHeight
      ? previewWidth / previewHeight
      : rawWidth / rawHeight;
  const videoAspect = rawWidth / rawHeight;

  let sx = 0;
  let sy = 0;
  let sWidth = rawWidth;
  let sHeight = rawHeight;

  if (videoAspect > targetAspect) {
    sWidth = rawHeight * targetAspect;
    sWidth = Math.min(rawWidth, sWidth * (1 + CAMERA_CROP_PADDING_RATIO));
    sx = (rawWidth - sWidth) / 2;
  } else {
    sHeight = rawWidth / targetAspect;
    sHeight = Math.min(rawHeight, sHeight * (1 + CAMERA_CROP_PADDING_RATIO));
    sy = (rawHeight - sHeight) / 2;
  }

  const outputWidth = Math.round(sWidth);
  const outputHeight = Math.round(sHeight);

  captureCanvas.width = outputWidth;
  captureCanvas.height = outputHeight;

  const ctx = captureCanvas.getContext('2d');
  ctx.clearRect(0, 0, outputWidth, outputHeight);

  ctx.drawImage(
    cameraEl,
    sx,
    sy,
    sWidth,
    sHeight,
    0,
    0,
    outputWidth,
    outputHeight,
  );

  const frameEl = getActiveFrameElement();
  if (frameEl && !frameEl.hidden) {
    try {
      ctx.drawImage(frameEl, 0, 0, outputWidth, outputHeight);
    } catch (error) {
      console.warn('フレームの描画に失敗しました:', error);
    }
  }

  return captureCanvas;
}

function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('画像データの生成に失敗しました'));
        }
      },
      'image/png',
      0.95,
    );
  });
}

function createCaptureFilename() {
  const timestamp = new Date()
    .toISOString()
    .replace(/[:.]/g, '-')
    .replace('T', '_')
    .replace('Z', '');
  return `arframe-photo-${timestamp}.png`;
}

function getActiveFrameElement() {
  const frame = FRAMES[activeFrameIndex];
  if (!frame) {
    return null;
  }
  return frame.type === 'video' ? frameVideoEl : frameImageEl;
}

function setFrame(frameConfig) {
  if (!frameConfig) {
    return;
  }

  if (frameConfig.type === 'video') {
    stopIosFrameAnimation();
    frameImageEl.hidden = true;
    frameVideoEl.hidden = false;
    swapVideoSources(frameConfig);
  } else {
    frameVideoEl.pause();
    frameVideoEl.hidden = true;
    frameImageEl.hidden = false;
    if (frameConfig.imageSrc) {
      frameImageEl.src = frameConfig.imageSrc;
    }
  }
}

function swapVideoSources(frameConfig) {
  const sources = frameVideoEl.querySelectorAll('source');
  if (sources.length >= 2) {
    sources[0].src = frameConfig.webmSrc || '';
    sources[1].src = frameConfig.mp4Src || '';
  }
  frameVideoEl.load();
  frameVideoEl
    .play()
    .catch((error) => {
      console.warn('フレーム動画の再生に失敗しました:', error);
      fallbackToStaticFrame();
    });
}

function fallbackToStaticFrame() {
  if (FALLBACK_FRAME_INDEX === -1) {
    console.warn('静止画フレームが定義されていません。');
    return;
  }
  if (activeFrameIndex === FALLBACK_FRAME_INDEX) {
    return;
  }
  activeFrameIndex = FALLBACK_FRAME_INDEX;
  setFrame(FRAMES[FALLBACK_FRAME_INDEX]);
}

function handleFrameVideoError(event) {
  console.warn('フレーム動画の読み込み/再生でエラーが発生しました:', event);
  fallbackToStaticFrame();
}

function setupCaptureModal() {
  shareModal = $('#share-modal');
  capturedImage = $('#captured-image');
  shareModalShareButton = $('#share-button');
  shareModalCloseButton = $('#close-modal');

  if (shareModal) {
    shareModal.addEventListener('click', (event) => {
      if (event.target === shareModal) {
        closeShareModal();
      }
    });
  }

  if (shareModalShareButton) {
    shareModalShareButton.addEventListener('click', handleShareButtonClick);
  }

  if (shareModalCloseButton) {
    shareModalCloseButton.addEventListener('click', closeShareModal);
  }

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      closeShareModal();
    }
  });
}

function openShareModal(objectUrl, blob, filename) {
  if (!shareModal || !capturedImage) {
    return;
  }
  if (currentCapture?.objectUrl) {
    URL.revokeObjectURL(currentCapture.objectUrl);
  }
  currentCapture = { objectUrl, blob, filename };
  capturedImage.src = objectUrl;
  shareModal.classList.remove('hidden');
  shareModal.setAttribute('aria-hidden', 'false');
}

function closeShareModal() {
  if (!shareModal) {
    return;
  }
  shareModal.classList.add('hidden');
  shareModal.setAttribute('aria-hidden', 'true');
  if (capturedImage) {
    capturedImage.removeAttribute('src');
  }
  if (currentCapture?.objectUrl) {
    URL.revokeObjectURL(currentCapture.objectUrl);
  }
  currentCapture = null;
}

async function handleShareButtonClick() {
  if (!currentCapture?.blob) {
    alert('共有できる画像がありません。');
    return;
  }
  const { blob, filename } = currentCapture;
  const file = new File([blob], filename, { type: 'image/png' });

  if (
    navigator.share &&
    (!navigator.canShare || navigator.canShare({ files: [file] }))
  ) {
    try {
      await navigator.share({
        files: [file],
        title: 'ARフォトフレームで撮影した写真',
        text: 'ブラウザで撮影したフォトフレーム画像です。',
      });
      return;
    } catch (error) {
      console.warn('共有をキャンセル/失敗:', error);
    }
  }

  const fallbackUrl = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = fallbackUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(fallbackUrl);
}

function handleVisibilityChange() {
  if (document.hidden) {
    shouldResumeCamera = Boolean(mediaStream);
    stopCamera();
  } else if (shouldResumeCamera && !mediaStream) {
    shouldResumeCamera = false;
    startCamera();
  }
}

function bindControls() {
  shutterButton.addEventListener('click', handleShutterClick);
}

function init() {
  cameraEl = $('#camera');
  frameImageEl = $('#frameImage');
  frameVideoEl = $('#frameVideo');
  shutterButton = $('#shutterButton');
  downloadLink = $('#downloadLink');
  captureCanvas = $('#captureCanvas');
  cameraStatusText = $('#camera-status-text');
  previewEl = document.querySelector('.preview');

  if (!cameraEl || !shutterButton || !captureCanvas) {
    console.error('必要な要素を取得できませんでした。');
    return;
  }

  if (frameVideoEl) {
    frameVideoEl.addEventListener('error', handleFrameVideoError);
  }

  const safari = isSafari();
  const useVideoFrame = canPlayVp9WebM() && !safari;
  if (!useVideoFrame && FALLBACK_FRAME_INDEX !== -1) {
    activeFrameIndex = FALLBACK_FRAME_INDEX;
  }

  setFrame(FRAMES[activeFrameIndex]);
  if (safari) {
    startIosFrameAnimation();
  } else {
    stopIosFrameAnimation();
  }
  setupCaptureModal();
  bindControls();
  updateCameraStatus(false);

  window.addEventListener('beforeunload', stopCamera);
  document.addEventListener('visibilitychange', handleVisibilityChange);

  startCamera();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
