import type { Keypoint } from "./yoloPose";

export interface BodyMeasurements {
  shoulder: number;
  hip: number;
  sleeve: number;
}

interface MeasurementConfig {
  hipFactor: number;
}

const DEFAULT_CONFIG: MeasurementConfig = {
  hipFactor: 1.9, // tweak between 1.7 - 2.2 after real-world testing
};

// COCO 17-keypoint indices
const KP = {
  NOSE: 0,
  LEFT_SHOULDER: 5,
  RIGHT_SHOULDER: 6,
  LEFT_ELBOW: 7,
  LEFT_WRIST: 9,
  LEFT_HIP: 11,
  RIGHT_HIP: 12,
  LEFT_ANKLE: 15,
  RIGHT_ANKLE: 16,
} as const;

export function distance(
  a: Keypoint,
  b: Keypoint,
  width: number,
  height: number
): number {
  const dx = (a.x - b.x) * width;
  const dy = (a.y - b.y) * height;
  return Math.sqrt(dx * dx + dy * dy);
}

function pixelBodyHeight(
  keypoints: Keypoint[],
  frameHeight: number
): number {
  const nose = keypoints[KP.NOSE];
  const leftAnkle = keypoints[KP.LEFT_ANKLE];
  const rightAnkle = keypoints[KP.RIGHT_ANKLE];

  const ankleY = Math.max(leftAnkle.y, rightAnkle.y);
  return (ankleY - nose.y) * frameHeight;
}

export function computeMeasurements(
  keypoints: Keypoint[],
  userHeightCm: number,
  frameWidth: number,
  frameHeight: number
): BodyMeasurements | null {
  const pxHeight = pixelBodyHeight(keypoints, frameHeight);
  if (pxHeight <= 0) {
    return null;
  }

  const scale = userHeightCm / pxHeight;

  const leftShoulder = keypoints[KP.LEFT_SHOULDER];
  const rightShoulder = keypoints[KP.RIGHT_SHOULDER];
  const leftHip = keypoints[KP.LEFT_HIP];
  const rightHip = keypoints[KP.RIGHT_HIP];
  const leftElbow = keypoints[KP.LEFT_ELBOW];
  const leftWrist = keypoints[KP.LEFT_WRIST];

  const shoulderPx = distance(leftShoulder, rightShoulder, frameWidth, frameHeight);
  const hipPx = distance(leftHip, rightHip, frameWidth, frameHeight);
  const upperArmPx = distance(leftShoulder, leftElbow, frameWidth, frameHeight);
  const forearmPx = distance(leftElbow, leftWrist, frameWidth, frameHeight);
  const sleevePx = upperArmPx + forearmPx;

  return {
    shoulder: Math.round(shoulderPx * scale * 100) / 100,
    hip: Math.round(hipPx * scale * DEFAULT_CONFIG.hipFactor * 100) / 100,
    sleeve: Math.round(sleevePx * scale * 100) / 100,
  };
}

export function isBodyFullyVisible(keypoints: Keypoint[]): boolean {
  const requiredIndices = [
    KP.NOSE,
    KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER,
    KP.LEFT_ELBOW, KP.LEFT_WRIST,
    KP.LEFT_HIP, KP.RIGHT_HIP,
    KP.LEFT_ANKLE, KP.RIGHT_ANKLE,
  ];

  return requiredIndices.every((i) => {
    const kp = keypoints[i];
    return (
      kp &&
      kp.confidence > 0.5 &&
      kp.x >= 0 &&
      kp.x <= 1 &&
      kp.y >= 0 &&
      kp.y <= 1
    );
  });
}

export function averageMeasurements(
  buffer: BodyMeasurements[]
): BodyMeasurements {
  const n = buffer.length;
  if (n === 0) {
    return { shoulder: 0, hip: 0, sleeve: 0 };
  }

  const sum = buffer.reduce(
    (acc, m) => ({
      shoulder: acc.shoulder + m.shoulder,
      hip: acc.hip + m.hip,
      sleeve: acc.sleeve + m.sleeve,
    }),
    { shoulder: 0, hip: 0, sleeve: 0 }
  );

  return {
    shoulder: Math.round((sum.shoulder / n) * 100) / 100,
    hip: Math.round((sum.hip / n) * 100) / 100,
    sleeve: Math.round((sum.sleeve / n) * 100) / 100,
  };
}
