import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export interface BodyMeasurements {
  shoulder: number;
  hip: number;
  sleeve: number;
}

export function distance(
  a: NormalizedLandmark,
  b: NormalizedLandmark,
  width: number,
  height: number
): number {
  const dx = (a.x - b.x) * width;
  const dy = (a.y - b.y) * height;
  return Math.sqrt(dx * dx + dy * dy);
}

function pixelBodyHeight(
  landmarks: NormalizedLandmark[],
  frameHeight: number
): number {
  const nose = landmarks[0];
  const leftAnkle = landmarks[27];
  const rightAnkle = landmarks[28];

  const ankleY = Math.max(leftAnkle.y, rightAnkle.y);
  return (ankleY - nose.y) * frameHeight;
}

export function computeMeasurements(
  landmarks: NormalizedLandmark[],
  userHeightCm: number,
  frameWidth: number,
  frameHeight: number
): BodyMeasurements | null {
  const pxHeight = pixelBodyHeight(landmarks, frameHeight);
  if (pxHeight <= 0) {
    return null;
  }

  const scale = userHeightCm / pxHeight;

  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];
  const leftElbow = landmarks[13];
  const leftWrist = landmarks[15];

  const shoulderPx = distance(leftShoulder, rightShoulder, frameWidth, frameHeight);
  const hipPx = distance(leftHip, rightHip, frameWidth, frameHeight);
  const upperArmPx = distance(leftShoulder, leftElbow, frameWidth, frameHeight);
  const forearmPx = distance(leftElbow, leftWrist, frameWidth, frameHeight);
  const sleevePx = upperArmPx + forearmPx;

  return {
    shoulder: Math.round(shoulderPx * scale * 100) / 100,
    hip: Math.round(hipPx * scale * 100) / 100,
    sleeve: Math.round(sleevePx * scale * 100) / 100,
  };
}

export function isBodyFullyVisible(landmarks: NormalizedLandmark[]): boolean {
  const keyIndices = [0, 11, 12, 13, 15, 23, 24, 27, 28];

  return keyIndices.every((i) => {
    const lm = landmarks[i];
    return (
      lm &&
      lm.visibility !== undefined &&
      lm.visibility > 0.5 &&
      lm.x >= 0 &&
      lm.x <= 1 &&
      lm.y >= 0 &&
      lm.y <= 1
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
