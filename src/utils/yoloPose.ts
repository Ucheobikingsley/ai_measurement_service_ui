import * as ort from "onnxruntime-web";

export interface Keypoint {
  x: number;
  y: number;
  confidence: number;
}

export interface PoseDetection {
  confidence: number;
  keypoints: Keypoint[];
}

const MODEL_INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;
const NUM_KEYPOINTS = 17;

export const COCO_SKELETON: [number, number][] = [
  [0, 1], [0, 2], [1, 3], [2, 4],
  [5, 6],
  [5, 7], [7, 9],
  [6, 8], [8, 10],
  [5, 11], [6, 12],
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
];

let session: ort.InferenceSession | null = null;
let preprocessCanvas: OffscreenCanvas | null = null;

function getPreprocessCanvas(): OffscreenCanvas {
  if (!preprocessCanvas) {
    preprocessCanvas = new OffscreenCanvas(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  }
  return preprocessCanvas;
}

export async function loadModel(modelUrl: string): Promise<void> {
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  ort.env.wasm.numThreads = 1;

  session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
  });
}

export function isModelLoaded(): boolean {
  return session !== null;
}

interface LetterboxInfo {
  scale: number;
  padX: number;
  padY: number;
}

function preprocess(
  video: HTMLVideoElement
): { data: Float32Array; info: LetterboxInfo } {
  const canvas = getPreprocessCanvas();
  const ctx = canvas.getContext("2d")!;
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const scale = Math.min(MODEL_INPUT_SIZE / vw, MODEL_INPUT_SIZE / vh);
  const nw = Math.round(vw * scale);
  const nh = Math.round(vh * scale);
  const padX = Math.round((MODEL_INPUT_SIZE - nw) / 2);
  const padY = Math.round((MODEL_INPUT_SIZE - nh) / 2);

  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  ctx.drawImage(video, padX, padY, nw, nh);

  const imageData = ctx.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  const { data } = imageData;

  const float32Data = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);
  const pixelCount = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

  for (let i = 0; i < pixelCount; i++) {
    float32Data[i] = data[i * 4] / 255;
    float32Data[pixelCount + i] = data[i * 4 + 1] / 255;
    float32Data[2 * pixelCount + i] = data[i * 4 + 2] / 255;
  }

  return { data: float32Data, info: { scale, padX, padY } };
}

function computeIou(a: number[], b: number[]): number {
  const aX1 = a[0] - a[2] / 2;
  const aY1 = a[1] - a[3] / 2;
  const aX2 = a[0] + a[2] / 2;
  const aY2 = a[1] + a[3] / 2;

  const bX1 = b[0] - b[2] / 2;
  const bY1 = b[1] - b[3] / 2;
  const bX2 = b[0] + b[2] / 2;
  const bY2 = b[1] + b[3] / 2;

  const interX1 = Math.max(aX1, bX1);
  const interY1 = Math.max(aY1, bY1);
  const interX2 = Math.min(aX2, bX2);
  const interY2 = Math.min(aY2, bY2);

  const inter =
    Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
  const aArea = a[2] * a[3];
  const bArea = b[2] * b[3];

  return inter / (aArea + bArea - inter);
}

export async function detectPose(
  video: HTMLVideoElement
): Promise<PoseDetection | null> {
  if (!session) {
    return null;
  }

  const { data, info } = preprocess(video);

  const inputTensor = new ort.Tensor("float32", data, [
    1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE,
  ]);

  const feeds: Record<string, ort.Tensor> = {};
  feeds[session.inputNames[0]] = inputTensor;

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]];
  const outputData = output.data as Float32Array;

  // Output shape: [1, 56, 8400]
  // Row layout: [cx, cy, w, h, conf, kp0_x, kp0_y, kp0_conf, kp1_x, ...]
  const numDetections = output.dims[2];

  const detections: Array<{
    box: number[];
    confidence: number;
    keypoints: Keypoint[];
  }> = [];

  for (let i = 0; i < numDetections; i++) {
    const conf = outputData[4 * numDetections + i];
    if (conf < CONF_THRESHOLD) {
      continue;
    }

    const cx = outputData[0 * numDetections + i];
    const cy = outputData[1 * numDetections + i];
    const w = outputData[2 * numDetections + i];
    const h = outputData[3 * numDetections + i];

    const keypoints: Keypoint[] = [];
    for (let k = 0; k < NUM_KEYPOINTS; k++) {
      const rawX = outputData[(5 + k * 3) * numDetections + i];
      const rawY = outputData[(5 + k * 3 + 1) * numDetections + i];
      const kpConf = outputData[(5 + k * 3 + 2) * numDetections + i];

      keypoints.push({
        x: (rawX - info.padX) / (video.videoWidth * info.scale),
        y: (rawY - info.padY) / (video.videoHeight * info.scale),
        confidence: kpConf,
      });
    }

    detections.push({ box: [cx, cy, w, h], confidence: conf, keypoints });
  }

  detections.sort((a, b) => b.confidence - a.confidence);

  const kept: typeof detections = [];
  for (const det of detections) {
    let suppress = false;

    for (const k of kept) {
      if (computeIou(det.box, k.box) > IOU_THRESHOLD) {
        suppress = true;
        break;
      }
    }

    if (!suppress) {
      kept.push(det);
    }
  }

  if (kept.length === 0) {
    return null;
  }

  return {
    confidence: kept[0].confidence,
    keypoints: kept[0].keypoints,
  };
}
