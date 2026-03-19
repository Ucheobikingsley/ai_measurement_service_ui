import { useRef, useEffect, useCallback, useState } from "react";
import {
  loadModel,
  detectPose as yoloDetectPose,
  isModelLoaded,
  COCO_SKELETON,
} from "../utils/yoloPose";
import type { Keypoint } from "../utils/yoloPose";
import {
  computeMeasurements,
  isBodyFullyVisible,
  averageMeasurements,
} from "../utils/measurement";
import type { BodyMeasurements } from "../utils/measurement";

interface CameraProps {
  userHeightCm: number;
  scanning: boolean;
  onMeasurement: (m: BodyMeasurements) => void;
  onStatusChange: (status: string) => void;
}

const SMOOTHING_BUFFER_SIZE = 10;
const STABLE_FRAMES_REQUIRED = 15;
const TARGET_FPS = 15;
const MODEL_URL = "/yolo11n-pose.onnx";

function drawKeypoints(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  w: number,
  h: number
): void {
  for (const kp of keypoints) {
    if (kp.confidence < 0.3) {
      continue;
    }

    ctx.beginPath();
    ctx.arc(kp.x * w, kp.y * h, 4, 0, 2 * Math.PI);
    ctx.fillStyle = "#00FF8899";
    ctx.fill();
    ctx.strokeStyle = "#00FF88";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  keypoints: Keypoint[],
  w: number,
  h: number
): void {
  ctx.strokeStyle = "#00CCFF";
  ctx.lineWidth = 2;

  for (const [i, j] of COCO_SKELETON) {
    const a = keypoints[i];
    const b = keypoints[j];

    if (a.confidence < 0.3 || b.confidence < 0.3) {
      continue;
    }

    ctx.beginPath();
    ctx.moveTo(a.x * w, a.y * h);
    ctx.lineTo(b.x * w, b.y * h);
    ctx.stroke();
  }
}

function isUserStable(keypoints: Keypoint[]): boolean {
  const leftShoulder = keypoints[5];
  const rightShoulder = keypoints[6];

  if (!leftShoulder || !rightShoulder) {
    return false;
  }

  if (leftShoulder.confidence < 0.5 || rightShoulder.confidence < 0.5) {
    return false;
  }

  return Math.abs(leftShoulder.y - rightShoulder.y) < 0.03;
}

export default function Camera({
  userHeightCm,
  scanning,
  onMeasurement,
  onStatusChange,
}: CameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const animFrameRef = useRef<number>(0);
  const lastFrameTimeRef = useRef(0);
  const inferringRef = useRef(false);

  const bufferRef = useRef<BodyMeasurements[]>([]);
  const stableFrameCountRef = useRef(0);

  const [cameraReady, setCameraReady] = useState(false);
  const [modelReady, setModelReady] = useState(false);

  const initModel = useCallback(async () => {
    try {
      await loadModel(MODEL_URL);
      setModelReady(true);
    } catch (err) {
      console.error("Failed to load YOLO pose model:", err);
      onStatusChange("Failed to load pose model");
    }
  }, [onStatusChange]);

  const startCamera = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
    });

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraReady(true);
    }
  }, []);

  const detectPose = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !isModelLoaded()) {
      return;
    }

    const now = performance.now();

    if (now - lastFrameTimeRef.current < 1000 / TARGET_FPS || inferringRef.current) {
      animFrameRef.current = requestAnimationFrame(detectPose);
      return;
    }

    lastFrameTimeRef.current = now;
    inferringRef.current = true;

    yoloDetectPose(video)
      .then((result) => {
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (result) {
          const { keypoints } = result;

          drawSkeleton(ctx, keypoints, canvas.width, canvas.height);
          drawKeypoints(ctx, keypoints, canvas.width, canvas.height);

          const fullyVisible = isBodyFullyVisible(keypoints);
          const stable = isUserStable(keypoints);

          if (fullyVisible && stable) {
            stableFrameCountRef.current++;

            onStatusChange(
              `Hold still... (${stableFrameCountRef.current}/${STABLE_FRAMES_REQUIRED})`
            );

            if (stableFrameCountRef.current >= STABLE_FRAMES_REQUIRED) {
              const m = computeMeasurements(
                keypoints,
                userHeightCm,
                video.videoWidth,
                video.videoHeight
              );

              if (m) {
                bufferRef.current.push(m);

                if (bufferRef.current.length > SMOOTHING_BUFFER_SIZE) {
                  bufferRef.current.shift();
                }

                const smoothed = averageMeasurements(bufferRef.current);
                onMeasurement(smoothed);
                onStatusChange("Measurement locked");
              }
            }
          } else {
            stableFrameCountRef.current = 0;

            if (!fullyVisible) {
              onStatusChange("Step back — full body not visible");
            } else {
              onStatusChange("Stand straight — don't tilt");
            }
          }
        } else {
          stableFrameCountRef.current = 0;
          onStatusChange("No body detected");
        }
      })
      .catch(console.error)
      .finally(() => {
        inferringRef.current = false;
      });

    animFrameRef.current = requestAnimationFrame(detectPose);
  }, [userHeightCm, onMeasurement, onStatusChange]);

  useEffect(() => {
    initModel().then(startCamera);

    return () => {
      cancelAnimationFrame(animFrameRef.current);

      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }
    };
  }, [initModel, startCamera]);

  useEffect(() => {
    if (scanning && cameraReady && modelReady) {
      bufferRef.current = [];
      stableFrameCountRef.current = 0;
      animFrameRef.current = requestAnimationFrame(detectPose);
    }

    if (!scanning) {
      cancelAnimationFrame(animFrameRef.current);
    }

    return () => cancelAnimationFrame(animFrameRef.current);
  }, [scanning, cameraReady, modelReady, detectPose]);

  return (
    <div className="camera-container" style={{ position: "relative" }}>
      <video
        ref={videoRef}
        playsInline
        muted
        style={{
          width: "100%",
          height: "100%",
          objectFit: "contain",
          borderRadius: "12px",
        }}
      />

      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
        }}
      />
    </div>
  );
}
