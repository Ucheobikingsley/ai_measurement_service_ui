import { useRef, useEffect, useCallback, useState } from "react";
import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import {
  computeMeasurements,
  isBodyFullyVisible,
  averageMeasurements,
  type BodyMeasurements,
} from "../utils/measurement";

interface CameraProps {
  userHeightCm: number;
  scanning: boolean;
  onMeasurement: (m: BodyMeasurements) => void;
  onStatusChange: (status: string) => void;
}

const SMOOTHING_BUFFER_SIZE = 10;
const STABLE_FRAMES_REQUIRED = 15;
const TARGET_FPS = 15;

export default function Camera({
  userHeightCm,
  scanning,
  onMeasurement,
  onStatusChange,
}: CameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const drawingUtilsRef = useRef<DrawingUtils | null>(null);

  const animFrameRef = useRef<number>(0);
  const lastFrameTimeRef = useRef(0);

  const bufferRef = useRef<BodyMeasurements[]>([]);
  const stableFrameCountRef = useRef(0);

  const [cameraReady, setCameraReady] = useState(false);

  // ------------------------
  // Init
  // ------------------------

  const initLandmarker = useCallback(async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );

    landmarkerRef.current = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  }, []);

  const startCamera = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 640,
        height: 480,
        facingMode: "user",
      },
    });

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setCameraReady(true);
    }
  }, []);

  // ------------------------
  // Pose Validation (NEW 🔥)
  // ------------------------

  function isUserStable(landmarks: NormalizedLandmark[]): boolean {
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];

    if (!leftShoulder || !rightShoulder) return false;

    // shoulders should be roughly horizontal
    const diffY = Math.abs(leftShoulder.y - rightShoulder.y);

    return diffY < 0.03; // tweak threshold
  }

  // ------------------------
  // Detection Loop
  // ------------------------

  const detectPose = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;

    if (!video || !canvas || !landmarker || !scanning) return;

    const now = performance.now();

    // 🔥 FPS throttling
    if (now - lastFrameTimeRef.current < 1000 / TARGET_FPS) {
      animFrameRef.current = requestAnimationFrame(detectPose);
      return;
    }
    lastFrameTimeRef.current = now;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    if (!drawingUtilsRef.current) {
      drawingUtilsRef.current = new DrawingUtils(ctx);
    }

    const result = landmarker.detectForVideo(video, now);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (result.landmarks?.length) {
      const landmarks = result.landmarks[0];
      const drawingUtils = drawingUtilsRef.current;

      // Draw
      drawingUtils.drawLandmarks(landmarks, {
        radius: 4,
        color: "#00FF88",
        fillColor: "#00FF8899",
      });

      drawingUtils.drawConnectors(
        landmarks,
        PoseLandmarker.POSE_CONNECTIONS,
        { color: "#00CCFF", lineWidth: 2 }
      );

      const fullyVisible = isBodyFullyVisible(landmarks);
      const stable = isUserStable(landmarks);

      if (fullyVisible && stable) {
        stableFrameCountRef.current++;

        onStatusChange(
          `Hold still... (${stableFrameCountRef.current}/${STABLE_FRAMES_REQUIRED})`
        );

        if (stableFrameCountRef.current >= STABLE_FRAMES_REQUIRED) {
          const m = computeMeasurements(
            landmarks,
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

            onStatusChange("Measurement locked ✅");
          }
        }
      } else {
        stableFrameCountRef.current = 0;

        if (!fullyVisible) {
          onStatusChange("Step back — full body not visible");
        } else if (!stable) {
          onStatusChange("Stand straight — don't tilt");
        }
      }
    } else {
      stableFrameCountRef.current = 0;
      onStatusChange("No body detected");
    }

    animFrameRef.current = requestAnimationFrame(detectPose);
  }, [scanning, userHeightCm, onMeasurement, onStatusChange]);

  // ------------------------
  // Lifecycle
  // ------------------------

  useEffect(() => {
    initLandmarker().then(startCamera);

    return () => {
      cancelAnimationFrame(animFrameRef.current);

      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }
    };
  }, [initLandmarker, startCamera]);

  useEffect(() => {
    if (scanning && cameraReady && landmarkerRef.current) {
      bufferRef.current = [];
      stableFrameCountRef.current = 0;
      animFrameRef.current = requestAnimationFrame(detectPose);
    }

    if (!scanning) {
      cancelAnimationFrame(animFrameRef.current);
    }

    return () => cancelAnimationFrame(animFrameRef.current);
  }, [scanning, cameraReady, detectPose]);

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