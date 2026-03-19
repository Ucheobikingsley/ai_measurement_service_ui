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

export default function Camera({
  userHeightCm,
  scanning,
  onMeasurement,
  onStatusChange,
}: CameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const animFrameRef = useRef<number>(0);
  const bufferRef = useRef<BodyMeasurements[]>([]);
  const [cameraReady, setCameraReady] = useState(false);

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
    const landmarker = landmarkerRef.current;

    if (!video || !canvas || !landmarker || !scanning) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const now = performance.now();
    const result = landmarker.detectForVideo(video, now);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (result.landmarks && result.landmarks.length > 0) {
      const landmarks: NormalizedLandmark[] = result.landmarks[0];
      const drawingUtils = new DrawingUtils(ctx);

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

      if (isBodyFullyVisible(landmarks)) {
        onStatusChange("Body detected — measuring...");

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
        }
      } else {
        onStatusChange("Adjust position — full body must be visible");
      }
    } else {
      onStatusChange("No body detected — step into the frame");
    }

    animFrameRef.current = requestAnimationFrame(detectPose);
  }, [scanning, userHeightCm, onMeasurement, onStatusChange]);

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
      animFrameRef.current = requestAnimationFrame(detectPose);
    }

    if (!scanning) {
      cancelAnimationFrame(animFrameRef.current);
    }

    return () => cancelAnimationFrame(animFrameRef.current);
  }, [scanning, cameraReady, detectPose]);

  return (
    <div className="camera-container">
      <video
        ref={videoRef}
        playsInline
        muted
        style={{ width: "100%", height: "100%", objectFit: "cover", display: "block", borderRadius: "12px" }}
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
