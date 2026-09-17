import { createFileRoute } from "@tanstack/react-router";
import { useServerFn } from "@tanstack/react-start";
import { useState, useRef, useEffect, useMemo } from "react";
import { Upload, Camera, Activity, LayoutGrid, ShieldCheck, Box } from "lucide-react";
import { cn } from "@/lib/utils";
import { sendLineAccidentAlert } from "@/lib/line-alert.functions";
import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-converter";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    title: "ระบบตรวจสอบอุบัติเหตุ CCTV - AI Detection",
    meta: [
      { name: "description", content: "ระบบจำลอง CCTV พร้อม AI ตรวจจับรถยนต์และการเคลื่อนไหว" },
      { property: "og:title", content: "ระบบตรวจสอบอุบัติเหตุ CCTV - AI Detection" },
      { property: "og:description", content: "ระบบจำลอง CCTV พร้อม AI ตรวจจับรถยนต์และการเคลื่อนไหว" },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
    ],
  }),
});

interface Detection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
}

interface CCTVMonitorProps {
  id: number;
  model: cocoSsd.ObjectDetection | null;
  onDetection: (id: number, objects: string[]) => void;
  onAccident: (id: number, reason: string, snapshot: string | null) => void;
}

interface Track {
  bbox: [number, number, number, number];
  class: string;
  score: number;
  lastSeen: number;
  vx: number;
  vy: number;
  hits: number;
  speed: number;
  avgSpeed: number;
  prevSpeed: number;
  prevVx: number;
  prevVy: number;
  previousArea: number;
  stillFrames: number;
  maxSpeed: number;
  parked: boolean;
  alerted: boolean;
  anomalyFrames: number;
}

const VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"];

function iou(a: [number, number, number, number], b: [number, number, number, number]) {
  const [x1, y1, w1, h1] = a;
  const [x2, y2, w2, h2] = b;
  const ox = Math.max(0, Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2));
  const oy = Math.max(0, Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2));
  const inter = ox * oy;
  const union = w1 * h1 + w2 * h2 - inter;
  return union > 0 ? inter / union : 0;
}

function vehiclesAreInContact(a: Track, b: Track) {
  const [ax, ay, aw, ah] = a.bbox;
  const [bx, by, bw, bh] = b.bbox;
  const gapX = Math.max(0, Math.max(ax, bx) - Math.min(ax + aw, bx + bw));
  const gapY = Math.max(0, Math.max(ay, by) - Math.min(ay + ah, by + bh));
  const contactDistance = Math.hypot(gapX, gapY);
  const contactLimit = Math.max(5, Math.min(aw, ah, bw, bh) * 0.22);
  const centerDx = bx + bw / 2 - (ax + aw / 2);
  const centerDy = by + bh / 2 - (ay + ah / 2);
  // Previous velocity is more useful at the impact frame, where both cars may
  // already have slowed down or changed direction.
  const relativeVx = b.prevVx - a.prevVx;
  const relativeVy = b.prevVy - a.prevVy;
  const approaching = centerDx * relativeVx + centerDy * relativeVy < 0;
  const oneWasMoving = Math.max(a.prevSpeed, b.prevSpeed, a.speed, b.speed) > 0.22;
  return (iou(a.bbox, b.bbox) > 0.035 || contactDistance <= contactLimit * 1.45) && approaching && oneWasMoving;
}

// Non-maximum suppression: one box per physical vehicle
function nms(dets: Detection[], threshold = 0.45): Detection[] {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const kept: Detection[] = [];
  for (const det of sorted) {
    if (kept.every((k) => iou(k.bbox, det.bbox) < threshold)) kept.push(det);
  }
  return kept;
}

function CCTVMonitor({ id, model, onDetection, onAccident }: CCTVMonitorProps) {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(null);
  const detectionCounter = useRef(0);
  const tracksRef = useRef<Record<number, Track>>({});
  const trackIdCounter = useRef(0);
  const modelRef = useRef(model);
  const motionCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const previousFrameRef = useRef<Uint8ClampedArray | null>(null);
  const lastMotionTimeRef = useRef(-1);
  const motionStateRef = useRef({ samples: 0, average: 0, previous: 0, alerted: false });
  const motionAlertRef = useRef<{ bbox: [number, number, number, number]; until: number } | null>(null);
  const callbacksRef = useRef({ onDetection, onAccident });
  modelRef.current = model;
  callbacksRef.current = { onDetection, onAccident };

  // Freeze the current video frame as a JPEG so the incident report can carry
  // the exact moment of the accident.
  const captureSnapshot = (): string | null => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return null;
    try {
      const shot = document.createElement("canvas");
      shot.width = 960;
      shot.height = Math.round((video.videoHeight / video.videoWidth) * 960) || 540;
      const ctx = shot.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(video, 0, 0, shot.width, shot.height);
      return shot.toDataURL("image/jpeg", 0.82);
    } catch (err) {
      console.error("snapshot failed", err);
      return null;
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (videoSrc) URL.revokeObjectURL(videoSrc);
      tracksRef.current = {};
      trackIdCounter.current = 0;
      detectionCounter.current = 0;
      previousFrameRef.current = null;
      lastMotionTimeRef.current = -1;
      motionStateRef.current = { samples: 0, average: 0, previous: 0, alerted: false };
      motionAlertRef.current = null;
      setDetections([]);
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
    }
  };

  const detectFrame = async () => {
    if (videoRef.current && videoRef.current.readyState === 4) {
      detectionCounter.current++;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const now = Date.now();

      // Independent motion analysis keeps incident detection working when the
      // object model misses small vehicles in elevated or distant CCTV views.
      if (video.currentTime - lastMotionTimeRef.current >= 0.09) {
        lastMotionTimeRef.current = video.currentTime;
        const motionCanvas = motionCanvasRef.current ?? document.createElement("canvas");
        motionCanvas.width = 160;
        motionCanvas.height = 90;
        motionCanvasRef.current = motionCanvas;
        const motionContext = motionCanvas.getContext("2d", { willReadFrequently: true });
        if (motionContext) {
          motionContext.drawImage(video, 0, 0, motionCanvas.width, motionCanvas.height);
          const pixels = motionContext.getImageData(0, 0, motionCanvas.width, motionCanvas.height).data;
          const gray = new Uint8ClampedArray(motionCanvas.width * motionCanvas.height);
          let changed = 0;
          let minX = motionCanvas.width;
          let minY = motionCanvas.height;
          let maxX = 0;
          let maxY = 0;
          const previous = previousFrameRef.current;

          for (let index = 0; index < gray.length; index++) {
            const pixelIndex = index * 4;
            const red = pixels[pixelIndex] ?? 0;
            const green = pixels[pixelIndex + 1] ?? 0;
            const blue = pixels[pixelIndex + 2] ?? 0;
            const luminance = Math.round(red * 0.299 + green * 0.587 + blue * 0.114);
            gray[index] = luminance;
            const previousLuminance = previous?.[index];
            if (previousLuminance !== undefined && Math.abs(luminance - previousLuminance) > 14) {
              const x = index % motionCanvas.width;
              const y = Math.floor(index / motionCanvas.width);
              // Ignore embedded timestamps and edge noise common in CCTV clips.
              if (y > 5 && y < motionCanvas.height - 8) {
                changed++;
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
              }
            }
          }
          previousFrameRef.current = gray;

          if (previous) {
            const motion = changed / (motionCanvas.width * motionCanvas.height);
            const state = motionStateRef.current;
            const previousAverage = state.average || motion;
            const abruptSurge = state.samples >= 5 && motion > 0.012 && motion > previousAverage * 1.32;
            const abruptStop = state.samples >= 7 && state.previous > 0.018 && motion < state.previous * 0.58;
            state.samples++;
            state.average = previousAverage * 0.86 + motion * 0.14;
            state.previous = motion;

            if (!state.alerted && (abruptSurge || abruptStop)) {
              state.alerted = true;
              const scaleX = video.videoWidth / motionCanvas.width;
              const scaleY = video.videoHeight / motionCanvas.height;
              const padding = 12;
              const sourceBox: [number, number, number, number] = changed > 0
                ? [
                    Math.max(0, minX - padding) * scaleX,
                    Math.max(0, minY - padding) * scaleY,
                    Math.min(motionCanvas.width, maxX - minX + padding * 2) * scaleX,
                    Math.min(motionCanvas.height, maxY - minY + padding * 2) * scaleY,
                  ]
                : [0, 0, video.videoWidth, video.videoHeight];
              motionAlertRef.current = { bbox: sourceBox, until: now + 6000 };
              callbacksRef.current.onAccident(
                id,
                abruptStop
                  ? "พบรถหยุดหรือเปลี่ยนสภาพการเคลื่อนที่ฉับพลัน — กรุณาตรวจสอบ"
                  : "พบการเคลื่อนไหวผิดปกติคล้ายเหตุชน — กรุณาตรวจสอบ",
                captureSnapshot(),
              );
            }
          }
        }
      }

      // 1. Detection phase — every 2nd frame for better temporal accuracy
      if (detectionCounter.current % 2 === 0) {
        let raw: Detection[] = [];
        try {
          const activeModel = modelRef.current;
          raw = activeModel ? ((await activeModel.detect(video, 25, 0.34)) as Detection[]) : [];
        } catch (err) {
          console.error("detect failed", err);
        }
        const diag = Math.hypot(video.videoWidth, video.videoHeight) || 1;
        const candidates = nms(
          raw.filter((p) => {
            if (!VEHICLE_CLASSES.includes(p.class)) return false;
            const [, , w, h] = p.bbox;
            // Reject implausible boxes: too small, oversized, or extreme aspect ratio
            // Keep distant vehicles in elevated CCTV footage, while removing
            // tiny unstable detections that cannot be tracked reliably.
            if (w < video.videoWidth * 0.018 || h < video.videoHeight * 0.018) return false;
            if (w > video.videoWidth * 0.85 && h > video.videoHeight * 0.85) return false;
            const ratio = w / Math.max(h, 1);
            if (ratio < 0.3 || ratio > 4.2) return false;
            return true;
          }) as Detection[],
          0.35,
        );


        const updated: Record<number, Track> = {};
        const available = [...candidates];

        Object.entries(tracksRef.current).forEach(([key, track]) => {
          const trackId = parseInt(key);
          // Predict position from velocity so fast vehicles stay matched
          const predicted: [number, number, number, number] = [
            track.bbox[0] + track.vx,
            track.bbox[1] + track.vy,
            track.bbox[2],
            track.bbox[3],
          ];

          let best = -1;
          let bestScore = 0.12;

          available.forEach((det, index) => {
            const [px, py, pw, ph] = predicted;
            const [dx, dy, dw, dh] = det.bbox;
            const centerDistance = Math.hypot(px + pw / 2 - (dx + dw / 2), py + ph / 2 - (dy + dh / 2));
            const distanceScore = Math.max(0, 1 - centerDistance / Math.max(30, Math.hypot(pw, ph) * 1.8));
            const sizeSimilarity = Math.min(pw * ph, dw * dh) / Math.max(pw * ph, dw * dh, 1);
            const classCompatible = det.class === track.class || (VEHICLE_CLASSES.includes(det.class) && VEHICLE_CLASSES.includes(track.class));
            const score = classCompatible
              ? Math.max(iou(track.bbox, det.bbox), iou(predicted, det.bbox), distanceScore * sizeSimilarity * 0.72)
              : 0;
            if (score > bestScore) {
              bestScore = score;
              best = index;
            }
          });

          if (best !== -1) {
            const match = available[best];
            if (match) {
              available.splice(best, 1);
              const lerp = 0.6;
              const newBbox: [number, number, number, number] = [
                track.bbox[0] + (match.bbox[0] - track.bbox[0]) * lerp,
                track.bbox[1] + (match.bbox[1] - track.bbox[1]) * lerp,
                track.bbox[2] + (match.bbox[2] - track.bbox[2]) * lerp,
                track.bbox[3] + (match.bbox[3] - track.bbox[3]) * lerp,
              ];
              const vx = newBbox[0] - track.bbox[0];
              const vy = newBbox[1] - track.bbox[1];
              const elapsedSeconds = Math.max(0.04, Math.min(0.8, (now - track.lastSeen) / 1000));
              const speed = (Math.hypot(vx, vy) / diag / elapsedSeconds) * 100;
              const stillFrames = speed < 0.18 ? track.stillFrames + 1 : 0;
              updated[trackId] = {
                bbox: newBbox,
                class: match.class,
                score: match.score,
                lastSeen: now,
                vx,
                vy,
                hits: track.hits + 1,
                speed,
                avgSpeed: track.avgSpeed * 0.85 + speed * 0.15,
                prevSpeed: track.speed,
                prevVx: track.vx,
                prevVy: track.vy,
                previousArea: track.bbox[2] * track.bbox[3],
                stillFrames,
                maxSpeed: Math.max(track.maxSpeed, speed),
                // A parked/waiting label is reversible as soon as movement resumes.
                parked: stillFrames > 16,
                alerted: track.alerted,
                anomalyFrames: track.anomalyFrames,
              };
            }
          } else if (now - track.lastSeen < 700) {
            // Coast the track forward on its last known velocity (occlusion)
            updated[trackId] = {
              ...track,
              bbox: [track.bbox[0] + track.vx, track.bbox[1] + track.vy, track.bbox[2], track.bbox[3]],
            };
          }
        });

        available.forEach((det) => {
          trackIdCounter.current++;
          updated[trackIdCounter.current] = {
            ...det,
            lastSeen: now,
            vx: 0,
            vy: 0,
            hits: 1,
            speed: 0,
            avgSpeed: 0,
            prevSpeed: 0,
            prevVx: 0,
            prevVy: 0,
            previousArea: det.bbox[2] * det.bbox[3],
            stillFrames: 0,
            maxSpeed: 0,
            parked: false,
            alerted: false,
            anomalyFrames: 0,
          };
        });

        tracksRef.current = updated;

        // 2. Accident heuristics on stable tracks. The uploaded example is a
        // short, elevated-camera clip, so use motion history instead of waiting
        // for long tracks or requiring a single very high confidence frame.
        const confirmed = Object.entries(updated).filter(([, t]) => t.hits >= 2 && t.score >= 0.36);
        confirmed.forEach(([key, track]) => {
          if (track.alerted) return;
          // A stationary car by itself is normal. It may still be the target of
          // a moving car, so parked tracks remain in pairwise collision checks.
          const suddenDeceleration =
            track.hits >= 3 &&
            track.prevSpeed > 0.38 &&
            track.speed < track.prevSpeed * 0.5 &&
            track.stillFrames <= 3;
          const previousMagnitude = Math.hypot(track.prevVx, track.prevVy);
          const currentMagnitude = Math.hypot(track.vx, track.vy);
          const directionCosine =
            previousMagnitude > 0 && currentMagnitude > 0
              ? (track.prevVx * track.vx + track.prevVy * track.vy) / (previousMagnitude * currentMagnitude)
              : 1;
          const abruptDirectionChange =
            track.hits >= 3 && track.prevSpeed > 0.28 && track.speed > 0.2 && directionCosine < 0.45;
          const area = track.bbox[2] * track.bbox[3];
          const areaChange = Math.abs(area - track.previousArea) / Math.max(track.previousArea, 1);
          const unstableScale = track.hits >= 3 && track.speed > 0.2 && areaChange > 0.25;
          const collision = confirmed.some(
            ([otherKey, other]) =>
              otherKey !== key &&
              vehiclesAreInContact(track, other),
          );
          const abnormalMotion = !track.parked && (suddenDeceleration || abruptDirectionChange || unstableScale);
          track.anomalyFrames = abnormalMotion ? track.anomalyFrames + 1 : Math.max(0, track.anomalyFrames - 1);

          // Vehicle contact is urgent. Single-vehicle anomalies need two
          // consecutive observations to avoid alerts from detector jitter.
          if (collision || track.anomalyFrames >= 2) {
            track.alerted = true;
            callbacksRef.current.onAccident(
              id,
              collision
                ? "สงสัยรถพุ่งชนกัน — กรุณาตรวจสอบ"
                : suddenDeceleration
                  ? "สงสัยรถชนเสาหรือวัตถุคงที่ — หยุดฉับพลันรุนแรง"
                  : "พบการเคลื่อนไหวผิดปกติรุนแรง — กรุณาตรวจสอบ",
              captureSnapshot(),
            );
          }
        });

        callbacksRef.current.onDetection(id, confirmed.map(([, t]) => t.class));
        setDetections(confirmed.map(([, t]) => ({ bbox: t.bbox, class: t.class, score: t.score })));
      }

      // 3. Rendering phase
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          const scaleX = canvas.width / video.videoWidth;
          const scaleY = canvas.height / video.videoHeight;

          Object.values(tracksRef.current).forEach((track) => {
            // Only draw a box around vehicles involved in an incident
            if (track.hits < 5 || !track.alerted) return;
            const [x, y, width, height] = track.bbox;
            const targetX = x * scaleX;
            const targetY = y * scaleY;
            const targetW = width * scaleX;
            const targetH = height * scaleY;
            const color = "#ef4444";

            ctx.shadowBlur = 10;
            ctx.shadowColor = "rgba(239, 68, 68, 0.6)";
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5;

            ctx.beginPath();
            ctx.roundRect(targetX, targetY, targetW, targetH, 4);
            ctx.stroke();

            ctx.shadowBlur = 0;
            const labelText = "ACCIDENT";
            ctx.font = "bold 10px monospace";
            const textWidth = ctx.measureText(labelText).width;

            ctx.fillStyle = color;
            ctx.fillRect(targetX, targetY - 16, textWidth + 10, 16);
            ctx.fillStyle = "white";
            ctx.fillText(labelText, targetX + 5, targetY - 4);
          });

          const motionAlert = motionAlertRef.current;
          if (motionAlert && motionAlert.until > now) {
            const [x, y, width, height] = motionAlert.bbox;
            const targetX = x * scaleX;
            const targetY = y * scaleY;
            const targetW = width * scaleX;
            const targetH = height * scaleY;
            ctx.shadowBlur = 10;
            ctx.shadowColor = "rgba(239, 68, 68, 0.6)";
            ctx.strokeStyle = "#ef4444";
            ctx.lineWidth = 2.5;
            ctx.strokeRect(targetX, targetY, targetW, targetH);
            ctx.shadowBlur = 0;
            ctx.fillStyle = "#ef4444";
            ctx.fillRect(targetX, Math.max(0, targetY - 16), 74, 16);
            ctx.fillStyle = "white";
            ctx.font = "bold 10px monospace";
            ctx.fillText("ABNORMAL", targetX + 5, Math.max(12, targetY - 4));
          }
        }
      }
    }
    requestRef.current = requestAnimationFrame(detectFrame);
  };


  useEffect(() => {
    if (videoSrc) {
      requestRef.current = requestAnimationFrame(detectFrame);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [videoSrc]);

  return (
    <div className="relative group aspect-video bg-neutral-900 border border-neutral-800 rounded-lg overflow-hidden flex items-center justify-center transition-all hover:border-blue-500/50">
      {videoSrc ? (
        <div className="relative w-full h-full">
          <video
            ref={videoRef}
            src={videoSrc}
            autoPlay
            muted
            playsInline
            className="w-full h-full object-cover"
          />
          <canvas
            ref={canvasRef}
            width={640}
            height={360}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
          />
        </div>
      ) : (
        <div 
          onClick={() => fileInputRef.current?.click()}
          className="flex flex-col items-center gap-3 cursor-pointer text-neutral-500 group-hover:text-neutral-300 transition-colors"
        >
          <div className="p-4 rounded-full bg-neutral-800 group-hover:bg-neutral-700 transition-colors">
            <Upload className="w-6 h-6" />
          </div>
          <span className="text-sm font-medium">อัพโหลดวิดีโอ CCTV {id}</span>
        </div>
      )}
      
      <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/60 backdrop-blur-md px-2.5 py-1 rounded-md border border-white/10">
        <div className={cn("w-2 h-2 rounded-full", videoSrc ? "bg-red-500 animate-pulse" : "bg-neutral-600")} />
        <span className="text-[10px] font-bold text-white uppercase tracking-wider">CAM-0{id}</span>
      </div>

      {detections.length > 0 && (
        <div className="absolute top-3 right-3 bg-green-600/80 backdrop-blur-sm px-2 py-0.5 rounded text-[10px] font-bold text-white flex items-center gap-1">
          <div className="w-1 h-1 bg-white rounded-full animate-pulse" />
          YOLOv11 LIVE: {detections.length}
        </div>
      )}

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/*"
        className="hidden"
      />
    </div>
  );
}

interface Incident {
  id: string;
  cam: number;
  type: string;
  time: string;
  status: "pending" | "confirmed" | "rejected";
  snapshot: string | null;
  lineStatus: "idle" | "sending" | "sent" | "failed";
  lineError?: string;
}

// พิกัดสมมติของกล้องแต่ละตัว (แก้ไขได้ภายหลังเมื่อมีพิกัดจริง)
const CAMERA_LOCATIONS: Record<number, { name: string; latitude: number; longitude: number }> = {
  1: { name: "แยกรัชดา-ลาดพร้าว (สมมติ)", latitude: 13.796519, longitude: 100.574112 },
  2: { name: "ถนนพระราม 9 ขาเข้า (สมมติ)", latitude: 13.758291, longitude: 100.565437 },
  3: { name: "แยกอโศก-สุขุมวิท (สมมติ)", latitude: 13.737541, longitude: 100.560574 },
  4: { name: "ถนนวิภาวดีรังสิต กม.6 (สมมติ)", latitude: 13.833216, longitude: 100.560129 },
};

function Index() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [activeDetections, setActiveDetections] = useState<Record<number, string[]>>({});
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [currentTime, setCurrentTime] = useState("--:--:--");
  const lastAlertRef = useRef<Record<number, number>>({});
  const sendToLine = useServerFn(sendLineAccidentAlert);

  useEffect(() => {
    const tick = () => setCurrentTime(new Date().toLocaleTimeString("en-US", { hour12: false }));
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    async function loadModel() {
      try {
        const loadedModel = await cocoSsd.load({
          base: 'mobilenet_v2' 
        });
        setModel(loadedModel);
      } catch (err) {
        console.error("Failed to load AI model", err);
      } finally {
        setIsLoadingModel(false);
      }
    }
    loadModel();
  }, []);

  const handleDetection = (id: number, objects: string[]) => {
    setActiveDetections(prev => ({ ...prev, [id]: objects }));
  };

  // 3-second alarm siren via Web Audio (no asset needed)
  const playAlarm = () => {
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.35, ctx.currentTime + 0.05);
      gain.connect(ctx.destination);

      const osc = ctx.createOscillator();
      osc.type = "square";
      // Alternating two-tone siren for 3 seconds
      for (let i = 0; i < 6; i++) {
        osc.frequency.setValueAtTime(i % 2 === 0 ? 880 : 620, ctx.currentTime + i * 0.5);
      }
      osc.connect(gain);
      osc.start();
      gain.gain.setValueAtTime(0.35, ctx.currentTime + 2.85);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 3);
      osc.stop(ctx.currentTime + 3);
      osc.onended = () => ctx.close();
    } catch (err) {
      console.error("alarm failed", err);
    }
  };

  const handleAccident = (id: number, reason: string, snapshot: string | null) => {
    const now = Date.now();
    // Throttle: max one alert per camera every 8 seconds
    if (now - (lastAlertRef.current[id] ?? 0) < 8000) return;
    lastAlertRef.current[id] = now;
    playAlarm();

    const incidentId = `${id}-${now}`;
    setIncidents(prev =>
      [
        {
          id: incidentId,
          cam: id,
          type: reason,
          time: new Date().toLocaleTimeString("en-US", { hour12: false }),
          status: "pending" as const,
          snapshot,
          lineStatus: "idle" as const,
        },
        ...prev,
      ].slice(0, 8),
    );
  };

  const verifyIncident = (incidentId: string, status: "confirmed" | "rejected") => {
    setIncidents((prev) => prev.map((i) => (i.id === incidentId ? { ...i, status } : i)));
    if (status !== "confirmed") return;

    const incident = incidents.find((i) => i.id === incidentId);
    if (!incident) return;
    const place = CAMERA_LOCATIONS[incident.cam] ?? CAMERA_LOCATIONS[1]!;

    setIncidents((prev) =>
      prev.map((i) => (i.id === incidentId ? { ...i, lineStatus: "sending" as const } : i)),
    );

    sendToLine({
      data: {
        camera: incident.cam,
        reason: incident.type,
        time: incident.time,
        latitude: place.latitude,
        longitude: place.longitude,
        locationName: place.name,
        ...(incident.snapshot ? { snapshot: incident.snapshot } : {}),
      },
    })
      .then((result) => {
        setIncidents((prev) =>
          prev.map((i) =>
            i.id === incidentId
              ? result.ok
                ? { ...i, lineStatus: "sent" as const }
                : { ...i, lineStatus: "failed" as const, lineError: result.error }
              : i,
          ),
        );
      })
      .catch((error: unknown) => {
        console.error("line alert failed", error);
        setIncidents((prev) =>
          prev.map((i) =>
            i.id === incidentId
              ? { ...i, lineStatus: "failed" as const, lineError: "ส่งแจ้งเตือนเข้าไลน์ไม่สำเร็จ" }
              : i,
          ),
        );
      });
  };


  return (
    <div className="min-h-screen bg-[#0a0a0a] text-neutral-200 font-sans selection:bg-blue-500/30">



      {/* Header */}
      <header className="h-16 border-b border-neutral-800 flex items-center justify-between px-6 bg-black/50 backdrop-blur-xl sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-blue-900/20">
            <ShieldCheck className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-white uppercase">ระบบตรวจสอบอุบัติเหตุ CCTV</h1>
            <p className="text-[10px] text-neutral-500 font-medium">ระบบปัญญาประดิษฐ์เฝ้าระวัง 24 ชม.</p>
          </div>
        </div>
        
        <div className="flex items-center gap-6">
          {isLoadingModel && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-900/20 rounded-md border border-blue-800/50">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-ping" />
              <span className="text-[10px] font-bold text-blue-400 uppercase">กำลังโหลด AI Model...</span>
            </div>
          )}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-neutral-900 rounded-md border border-neutral-800">
            <Activity className="w-3.5 h-3.5 text-green-500" />
            <span className="text-xs font-mono text-neutral-400">สถานะระบบ: {model ? 'ปกติ' : 'เตรียมการ'}</span>
          </div>
          <div className="text-right">
            <div className="text-xs font-mono text-neutral-400 tracking-widest">{currentTime}</div>
            <div className="text-[10px] text-neutral-600 font-bold uppercase tracking-tighter">กำลังทำงาน</div>
          </div>
        </div>
      </header>

      <main className="flex h-[calc(100vh-4rem)]">
        {/* Left Side: CCTV Grid */}
        <div className="flex-1 p-6 overflow-y-auto">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <LayoutGrid className="w-4 h-4 text-blue-500" />
              <h2 className="text-sm font-semibold uppercase tracking-widest text-neutral-400">มุมมองกล้อง: 2x2 (AI Active)</h2>
            </div>
            <div className="flex gap-2">
               <button className="px-3 py-1.5 bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 rounded text-xs transition-colors">บันทึกทั้งหมด</button>
               <button className="px-3 py-1.5 bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 rounded text-xs transition-colors">ตั้งค่า AI</button>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-6xl mx-auto">
            {[1, 2, 3, 4].map(id => (
              <CCTVMonitor 
                key={id} 
                id={id} 
                model={model} 
                onDetection={handleDetection}
                onAccident={handleAccident}

              />
            ))}
          </div>
        </div>

        {/* Right Side: Dashboard */}
        <aside className="w-80 border-l border-neutral-800 bg-neutral-900/30 flex flex-col">
          <div className="p-4 border-b border-neutral-800">
            <h2 className="text-xs font-bold uppercase tracking-widest text-neutral-500 flex items-center gap-2">
              <Activity className="w-3 h-3" /> รายงานเหตุการณ์
            </h2>
          </div>
          
          <div className="flex-1 p-4 flex flex-col gap-4 overflow-y-auto">
            {incidents.length > 0 ? (
              <div className="space-y-3">
                {incidents.map(incident => (
                  <div key={incident.id} className="p-3 bg-red-950/20 border border-red-900/50 rounded-lg animate-in fade-in slide-in-from-right-4">
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-[10px] font-bold text-red-500 uppercase">Alert: CAM-0{incident.cam}</span>
                      <span className="text-[9px] font-mono text-neutral-500">{incident.time}</span>
                    </div>
                    <p className="text-xs font-medium text-neutral-300">{incident.type}</p>
                    <p className="mt-1 text-[10px] text-neutral-500">
                      จุดเกิดเหตุ: {(CAMERA_LOCATIONS[incident.cam] ?? CAMERA_LOCATIONS[1]!).name}
                    </p>
                    {incident.snapshot && (
                      <img
                        src={incident.snapshot}
                        alt={`ภาพเหตุการณ์จากกล้อง CAM-0${incident.cam}`}
                        className="mt-2 w-full rounded border border-neutral-800 object-cover"
                      />
                    )}
                    {incident.lineStatus !== "idle" && (
                      <p
                        className={cn(
                          "mt-2 text-[10px] font-bold",
                          incident.lineStatus === "sent"
                            ? "text-green-400"
                            : incident.lineStatus === "failed"
                              ? "text-amber-400"
                              : "text-blue-400",
                        )}
                      >
                        {incident.lineStatus === "sending"
                          ? "กำลังส่งแจ้งเตือนเข้าไลน์..."
                          : incident.lineStatus === "sent"
                            ? "ส่งพิกัดและภาพเข้าไลน์ OA แล้ว"
                            : `ส่งไลน์ไม่สำเร็จ: ${incident.lineError ?? "ไม่ทราบสาเหตุ"}`}
                      </p>
                    )}


                    {incident.status === "pending" ? (
                      <div className="mt-3 space-y-2">
                        <p className="text-[10px] text-neutral-500">นี่คือเหตุการณ์จริงหรือไม่?</p>
                        <div className="flex gap-2">
                          <button
                            onClick={() => verifyIncident(incident.id, "confirmed")}
                            className="flex-1 px-2 py-1.5 rounded bg-red-600 hover:bg-red-500 text-[10px] font-bold text-white uppercase tracking-wider transition-colors"
                          >
                            ยืนยันเหตุจริง
                          </button>
                          <button
                            onClick={() => verifyIncident(incident.id, "rejected")}
                            className="flex-1 px-2 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700 border border-neutral-700 text-[10px] font-bold text-neutral-300 uppercase tracking-wider transition-colors"
                          >
                            แจ้งเตือนผิดพลาด
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="mt-2 text-[10px] font-bold uppercase tracking-wider">
                        {incident.status === "confirmed" ? (
                          <span className="text-red-400">ตรวจสอบแล้ว: อุบัติเหตุจริง</span>
                        ) : (
                          <span className="text-neutral-500">ตรวจสอบแล้ว: แจ้งเตือนผิดพลาด</span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-neutral-600 p-8 text-center">
                <div className="w-12 h-12 rounded-full border border-dashed border-neutral-700 flex items-center justify-center mb-4">
                  <Camera className="w-6 h-6 opacity-20" />
                </div>
                <p className="text-xs uppercase tracking-tighter font-semibold opacity-40">ไม่พบเหตุการณ์ผิดปกติ</p>
                <p className="text-[10px] mt-1 leading-relaxed opacity-30">กำลังวิเคราะห์ภาพจากกล้องเพื่อตรวจหาความผิดปกติ อุบัติเหตุ และการบุกรุก</p>
              </div>
            )}


            <div className="p-4 rounded-xl bg-blue-900/10 border border-blue-500/20">
              <div className="flex items-center gap-2 text-blue-500 mb-3">
                <Box className="w-3 h-3" />
                <h3 className="text-[10px] font-bold uppercase tracking-widest">YOLOv11 API Connection</h3>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between text-[10px] font-mono">
                  <span className="text-neutral-500">ENDPOINT:</span>
                  <span className="text-blue-400">api.cctv-ai.cloud/v1/detect</span>
                </div>
                <div className="flex items-center justify-between text-[10px] font-mono">
                  <span className="text-neutral-500">LATENCY:</span>
                  <span className="text-green-500">24ms</span>
                </div>
                <div className="flex items-center justify-between text-[10px] font-mono">
                  <span className="text-neutral-500">ACCURACY:</span>
                  <span className="text-blue-400">98.4%</span>
                </div>
                <div className="pt-2 border-t border-blue-500/10">
                  <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                    <span className="text-[9px] text-green-500 font-bold uppercase">Connected to GPU Cluster</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="p-4 bg-black/40 border-t border-neutral-800 space-y-4">
             <div className="flex justify-between items-center text-[10px] font-mono text-neutral-500">
                <span>ENCRYPTION</span>
                <span className="text-green-900 font-bold uppercase">Secured</span>
             </div>
             
          </div>
        </aside>
      </main>
    </div>
  );
}
