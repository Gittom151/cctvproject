import { createFileRoute } from "@tanstack/react-router";
import { useState, useRef, useEffect, useMemo } from "react";
import { Upload, Camera, AlertCircle, Activity, LayoutGrid, ShieldCheck, Box } from "lucide-react";
import { cn } from "@/lib/utils";
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
  onAccident: (id: number, reason: string) => void;
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
  const relativeVx = b.vx - a.vx;
  const relativeVy = b.vy - a.vy;
  const approaching = centerDx * relativeVx + centerDy * relativeVy < 0;
  return (iou(a.bbox, b.bbox) > 0.08 || contactDistance <= contactLimit) && approaching;
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
  const callbacksRef = useRef({ onDetection, onAccident });
  callbacksRef.current = { onDetection, onAccident };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
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

      // 1. Detection phase — every 2nd frame for better temporal accuracy
      if (detectionCounter.current % 2 === 0) {
        let raw: Detection[] = [];
        try {
          raw = model ? ((await model.detect(video, 15, 0.55)) as Detection[]) : [];
        } catch (err) {
          console.error("detect failed", err);
        }
        const diag = Math.hypot(video.videoWidth, video.videoHeight) || 1;
        const candidates = nms(
          raw.filter((p) => {
            if (!VEHICLE_CLASSES.includes(p.class)) return false;
            const [, , w, h] = p.bbox;
            // Reject implausible boxes: too small, oversized, or extreme aspect ratio
            if (w < video.videoWidth * 0.03 || h < video.videoHeight * 0.03) return false;
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
          let bestScore = 0.22;

          available.forEach((det, index) => {
            const score = Math.max(iou(track.bbox, det.bbox), iou(predicted, det.bbox));
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
              const speed = (Math.hypot(vx, vy) / diag) * 100;
              const stillFrames = speed < 0.06 ? track.stillFrames + 1 : 0;
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
                // Vehicles that stay still for ~1.5s are parked or waiting at a red light
                parked: track.parked || stillFrames > 25,
                alerted: track.alerted,
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
          };
        });

        tracksRef.current = updated;

        // 2. Accident heuristics on confirmed tracks (stricter confirmation = fewer false boxes)
        const confirmed = Object.entries(updated).filter(([, t]) => t.hits >= 5 && t.score >= 0.6);
        confirmed.forEach(([key, track]) => {
          if (track.alerted) return;
          // Parked cars / cars waiting at a red light stay still smoothly — never alert on them.
          if (track.parked || track.maxSpeed < 0.4) return;
          // A violent loss of speed catches impacts with poles/walls, which are
          // not object classes available in COCO-SSD. Gentle braking is ignored.
          const suddenDeceleration =
            track.hits >= 10 &&
            track.prevSpeed > 0.7 &&
            track.speed < track.prevSpeed * 0.12 &&
            track.stillFrames <= 2;
          const previousMagnitude = Math.hypot(track.prevVx, track.prevVy);
          const currentMagnitude = Math.hypot(track.vx, track.vy);
          const directionCosine =
            previousMagnitude > 0 && currentMagnitude > 0
              ? (track.prevVx * track.vx + track.prevVy * track.vy) / (previousMagnitude * currentMagnitude)
              : 1;
          const abruptDirectionChange =
            track.hits >= 10 && previousMagnitude > 0.6 && currentMagnitude > 0.45 && directionCosine < -0.35;
          // Detect contact before boxes heavily overlap, while the vehicles are approaching at speed.
          const collision = confirmed.some(
            ([otherKey, other]) =>
              otherKey !== key &&
              other.hits >= 5 &&
              !other.parked &&
              (track.speed > 0.5 || other.speed > 0.5) &&
              vehiclesAreInContact(track, other),
          );

          if (collision || suddenDeceleration || abruptDirectionChange) {
            track.alerted = true;
            callbacksRef.current.onAccident(
              id,
              collision
                ? "สงสัยรถพุ่งชนกัน — กรุณาตรวจสอบ"
                : suddenDeceleration
                  ? "สงสัยรถชนเสาหรือวัตถุคงที่ — หยุดฉับพลันรุนแรง"
                  : "พบการเคลื่อนไหวผิดปกติรุนแรง — กรุณาตรวจสอบ",
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
            if (track.hits < 3 || !track.alerted) return;
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

function Index() {
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [activeDetections, setActiveDetections] = useState<Record<number, string[]>>({});
  const [incidents, setIncidents] = useState<{id: string, cam: number, type: string, time: string, status: "pending" | "confirmed" | "rejected"}[]>([]);
  const [currentTime, setCurrentTime] = useState("--:--:--");
  const lastAlertRef = useRef<Record<number, number>>({});

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

  const handleAccident = (id: number, reason: string) => {
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
        },
        ...prev,
      ].slice(0, 8),
    );
  };

  const verifyIncident = (incidentId: string, status: "confirmed" | "rejected") => {
    setIncidents((prev) => prev.map((i) => (i.id === incidentId ? { ...i, status } : i)));
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
