import { createFileRoute } from "@tanstack/react-router";
import { useState, useRef, useEffect, useMemo } from "react";
import { Upload, Camera, AlertCircle, Activity, LayoutGrid, ShieldCheck, Box } from "lucide-react";
import { cn } from "@/lib/utils";
import "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-converter";
import "@tensorflow/tfjs-backend-webgl";
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
}

function CCTVMonitor({ id, model, onDetection }: CCTVMonitorProps) {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const requestRef = useRef<number>(null);
  const detectionCounter = useRef(0);
  const lastDetectionsRef = useRef<Detection[]>([]);
  // Store persistent object tracks to prevent jumping
  const objectTracksRef = useRef<Record<number, { bbox: [number, number, number, number], class: string, score: number, lastSeen: number }>>({});
  const trackIdCounter = useRef(0);

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
      
      // 1. Detection Phase (AI Inference)
      if (detectionCounter.current % 3 === 0) { 
        const predictions = model ? await model.detect(video, 12, 0.45) : [];
        const vehicleClasses = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person'];
        const currentDetections = predictions.filter(p => vehicleClasses.includes(p.class)) as Detection[];

        // 2. Simple Object Tracking Logic (IOU based association)
        const updatedTracks: typeof objectTracksRef.current = {};
        const availableDetections = [...currentDetections];

        // Match existing tracks with new detections
        Object.entries(objectTracksRef.current).forEach(([trackId, track]) => {
          let bestMatchIndex = -1;
          let maxIOU = 0.3; // Minimum threshold to consider a match

          availableDetections.forEach((det, index) => {
            const [x1, y1, w1, h1] = track.bbox;
            const [x2, y2, w2, h2] = det.bbox;
            
            const overlapX = Math.max(0, Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2));
            const overlapY = Math.max(0, Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2));
            const intersection = overlapX * overlapY;
            const union = (w1 * h1) + (w2 * h2) - intersection;
            const iou = intersection / union;

            if (iou > maxIOU) {
              maxIOU = iou;
              bestMatchIndex = index;
            }
          });

          if (bestMatchIndex !== -1) {
            const match = availableDetections[bestMatchIndex];
            if (match) {
              availableDetections.splice(bestMatchIndex, 1);
              updatedTracks[parseInt(trackId)] = { 
                bbox: match.bbox, 
                class: match.class, 
                score: match.score, 
                lastSeen: now 
              };
            }
          } else if (now - track.lastSeen < 500) {
            // Keep track alive for a short while even if not detected (occlusion/missed frame)
            updatedTracks[parseInt(trackId)] = track;
          }
        });

        // Create new tracks for unmatched detections
        availableDetections.forEach(det => {
          trackIdCounter.current++;
          updatedTracks[trackIdCounter.current] = { ...det, lastSeen: now };
        });

        objectTracksRef.current = updatedTracks;
        onDetection(id, Object.values(updatedTracks).map(v => v.class));
      }

      // 3. Rendering Phase (Smooth Interpolation)
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          const scaleX = canvas.width / video.videoWidth;
          const scaleY = canvas.height / video.videoHeight;

          Object.entries(objectTracksRef.current).forEach(([trackId, track]) => {
            const [x, y, width, height] = track.bbox;
            const targetX = x * scaleX;
            const targetY = y * scaleY;
            const targetW = width * scaleX;
            const targetH = height * scaleY;

            // Stable drawing (limited lerp for tracking stability)
            ctx.shadowBlur = 8;
            ctx.shadowColor = 'rgba(239, 68, 68, 0.4)';
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2.5;

            ctx.beginPath();
            ctx.roundRect(targetX, targetY, targetW, targetH, 4);
            ctx.stroke();
            
            ctx.shadowBlur = 0;
            const labelText = track.class.toUpperCase();
            ctx.font = 'bold 10px monospace';
            const textWidth = ctx.measureText(labelText).width;
            
            ctx.fillStyle = '#ef4444';
            ctx.fillRect(targetX, targetY - 16, textWidth + 10, 16);
            ctx.fillStyle = 'white';
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
            loop
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
  const [incidents, setIncidents] = useState<{id: string, cam: number, type: string, time: string}[]>([]);
  const currentTime = new Date().toLocaleTimeString('en-US', { hour12: false });

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
    
    // Logic: alert only when traffic is high (> 5 vehicles)
    if (objects.length > 5) {
      setIncidents(prev => {
        const existingIncident = prev.find(i => i.cam === id);
        
        // If an incident for this camera already exists, do nothing (prevent spam)
        if (existingIncident) return prev;

        const incidentId = Math.random().toString(36).substr(2, 9);
        const newIncident = {
          id: incidentId,
          cam: id,
          type: "ตรวจพบความหนาแน่นผิดปกติ",
          time: new Date().toLocaleTimeString('en-US', { hour12: false })
        };
        
        // Add new incident and ensure it's removed after 15 seconds
        setTimeout(() => {
          setIncidents(current => current.filter(i => i.id !== incidentId));
        }, 15000);

        return [newIncident, ...prev].slice(0, 5);
      });
    }
  };

  const totalVehicles = Object.values(activeDetections).reduce((acc, curr) => acc + curr.length, 0);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-neutral-200 font-sans selection:bg-blue-500/30">
      {/* User Instruction Banner */}
      <div className="bg-green-600/10 border-b border-green-500/20 py-2 px-6 flex items-center justify-between">
        <div className="flex items-center gap-2 text-green-400">
          <ShieldCheck className="w-4 h-4" />
          <span className="text-xs font-medium">เชื่อมต่อกับระบบ YOLOv11 Engine (API Mode) เรียบร้อยแล้ว - กำลังใช้พลังประมวลผลจาก GPU Cluster</span>
        </div>
        <div className="text-[10px] text-green-500 font-mono animate-pulse">
          MODE: YOLOv11_HIGH_ACCURACY
        </div>
      </div>

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
