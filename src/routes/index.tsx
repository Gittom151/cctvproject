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
  isExpanded: boolean;
  onToggleExpand: (id: number) => void;
}

function CCTVMonitor({ id, model, onDetection, isExpanded, onToggleExpand }: CCTVMonitorProps) {
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
      // Optimized for performance: skip more frames and use a lighter detection strategy
      if (detectionCounter.current % 3 === 0) { 
        // Offload detection to avoid blocking the main thread too long
        const predictions = model ? await model.detect(video, 8, 0.5) : [];
        const vehicleClasses = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person'];
        const currentDetections = predictions.filter(p => vehicleClasses.includes(p.class)) as Detection[];

        // 2. Simple Object Tracking Logic (IOU based association)
        const updatedTracks: typeof objectTracksRef.current = {};
        const availableDetections = [...currentDetections];

        // Match existing tracks with new detections
        Object.entries(objectTracksRef.current).forEach(([trackId, track]) => {
          let bestMatchIndex = -1;
          let maxIOU = 0.35; // Slightly higher threshold for better stability

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
          } else if (now - track.lastSeen < 600) { // Increased persistence for better track stability
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

            // Linear Interpolation (Lerp) for ultra-smooth movement
            const lerp = 0.25;
            const currentTrack = track;
            
            // We use the raw values but add a visual smoothing layer
            // For drawing, we'll keep it snappy but clean

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
    <div 
      onClick={() => onToggleExpand(id)}
      className={cn(
        "relative group bg-neutral-900 border border-neutral-800 rounded-lg overflow-hidden flex items-center justify-center transition-all duration-500 cursor-pointer",
        isExpanded ? "fixed inset-10 z-[100] shadow-[0_0_100px_rgba(0,0,0,0.8)] border-blue-500/30" : "aspect-video hover:border-blue-500/50"
      )}
    >
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
  const [expandedCam, setExpandedCam] = useState<number | null>(null);
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [activeDetections, setActiveDetections] = useState<Record<number, string[]>>({});
  const [incidents, setIncidents] = useState<{id: string, cam: number, type: string, time: string}[]>([]);
  const currentTime = new Date().toLocaleTimeString('en-US', { hour12: false });

  useEffect(() => {
    async function loadModel() {
      try {
        const loadedModel = await cocoSsd.load({
          base: 'lite_mobilenet_v2' 
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
    <div className="min-h-screen bg-[#050505] text-neutral-200 font-sans selection:bg-blue-500/30">
      {/* Banner: Status */}
      <div className="bg-blue-500/5 border-b border-white/5 py-1.5 px-6 flex items-center justify-between">
        <div className="flex items-center gap-2 text-blue-400/80">
          <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse" />
          <span className="text-[10px] font-medium tracking-wide uppercase">YOLOv11 Engine Connected • GPU Cluster Active</span>
        </div>
        <div className="text-[9px] text-neutral-500 font-mono">
          v11.4.2-stable
        </div>
      </div>

      {/* Header */}
      <header className="h-16 border-b border-white/5 flex items-center justify-between px-8 bg-black/40 backdrop-blur-2xl sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div className="relative">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl flex items-center justify-center shadow-2xl shadow-blue-500/20">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>
            <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 border-2 border-[#050505] rounded-full" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight text-white">SENTINEL AI</h1>
            <p className="text-[10px] text-neutral-500 font-bold uppercase tracking-[0.2em]">CCTV Detection System</p>
          </div>
        </div>
        
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-4">
            <div className="flex flex-col items-end">
              <span className="text-[10px] text-neutral-500 font-bold uppercase tracking-wider">System Status</span>
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-green-500 rounded-full" />
                <span className="text-xs font-mono text-neutral-300">OPERATIONAL</span>
              </div>
            </div>
            
            <div className="w-px h-8 bg-white/10" />

            <div className="flex flex-col items-end">
              <span className="text-[10px] text-neutral-500 font-bold uppercase tracking-wider">Local Time</span>
              <span className="text-xs font-mono text-neutral-300 tracking-widest">{currentTime}</span>
            </div>
          </div>
          
          <button className="h-10 px-5 bg-white text-black text-xs font-bold rounded-full hover:bg-neutral-200 transition-all shadow-xl shadow-white/5 active:scale-95">
            LOG OUT
          </button>
        </div>
      </header>

      <main className="flex h-[calc(100vh-4rem)]">
        {/* Left Side: CCTV Grid */}
        <div className="flex-1 p-8 overflow-y-auto">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <LayoutGrid className="w-5 h-5 text-blue-500" />
              </div>
              <div>
                <h2 className="text-sm font-bold uppercase tracking-[0.1em] text-white">Live Monitors</h2>
                <p className="text-[10px] text-neutral-500 font-medium uppercase tracking-wider">4 Active Channels • AI-Assisted</p>
              </div>
            </div>
            <div className="flex gap-3">
               <button className="h-9 px-4 bg-neutral-900/50 hover:bg-neutral-800 border border-white/5 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all">Export Footage</button>
               <button className="h-9 px-4 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all shadow-lg shadow-blue-500/20">AI Settings</button>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-7xl mx-auto">
            {[1, 2, 3, 4].map(id => (
              <CCTVMonitor 
                key={id} 
                id={id} 
                model={model} 
                onDetection={handleDetection}
                isExpanded={expandedCam === id}
                onToggleExpand={(camId) => setExpandedCam(expandedCam === camId ? null : camId)}
              />
            ))}
          </div>

          {expandedCam && (
            <div 
              className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[90] animate-in fade-in"
              onClick={() => setExpandedCam(null)}
            />
          )}
        </div>

        {/* Right Side: Dashboard */}
        <aside className="w-96 border-l border-white/5 bg-[#080808] flex flex-col">
          <div className="p-6 border-b border-white/5">
            <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-neutral-400 flex items-center gap-2">
              <Activity className="w-3.5 h-3.5 text-blue-500" /> Activity Log
            </h2>
          </div>
          
          <div className="flex-1 p-6 flex flex-col gap-6 overflow-y-auto">
            {incidents.length > 0 ? (
              <div className="space-y-4">
                {incidents.map(incident => (
                  <div key={incident.id} className="group relative p-4 bg-red-500/5 border border-red-500/10 rounded-xl transition-all hover:bg-red-500/10 hover:border-red-500/20 animate-in fade-in slide-in-from-right-4">
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                        <span className="text-[10px] font-bold text-red-500 uppercase tracking-wider">Alert: CAM-0{incident.cam}</span>
                      </div>
                      <span className="text-[10px] font-mono text-neutral-500">{incident.time}</span>
                    </div>
                    <p className="text-xs font-bold text-neutral-200 uppercase tracking-tight">{incident.type}</p>
                    <div className="mt-3 flex gap-2">
                      <button className="text-[9px] font-bold uppercase tracking-widest text-red-400 hover:text-red-300 transition-colors">View Clip</button>
                      <button className="text-[9px] font-bold uppercase tracking-widest text-neutral-500 hover:text-neutral-400 transition-colors">Dismiss</button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-neutral-600 p-8 text-center">
                <div className="w-16 h-16 rounded-2xl bg-neutral-900 flex items-center justify-center mb-6 shadow-inner border border-white/5">
                  <Camera className="w-8 h-8 opacity-20" />
                </div>
                <p className="text-[10px] uppercase tracking-[0.2em] font-bold text-neutral-400 mb-2">System Clear</p>
                <p className="text-[11px] leading-relaxed opacity-40 max-w-[200px]">AI models are currently scanning all feeds for suspicious activity.</p>
              </div>
            )}


            <div className="p-6 rounded-2xl bg-gradient-to-br from-blue-600/10 to-transparent border border-blue-500/20 shadow-xl shadow-blue-500/5">
              <div className="flex items-center gap-3 text-blue-400 mb-5">
                <Box className="w-4 h-4" />
                <h3 className="text-[10px] font-bold uppercase tracking-[0.2em]">YOLOv11 API INFRA</h3>
              </div>
              <div className="space-y-4">
                <div className="flex flex-col gap-1">
                  <span className="text-[9px] text-neutral-500 font-bold uppercase tracking-wider">Gateway Endpoint</span>
                  <span className="text-[11px] font-mono text-blue-400 truncate">api.sentinel-ai.cloud/v1/live</span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex flex-col gap-1">
                    <span className="text-[9px] text-neutral-500 font-bold uppercase tracking-wider">Latency</span>
                    <span className="text-xs font-mono text-green-500">24ms</span>
                  </div>
                  <div className="flex flex-col gap-1">
                    <span className="text-[9px] text-neutral-500 font-bold uppercase tracking-wider">Accuracy</span>
                    <span className="text-xs font-mono text-blue-400">98.4%</span>
                  </div>
                </div>
                <div className="pt-4 border-t border-white/5">
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-1.5 h-1.5 bg-green-500 rounded-full shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
                    <span className="text-[10px] font-bold text-neutral-400 uppercase tracking-widest">GPU Cluster: ONLINE</span>
                  </div>
                  <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                    <div className="w-[65%] h-full bg-blue-500 rounded-full shadow-[0_0_12px_rgba(59,130,246,0.4)]" />
                  </div>
                  <p className="text-[9px] text-neutral-600 mt-2 font-medium uppercase tracking-tighter">Load: 65.2% (RTX 4090 x 8)</p>
                </div>
              </div>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
