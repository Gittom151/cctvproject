import { createFileRoute } from "@tanstack/react-router";
import { useState, useRef } from "react";
import { Upload, Camera, AlertCircle, Activity, LayoutGrid, ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    title: "CCTV Monitoring System",
    meta: [
      { name: "description", content: "AI-powered CCTV monitoring and incident detection dashboard." },
      { property: "og:title", content: "CCTV Monitoring System" },
      { property: "og:description", content: "Advanced 2x2 CCTV monitoring layout with real-time analytics." },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
});

interface CCTVMonitorProps {
  id: number;
}

function CCTVMonitor({ id }: CCTVMonitorProps) {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
    }
  };

  return (
    <div className="relative group aspect-video bg-neutral-900 border border-neutral-800 rounded-lg overflow-hidden flex flex-center items-center justify-center transition-all hover:border-blue-500/50">
      {videoSrc ? (
        <video
          src={videoSrc}
          autoPlay
          muted
          loop
          className="w-full h-full object-cover"
        />
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

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/*"
        className="hidden"
      />
      
      {videoSrc && (
        <button 
          onClick={() => setVideoSrc(null)}
          className="absolute bottom-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity bg-black/60 hover:bg-red-900/40 text-white p-1.5 rounded-md border border-white/10"
        >
          <Upload className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}

function Index() {
  const currentTime = new Date().toLocaleTimeString('en-US', { hour12: false });

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
          <div className="flex items-center gap-2 px-3 py-1.5 bg-neutral-900 rounded-md border border-neutral-800">
            <Activity className="w-3.5 h-3.5 text-green-500" />
            <span className="text-xs font-mono text-neutral-400">สถานะระบบ: ปกติ</span>
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
              <h2 className="text-sm font-semibold uppercase tracking-widest text-neutral-400">มุมมองกล้อง: 2x2</h2>
            </div>
            <div className="flex gap-2">
               <button className="px-3 py-1.5 bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 rounded text-xs transition-colors">บันทึกทั้งหมด</button>
               <button className="px-3 py-1.5 bg-neutral-900 hover:bg-neutral-800 border border-neutral-800 rounded text-xs transition-colors">ควบคุม PTZ</button>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-6xl mx-auto">
            <CCTVMonitor id={1} />
            <CCTVMonitor id={2} />
            <CCTVMonitor id={3} />
            <CCTVMonitor id={4} />
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
            {/* Empty State / Placeholder for incidents */}
            <div className="flex-1 flex flex-col items-center justify-center text-neutral-600 p-8 text-center">
              <div className="w-12 h-12 rounded-full border border-dashed border-neutral-700 flex items-center justify-center mb-4">
                <Camera className="w-6 h-6 opacity-20" />
              </div>
              <p className="text-xs uppercase tracking-tighter font-semibold opacity-40">ไม่พบเหตุการณ์รถชน</p>
              <p className="text-[10px] mt-1 leading-relaxed opacity-30">กำลังวิเคราะห์ภาพจากกล้องเพื่อตรวจหาความผิดปกติ อุบัติเหตุ และการบุกรุก</p>
            </div>

            {/* Placeholder Analytics Card */}
            <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800/50">
              <h3 className="text-[10px] font-bold text-neutral-500 uppercase mb-3">ความหนาแน่นของการจราจร</h3>
              <div className="space-y-2">
                <div className="h-1 w-full bg-neutral-800 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 w-1/3 shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
                </div>
                <div className="flex justify-between text-[10px] font-mono text-neutral-500">
                  <span>โซน A</span>
                  <span>น้อย</span>
                </div>
              </div>
            </div>

            <div className="p-4 rounded-xl bg-neutral-900 border border-neutral-800/50">
              <div className="flex items-center gap-2 text-amber-500/50 mb-3">
                <AlertCircle className="w-3 h-3" />
                <h3 className="text-[10px] font-bold uppercase">การแจ้งเตือนระบบ</h3>
              </div>
              <div className="text-[10px] font-mono text-neutral-600 italic">
                รอรับข้อมูลวิเคราะห์...
              </div>
            </div>
          </div>

          <div className="p-4 bg-black/40 border-t border-neutral-800">
             <div className="flex justify-between items-center text-[10px] font-mono text-neutral-500">
                <span>ENCRYPTION</span>
                <span className="text-green-900 font-bold">AES-256</span>
             </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
