"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useParams } from "next/navigation";
import { Loader2, CheckCircle2, AlertCircle, Play, Download, Trophy, Film, QrCode, RefreshCw, Upload } from "lucide-react";
import { supabase } from "@/lib/supabase";
import type { Session, Highlight } from "@/types/database";
import { cn } from "@/lib/cn";
import { QRCodeSVG } from "qrcode.react";

interface SessionWithUrls extends Omit<Session, "highlights"> {
  urls?: {
    stitched_video?: string;
    tracked_video?: string;
    left_raw?: string;
    right_raw?: string;
  };
  highlights?: (Highlight & { clip_url?: string })[] | null;
}

const STATUS_STEPS = ["created", "uploading", "processing", "done"] as const;

const PROGRESS_LABELS: Record<string, string> = {
  created: "Session created",
  uploading: "Uploading videos",
  processing: "AI processing",
  done: "Complete",
};

export default function SessionPage() {
  const { id } = useParams<{ id: string }>();
  const [session, setSession] = useState<SessionWithUrls | null>(null);
  const [activeTab, setActiveTab] = useState<"panoramic" | "tracked" | "highlights">("panoramic");
  const [activeHighlight, setActiveHighlight] = useState<number>(0);
  const [retrying, setRetrying] = useState(false);

  async function retryProcessing() {
    setRetrying(true);
    try {
      const res = await fetch(`/api/sessions/${id}/retry`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) alert(`Failed to start processing: ${data.error || res.status}`);
    } catch (e) {
      alert(`Failed to start processing: ${e}`);
    }
    await fetchSession();
    setRetrying(false);
  }

  const fetchSession = useCallback(async () => {
    const res = await fetch(`/api/sessions/${id}`);
    if (res.ok) {
      const data = await res.json();
      setSession(data);
    }
  }, [id]);

  useEffect(() => {
    fetchSession();
  }, [fetchSession]);

  // Poll every 4 seconds while processing, or while done but tracking not yet available
  useEffect(() => {
    if (!session) return;
    if (session.status === "error") return;
    if (session.status === "done" && session.tracked_video_key) return;
    const interval = setInterval(fetchSession, 4000);
    return () => clearInterval(interval);
  }, [session?.status, session?.tracked_video_key, fetchSession]);

  // Realtime subscription for live status updates
  useEffect(() => {
    const channel = supabase
      .channel(`session:${id}`)
      .on(
        "postgres_changes",
        { event: "UPDATE", schema: "public", table: "sessions", filter: `id=eq.${id}` },
        () => { fetchSession(); }
      )
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, [id, fetchSession]);

  if (!session) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="text-green-400 animate-spin" size={32} />
      </div>
    );
  }

  const currentStepIndex = STATUS_STEPS.indexOf(session.status as typeof STATUS_STEPS[number]);
  const isProcessing = session.status === "processing";
  const isDone = session.status === "done";
  const isError = session.status === "error";

  return (
    <div className="min-h-screen px-4 py-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <a href="/" className="text-green-500 text-sm font-medium hover:text-green-400 mb-2 inline-block">&larr; FieldVision</a>
          <h1 className="text-2xl font-bold">{session.team_name || "Match Session"}</h1>
          <p className="text-green-200/50 text-sm">{session.match_date || ""} · <span className="font-mono text-xs text-green-700">{id}</span></p>
        </div>
        <StatusBadge status={session.status} />
      </div>

      {/* Progress steps */}
      {!isDone && !isError && (
        <div className="bg-green-950/30 border border-green-900/40 rounded-2xl p-6 mb-8">
          <div className="flex items-center gap-0 mb-6">
            {STATUS_STEPS.map((s, i) => (
              <div key={s} className="flex items-center flex-1 last:flex-none">
                <div className="flex flex-col items-center gap-1">
                  <div className={cn(
                    "w-7 h-7 rounded-full flex items-center justify-center text-xs transition-all",
                    i < currentStepIndex ? "bg-green-500 text-black" :
                      i === currentStepIndex ? "bg-green-500/30 border-2 border-green-500 text-green-300" :
                        "bg-green-900/40 text-green-700 border border-green-800/40"
                  )}>
                    {i < currentStepIndex ? <CheckCircle2 size={14} /> : i + 1}
                  </div>
                  <span className={cn(
                    "text-[10px] font-medium whitespace-nowrap",
                    i === currentStepIndex ? "text-green-300" : i < currentStepIndex ? "text-green-500" : "text-green-800"
                  )}>
                    {PROGRESS_LABELS[s]}
                  </span>
                </div>
                {i < STATUS_STEPS.length - 1 && (
                  <div className={cn("flex-1 h-px mx-2 mb-4", i < currentStepIndex ? "bg-green-500/60" : "bg-green-900/40")} />
                )}
              </div>
            ))}
          </div>

          {isProcessing && (
            <div className="flex flex-col gap-2">
              <div className="flex justify-between text-xs text-green-500">
                <span>{(session.progress ?? 0) <= 1 ? "Starting worker — takes ~2 min to boot..." : "Processing with AI..."}</span>
                <span>{session.progress ?? 0}%</span>
              </div>
              <div className="w-full bg-green-900/30 rounded-full h-2">
                <div
                  className="bg-green-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${Math.max(session.progress ?? 0, 2)}%` }}
                />
              </div>
              <p className="text-xs text-green-700 mt-1">
                {(session.progress ?? 0) <= 1 ? "Worker is cold-starting, progress will appear shortly..." : "Stitching videos, running ball tracking, detecting highlights..."}
              </p>
            </div>
          )}

              {/* QR codes for live recording */}
          {(session.status === "created" || session.status === "uploading") && !isProcessing && (
            <RecordingQRCodes sessionId={id} />
          )}

          {/* Direct upload buttons for existing footage */}
          {(session.status === "created" || session.status === "uploading" || session.status === "error") && !isProcessing && (
            <VideoUploader sessionId={id} />
          )}
        </div>
      )}

      {/* Error / stuck state */}
      {isError && (
        <div className="bg-red-950/30 border border-red-800/40 rounded-2xl p-6 mb-8 flex items-start gap-3">
          <AlertCircle size={20} className="text-red-400 mt-0.5 shrink-0" />
          <div className="flex-1">
            <p className="font-semibold text-red-300">Processing failed</p>
            <p className="text-sm text-red-400/70 mt-1">{session.error_message || "An unknown error occurred."}</p>
          </div>
          {(session.left_video_key && session.right_video_key) && (
            <button
              onClick={retryProcessing}
              disabled={retrying}
              className="flex items-center gap-2 bg-red-900/40 hover:bg-red-800/50 border border-red-700/40 text-red-300 text-sm font-medium px-4 py-2 rounded-xl transition-colors disabled:opacity-50 shrink-0"
            >
              {retrying ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
              Retry
            </button>
          )}
        </div>
      )}

      {/* Reprocess / Start Processing — show whenever both videos exist */}
      {session.left_video_key && session.right_video_key && (
        <div className="flex justify-end mb-4">
          <button
            onClick={retryProcessing}
            disabled={retrying}
            className="flex items-center gap-2 bg-green-950/40 hover:bg-green-900/50 border border-green-800/40 text-green-500 text-xs font-medium px-3 py-1.5 rounded-lg transition-colors disabled:opacity-50"
          >
            {retrying ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
            {isDone ? "Reprocess with latest algorithm" : isProcessing ? "Restart processing" : "Start Processing"}
          </button>
        </div>
      )}

      {/* Results */}
      {isDone && (
        <>
          {/* Tabs */}
          <div className="flex gap-1 p-1 bg-green-950/30 rounded-xl border border-green-900/30 mb-6 w-fit">
            {[
              { key: "panoramic", label: "Full Panorama", icon: Film },
              { key: "tracked", label: "Ball Tracking", icon: Play },
              { key: "highlights", label: "Highlights", icon: Trophy },
            ].map(({ key, label, icon: Icon }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key as typeof activeTab)}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
                  activeTab === key
                    ? "bg-green-500 text-black"
                    : "text-green-400 hover:text-green-300"
                )}
              >
                <Icon size={14} />
                {label}
              </button>
            ))}
          </div>

          {/* Panoramic view */}
          {activeTab === "panoramic" && session.urls?.stitched_video && (
            <div className="rounded-2xl overflow-hidden border border-green-900/40 bg-black">
              <video
                src={session.urls.stitched_video}
                controls
                className="w-full aspect-video"
                playsInline
              />
              <div className="flex items-center justify-between px-4 py-3 border-t border-green-900/30">
                <span className="text-xs text-green-600 font-mono">STITCHED PANORAMA</span>
                <a
                  href={session.urls.stitched_video}
                  download
                  className="flex items-center gap-1 text-green-400 hover:text-green-300 text-xs font-medium"
                >
                  <Download size={12} /> Download
                </a>
              </div>
            </div>
          )}

          {/* Ball tracking */}
          {activeTab === "tracked" && session.urls?.tracked_video && (
            <div className="rounded-2xl overflow-hidden border border-green-900/40 bg-black">
              <video
                src={session.urls.tracked_video}
                controls
                className="w-full aspect-video"
                playsInline
              />
              <div className="flex items-center justify-between px-4 py-3 border-t border-green-900/30">
                <span className="text-xs text-green-600 font-mono">AI BALL + PLAYER TRACKING</span>
                <a
                  href={session.urls.tracked_video}
                  download
                  className="flex items-center gap-1 text-green-400 hover:text-green-300 text-xs font-medium"
                >
                  <Download size={12} /> Download
                </a>
              </div>
            </div>
          )}

          {/* Highlights */}
          {activeTab === "highlights" && session.highlights && session.highlights.length > 0 && (
            <div className="grid md:grid-cols-[1fr_300px] gap-4">
              <div className="rounded-2xl overflow-hidden border border-green-900/40 bg-black">
                <video
                  src={session.highlights[activeHighlight]?.clip_url}
                  controls
                  autoPlay
                  className="w-full aspect-video"
                  playsInline
                />
                <div className="px-4 py-3 border-t border-green-900/30">
                  <p className="text-sm font-semibold text-green-300">
                    {session.highlights[activeHighlight]?.label}
                  </p>
                </div>
              </div>

              <div className="flex flex-col gap-2">
                {session.highlights.map((h, i) => (
                  <button
                    key={i}
                    onClick={() => setActiveHighlight(i)}
                    className={cn(
                      "flex items-center gap-3 p-3 rounded-xl border text-left transition-colors",
                      activeHighlight === i
                        ? "border-green-500/60 bg-green-900/30"
                        : "border-green-900/30 bg-green-950/20 hover:border-green-700/40"
                    )}
                  >
                    <div className="bg-green-500/20 rounded-lg p-2 shrink-0">
                      <Trophy size={14} className="text-green-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-green-200">{h.label}</p>
                      <p className="text-xs text-green-600 font-mono">
                        {Math.floor(h.start_sec / 60)}:{String(h.start_sec % 60).padStart(2, "0")}
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {activeTab === "highlights" && (!session.highlights || session.highlights.length === 0) && (
            <div className="text-center py-16 text-green-700">
              <Trophy size={32} className="mx-auto mb-3 opacity-30" />
              <p>No highlights detected in this match.</p>
            </div>
          )}
        </>
      )}

      {/* Raw footage downloads — shown whenever either raw video exists */}
      {(session.urls?.left_raw || session.urls?.right_raw) && (
        <div className="mt-8 border border-green-900/40 rounded-2xl p-5">
          <p className="text-xs font-semibold text-green-600 uppercase tracking-widest mb-4">Raw Footage</p>
          <div className="flex flex-col sm:flex-row gap-3">
            {session.urls?.left_raw && (
              <a
                href={session.urls.left_raw}
                download
                className="flex-1 flex items-center justify-center gap-2 bg-green-950/40 hover:bg-green-900/40 border border-green-800/40 hover:border-green-700/60 text-green-300 text-sm font-medium py-3 px-4 rounded-xl transition-colors"
              >
                <Download size={14} /> Left Camera
              </a>
            )}
            {session.urls?.right_raw && (
              <a
                href={session.urls.right_raw}
                download
                className="flex-1 flex items-center justify-center gap-2 bg-green-950/40 hover:bg-green-900/40 border border-green-800/40 hover:border-green-700/60 text-green-300 text-sm font-medium py-3 px-4 rounded-xl transition-colors"
              >
                <Download size={14} /> Right Camera
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function RecordingQRCodes({ sessionId }: { sessionId: string }) {
  const [origin, setOrigin] = useState("");
  const [leftReady, setLeftReady] = useState(false);
  const [rightReady, setRightReady] = useState(false);

  useEffect(() => {
    setOrigin(window.location.origin);

    const channel = supabase
      .channel(`recording:${sessionId}`)
      .on("broadcast", { event: "state" }, ({ payload }) => {
        if (payload.side === "left" && ["ready","recording","uploading","done"].includes(payload.status)) setLeftReady(true);
        if (payload.side === "right" && ["ready","recording","uploading","done"].includes(payload.status)) setRightReady(true);
      })
      .subscribe();

    return () => { supabase.removeChannel(channel); };
  }, [sessionId]);

  if (!origin) return null;

  const leftUrl  = `${origin}/session/${sessionId}/record?side=left`;
  const rightUrl = `${origin}/session/${sessionId}/record?side=right`;

  return (
    <div className="mt-4 border-t border-green-900/40 pt-4">
      <div className="flex items-center gap-2 mb-4">
        <QrCode size={16} className="text-green-500" />
        <p className="text-sm font-semibold text-green-300">Record from phones</p>
        <span className="text-xs text-green-700 ml-auto">Scan with each phone's camera</span>
      </div>
      <div className="grid grid-cols-2 gap-4">
        {(["left", "right"] as const).map((side) => {
          const url   = side === "left" ? leftUrl : rightUrl;
          const ready = side === "left" ? leftReady : rightReady;
          return (
            <div key={side} className="flex flex-col items-center gap-2">
              <div className={cn(
                "p-3 rounded-xl border transition-colors",
                ready ? "border-green-500/60 bg-green-900/20" : "border-green-900/40 bg-black"
              )}>
                <QRCodeSVG value={url} size={120} bgColor="transparent" fgColor={ready ? "#4ade80" : "#ffffff"} />
              </div>
              <div className="flex items-center gap-1.5">
                <span className={cn("w-2 h-2 rounded-full", ready ? "bg-green-500" : "bg-gray-600")} />
                <span className="text-xs font-medium capitalize text-green-300">{side} Camera</span>
                {ready && <span className="text-xs text-green-500">· Connected</span>}
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-green-800 text-center mt-3">
        Both phones must scan and connect before recording can start
      </p>
    </div>
  );
}

function VideoUploader({ sessionId }: { sessionId: string }) {
  const [uploading, setUploading] = useState<"left" | "right" | null>(null);
  const [done, setDone] = useState<{ left: boolean; right: boolean }>({ left: false, right: false });
  const [progress, setProgress] = useState(0);
  const [expandedQR, setExpandedQR] = useState<"left" | "right" | null>(null);
  const leftRef = useRef<HTMLInputElement>(null);
  const rightRef = useRef<HTMLInputElement>(null);

  const origin = typeof window !== "undefined" ? window.location.origin : "";

  async function upload(side: "left" | "right", file: File) {
    setUploading(side);
    setProgress(0);
    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, camera: side }),
      });
      if (!res.ok) throw new Error("Failed to get upload URL");
      const { url, key } = await res.json();

      await new Promise<void>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.upload.onprogress = (e) => { if (e.lengthComputable) setProgress(Math.round(e.loaded / e.total * 100)); };
        xhr.onload = async () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            await fetch(`/api/sessions/${sessionId}/upload-done`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ side, key }),
            });
            resolve();
          } else reject(new Error(`Upload failed: ${xhr.status} ${xhr.responseText.slice(0, 200)}`));
        };
        xhr.onerror = () => reject(new Error("Network error — check your connection"));
        xhr.open("PUT", url);
        xhr.setRequestHeader("Content-Type", "video/mp4");
        xhr.send(file);
      });

      setDone(d => ({ ...d, [side]: true }));
    } catch (e) {
      alert(`Upload failed: ${e}`);
    } finally {
      setUploading(null);
    }
  }

  return (
    <div className="mt-4 border-t border-green-900/40 pt-4">
      <div className="flex items-center gap-2 mb-3">
        <Upload size={14} className="text-green-500" />
        <p className="text-sm font-semibold text-green-300">Upload existing footage</p>
      </div>
      <p className="text-xs text-green-700 mb-3">Upload from this phone, or scan the QR code with the other phone to upload directly from there.</p>
      <div className="grid grid-cols-2 gap-3">
        {(["left", "right"] as const).map((side) => {
          const isActive = uploading === side;
          const isDone = done[side];
          const uploadUrl = `${origin}/session/${sessionId}/upload?side=${side}`;
          const showQR = expandedQR === side;

          return (
            <div key={side} className="flex flex-col gap-2">
              <input ref={side === "left" ? leftRef : rightRef} type="file" accept="video/*" className="hidden"
                onChange={e => { const f = e.target.files?.[0]; if (f) upload(side, f); }} />

              {/* Upload button */}
              <button
                onClick={() => (side === "left" ? leftRef : rightRef).current?.click()}
                disabled={!!uploading || isDone}
                className={cn(
                  "w-full flex flex-col items-center gap-2 py-4 rounded-xl border text-sm font-medium transition-colors",
                  isDone ? "border-green-500/60 bg-green-900/20 text-green-300" :
                  isActive ? "border-green-700/40 bg-green-900/10 text-green-400" :
                  "border-green-900/40 bg-green-950/20 text-green-500 hover:border-green-700/50"
                )}
              >
                {isDone ? <CheckCircle2 size={18} /> : isActive ? <Loader2 size={18} className="animate-spin" /> : <Upload size={18} />}
                <span className="capitalize">{side} Camera</span>
                {isActive && <span className="text-xs text-green-600">{progress}%</span>}
                {isDone && <span className="text-xs text-green-500">Uploaded ✓</span>}
              </button>

              {/* QR code toggle */}
              {!isDone && (
                <button
                  onClick={() => setExpandedQR(showQR ? null : side)}
                  className="text-xs text-green-700 hover:text-green-500 text-center underline"
                >
                  {showQR ? "Hide QR" : "Upload from other phone →"}
                </button>
              )}
              {showQR && !isDone && (
                <div className="flex flex-col items-center gap-2 p-3 bg-white rounded-xl">
                  <QRCodeSVG value={uploadUrl} size={140} bgColor="#ffffff" fgColor="#000000" />
                  <p className="text-xs text-black/60 text-center">Scan with {side} camera phone</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const configs: Record<string, { label: string; className: string }> = {
    created: { label: "Created", className: "bg-green-900/30 text-green-500 border-green-800/40" },
    uploading: { label: "Uploading", className: "bg-blue-900/30 text-blue-400 border-blue-800/40" },
    processing: { label: "Processing", className: "bg-yellow-900/30 text-yellow-400 border-yellow-800/40" },
    done: { label: "Ready", className: "bg-green-500/20 text-green-300 border-green-500/40" },
    error: { label: "Error", className: "bg-red-900/30 text-red-400 border-red-800/40" },
  };
  const c = configs[status] ?? configs.created;
  return (
    <span className={cn("text-xs font-semibold px-3 py-1 rounded-full border", c.className)}>
      {c.label}
    </span>
  );
}
