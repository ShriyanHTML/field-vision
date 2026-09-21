"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { Camera, Upload, ChevronRight, CheckCircle2, Loader2, Smartphone } from "lucide-react";
import { cn } from "@/lib/cn";

type UploadState = "idle" | "uploading" | "done" | "error";

interface VideoUpload {
  file: File | null;
  state: UploadState;
  progress: number;
}

export default function NewSessionPage() {
  const router = useRouter();
  const [teamName, setTeamName] = useState("");
  const [matchDate, setMatchDate] = useState("");
  const [left, setLeft] = useState<VideoUpload>({ file: null, state: "idle", progress: 0 });
  const [right, setRight] = useState<VideoUpload>({ file: null, state: "idle", progress: 0 });
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [step, setStep] = useState<"details" | "upload" | "processing">("details");
  const [error, setError] = useState<string | null>(null);

  const leftRef = useRef<HTMLInputElement>(null);
  const rightRef = useRef<HTMLInputElement>(null);

  async function createSession() {
    setError(null);
    const res = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ team_name: teamName, match_date: matchDate }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      setError(body?.error ? `Failed to create session: ${body.error}` : "Failed to create session");
      return;
    }
    const { id } = await res.json();
    setSessionId(id);
    setStep("upload");
  }

  async function uploadVideo(camera: "left" | "right", file: File) {
    const setter = camera === "left" ? setLeft : setRight;
    setter(s => ({ ...s, state: "uploading", progress: 0 }));

    // Get presigned URL
    const res = await fetch("/api/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, camera }),
    });
    if (!res.ok) { setter(s => ({ ...s, state: "error" })); return; }
    const { url } = await res.json();

    // Upload directly to R2 via XHR so we get progress
    await new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          setter(s => ({ ...s, progress: Math.round((e.loaded / e.total) * 100) }));
        }
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          setter(s => ({ ...s, state: "done", progress: 100 }));
          resolve();
        } else {
          setter(s => ({ ...s, state: "error" }));
          reject();
        }
      };
      xhr.onerror = () => { setter(s => ({ ...s, state: "error" })); reject(); };
      xhr.open("PUT", url);
      xhr.setRequestHeader("Content-Type", "video/mp4");
      xhr.send(file);
    });
  }

  function handleFileChange(camera: "left" | "right", file: File | null) {
    if (!file) return;
    if (camera === "left") setLeft({ file, state: "idle", progress: 0 });
    else setRight({ file, state: "idle", progress: 0 });
  }

  async function startProcessing() {
    if (!sessionId) return;
    setStep("processing");
    const res = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!res.ok) {
      setError("Failed to start processing");
      setStep("upload");
      return;
    }
    router.push(`/session/${sessionId}`);
  }

  const bothUploaded = left.state === "done" && right.state === "done";

  async function createAndRecord(side: "left" | "right") {
    setError(null);
    const res = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ team_name: teamName, match_date: matchDate }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      setError(body?.error ? `Failed to create session: ${body.error}` : "Failed to create session");
      return;
    }
    const { id } = await res.json();
    router.push(`/session/${id}/record?side=${side}&host=1`);
  }

  return (
    <div className="min-h-screen px-4 py-12 flex flex-col items-center">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="mb-8">
          <a href="/" className="text-green-500 text-sm font-medium hover:text-green-400 mb-4 inline-block">&larr; FieldVision</a>
          <h1 className="text-3xl font-bold">New Match Session</h1>
          <p className="text-green-200/50 text-sm mt-1">Record live from two phones or upload existing footage.</p>
        </div>

        {/* Step indicator */}
        <div className="flex items-center gap-2 mb-8">
          {["details", "upload", "processing"].map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <div className={cn(
                "w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold transition-colors",
                step === s ? "bg-green-500 text-black" :
                  (["details", "upload", "processing"].indexOf(step) > i ? "bg-green-900 text-green-400" : "bg-green-900/30 text-green-700")
              )}>
                {i + 1}
              </div>
              <span className={cn("text-xs capitalize", step === s ? "text-green-300" : "text-green-700")}>
                {s}
              </span>
              {i < 2 && <div className="w-8 h-px bg-green-900/50" />}
            </div>
          ))}
        </div>

        {/* Step 1: Details */}
        {step === "details" && (
          <div className="flex flex-col gap-4">
            <div>
              <label className="block text-sm font-medium text-green-300 mb-1">Team / Match name</label>
              <input
                value={teamName}
                onChange={e => setTeamName(e.target.value)}
                placeholder="e.g. FC Galaxy vs United SC"
                className="w-full bg-green-950/30 border border-green-800/50 rounded-lg px-4 py-2.5 text-sm text-green-100 placeholder:text-green-700 focus:outline-none focus:border-green-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-green-300 mb-1">Match date</label>
              <input
                type="date"
                value={matchDate}
                onChange={e => setMatchDate(e.target.value)}
                className="w-full bg-green-950/30 border border-green-800/50 rounded-lg px-4 py-2.5 text-sm text-green-100 focus:outline-none focus:border-green-500"
              />
            </div>
            {error && <p className="text-red-400 text-sm">{error}</p>}

            {/* Record from phones */}
            <div className="border border-green-800/50 rounded-xl p-4 mt-2">
              <div className="flex items-center gap-2 mb-3">
                <Smartphone size={16} className="text-green-400" />
                <p className="text-sm font-semibold text-green-300">Record from this phone</p>
              </div>
              <p className="text-xs text-green-700 mb-4">You'll record one half. A QR code will appear for the second phone to scan and record the other half.</p>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => createAndRecord("left")}
                  disabled={!teamName}
                  className="flex flex-col items-center gap-2 bg-green-950/50 border border-green-800/50 hover:border-green-500/60 disabled:opacity-40 rounded-xl p-4 transition-colors"
                >
                  <span className="text-2xl">⬅️</span>
                  <span className="text-sm font-semibold text-green-200">I'm Left Camera</span>
                  <span className="text-xs text-green-700">Covers left half</span>
                </button>
                <button
                  onClick={() => createAndRecord("right")}
                  disabled={!teamName}
                  className="flex flex-col items-center gap-2 bg-green-950/50 border border-green-800/50 hover:border-green-500/60 disabled:opacity-40 rounded-xl p-4 transition-colors"
                >
                  <span className="text-2xl">➡️</span>
                  <span className="text-sm font-semibold text-green-200">I'm Right Camera</span>
                  <span className="text-xs text-green-700">Covers right half</span>
                </button>
              </div>
            </div>

            {/* Divider */}
            <div className="flex items-center gap-3 my-1">
              <div className="flex-1 h-px bg-green-900/40" />
              <span className="text-xs text-green-800">or upload existing footage</span>
              <div className="flex-1 h-px bg-green-900/40" />
            </div>

            <button
              onClick={createSession}
              className="flex items-center justify-center gap-2 bg-green-950/50 border border-green-800/50 hover:border-green-500/50 text-green-300 font-semibold px-6 py-3 rounded-xl transition-colors"
            >
              <Upload size={16} /> Upload video files <ChevronRight size={16} />
            </button>
          </div>
        )}

        {/* Step 2: Upload */}
        {step === "upload" && (
          <div className="flex flex-col gap-6">
            <p className="text-sm text-green-200/60">Upload the left-half footage and right-half footage. Start both recordings at the same time for best sync.</p>

            {(["left", "right"] as const).map((camera) => {
              const state = camera === "left" ? left : right;
              const inputRef = camera === "left" ? leftRef : rightRef;
              const color = camera === "left" ? "green" : "blue";

              return (
                <div
                  key={camera}
                  className={cn(
                    "border rounded-xl p-5 flex flex-col gap-3 transition-colors cursor-pointer",
                    state.state === "done"
                      ? "border-green-500/60 bg-green-900/20"
                      : `border-${color}-800/40 bg-${color}-950/20 hover:border-${color}-600/40`
                  )}
                  onClick={() => !state.file && inputRef.current?.click()}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`bg-${color}-500/20 border border-${color}-500/30 rounded-lg p-2`}>
                        <Camera size={16} className={`text-${color}-400`} />
                      </div>
                      <div>
                        <p className="font-semibold text-sm capitalize">{camera} Camera</p>
                        <p className={`text-xs text-${color}-300/50`}>{camera === "left" ? "Left half of pitch" : "Right half of pitch"}</p>
                      </div>
                    </div>
                    {state.state === "done" && <CheckCircle2 size={18} className="text-green-400" />}
                    {state.state === "uploading" && <Loader2 size={18} className="text-green-400 animate-spin" />}
                  </div>

                  {state.file && state.state !== "done" && (
                    <>
                      <p className="text-xs text-green-300/50 truncate">{state.file.name}</p>
                      {state.state === "idle" && (
                        <button
                          onClick={(e) => { e.stopPropagation(); uploadVideo(camera, state.file!); }}
                          className="flex items-center gap-2 bg-green-500/20 hover:bg-green-500/30 border border-green-500/30 text-green-300 text-xs font-semibold px-4 py-2 rounded-lg w-fit transition-colors"
                        >
                          <Upload size={12} /> Upload
                        </button>
                      )}
                      {state.state === "uploading" && (
                        <div className="w-full bg-green-900/30 rounded-full h-1.5">
                          <div
                            className="bg-green-500 h-1.5 rounded-full transition-all"
                            style={{ width: `${state.progress}%` }}
                          />
                        </div>
                      )}
                    </>
                  )}

                  {!state.file && (
                    <p className="text-xs text-green-600">Click to select video file (MP4)</p>
                  )}

                  <input
                    ref={inputRef}
                    type="file"
                    accept="video/mp4,video/*"
                    className="hidden"
                    onChange={e => handleFileChange(camera, e.target.files?.[0] ?? null)}
                  />
                </div>
              );
            })}

            {error && <p className="text-red-400 text-sm">{error}</p>}

            <button
              disabled={!bothUploaded}
              onClick={startProcessing}
              className="flex items-center justify-center gap-2 bg-green-500 hover:bg-green-400 disabled:opacity-40 disabled:cursor-not-allowed text-black font-bold px-6 py-3 rounded-xl transition-colors"
            >
              Start AI Processing <ChevronRight size={16} />
            </button>
          </div>
        )}

        {/* Step 3: Redirecting */}
        {step === "processing" && (
          <div className="flex flex-col items-center gap-4 py-12">
            <Loader2 size={32} className="text-green-400 animate-spin" />
            <p className="text-green-300">Kicking off your processing job...</p>
          </div>
        )}
      </div>
    </div>
  );
}
