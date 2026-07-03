import { NextRequest, NextResponse } from "next/server";
import { supabaseAdmin } from "@/lib/supabase";

export async function POST(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const body = await req.json();

  const allowed = ["status", "progress", "error_message", "stitched_video_key",
                   "tracked_video_key", "highlights"];
  const update: Record<string, unknown> = {};
  for (const key of allowed) {
    if (key in body) update[key] = body[key];
  }

  if (Object.keys(update).length === 0) {
    return NextResponse.json({ error: "Nothing to update" }, { status: 400 });
  }

  // Never overwrite a completed session with a processing/error status
  if (update.status && update.status !== "done") {
    const { data } = await supabaseAdmin
      .from("sessions").select("status").eq("id", id).single();
    if (data?.status === "done") {
      return NextResponse.json({ ok: true, skipped: true });
    }
  }

  const { error } = await supabaseAdmin.from("sessions").update(update).eq("id", id);
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ ok: true });
}
