export async function triggerProcessingJob(
  sessionId: string,
  leftKey: string,
  rightKey: string,
  leftUrl?: string,
  rightUrl?: string,
): Promise<{ jobId: string }> {
  const webhookUrl = (process.env.MODAL_WEBHOOK_URL ?? "").trim();
  const res = await fetch(webhookUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      left_key: leftKey,
      right_key: rightKey,
      left_url: leftUrl,
      right_url: rightUrl,
      _auth: process.env.MODAL_AUTH_TOKEN,
    }),
  });

  const text = await res.text();
  if (!res.ok) {
    throw new Error(`Modal trigger failed: ${res.status} ${text.slice(0, 300)}`);
  }

  try {
    return text ? JSON.parse(text) : { jobId: "modal-ok" };
  } catch {
    throw new Error(`Modal returned non-JSON: ${text.slice(0, 300)}`);
  }
}
