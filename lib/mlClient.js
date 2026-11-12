// lib/mlClient.js — server-side helper for talking to the Python ML service.
const ML_API_URL = (process.env.ML_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export async function mlFetch(path, { method = "GET", body, headers = {}, timeoutMs = 120000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`${ML_API_URL}${path}`, {
      method,
      body,
      headers,
      signal: controller.signal,
      cache: "no-store",
    });
  } finally {
    clearTimeout(timer);
  }
}

export function serviceUnavailable(err) {
  const reason = err?.name === "AbortError" ? "timed out" : "is not reachable";
  return Response.json(
    { error: `ML service ${reason} at ${ML_API_URL}. Start it with "npm run ml" (see README).` },
    { status: 503 }
  );
}

// FastAPI returns {detail: "..."} (or a validation array); the UI expects {error: "..."}.
export async function relayJson(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    let msg = data?.detail ?? data?.error ?? `ML service error (${res.status})`;
    if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
    return Response.json({ error: String(msg) }, { status: res.status });
  }
  return Response.json(data, { status: res.status });
}
