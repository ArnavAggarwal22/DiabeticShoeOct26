// app/api/admin/[...path]/route.js
// Authenticated pass-through to the ML service's /admin/* endpoints. The admin token typed on
// the /admin page is forwarded as X-Admin-Token and checked by the ML service.
import { mlFetch, relayJson, serviceUnavailable } from "../../../../lib/mlClient";

async function proxy(request, { params }) {
  const { path } = await params;
  const search = new URL(request.url).search;
  const target = `/admin/${path.map(encodeURIComponent).join("/")}${search}`;
  const token = request.headers.get("x-admin-token") || "";
  const hasBody = !["GET", "HEAD", "DELETE"].includes(request.method);

  try {
    const res = await mlFetch(target, {
      method: request.method,
      headers: { "X-Admin-Token": token, ...(hasBody ? { "Content-Type": "application/json" } : {}) },
      body: hasBody ? await request.text() : undefined,
      timeoutMs: 600000, // training can take a while on large datasets
    });
    const type = res.headers.get("content-type") || "";
    if (res.ok && type.includes("text/csv")) {
      return new Response(await res.text(), {
        status: 200,
        headers: {
          "Content-Type": "text/csv; charset=utf-8",
          "Content-Disposition": res.headers.get("content-disposition") || "attachment",
        },
      });
    }
    return relayJson(res);
  } catch (err) {
    return serviceUnavailable(err);
  }
}

export const GET = proxy;
export const POST = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const dynamic = "force-dynamic";
