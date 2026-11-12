// app/api/demo/[scenario]/route.js — demo recordings (healthy | moderate | critical) from the ML service.
import { mlFetch, relayJson, serviceUnavailable } from "../../../../lib/mlClient";

export async function GET(_request, { params }) {
  const { scenario } = await params;
  try {
    return relayJson(await mlFetch(`/demo/${encodeURIComponent(scenario)}`));
  } catch (err) {
    return serviceUnavailable(err);
  }
}

export const dynamic = "force-dynamic";
