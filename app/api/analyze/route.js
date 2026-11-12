// app/api/analyze/route.js
// Thin proxy: the risk assessment is produced by the Python ML service (ml/), not computed here.
import { mlFetch, relayJson, serviceUnavailable } from "../../../lib/mlClient";

const CLINICAL_KEYS = [
  "height_cm", "diabetes_years", "hba1c", "neuropathy", "pad", "deformity", "prior_ulcer", "prior_amputation",
];

export async function POST(request) {
  let payload;
  try {
    payload = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }
  const { age, gender, weight, leftCsvData, rightCsvData, clinical = {} } = payload;
  if (!leftCsvData || !rightCsvData) {
    return Response.json({ error: "Both left and right shoe logs are required." }, { status: 400 });
  }

  const meta = { age, sex: gender, weight_kg: weight };
  for (const k of CLINICAL_KEYS) if (clinical[k] !== undefined && clinical[k] !== "") meta[k] = clinical[k];

  try {
    const res = await mlFetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ meta, left_csv: leftCsvData, right_csv: rightCsvData }),
    });
    const out = await relayJson(res);
    if (!res.ok) return out;
    const data = await out.json();
    return Response.json({ age, gender, weight, ...data });
  } catch (err) {
    return serviceUnavailable(err);
  }
}
