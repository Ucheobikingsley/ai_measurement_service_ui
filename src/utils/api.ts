const API_BASE = "  https://7073-102-216-181-52.ngrok-free.app";

export interface MeasurementPayload {
  height: number;
  shoulder: number;
  hip: number;
  sleeve: number;
}

export interface SavedMeasurement extends MeasurementPayload {
  id: string;
  created_at: string;
}

export async function saveMeasurement(
  data: MeasurementPayload
): Promise<SavedMeasurement> {
  const res = await fetch(`${API_BASE}/measurements`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Failed to save: ${detail}`);
  }

  return res.json();
}

export async function getMeasurements(): Promise<SavedMeasurement[]> {
  const res = await fetch(`${API_BASE}/measurements`);

  if (!res.ok) {
    throw new Error("Failed to fetch measurements");
  }

  return res.json();
}
