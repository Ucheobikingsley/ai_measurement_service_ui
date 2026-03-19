import { useState, useCallback } from "react";
import Camera from "./components/Camera";
import { saveMeasurement, type SavedMeasurement } from "./utils/api";
import type { BodyMeasurements } from "./utils/measurement";
import "./App.css";

function App() {
  const [heightCm, setHeightCm] = useState("");
  const [scanning, setScanning] = useState(false);
  const [measurements, setMeasurements] = useState<BodyMeasurements | null>(null);
  const [status, setStatus] = useState("Enter your height and press Start Scan");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<SavedMeasurement | null>(null);
  const [error, setError] = useState("");

  const heightNum = parseFloat(heightCm);
  const heightValid = !isNaN(heightNum) && heightNum > 0;

  const handleStart = () => {
    if (!heightValid) {
      setError("Please enter a valid height greater than 0");
      return;
    }

    setError("");
    setSaved(null);
    setMeasurements(null);
    setScanning(true);
  };

  const handleStop = () => {
    setScanning(false);
    setStatus("Scan stopped");
  };

  const handleMeasurement = useCallback((m: BodyMeasurements) => {
    setMeasurements(m);
  }, []);

  const handleStatusChange = useCallback((s: string) => {
    setStatus(s);
  }, []);

  const handleSave = async () => {
    if (!measurements || !heightValid) {
      return;
    }

    setSaving(true);
    setError("");

    try {
      const result = await saveMeasurement({
        height: heightNum,
        shoulder: measurements.shoulder,
        hip: measurements.hip,
        sleeve: measurements.sleeve,
      });

      setSaved(result);
      setScanning(false);
      setStatus("Measurements saved successfully!");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Remote Tailor</h1>
        <p className="subtitle">AI-Powered Body Measurement System</p>
      </header>

      <main className="app-main">
        <div className="camera-panel">
          <Camera
            userHeightCm={heightValid ? heightNum : 170}
            scanning={scanning}
            onMeasurement={handleMeasurement}
            onStatusChange={handleStatusChange}
          />

          <div className={`status-bar ${scanning ? "active" : ""}`}>
            {status}
          </div>
        </div>

        <div className="controls-panel">
          <div className="card">
            <h2>Setup</h2>

            <label className="input-label" htmlFor="height-input">
              Your Height (cm)
            </label>
            <input
              id="height-input"
              type="number"
              min="1"
              placeholder="e.g. 175"
              value={heightCm}
              onChange={(e) => setHeightCm(e.target.value)}
              disabled={scanning}
              className="input-field"
            />

            <div className="button-row">
              {!scanning ? (
                <button
                  className="btn btn-primary"
                  onClick={handleStart}
                  disabled={!heightValid}
                >
                  Start Scan
                </button>
              ) : (
                <button className="btn btn-secondary" onClick={handleStop}>
                  Stop Scan
                </button>
              )}
            </div>
          </div>

          {measurements && (
            <div className="card measurements-card">
              <h2>Measurements</h2>

              <div className="measurement-grid">
                <div className="measurement-item">
                  <span className="measurement-label">Shoulder Width</span>
                  <span className="measurement-value">
                    {measurements.shoulder} cm
                  </span>
                </div>

                <div className="measurement-item">
                  <span className="measurement-label">Hip Width</span>
                  <span className="measurement-value">
                    {measurements.hip} cm
                  </span>
                </div>

                <div className="measurement-item">
                  <span className="measurement-label">Sleeve Length</span>
                  <span className="measurement-value">
                    {measurements.sleeve} cm
                  </span>
                </div>
              </div>

              <button
                className="btn btn-save"
                onClick={handleSave}
                disabled={saving}
              >
                {saving ? "Saving..." : "Save Measurements"}
              </button>
            </div>
          )}

          {saved && (
            <div className="card saved-card">
              <h2>Saved</h2>
              <p>
                ID: <code>{saved.id}</code>
              </p>
              <p>
                Saved at:{" "}
                {new Date(saved.created_at).toLocaleString()}
              </p>
            </div>
          )}

          {error && <div className="error-banner">{error}</div>}
        </div>
      </main>
    </div>
  );
}

export default App;
