import { type FormEvent, useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiJson, apiUploadCsv } from "../api/client";
import { usePublicConstraints } from "../hooks/usePublicConstraints";
import {
  coerceIntWithNumConstraint,
  intFieldBindingFromNumConstraint,
} from "../lib/publicConstraints";

type CsvRow = { id: string; name: string; created_at: number; is_sample: boolean };

export default function NewKohonenPage() {
  const nav = useNavigate();
  const { constraints, error: cErr, loading: cLoad } = usePublicConstraints();
  const [files, setFiles] = useState<CsvRow[]>([]);
  const [fileId, setFileId] = useState("");
  const [inputSize, setInputSize] = useState(4);
  const [inputSizeStr, setInputSizeStr] = useState("4");
  const [mapSize, setMapSize] = useState(16);
  const [err, setErr] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const loadCsv = useCallback(async () => {
    try {
      const res = await apiJson<{ files: CsvRow[] }>("/csv/");
      setFiles(res.files);
      setFileId((prev) => prev || res.files[0]?.id || "");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void loadCsv();
  }, [loadCsv]);

  useEffect(() => {
    if (!constraints) return;
    const allowed = constraints.KOHONEN_OUTPUT_LAYER_SIZE_RANGE.allowed_values;
    if (!allowed?.length) return;
    setMapSize((prev) => (allowed.includes(prev) ? prev : allowed[0]!));
  }, [constraints]);

  function syncInputSizeFromString(raw: string) {
    if (!constraints) return;
    const next = coerceIntWithNumConstraint(raw, inputSize, constraints.KOHONEN_INPUT_LAYER_SIZE_RANGE);
    setInputSize(next);
    setInputSizeStr(String(next));
  }

  async function onUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setUploading(true);
    setErr(null);
    try {
      const res = await apiUploadCsv(f);
      await loadCsv();
      setFileId(res.id);
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setErr(null);
    if (!constraints) return;
    try {
      const res = await apiJson<{ project: { id: string }; image_id: string }>(
        "/actions/kohonen/init",
        {
          method: "POST",
          json: {
            file_id: fileId,
            input_layer_size: inputSize,
            output_layer_size: mapSize,
          },
        },
      );
      nav(`/projects/${res.project.id}`, { state: { imageId: res.image_id } });
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    }
  }

  const outBinding = constraints
    ? intFieldBindingFromNumConstraint(constraints.KOHONEN_OUTPUT_LAYER_SIZE_RANGE)
    : null;
  const inBinding = constraints
    ? intFieldBindingFromNumConstraint(constraints.KOHONEN_INPUT_LAYER_SIZE_RANGE)
    : null;

  return (
    <div className="card">
      <h1>Новый проект: сеть Кохонена</h1>
      <p>
        <Link to="/projects">← к списку</Link>
      </p>
      {cLoad ? <p>Загрузка ограничений…</p> : null}
      {cErr ? <p className="err">{cErr}</p> : null}
      <form onSubmit={onSubmit}>
        <div className="field">
          <label>CSV (только числовые признаки; число колонок = размерность входа)</label>
          <select value={fileId} onChange={(e) => setFileId(e.target.value)} required>
            {files.map((f) => (
              <option key={f.id} value={f.id}>
                {f.name}
                {f.is_sample ? " (образец)" : ""}
              </option>
            ))}
          </select>
          <input type="file" accept=".csv" disabled={uploading} onChange={(e) => void onUpload(e)} />
        </div>

        <div className="field">
          <label htmlFor="in-size">Размерность входа (число признаков в CSV)</label>
          <input
            id="in-size"
            type="number"
            value={inputSizeStr}
            onChange={(e) => setInputSizeStr(e.target.value)}
            onBlur={() => syncInputSizeFromString(inputSizeStr)}
            min={inBinding?.kind === "range" ? inBinding.min : undefined}
            max={inBinding?.kind === "range" ? inBinding.max : undefined}
            required
          />
          {inBinding?.kind === "range" ? (
            <small>
              Допустимо {inBinding.min}…{inBinding.max} (согласно /public-constraints).
            </small>
          ) : null}
        </div>

        <div className="field">
          <label htmlFor="map-size">Размер карты (число нейронов, полный квадрат)</label>
          {outBinding?.kind === "enum" ? (
            <select
              id="map-size"
              value={mapSize}
              onChange={(e) => setMapSize(Number(e.target.value))}
            >
              {outBinding.values.map((v) => (
                <option key={v} value={v}>
                  {v} ({Math.sqrt(v)}×{Math.sqrt(v)})
                </option>
              ))}
            </select>
          ) : (
            <input
              id="map-size"
              type="number"
              value={mapSize}
              onChange={(e) => setMapSize(Number(e.target.value))}
            />
          )}
        </div>

        {err ? <p className="err">{err}</p> : null}
        <button type="submit" className="btn" disabled={!constraints || !fileId}>
          Создать
        </button>
      </form>
    </div>
  );
}
