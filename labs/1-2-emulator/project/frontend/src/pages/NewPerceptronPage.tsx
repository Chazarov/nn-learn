import { type FormEvent, useCallback, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiJson, apiUploadCsv } from "../api/client";
import { usePublicConstraints } from "../hooks/usePublicConstraints";
import {
  intFieldBindingFromNumConstraint,
  parseHiddenLayersFromText,
} from "../lib/publicConstraints";

type CsvRow = { id: string; name: string; created_at: number; is_sample: boolean };

export default function NewPerceptronPage() {
  const nav = useNavigate();
  const { constraints, error: cErr, loading: cLoad } = usePublicConstraints();
  const [files, setFiles] = useState<CsvRow[]>([]);
  const [fileId, setFileId] = useState("");
  const [hiddenText, setHiddenText] = useState("8");
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
    if (!constraints) {
      setErr("Ограничения ещё не загружены");
      return;
    }
    const parsed = parseHiddenLayersFromText(hiddenText, constraints.PERCEPTRON_HIDDEN_LAYERS);
    if (!parsed.ok) {
      setErr(parsed.error);
      return;
    }
    try {
      const res = await apiJson<{ project: { id: string }; image_id: string }>(
        "/actions/perceptron/init",
        {
          method: "POST",
          json: { file_id: fileId, hidden_layers_architecture: parsed.layers },
        },
      );
      nav(`/projects/${res.project.id}`, { state: { imageId: res.image_id } });
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    }
  }

  const nc = constraints?.PERCEPTRON_HIDDEN_LAYERS.neurons_per_hidden_layer;
  const binding = nc ? intFieldBindingFromNumConstraint(nc) : null;

  return (
    <div className="card">
      <h1>Новый проект: перцептрон</h1>
      <p>
        <Link to="/projects">← к списку</Link>
      </p>
      {cLoad ? <p>Загрузка ограничений…</p> : null}
      {cErr ? <p className="err">{cErr}</p> : null}
      <form onSubmit={onSubmit}>
        <div className="field">
          <label>CSV для обучения</label>
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
          <label>Скрытые слои (через запятую)</label>
          <input
            value={hiddenText}
            onChange={(e) => setHiddenText(e.target.value)}
            placeholder="например: 16, 8"
            required
          />
          {binding?.kind === "range" ? (
            <small>
              Каждый слой: целое от {binding.min} до {binding.max}, не более{" "}
              {constraints?.PERCEPTRON_HIDDEN_LAYERS.max_hidden_layers} слоёв.
            </small>
          ) : binding?.kind === "enum" ? (
            <small>Допустимые размеры слоя: {binding.values.join(", ")}</small>
          ) : null}
        </div>
        {err ? <p className="err">{err}</p> : null}
        <button type="submit" className="btn" disabled={!constraints || !fileId}>
          Создать
        </button>
      </form>

      {constraints ? (
        <section style={{ marginTop: "2rem" }}>
          <h2>Параметры обучения (подсказка)</h2>
          <p>
            Эпохи и скорость обучения на странице проекта ограничиваются так же, как на
            сервере: эпохи {constraints.PERCEPTRON_LEARN_EPOCHS_RANGE.min_value}…
            {constraints.PERCEPTRON_LEARN_EPOCHS_RANGE.max_value}, learning rate в диапазоне
            от {constraints.PERCEPTRON_LEARN_LEARNING_RATE_RANGE.min_value} до{" "}
            {constraints.PERCEPTRON_LEARN_LEARNING_RATE_RANGE.max_value}.
          </p>
        </section>
      ) : null}
    </div>
  );
}