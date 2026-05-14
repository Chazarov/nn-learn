import { type FormEvent, useCallback, useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { apiJson } from "../api/client";
import { AuthImage } from "../components/AuthImage";
import { ProjectTypeIcon } from "../components/ProjectTypeIcon";
import { usePublicConstraints } from "../hooks/usePublicConstraints";
import {
  coerceFloatWithFloatConstraint,
  coerceIntWithNumConstraint,
  type PublicConstraints,
} from "../lib/publicConstraints";
import type { ProjectWithData } from "../types";

const ACTIVATIONS = ["RELLU", "SIGMOID", "SOFTMAX"] as const;
const LOSSES = ["MSE", "CROSS_ENTROPY"] as const;
const NEIGH = ["GAUSSIAN", "MEXICAN_HAT"] as const;
const TOPO = ["EUCLIDEAN", "MANHATTAN"] as const;

export default function ProjectDetailPage() {
  const { id } = useParams<{ id: string }>();
  const loc = useLocation();
  const { constraints, error: cErr } = usePublicConstraints();
  const [project, setProject] = useState<ProjectWithData | null>(null);
  const [imageId, setImageId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const s = loc.state as { imageId?: string } | null;
    setImageId(s?.imageId ?? null);
  }, [id, loc.state]);

  const load = useCallback(async () => {
    if (!id) return;
    setErr(null);
    try {
      const res = await apiJson<{ project: ProjectWithData }>(`/actions/project/${id}`);
      setProject(res.project);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, [id]);

  useEffect(() => {
    void load();
  }, [load]);

  if (!id) return <p>Нет id</p>;

  if (err && !project) return <p className="err">{err}</p>;
  if (!project) return <p>Загрузка…</p>;

  const title =
    project.project_type === "PERCEPTRON"
      ? "Перцептрон"
      : "Сеть Кохонена";

  return (
    <div className="card" style={{ maxWidth: 960 }}>
      <div className="row" style={{ marginBottom: "1rem" }}>
        <ProjectTypeIcon type={project.project_type} />
        <h1 style={{ margin: 0 }}>
          {title} · {project.id.slice(0, 8)}…
        </h1>
      </div>
      <p>
        <Link to="/projects">← все проекты</Link>
      </p>
      {cErr ? <p className="err">Ограничения: {cErr}</p> : null}
      <p>
        Вход: {project.nn_data.input_size}, классов: {project.nn_data.classes.length}{" "}
        ({project.nn_data.classes.join(", ")})
      </p>

      <AuthImage imageId={imageId} />

      {project.project_type === "PERCEPTRON" && constraints ? (
        <PerceptronPanel
          projectId={project.id}
          constraints={constraints}
          onTrained={(img) => {
            setImageId(img);
            void load();
          }}
        />
      ) : null}

      {project.project_type === "KOHONEN" && constraints ? (
        <KohonenPanel
          projectId={project.id}
          constraints={constraints}
          inputSize={project.nn_data.input_size}
          onTrained={(img) => {
            setImageId(img);
            void load();
          }}
        />
      ) : null}

      {err ? <p className="err">{err}</p> : null}
    </div>
  );
}

function PerceptronPanel({
  projectId,
  constraints,
  onTrained,
}: {
  projectId: string;
  constraints: PublicConstraints;
  onTrained: (imageId: string) => void;
}) {
  const [activation, setActivation] = useState<string>("SIGMOID");
  const [softmax, setSoftmax] = useState(false);
  const [loss, setLoss] = useState<string>("MSE");
  const [epochs, setEpochs] = useState(100);
  const [epochsStr, setEpochsStr] = useState("100");
  const [lr, setLr] = useState(0.05);
  const [lrStr, setLrStr] = useState("0.05");
  const [vec, setVec] = useState("");
  const [pred, setPred] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  function syncEpochs() {
    const n = coerceIntWithNumConstraint(
      epochsStr,
      epochs,
      constraints.PERCEPTRON_LEARN_EPOCHS_RANGE,
    );
    setEpochs(n);
    setEpochsStr(String(n));
  }

  function syncLr() {
    const n = coerceFloatWithFloatConstraint(
      lrStr,
      lr,
      constraints.PERCEPTRON_LEARN_LEARNING_RATE_RANGE,
    );
    setLr(n);
    setLrStr(String(n));
  }

  async function learn(e: FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    syncEpochs();
    syncLr();
    try {
      const res = await apiJson<{ image_id: string }>("/actions/perceptron/learn", {
        method: "POST",
        json: {
          project_id: projectId,
          activation_type: activation,
          softmax_use: softmax,
          loss_type: loss,
          epochs: coerceIntWithNumConstraint(
            epochsStr,
            epochs,
            constraints.PERCEPTRON_LEARN_EPOCHS_RANGE,
          ),
          learning_rate: coerceFloatWithFloatConstraint(
            lrStr,
            lr,
            constraints.PERCEPTRON_LEARN_LEARNING_RATE_RANGE,
          ),
        },
      });
      onTrained(res.image_id);
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  }

  async function predict(e: FormEvent) {
    e.preventDefault();
    setErr(null);
    const parts = vec.split(/[,;\s]+/).map((s) => s.trim()).filter(Boolean);
    const nums = parts.map((p) => Number.parseFloat(p.replace(",", ".")));
    if (nums.some((x) => !Number.isFinite(x))) {
      setErr("Вектор должен содержать только числа");
      return;
    }
    if (nums.length > constraints.PERCEPTRON_GET_ANSWER_INPUT_VECTOR_MAX_LEN) {
      setErr(`Не более ${constraints.PERCEPTRON_GET_ANSWER_INPUT_VECTOR_MAX_LEN} элементов`);
      return;
    }
    setBusy(true);
    try {
      const res = await apiJson<{ predicted: string }>("/actions/perceptron/get_answer", {
        method: "POST",
        json: {
          perceptron_id: projectId,
          input_vector: nums,
          activation_type: activation,
          softmax_use: softmax,
        },
      });
      setPred(res.predicted);
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <h2>Обучение</h2>
      <form onSubmit={learn}>
        <div className="field">
          <label>Активация</label>
          <select value={activation} onChange={(e) => setActivation(e.target.value)}>
            {ACTIVATIONS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>
            <input
              type="checkbox"
              checked={softmax}
              onChange={(e) => setSoftmax(e.target.checked)}
            />{" "}
            Softmax на выходе
          </label>
        </div>
        <div className="field">
          <label>Функция потерь</label>
          <select value={loss} onChange={(e) => setLoss(e.target.value)}>
            {LOSSES.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Эпохи</label>
          <input
            type="number"
            value={epochsStr}
            onChange={(e) => setEpochsStr(e.target.value)}
            onBlur={syncEpochs}
          />
        </div>
        <div className="field">
          <label>Learning rate</label>
          <input
            type="text"
            inputMode="decimal"
            value={lrStr}
            onChange={(e) => setLrStr(e.target.value)}
            onBlur={syncLr}
          />
        </div>
        {err ? <p className="err">{err}</p> : null}
        <button type="submit" className="btn" disabled={busy}>
          Обучить
        </button>
      </form>

      <h2>Классификация</h2>
      <form onSubmit={predict}>
        <div className="field">
          <label>Входной вектор (числа через запятую)</label>
          <textarea value={vec} onChange={(e) => setVec(e.target.value)} rows={2} />
        </div>
        <button type="submit" className="btn secondary" disabled={busy}>
          Ответ
        </button>
        {pred ? <p>Класс: {pred}</p> : null}
      </form>
    </>
  );
}

function KohonenPanel({
  projectId,
  constraints,
  inputSize,
  onTrained,
}: {
  projectId: string;
  constraints: PublicConstraints;
  inputSize: number;
  onTrained: (imageId: string) => void;
}) {
  const [epochs, setEpochs] = useState(200);
  const [epochsStr, setEpochsStr] = useState("200");
  const [lr, setLr] = useState(0.1);
  const [lrStr, setLrStr] = useState("0.1");
  const [sigma, setSigma] = useState(2.0);
  const [sigmaStr, setSigmaStr] = useState("2");
  const [nf, setNf] = useState<string>("GAUSSIAN");
  const [topo, setTopo] = useState<string>("EUCLIDEAN");
  const [vec, setVec] = useState("");
  const [ans, setAns] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  function syncEpochs() {
    const n = coerceIntWithNumConstraint(epochsStr, epochs, constraints.KOHONEN_LEARN_EPOCHS_RANGE);
    setEpochs(n);
    setEpochsStr(String(n));
  }
  function syncLr() {
    const n = coerceFloatWithFloatConstraint(
      lrStr,
      lr,
      constraints.KOHONEN_LEARN_LEARNING_RATE_RANGE,
    );
    setLr(n);
    setLrStr(String(n));
  }
  function syncSigma() {
    const n = coerceFloatWithFloatConstraint(
      sigmaStr,
      sigma,
      constraints.KOHONEN_INITIAL_NEIGHBORHOOD_RADIUS_RANGE,
    );
    setSigma(n);
    setSigmaStr(String(n));
  }

  async function learn(e: FormEvent) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    syncEpochs();
    syncLr();
    syncSigma();
    try {
      const res = await apiJson<{ image_id: string }>("/actions/kohonen/learn", {
        method: "POST",
        json: {
          project_id: projectId,
          epochs: coerceIntWithNumConstraint(
            epochsStr,
            epochs,
            constraints.KOHONEN_LEARN_EPOCHS_RANGE,
          ),
          learning_rate: coerceFloatWithFloatConstraint(
            lrStr,
            lr,
            constraints.KOHONEN_LEARN_LEARNING_RATE_RANGE,
          ),
          initial_neighborhood_radius: coerceFloatWithFloatConstraint(
            sigmaStr,
            sigma,
            constraints.KOHONEN_INITIAL_NEIGHBORHOOD_RADIUS_RANGE,
          ),
          neighbourhood_function: nf,
          topology_distance: topo,
        },
      });
      onTrained(res.image_id);
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  }

  async function winner(e: FormEvent) {
    e.preventDefault();
    setErr(null);
    const parts = vec.split(/[,;\s]+/).map((s) => s.trim()).filter(Boolean);
    const nums = parts.map((p) => Number.parseFloat(p.replace(",", ".")));
    if (nums.length !== inputSize) {
      setErr(`Ожидается ${inputSize} чисел`);
      return;
    }
    if (nums.some((x) => !Number.isFinite(x))) {
      setErr("Только конечные числа");
      return;
    }
    if (nums.length > constraints.KOHONEN_GET_ANSWER_INPUT_VECTOR_MAX_LEN) {
      setErr("Слишком длинный вектор");
      return;
    }
    setBusy(true);
    try {
      const res = await apiJson<{ winner_neuron_index: number; map_rows: number; map_cols: number }>(
        "/actions/kohonen/get_answer",
        {
          method: "POST",
          json: { project_id: projectId, input_vector: nums },
        },
      );
      setAns(
        `Победитель: нейрон #${res.winner_neuron_index} на сетке ${res.map_rows}×${res.map_cols}`,
      );
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <h2>Обучение SOM</h2>
      <form onSubmit={learn}>
        <div className="field">
          <label>Эпохи</label>
          <input
            type="number"
            value={epochsStr}
            onChange={(e) => setEpochsStr(e.target.value)}
            onBlur={syncEpochs}
          />
        </div>
        <div className="field">
          <label>Learning rate</label>
          <input value={lrStr} onChange={(e) => setLrStr(e.target.value)} onBlur={syncLr} />
        </div>
        <div className="field">
          <label>Начальный радиус соседства σ</label>
          <input value={sigmaStr} onChange={(e) => setSigmaStr(e.target.value)} onBlur={syncSigma} />
        </div>
        <div className="field">
          <label>Функция соседства</label>
          <select value={nf} onChange={(e) => setNf(e.target.value)}>
            {NEIGH.map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Топологическое расстояние</label>
          <select value={topo} onChange={(e) => setTopo(e.target.value)}>
            {TOPO.map((x) => (
              <option key={x} value={x}>
                {x}
              </option>
            ))}
          </select>
        </div>
        {err ? <p className="err">{err}</p> : null}
        <button type="submit" className="btn" disabled={busy}>
          Обучить
        </button>
      </form>

      <h2>Позиция на карте</h2>
      <form onSubmit={winner}>
        <div className="field">
          <label>Вектор ({inputSize} числа через запятую)</label>
          <textarea value={vec} onChange={(e) => setVec(e.target.value)} rows={2} />
        </div>
        <button type="submit" className="btn secondary" disabled={busy}>
          Найти нейрон-победитель
        </button>
        {ans ? <p>{ans}</p> : null}
      </form>
    </>
  );
}
