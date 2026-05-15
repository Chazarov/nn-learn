import { useState, useEffect, useRef, useCallback } from "react";
import * as api from "../api";
import { learnPerceptronWS } from "../api";
import {
  clampIntByNumConstraint,
  clampFloatByFloatConstraint,
  describeNumConstraint,
  numberInputPropsFromNumConstraint,
  numberInputPropsFromFloatConstraint,
  hiddenLayersConstraintBundle,
} from "../lib/publicConstraints";

const HELP_URL =
  "https://github.com/Chazarov/nn-learn/tree/master/labs/1-development-of-a-multilayer-perceptron-neuroemulator/resolve";

function ProjectTypeGlyph({ projectType }) {
  const isKoh = projectType === "KOHONEN";
  return (
    <span
      className={`project-type-glyph ${isKoh ? "kohonen" : "perceptron"}`}
      title={isKoh ? "Сеть Кохонена (SOM)" : "Многослойный перцептрон"}
      aria-hidden
    >
      {isKoh ? (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <path d="M2 2h3v3H2V2zm4.5 0h3v3h-3V2zm4.5 0h3v3h-3V2zM2 6.5h3v3H2v-3zm4.5 0h3v3h-3v-3zm4.5 0h3v3h-3v-3zM2 11h3v3H2v-3zm4.5 0h3v3h-3v-3zm4.5 0h3v3h-3v-3z" />
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2">
          <circle cx="3" cy="4" r="1.4" fill="currentColor" stroke="none" />
          <circle cx="3" cy="12" r="1.4" fill="currentColor" stroke="none" />
          <circle cx="13" cy="8" r="1.4" fill="currentColor" stroke="none" />
          <line x1="4.2" y1="4.3" x2="11.5" y2="7.2" />
          <line x1="4.2" y1="11.7" x2="11.5" y2="8.8" />
        </svg>
      )}
    </span>
  );
}

export default function MainPage({ token, onLogout }) {
  const [projects, setProjects] = useState([]);
  const [csvFiles, setCsvFiles] = useState([]);
  const [selectedProjectId, setSelectedProjectId] = useState(null);
  const [projectData, setProjectData] = useState(null);

  const [constraints, setConstraints] = useState(null);

  /** null | 'pick' | 'PERCEPTRON' | 'KOHONEN' */
  const [createFlow, setCreateFlow] = useState(null);

  const [hiddenLayers, setHiddenLayers] = useState([4]);
  const [selectedCsvId, setSelectedCsvId] = useState("");
  const [activationType, setActivationType] = useState("RELLU");
  const [softmaxUse, setSoftmaxUse] = useState(false);
  const [lossType, setLossType] = useState("MSE");

  const [epochs, setEpochs] = useState(100);
  const [learningRate, setLearningRate] = useState(0.1);

  const [kohCsvId, setKohCsvId] = useState("");
  const [kohInputLayer, setKohInputLayer] = useState(2);
  const [kohOutputLayer, setKohOutputLayer] = useState(9);
  const [kohEpochs, setKohEpochs] = useState(100);
  const [kohLR, setKohLR] = useState(0.1);
  const [kohSigma, setKohSigma] = useState(1.0);
  const [kohNeighbour, setKohNeighbour] = useState("GAUSSIAN");
  const [kohTopology, setKohTopology] = useState("EUCLIDEAN");

  const [trainingProgress, setTrainingProgress] = useState(null);

  const [imageUrl, setImageUrl] = useState(null);
  const imageUrlRef = useRef(null);
  const setImageUrlSafe = useCallback((url) => {
    if (imageUrlRef.current) {
      URL.revokeObjectURL(imageUrlRef.current);
      imageUrlRef.current = null;
    }
    imageUrlRef.current = url;
    setImageUrl(url);
  }, []);

  const [inputSize, setInputSize] = useState(0);
  const [inputVector, setInputVector] = useState([]);
  const [answerResult, setAnswerResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isNarrow, setIsNarrow] = useState(false);

  const csvInputRef = useRef(null);

  useEffect(() => {
    return () => {
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
    };
  }, []);

  useEffect(() => {
    const check = () => setIsNarrow(window.innerWidth < 700);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const c = await api.getPublicConstraints();
        if (!cancelled) setConstraints(c);
      } catch {
        if (!cancelled) setConstraints(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const hlBundle = hiddenLayersConstraintBundle(constraints || {});
  const perceptronEpochsDesc = describeNumConstraint(
    constraints?.PERCEPTRON_LEARN_EPOCHS_RANGE ?? { min_value: 1, max_value: 100_000 },
  );
  const perceptronLRProps = numberInputPropsFromFloatConstraint(
    constraints?.PERCEPTRON_LEARN_LEARNING_RATE_RANGE ?? { min_value: 1e-8, max_value: 1 },
  );
  const kohEpochsDesc = describeNumConstraint(
    constraints?.KOHONEN_LEARN_EPOCHS_RANGE ?? { min_value: 1, max_value: 100_000 },
  );
  const kohLRProps = numberInputPropsFromFloatConstraint(
    constraints?.KOHONEN_LEARN_LEARNING_RATE_RANGE ?? { min_value: 1e-8, max_value: 1 },
  );
  const kohSigmaProps = numberInputPropsFromFloatConstraint(
    constraints?.KOHONEN_INITIAL_NEIGHBORHOOD_RADIUS_RANGE ?? {
      min_value: 1e-8,
      max_value: 1000,
    },
  );
  const kohInputDesc = describeNumConstraint(
    constraints?.KOHONEN_INPUT_LAYER_SIZE_RANGE ?? { min_value: 1, max_value: 20 },
  );
  const kohOutputDesc = describeNumConstraint(
    constraints?.KOHONEN_OUTPUT_LAYER_SIZE_RANGE ?? {
      allowed_values: [4, 9, 16, 25, 36, 49],
    },
  );

  const loadProjects = useCallback(async () => {
    try {
      const data = await api.getProjects(token);
      setProjects(data.projects || []);
    } catch (err) {
      if (String(err.message).includes("401")) onLogout();
    }
  }, [token, onLogout]);

  const loadCsvFiles = useCallback(async () => {
    try {
      const data = await api.getCsvFiles(token);
      setCsvFiles(data.files || []);
    } catch {
      /* ignore */
    }
  }, [token]);

  useEffect(() => {
    loadProjects();
    loadCsvFiles();
  }, [loadProjects, loadCsvFiles]);

  function resetWorkspace() {
    setSelectedProjectId(null);
    setProjectData(null);
    setCreateFlow(null);
    setImageUrlSafe(null);
    setInputSize(0);
    setInputVector([]);
    setAnswerResult(null);
    setStatusMsg(null);
  }

  async function selectProject(projectId) {
    resetWorkspace();
    setSelectedProjectId(projectId);
    setLoading(true);
    try {
      const data = await api.getProjectData(token, projectId);
      const proj = data.project;
      setProjectData(proj);
      setInputSize(proj.nn_data?.input_size || 0);
      setInputVector(new Array(proj.nn_data?.input_size || 0).fill(0));

      const imgUrl = await api.fetchImageBlob(token, projectId);
      setImageUrlSafe(imgUrl);
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleDeleteProject(e, projectId) {
    e.stopPropagation();
    try {
      await api.deleteProject(token, projectId);
      if (selectedProjectId === projectId) resetWorkspace();
      loadProjects();
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  async function handleUploadCsv(e) {
    const file = e.target.files[0];
    if (!file) return;
    try {
      await api.uploadCsv(token, file);
      loadCsvFiles();
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
    e.target.value = "";
  }

  async function handleDeleteCsv(e, fileId) {
    e.stopPropagation();
    try {
      await api.deleteCsv(token, fileId);
      loadCsvFiles();
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  async function handleDownloadCsv(e, fileId) {
    e.stopPropagation();
    try {
      await api.downloadCsv(token, fileId);
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    }
  }

  function startCreating() {
    resetWorkspace();
    setCreateFlow("pick");
    setHiddenLayers([4]);
    const first = csvFiles.length > 0 ? csvFiles[0].id : "";
    setSelectedCsvId(first);
    setKohCsvId(first);
    if (kohOutputDesc.kind === "allowed" && kohOutputDesc.values.length) {
      setKohOutputLayer(kohOutputDesc.values[0]);
    }
  }

  function addLayer() {
    if (hiddenLayers.length >= hlBundle.maxLayers) return;
    const n = clampIntByNumConstraint(hlBundle.neuronConstraint, 4);
    setHiddenLayers([...hiddenLayers, n]);
  }

  function removeLayer(idx) {
    setHiddenLayers(hiddenLayers.filter((_, i) => i !== idx));
  }

  function setLayerNeurons(idx, val) {
    const n = clampIntByNumConstraint(hlBundle.neuronConstraint, val);
    const updated = [...hiddenLayers];
    updated[idx] = n;
    setHiddenLayers(updated);
  }

  async function handleCreatePerceptron() {
    if (!selectedCsvId) {
      setStatusMsg({ type: "error", text: "Select a CSV file first" });
      return;
    }
    setLoading(true);
    setStatusMsg(null);
    try {
      const data = await api.initPerceptron(token, selectedCsvId, hiddenLayers);
      const proj = data.project;
      const imgId = data.image_id;

      setProjectData(proj);
      setSelectedProjectId(proj.id);
      setCreateFlow(null);
      setInputSize(proj.nn_data?.input_size || 0);
      setInputVector(new Array(proj.nn_data?.input_size || 0).fill(0));

      const imgUrl = await api.fetchImageBlob(token, imgId);
      setImageUrlSafe(imgUrl);

      loadProjects();
      setStatusMsg({ type: "success", text: "Perceptron created!" });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleCreateKohonen() {
    if (!kohCsvId) {
      setStatusMsg({ type: "error", text: "Выберите CSV" });
      return;
    }
    setLoading(true);
    setStatusMsg(null);
    try {
      const inSz = clampIntByNumConstraint(
        constraints?.KOHONEN_INPUT_LAYER_SIZE_RANGE ?? { min_value: 1, max_value: 20 },
        kohInputLayer,
      );
      const outSz = clampIntByNumConstraint(
        constraints?.KOHONEN_OUTPUT_LAYER_SIZE_RANGE ?? {
          allowed_values: [4, 9, 16, 25, 36, 49],
        },
        kohOutputLayer,
      );
      const data = await api.initKohonen(token, kohCsvId, inSz, outSz);
      const proj = data.project;
      const imgId = data.image_id;

      setProjectData(proj);
      setSelectedProjectId(proj.id);
      setCreateFlow(null);
      setInputSize(proj.nn_data?.input_size || 0);
      setInputVector(new Array(proj.nn_data?.input_size || 0).fill(0));

      const imgUrl = await api.fetchImageBlob(token, imgId);
      setImageUrlSafe(imgUrl);

      loadProjects();
      setStatusMsg({ type: "success", text: "Сеть Кохонена создана" });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleTrainPerceptron() {
    if (!selectedProjectId) return;
    setLoading(true);
    setStatusMsg(null);
    setTrainingProgress({ type: "queue_update", position: "..." });
    try {
      const result = await learnPerceptronWS(
        token,
        {
          project_id: selectedProjectId,
          activation_type: activationType,
          softmax_use: softmaxUse,
          loss_type: lossType,
          epochs,
          learning_rate: learningRate,
        },
        (msg) => {
          if (msg.type === "queue_update") setTrainingProgress(msg);
        },
      );

      const proj = result.project;
      const imgId = result.image_id;

      setProjectData(proj);
      setInputSize(proj.nn_data?.input_size || 0);
      setInputVector(new Array(proj.nn_data?.input_size || 0).fill(0));

      const imgUrl = await api.fetchImageBlob(token, imgId);
      setImageUrlSafe(imgUrl);

      setStatusMsg({
        type: "success",
        text: `Обучение завершено. Loss: ${result.loss?.toFixed(4)} | Эпохи: ${result.epochs}`,
      });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setLoading(false);
      setTrainingProgress(null);
    }
  }

  async function handleTrainKohonen() {
    if (!selectedProjectId) return;
    setLoading(true);
    setStatusMsg(null);
    try {
      const result = await api.learnKohonen(token, {
        project_id: selectedProjectId,
        epochs: kohEpochs,
        learning_rate: kohLR,
        initial_neighborhood_radius: kohSigma,
        neighbourhood_function: kohNeighbour,
        topology_distance: kohTopology,
      });
      const proj = result.project;
      const imgId = result.image_id;
      setProjectData(proj);
      const imgUrl = await api.fetchImageBlob(token, imgId);
      setImageUrlSafe(imgUrl);
      setStatusMsg({ type: "success", text: "Обучение Кохонена завершено" });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleGetAnswer() {
    if (!selectedProjectId || !projectData) return;
    setLoading(true);
    setAnswerResult(null);
    try {
      if (projectData.project_type === "KOHONEN") {
        const data = await api.getAnswerKohonen(
          token,
          selectedProjectId,
          inputVector.map(Number),
        );
        setAnswerResult(data);
      } else {
        const data = await api.getAnswerPerceptron(
          token,
          selectedProjectId,
          inputVector.map(Number),
          activationType,
          softmaxUse,
        );
        setAnswerResult(data);
      }
    } catch (err) {
      setAnswerResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  }

  function setInputValue(idx, val) {
    const updated = [...inputVector];
    updated[idx] = val;
    setInputVector(updated);
  }

  const creating = createFlow !== null;
  const isProjectLoaded = projectData && selectedProjectId && !creating;
  const isPerceptronProject = isProjectLoaded && projectData.project_type === "PERCEPTRON";
  const isKohonenProject = isProjectLoaded && projectData.project_type === "KOHONEN";

  const perceptronEpochProps =
    perceptronEpochsDesc.kind === "range"
      ? { min: perceptronEpochsDesc.min, max: perceptronEpochsDesc.max, step: 1 }
      : { min: 1, max: 100_000, step: 1 };
  const kohEpochProps =
    kohEpochsDesc.kind === "range"
      ? { min: kohEpochsDesc.min, max: kohEpochsDesc.max, step: 1 }
      : { min: 1, max: 100_000, step: 1 };

  return (
    <div className="main-page">
      <header className="main-header">
        <h1>Neural Network Emulator</h1>
        <div className="header-right">
          <a
            className="help-icon"
            href={HELP_URL}
            target="_blank"
            rel="noreferrer"
            title="Help"
          >
            ?
          </a>
          <button className="logout-btn" onClick={onLogout}>
            Logout
          </button>
        </div>
      </header>

      <div className="main-content">
        {isNarrow && (
          <button
            className={`sidebar-tab ${sidebarOpen ? "tab-right" : "tab-left"}`}
            onClick={() => setSidebarOpen((v) => !v)}
            title={sidebarOpen ? "Свернуть" : "Развернуть"}
            aria-label={sidebarOpen ? "Свернуть" : "Развернуть"}
          >
            {sidebarOpen ? "‹" : "›"}
          </button>
        )}
        <aside
          className={`sidebar ${sidebarOpen ? "" : "sidebar-collapsed"} ${isNarrow ? "sidebar-overlay" : ""}`}
        >
          <div className="sidebar-section">
            <h2>Projects</h2>
            <ul className="sidebar-list">
              {projects.map((p) => (
                <li
                  key={p.id}
                  className={`sidebar-item ${selectedProjectId === p.id ? "active" : ""}`}
                  onClick={() => selectProject(p.id)}
                >
                  <ProjectTypeGlyph projectType={p.project_type} />
                  <span className="item-name">
                    {p.project_type === "KOHONEN" ? "SOM " : "MLP "}
                    {p.id.slice(0, 6)}…
                  </span>
                  <span
                    className="delete-btn"
                    onClick={(e) => handleDeleteProject(e, p.id)}
                    title="Delete"
                  >
                    ✕
                  </span>
                </li>
              ))}
            </ul>
            <button className="add-btn" onClick={startCreating} title="New project">
              +
            </button>
          </div>

          <div className="sidebar-section">
            <h2>CSV Files</h2>
            <ul className="sidebar-list">
              {csvFiles.map((f) => (
                <li
                  key={f.id}
                  className="sidebar-item csv-file-item"
                  onClick={(e) => handleDownloadCsv(e, f.id)}
                >
                  <span className="item-name">{f.name || f.id}</span>
                  <span className="item-actions">
                    <span className="icon-btn download-btn" title="Download">
                      ↓
                    </span>
                    <span
                      className="delete-btn"
                      onClick={(e) => handleDeleteCsv(e, f.id)}
                      title="Delete"
                    >
                      ✕
                    </span>
                  </span>
                </li>
              ))}
            </ul>
            <input
              ref={csvInputRef}
              type="file"
              accept=".csv"
              className="upload-input"
              onChange={handleUploadCsv}
            />
            <button
              className="add-btn"
              onClick={() => csvInputRef.current?.click()}
              title="Upload CSV"
            >
              +
            </button>
          </div>
        </aside>

        <div
          className="workspace"
          style={{
            marginLeft: isNarrow && !sidebarOpen ? 0 : 260,
          }}
        >
          {loading && <div className="loading">Processing...</div>}

          {statusMsg && (
            <div className={`status-msg ${statusMsg.type}`}>{statusMsg.text}</div>
          )}

          {!creating && !isProjectLoaded && !loading && (
            <div className="workspace-empty">
              Select a project or create a new one
            </div>
          )}

          {createFlow === "pick" && (
            <div className="project-editor create-pick">
              <h3 className="create-pick-title">Тип проекта</h3>
              <p className="create-pick-hint">
                Выберите архитектуру. Ограничения полей подгружаются с сервера (
                <code>/api/public-constraints</code>).
              </p>
              <div className="create-pick-buttons">
                <button
                  type="button"
                  className="big-btn"
                  onClick={() => {
                    setCreateFlow("PERCEPTRON");
                    const first = csvFiles.length > 0 ? csvFiles[0].id : "";
                    setSelectedCsvId(first);
                  }}
                >
                  Перцептрон
                </button>
                <button
                  type="button"
                  className="big-btn train"
                  onClick={() => {
                    setCreateFlow("KOHONEN");
                    const first = csvFiles.length > 0 ? csvFiles[0].id : "";
                    setKohCsvId(first);
                    if (kohOutputDesc.kind === "allowed" && kohOutputDesc.values.length) {
                      setKohOutputLayer(kohOutputDesc.values[0]);
                    }
                  }}
                >
                  Кохонен (SOM)
                </button>
              </div>
              <button type="button" className="text-btn" onClick={() => setCreateFlow(null)}>
                Отмена
              </button>
            </div>
          )}

          {createFlow === "PERCEPTRON" && (
            <div className="project-editor">
              {!constraints && (
                <div className="constraints-hint">Загрузка ограничений с сервера…</div>
              )}
              <div className="editor-section">
                <h3>Скрытые слои</h3>
                <div className="layers-config">
                  {hiddenLayers.map((neurons, idx) => {
                    const neuronDesc = describeNumConstraint(hlBundle.neuronConstraint);
                    return (
                      <div className="layer-row" key={idx}>
                        <label>Слой {idx + 1}</label>
                        {neuronDesc.kind === "allowed" ? (
                          <select
                            value={neurons}
                            onChange={(e) => setLayerNeurons(idx, e.target.value)}
                          >
                            {neuronDesc.values.map((v) => (
                              <option key={v} value={v}>
                                {v}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="number"
                            {...numberInputPropsFromNumConstraint(hlBundle.neuronConstraint)}
                            value={neurons}
                            onChange={(e) => setLayerNeurons(idx, e.target.value)}
                          />
                        )}
                        <span className="delete-btn" onClick={() => removeLayer(idx)}>
                          ✕
                        </span>
                      </div>
                    );
                  })}
                </div>
                {hiddenLayers.length < hlBundle.maxLayers && (
                  <button className="add-layer-btn" onClick={addLayer}>
                    + Add Layer
                  </button>
                )}
              </div>

              <div className="editor-section">
                <div className="config-row">
                  <div className="config-field">
                    <label>Активация</label>
                    <select
                      value={activationType}
                      onChange={(e) => setActivationType(e.target.value)}
                    >
                      <option value="RELLU">RELLU</option>
                      <option value="SIGMOID">SIGMOID</option>
                    </select>
                  </div>
                  <div className="config-field">
                    <label>CSV</label>
                    <select
                      value={selectedCsvId}
                      onChange={(e) => setSelectedCsvId(e.target.value)}
                    >
                      <option value="">-- select --</option>
                      {csvFiles.map((f) => (
                        <option key={f.id} value={f.id}>
                          {f.name || f.id}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <button className="big-btn" onClick={handleCreatePerceptron} disabled={loading}>
                Создать перцептрон
              </button>
              <button type="button" className="text-btn" onClick={() => setCreateFlow(null)}>
                Назад
              </button>
            </div>
          )}

          {createFlow === "KOHONEN" && (
            <div className="project-editor">
              <div className="editor-section">
                <h3>Сеть Кохонена</h3>
                <p className="create-pick-hint">
                  Размер входа должен совпадать с числом признаков в CSV. Размер карты — квадратное
                  число нейронов (4, 9, 16…).
                </p>
                <div className="config-row">
                  <div className="config-field">
                    <label>CSV</label>
                    <select value={kohCsvId} onChange={(e) => setKohCsvId(e.target.value)}>
                      <option value="">-- select --</option>
                      {csvFiles.map((f) => (
                        <option key={f.id} value={f.id}>
                          {f.name || f.id}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="config-field">
                    <label>Входной слой (число признаков)</label>
                    {kohInputDesc.kind === "allowed" ? (
                      <select
                        value={kohInputLayer}
                        onChange={(e) =>
                          setKohInputLayer(
                            clampIntByNumConstraint(
                              constraints?.KOHONEN_INPUT_LAYER_SIZE_RANGE ?? {
                                min_value: 1,
                                max_value: 20,
                              },
                              e.target.value,
                            ),
                          )
                        }
                      >
                        {kohInputDesc.values.map((v) => (
                          <option key={v} value={v}>
                            {v}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        {...numberInputPropsFromNumConstraint(
                          constraints?.KOHONEN_INPUT_LAYER_SIZE_RANGE ?? {
                            min_value: 1,
                            max_value: 20,
                          },
                        )}
                        value={kohInputLayer}
                        onChange={(e) =>
                          setKohInputLayer(
                            clampIntByNumConstraint(
                              constraints?.KOHONEN_INPUT_LAYER_SIZE_RANGE ?? {
                                min_value: 1,
                                max_value: 20,
                              },
                              e.target.value,
                            ),
                          )
                        }
                      />
                    )}
                  </div>
                  <div className="config-field">
                    <label>Карта (нейронов)</label>
                    {kohOutputDesc.kind === "allowed" ? (
                      <select
                        value={kohOutputLayer}
                        onChange={(e) => setKohOutputLayer(Number(e.target.value))}
                      >
                        {kohOutputDesc.values.map((v) => (
                          <option key={v} value={v}>
                            {v} ({Math.sqrt(v)}×{Math.sqrt(v)})
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        {...numberInputPropsFromNumConstraint(
                          constraints?.KOHONEN_OUTPUT_LAYER_SIZE_RANGE ?? {
                            allowed_values: [4, 9, 16, 25, 36, 49],
                          },
                        )}
                        value={kohOutputLayer}
                        onChange={(e) =>
                          setKohOutputLayer(
                            clampIntByNumConstraint(
                              constraints?.KOHONEN_OUTPUT_LAYER_SIZE_RANGE ?? {
                                allowed_values: [4, 9, 16, 25, 36, 49],
                              },
                              e.target.value,
                            ),
                          )
                        }
                      />
                    )}
                  </div>
                </div>
              </div>
              <button className="big-btn train" onClick={handleCreateKohonen} disabled={loading}>
                Создать Кохонена
              </button>
              <button type="button" className="text-btn" onClick={() => setCreateFlow(null)}>
                Назад
              </button>
            </div>
          )}

          {isPerceptronProject && (
            <div className="project-editor">
              {imageUrl && (
                <div className="nn-image-wrapper">
                  {trainingProgress && (
                    <div className="training-overlay">
                      <div className="training-overlay-status">
                        {trainingProgress.type === "queue_update"
                          ? `Queue position: ${trainingProgress.position}`
                          : "Training..."}
                      </div>
                    </div>
                  )}
                  <img src={imageUrl} alt="Neural network visualization" />
                </div>
              )}

              <div className="editor-section">
                <div className="config-row">
                  <div className="config-field">
                    <label>Активация</label>
                    <select
                      value={activationType}
                      onChange={(e) => setActivationType(e.target.value)}
                    >
                      <option value="RELLU">RELLU</option>
                      <option value="SIGMOID">SIGMOID</option>
                    </select>
                  </div>
                  <div className="config-field">
                    <label>Функция потерь</label>
                    <select value={lossType} onChange={(e) => setLossType(e.target.value)}>
                      <option value="MSE">MSE</option>
                      <option value="CROSS_ENTROPY">CROSS_ENTROPY</option>
                    </select>
                  </div>
                  <div className="config-field">
                    <label>Эпохи</label>
                    <input
                      type="number"
                      {...perceptronEpochProps}
                      value={epochs}
                      onChange={(e) =>
                        setEpochs(
                          clampIntByNumConstraint(
                            constraints?.PERCEPTRON_LEARN_EPOCHS_RANGE ?? {
                              min_value: 1,
                              max_value: 100_000,
                            },
                            e.target.value,
                          ),
                        )
                      }
                    />
                  </div>
                  <div className="config-field">
                    <label>Learning rate</label>
                    <input
                      type="number"
                      {...perceptronLRProps}
                      value={learningRate}
                      onChange={(e) =>
                        setLearningRate(
                          clampFloatByFloatConstraint(
                            constraints?.PERCEPTRON_LEARN_LEARNING_RATE_RANGE ?? {
                              min_value: 1e-8,
                              max_value: 1,
                            },
                            e.target.value,
                          ),
                        )
                      }
                    />
                  </div>
                  <div className="config-field config-field-checkbox">
                    <label>Softmax</label>
                    <input
                      type="checkbox"
                      checked={softmaxUse}
                      onChange={(e) => setSoftmaxUse(e.target.checked)}
                    />
                  </div>
                </div>
              </div>

              <button className="big-btn train" onClick={handleTrainPerceptron} disabled={loading}>
                Train
              </button>

              {inputSize > 0 && (
                <div className="editor-section answer-section">
                  <h3>Тестовый вектор</h3>
                  <div className="input-vector-row">
                    {inputVector.map((val, idx) => (
                      <div className="input-vector-field" key={idx}>
                        <label>x{idx}</label>
                        <input
                          type="number"
                          step="any"
                          value={val}
                          onChange={(e) => setInputValue(idx, e.target.value)}
                        />
                      </div>
                    ))}
                  </div>

                  <button className="big-btn" onClick={handleGetAnswer} disabled={loading}>
                    GetAnswer
                  </button>

                  {answerResult && (
                    <div className="answer-output">
                      {JSON.stringify(answerResult, null, 2)}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {isKohonenProject && (
            <div className="project-editor">
              {imageUrl && (
                <div className="nn-image-wrapper">
                  <img src={imageUrl} alt="U-matrix / карта Кохонена" />
                </div>
              )}

              <div className="editor-section">
                <h3>Обучение SOM</h3>
                <div className="config-row">
                  <div className="config-field">
                    <label>Эпохи</label>
                    <input
                      type="number"
                      {...kohEpochProps}
                      value={kohEpochs}
                      onChange={(e) =>
                        setKohEpochs(
                          clampIntByNumConstraint(
                            constraints?.KOHONEN_LEARN_EPOCHS_RANGE ?? {
                              min_value: 1,
                              max_value: 100_000,
                            },
                            e.target.value,
                          ),
                        )
                      }
                    />
                  </div>
                  <div className="config-field">
                    <label>Learning rate</label>
                    <input
                      type="number"
                      {...kohLRProps}
                      value={kohLR}
                      onChange={(e) =>
                        setKohLR(
                          clampFloatByFloatConstraint(
                            constraints?.KOHONEN_LEARN_LEARNING_RATE_RANGE ?? {
                              min_value: 1e-8,
                              max_value: 1,
                            },
                            e.target.value,
                          ),
                        )
                      }
                    />
                  </div>
                  <div className="config-field">
                    <label>Начальный радиус соседства σ</label>
                    <input
                      type="number"
                      {...kohSigmaProps}
                      value={kohSigma}
                      onChange={(e) =>
                        setKohSigma(
                          clampFloatByFloatConstraint(
                            constraints?.KOHONEN_INITIAL_NEIGHBORHOOD_RADIUS_RANGE ?? {
                              min_value: 1e-8,
                              max_value: 1000,
                            },
                            e.target.value,
                          ),
                        )
                      }
                    />
                  </div>
                  <div className="config-field">
                    <label>Функция соседства</label>
                    <select value={kohNeighbour} onChange={(e) => setKohNeighbour(e.target.value)}>
                      <option value="GAUSSIAN">GAUSSIAN</option>
                      <option value="MEXICAN_HAT">MEXICAN_HAT</option>
                    </select>
                  </div>
                  <div className="config-field">
                    <label>Топология</label>
                    <select value={kohTopology} onChange={(e) => setKohTopology(e.target.value)}>
                      <option value="EUCLIDEAN">EUCLIDEAN</option>
                      <option value="MANHATTAN">MANHATTAN</option>
                    </select>
                  </div>
                </div>
              </div>

              <button className="big-btn train" onClick={handleTrainKohonen} disabled={loading}>
                Train
              </button>

              {inputSize > 0 && (
                <div className="editor-section answer-section">
                  <h3>Вектор для поиска BMU</h3>
                  <div className="input-vector-row">
                    {inputVector.map((val, idx) => (
                      <div className="input-vector-field" key={idx}>
                        <label>x{idx}</label>
                        <input
                          type="number"
                          step="any"
                          value={val}
                          onChange={(e) => setInputValue(idx, e.target.value)}
                        />
                      </div>
                    ))}
                  </div>
                  <button className="big-btn" onClick={handleGetAnswer} disabled={loading}>
                    Get Answer
                  </button>
                  {answerResult && (
                    <div className="answer-output">
                      {JSON.stringify(answerResult, null, 2)}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
