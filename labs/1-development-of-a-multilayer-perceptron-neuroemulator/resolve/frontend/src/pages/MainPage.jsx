import { useState, useEffect, useRef, useCallback } from "react";
import * as api from "../api";

const HELP_URL = "https://github.com/Chazarov/nn-learn/tree/master/labs/1-development-of-a-multilayer-perceptron-neuroemulator/resolve";
const MAX_LAYERS = 10;
const MAX_NEURONS = 20;

export default function MainPage({ token, onLogout }) {
  const [projects, setProjects] = useState([]);
  const [csvFiles, setCsvFiles] = useState([]);
  const [selectedProjectId, setSelectedProjectId] = useState(null);
  const [projectData, setProjectData] = useState(null);

  const [creating, setCreating] = useState(false);
  const [hiddenLayers, setHiddenLayers] = useState([4]);
  const [selectedCsvId, setSelectedCsvId] = useState("");
  const [activationType, setActivationType] = useState("RELLU");

  const [epochs, setEpochs] = useState(100);
  const [learningRate, setLearningRate] = useState(0.1);

  const [imageUrl, setImageUrl] = useState(null);
  const [inputSize, setInputSize] = useState(0);
  const [inputVector, setInputVector] = useState([]);
  const [answerResult, setAnswerResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isNarrow, setIsNarrow] = useState(false);

  const csvInputRef = useRef(null);

  useEffect(() => {
    const check = () => setIsNarrow(window.innerWidth < 700);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  const loadProjects = useCallback(async () => {
    try {
      const data = await api.getProjects(token);
      setProjects(data.projects || []);
    } catch (err) {
      if (err.message.includes("401")) onLogout();
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
    setCreating(false);
    setImageUrl(null);
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
      setImageUrl(imgUrl);
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
    setCreating(true);
    setHiddenLayers([4]);
    setSelectedCsvId(csvFiles.length > 0 ? csvFiles[0].id : "");
  }

  function addLayer() {
    if (hiddenLayers.length >= MAX_LAYERS) return;
    setHiddenLayers([...hiddenLayers, 4]);
  }

  function removeLayer(idx) {
    setHiddenLayers(hiddenLayers.filter((_, i) => i !== idx));
  }

  function setLayerNeurons(idx, val) {
    const n = Math.max(1, Math.min(MAX_NEURONS, Number(val) || 1));
    const updated = [...hiddenLayers];
    updated[idx] = n;
    setHiddenLayers(updated);
  }

  async function handleCreate() {
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
      setCreating(false);
      setInputSize(proj.nn_data?.input_size || 0);
      setInputVector(new Array(proj.nn_data?.input_size || 0).fill(0));

      const imgUrl = await api.fetchImageBlob(token, imgId);
      setImageUrl(imgUrl);

      loadProjects();
      setStatusMsg({ type: "success", text: "Perceptron created!" });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleTrain() {
    if (!selectedProjectId) return;
    setLoading(true);
    setStatusMsg(null);
    try {
      const data = await api.learnPerceptron(
        token,
        selectedProjectId,
        activationType,
        epochs,
        learningRate,
      );
      const proj = data.project;
      const imgId = data.image_id;

      setProjectData(proj);
      setInputSize(proj.nn_data?.input_size || 0);
      setInputVector(new Array(proj.nn_data?.input_size || 0).fill(0));

      const imgUrl = await api.fetchImageBlob(token, imgId);
      setImageUrl(imgUrl);

      setStatusMsg({ type: "success", text: "Training complete!" });
    } catch (err) {
      setStatusMsg({ type: "error", text: err.message });
    } finally {
      setLoading(false);
    }
  }

  async function handleGetAnswer() {
    if (!selectedProjectId) return;
    setLoading(true);
    setAnswerResult(null);
    try {
      const data = await api.getAnswer(
        token,
        selectedProjectId,
        inputVector.map(Number),
        activationType,
      );
      setAnswerResult(data);
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

  const isProjectLoaded = projectData && selectedProjectId && !creating;

  return (
    <div className="main-page">
      {/* HEADER */}
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
        {/* SIDEBAR */}
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
                  <span className="item-name">
                    Project {p.id.slice(0, 6)}...
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

        {/* WORKSPACE */}
        <div
          className="workspace"
          style={{
            marginLeft:
              isNarrow && !sidebarOpen ? 0 : 260,
          }}
        >
          {loading && <div className="loading">Processing...</div>}

          {statusMsg && (
            <div className={`status-msg ${statusMsg.type}`}>
              {statusMsg.text}
            </div>
          )}

          {!creating && !isProjectLoaded && !loading && (
            <div className="workspace-empty">
              Select a project or create a new one
            </div>
          )}

          {/* CREATION FORM */}
          {creating && (
            <div className="project-editor">
              <div className="editor-section">
                <h3>Hidden Layers</h3>
                <div className="layers-config">
                  {hiddenLayers.map((neurons, idx) => (
                    <div className="layer-row" key={idx}>
                      <label>Layer {idx + 1}</label>
                      <input
                        type="number"
                        min={1}
                        max={MAX_NEURONS}
                        value={neurons}
                        onChange={(e) => setLayerNeurons(idx, e.target.value)}
                      />
                      <span className="delete-btn" onClick={() => removeLayer(idx)}>
                        ✕
                      </span>
                    </div>
                  ))}
                </div>
                {hiddenLayers.length < MAX_LAYERS && (
                  <button className="add-layer-btn" onClick={addLayer}>
                    + Add Layer
                  </button>
                )}
              </div>

              <div className="editor-section">
                <div className="config-row">
                  <div className="config-field">
                    <label>Activation</label>
                    <select
                      value={activationType}
                      onChange={(e) => setActivationType(e.target.value)}
                    >
                      <option value="RELLU">RELLU</option>
                      <option value="SIGMOID">SIGMOID</option>
                    </select>
                  </div>
                  <div className="config-field">
                    <label>CSV File</label>
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

              <button
                className="big-btn"
                onClick={handleCreate}
                disabled={loading}
              >
                Create
              </button>
            </div>
          )}

          {/* PROJECT VIEW */}
          {isProjectLoaded && (
            <div className="project-editor">
              {/* IMAGE */}
              {imageUrl && (
                <div className="nn-image-wrapper">
                  <img src={imageUrl} alt="Neural network visualization" />
                </div>
              )}

              {/* ACTIVATION + TRAIN CONTROLS */}
              <div className="editor-section">
                <div className="config-row">
                  <div className="config-field">
                    <label>Activation</label>
                    <select
                      value={activationType}
                      onChange={(e) => setActivationType(e.target.value)}
                    >
                      <option value="RELLU">RELLU</option>
                      <option value="SIGMOID">SIGMOID</option>
                    </select>
                  </div>
                  <div className="config-field">
                    <label>Epochs</label>
                    <input
                      type="number"
                      min={1}
                      max={10000}
                      value={epochs}
                      onChange={(e) => setEpochs(Number(e.target.value) || 1)}
                    />
                  </div>
                  <div className="config-field">
                    <label>Learning Rate</label>
                    <input
                      type="number"
                      step="0.01"
                      min={0.001}
                      max={10}
                      value={learningRate}
                      onChange={(e) =>
                        setLearningRate(Number(e.target.value) || 0.01)
                      }
                    />
                  </div>
                </div>
              </div>

              <button
                className="big-btn train"
                onClick={handleTrain}
                disabled={loading}
              >
                Train
              </button>

              {/* INPUT VECTOR + GET ANSWER */}
              {inputSize > 0 && (
                <div className="editor-section answer-section">
                  <h3>Test Input Vector</h3>
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

                  <button
                    className="big-btn"
                    onClick={handleGetAnswer}
                    disabled={loading}
                  >
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
        </div>
      </div>
    </div>
  );
}
