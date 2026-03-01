const BASE = "/api";

function authHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}

async function handleResponse(res) {
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `Error ${res.status}`);
  }
  return data;
}

export async function signUp(email, name, password) {
  const res = await fetch(`${BASE}/auth/sign-up`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, name, password }),
  });
  return handleResponse(res);
}

export async function login(email, password) {
  const res = await fetch(`${BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return handleResponse(res);
}

export async function getMe(token) {
  const res = await fetch(`${BASE}/auth/getme`, {
    method: "POST",
    headers: authHeaders(token),
  });
  return handleResponse(res);
}

export async function getProjects(token) {
  const res = await fetch(`${BASE}/actions/projects`, {
    headers: authHeaders(token),
  });
  return handleResponse(res);
}

export async function getProjectData(token, projectId) {
  const res = await fetch(`${BASE}/actions/project/${projectId}`, {
    headers: authHeaders(token),
  });
  return handleResponse(res);
}

export async function initPerceptron(token, fileId, hiddenLayersArchitecture) {
  const res = await fetch(`${BASE}/actions/init`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({
      file_id: fileId,
      hidden_layers_architecture: hiddenLayersArchitecture,
    }),
  });
  return handleResponse(res);
}

export async function learnPerceptron(
  token,
  projectId,
  activationType,
  epochs,
  learningRate,
) {
  const res = await fetch(`${BASE}/actions/learn/`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({
      project_id: projectId,
      activation_type: activationType,
      epochs,
      learning_rate: learningRate,
    }),
  });
  return handleResponse(res);
}

export async function getAnswer(
  token,
  perceptronId,
  inputVector,
  activationType,
) {
  const res = await fetch(`${BASE}/actions/get_answer`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({
      perceptron_id: perceptronId,
      input_vector: inputVector,
      activation_type: activationType,
    }),
  });
  return handleResponse(res);
}

export async function deleteProject(token, projectId) {
  const res = await fetch(`${BASE}/actions/projects/${projectId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  return handleResponse(res);
}

export async function getCsvFiles(token) {
  const res = await fetch(`${BASE}/csv/`, {
    headers: authHeaders(token),
  });
  return handleResponse(res);
}

export async function uploadCsv(token, file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/csv/upload`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  return handleResponse(res);
}

export async function deleteCsv(token, fileId) {
  const res = await fetch(`${BASE}/csv/${fileId}`, {
    method: "DELETE",
    headers: authHeaders(token),
  });
  return handleResponse(res);
}

export async function downloadCsv(token, fileId) {
  const res = await fetch(`${BASE}/csv/${fileId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || `Error ${res.status}`);
  }
  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition");
  let filename = `${fileId}.csv`;
  if (disposition) {
    const match = disposition.match(/filename="?([^";\n]+)"?/);
    if (match) filename = match[1];
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export async function fetchImageBlob(token, imageId) {
  const res = await fetch(`${BASE}/images/${imageId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error("Failed to load image");
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
