import { API_BASE } from "../config";

const TOKEN_KEY = "nnemu_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(): HeadersInit {
  const t = getToken();
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (t) h.Authorization = `Bearer ${t}`;
  return h;
}

export async function apiJson<T>(
  path: string,
  init?: RequestInit & { json?: unknown },
): Promise<T> {
  const base = API_BASE;
  const { json, headers: extra, ...rest } = init ?? {};
  const headers = { ...authHeaders(), ...extra } as Record<string, string>;
  const body = json !== undefined ? JSON.stringify(json) : rest.body;
  const res = await fetch(`${base}${path}`, { ...rest, headers, body });
  const text = await res.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text) as unknown;
    } catch {
      parsed = text;
    }
  }
  if (!res.ok) {
    const detail =
      typeof parsed === "object" && parsed !== null && "detail" in parsed
        ? String((parsed as { detail: unknown }).detail)
        : text || res.statusText;
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return parsed as T;
}

export function imageUrl(imageId: string): string {
  return `${API_BASE}/images/${imageId}`;
}

export async function apiUploadCsv(file: File): Promise<{ id: string; name: string }> {
  const form = new FormData();
  form.append("file", file);
  const t = getToken();
  const headers: Record<string, string> = {};
  if (t) headers.Authorization = `Bearer ${t}`;
  const res = await fetch(`${API_BASE}/csv/upload`, {
    method: "POST",
    headers,
    body: form,
  });
  const text = await res.text();
  let parsed: unknown = null;
  if (text) {
    try {
      parsed = JSON.parse(text) as unknown;
    } catch {
      parsed = text;
    }
  }
  if (!res.ok) {
    const detail =
      typeof parsed === "object" && parsed !== null && "detail" in parsed
        ? String((parsed as { detail: unknown }).detail)
        : text || res.statusText;
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return parsed as { id: string; name: string };
}
