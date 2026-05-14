import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { apiJson } from "../api/client";
import { ProjectTypeIcon } from "../components/ProjectTypeIcon";
import type { ProjectSummary } from "../types";

export default function ProjectsListPage() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(async () => {
    setErr(null);
    try {
      const res = await apiJson<{ projects: ProjectSummary[] }>("/actions/projects");
      setProjects(res.projects);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function remove(id: string) {
    if (!confirm("Удалить проект?")) return;
    try {
      await apiJson(`/actions/projects/${id}`, { method: "DELETE" });
      await load();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="card" style={{ maxWidth: 900 }}>
      <h1>Проекты</h1>
      <div className="row" style={{ marginBottom: "1rem" }}>
        <Link className="btn" to="/projects/new/perceptron" style={{ textDecoration: "none" }}>
          Новый перцептрон
        </Link>
        <Link className="btn" to="/projects/new/kohonen" style={{ textDecoration: "none" }}>
          Новая сеть Кохонена
        </Link>
        <button type="button" className="btn secondary" onClick={() => void load()}>
          Обновить
        </button>
      </div>
      {err ? <p className="err">{err}</p> : null}
      {projects.length === 0 ? (
        <p>Пока нет проектов — создайте перцептрон или карту Кохонена.</p>
      ) : (
        <ul style={{ listStyle: "none", padding: 0 }}>
          {projects.map((p) => (
            <li
              key={p.id}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
                padding: "0.6rem 0",
                borderBottom: "1px solid #e2e8f0",
              }}
            >
              <ProjectTypeIcon type={p.project_type} />
              <Link to={`/projects/${p.id}`} style={{ flex: 1, fontWeight: 600 }}>
                {p.project_type === "PERCEPTRON" ? "Перцептрон" : "Кохонен"} · {p.id.slice(0, 8)}…
              </Link>
              <span style={{ color: "#64748b", fontSize: "0.85rem" }}>
                {new Date(p.created_at * 1000).toLocaleString("ru-RU")}
              </span>
              <button type="button" className="btn secondary" onClick={() => void remove(p.id)}>
                Удалить
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
