import { Link, Outlet, useNavigate } from "react-router-dom";
import { getToken, setToken } from "../api/client";

export function Layout() {
  const nav = useNavigate();
  const authed = !!getToken();

  return (
    <>
      <header className="topbar">
        <strong>
          <Link to="/" style={{ color: "inherit", textDecoration: "none" }}>
            Эмулятор НС
          </Link>
        </strong>
        <nav className="row">
          {authed ? (
            <>
              <Link to="/projects">Проекты</Link>
              <Link to="/projects/new/perceptron">+ Перцептрон</Link>
              <Link to="/projects/new/kohonen">+ Кохонен</Link>
              <button
                type="button"
                className="btn secondary"
                onClick={() => {
                  setToken(null);
                  nav("/login");
                }}
              >
                Выйти
              </button>
            </>
          ) : (
            <>
              <Link to="/login">Вход</Link>
              <Link to="/register">Регистрация</Link>
            </>
          )}
        </nav>
      </header>
      <main className="layout">
        <Outlet />
      </main>
    </>
  );
}
