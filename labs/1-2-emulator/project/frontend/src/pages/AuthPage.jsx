import { useState, useEffect } from "react";
import { signUp, login } from "../api";

const COOKIE_STORAGE_KEY = "nn_emulator_cookie_notice_v1";

export default function AuthPage({ onAuth }) {
  const [isLogin, setIsLogin] = useState(false);
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const [cookieNoticeAccepted, setCookieNoticeAccepted] = useState(() =>
    typeof localStorage !== "undefined"
      ? localStorage.getItem(COOKIE_STORAGE_KEY) === "1"
      : false,
  );
  const [pdConsent, setPdConsent] = useState(false);

  useEffect(() => {
    if (cookieNoticeAccepted) {
      localStorage.setItem(COOKIE_STORAGE_KEY, "1");
    }
  }, [cookieNoticeAccepted]);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    if (!isLogin && !pdConsent) {
      setError("Нужно согласие на обработку персональных данных для регистрации.");
      return;
    }
    setLoading(true);
    try {
      let data;
      if (isLogin) {
        data = await login(email, password);
      } else {
        data = await signUp(email, name, password);
      }
      onAuth(data.token);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      {!cookieNoticeAccepted && (
        <div className="cookie-banner" role="dialog" aria-label="Уведомление о cookie">
          <div className="cookie-banner-text">
            <strong>Файлы cookie.</strong> Сервис может сохранять в браузере технические данные
            (например, идентификатор сессии после входа), необходимые для работы личного кабинета.
            Продолжая пользоваться сайтом, вы соглашаетесь с таким использованием. Подробнее — в
            блоке «Персональные данные и cookie» ниже.
          </div>
          <button
            type="button"
            className="cookie-banner-btn"
            onClick={() => setCookieNoticeAccepted(true)}
          >
            Понятно
          </button>
        </div>
      )}

      <div className="auth-box">
        <h1>{isLogin ? "Login" : "Sign Up"}</h1>

        {error && <div className="auth-error">{error}</div>}

        <form onSubmit={handleSubmit}>
          {!isLogin && (
            <div className="field">
              <label>Name</label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required={!isLogin}
                placeholder="your name"
              />
            </div>
          )}

          <div className="field">
            <div className="auth-note">
              Друг, я не буду отправлять тебе ничего на почту. Она нужна чтобы ты не потерял доступ
              к своей работе на моем сервисе.
            </div>
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="your@email.com"
            />
          </div>

          <div className="field">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              placeholder="••••••••"
            />
          </div>

          {!isLogin && (
            <div className="field auth-legal-block">
              <label className="auth-checkbox-row">
                <input
                  type="checkbox"
                  checked={pdConsent}
                  onChange={(e) => setPdConsent(e.target.checked)}
                />
                <span>
                  Я даю{" "}
                  <strong>согласие на обработку персональных данных</strong> (включая фамилию,
                  имя, адрес электронной почты), в том объёме, который необходим для регистрации и
                  работы сервиса, в соответствии с Федеральным законом № 152-ФЗ «О персональных
                  данных». Согласие даётся до достижения цели обработки или до его отзыва.
                </span>
              </label>
              <details className="auth-policy-details">
                <summary>Персональные данные и cookie (развёрнуть)</summary>
                <div className="auth-policy-body">
                  <p>
                    Оператор обрабатывает персональные данные, которые вы указали при регистрации,
                    для создания учётной записи и предоставления функций приложения. Передача
                    третьим лицам не осуществляется, кроме случаев, предусмотренных законом.
                  </p>
                  <p>
                    Для авторизации может использоваться локальное хранилище браузера и/или cookie
                    с техническим токеном сессии. Вы можете удалить данные сайта в настройках
                    браузера; в этом случае повторный вход может потребоваться снова.
                  </p>
                </div>
              </details>
            </div>
          )}

          <button type="submit" disabled={loading}>
            {loading ? "..." : isLogin ? "Login" : "Sign Up"}
          </button>
        </form>

        <div className="switch-link">
          {isLogin ? (
            <a onClick={() => setIsLogin(false)}>Create an account</a>
          ) : (
            <a onClick={() => setIsLogin(true)}>Already have an account?</a>
          )}
        </div>
      </div>
    </div>
  );
}
