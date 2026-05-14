import { type FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { apiJson, setToken } from "../api/client";

export default function RegisterPage() {
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [consentPd, setConsentPd] = useState(false);
  const [consentCookies, setConsentCookies] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setErr(null);
    if (!consentPd) {
      setErr("Нужно согласие на обработку персональных данных.");
      return;
    }
    if (!consentCookies) {
      setErr("Подтвердите ознакомление с информацией об использовании файлов cookie.");
      return;
    }
    try {
      const res = await apiJson<{ token: string }>("/auth/sign-up", {
        method: "POST",
        json: { email, name, password },
      });
      setToken(res.token);
      nav("/projects", { replace: true });
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    }
  }

  const canSubmit = consentPd && consentCookies;

  return (
    <div className="card">
      <h1>Регистрация</h1>
      <p>
        Перед созданием учётной записи ознакомьтесь с документами по ссылкам ниже. Они
        относятся к требованиям законодательства РФ (в т.ч. к сфере деятельности РКН) к
        обработке персональных данных и к информированию пользователей о cookie.
      </p>
      <form onSubmit={onSubmit}>
        <div className="field">
          <label htmlFor="reg-email">Email</label>
          <input
            id="reg-email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="field">
          <label htmlFor="reg-name">Имя (отображаемое)</label>
          <input
            id="reg-name"
            name="name"
            autoComplete="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />
        </div>
        <div className="field">
          <label htmlFor="reg-password">Пароль</label>
          <input
            id="reg-password"
            type="password"
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={6}
          />
        </div>

        <div className="field">
          <label className="row" style={{ alignItems: "flex-start", fontWeight: 600 }}>
            <input
              type="checkbox"
              checked={consentPd}
              onChange={(e) => setConsentPd(e.target.checked)}
            />
            <span>
              Я даю согласие на обработку моих персональных данных (email, имя) в целях
              работы сервиса. Политика конфиденциальности:{" "}
              <Link to="/legal/privacy-policy" target="_blank" rel="noopener noreferrer">
                открыть
              </Link>
              .
            </span>
          </label>
        </div>

        <div className="field">
          <label className="row" style={{ alignItems: "flex-start", fontWeight: 600 }}>
            <input
              type="checkbox"
              checked={consentCookies}
              onChange={(e) => setConsentCookies(e.target.checked)}
            />
            <span>
              Я подтверждаю ознакомление с информацией об использовании файлов cookie и
              локального хранилища браузера (для сохранения сессии).{" "}
              <Link to="/legal/cookie-policy" target="_blank" rel="noopener noreferrer">
                Политика cookie
              </Link>
              .
            </span>
          </label>
        </div>

        {err ? <p className="err">{err}</p> : null}
        <button type="submit" className="btn" disabled={!canSubmit}>
          Зарегистрироваться
        </button>
      </form>
      <p>
        Уже есть аккаунт? <Link to="/login">Вход</Link>
      </p>
    </div>
  );
}
