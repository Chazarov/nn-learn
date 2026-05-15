import { useState } from "react";
import { signUp, login } from "../api";

export default function AuthPage({ onAuth }) {
  const [isLogin, setIsLogin] = useState(false);
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
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
              Друг, я не буду отправлять тебе ничего на почту. Она нужна чтобы
              ты не потерял доступ к своей работе на моем сервисе.
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
