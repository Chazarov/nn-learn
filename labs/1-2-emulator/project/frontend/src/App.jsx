import { useState, useEffect } from "react";
import AuthPage from "./pages/AuthPage";
import MainPage from "./pages/MainPage";
import { getMe } from "./api";

const TOKEN_KEY = "nn_emulator_token";

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    if (!token) {
      setChecked(true);
      return;
    }
    getMe(token)
      .then(() => setChecked(true))
      .catch(() => {
        localStorage.removeItem(TOKEN_KEY);
        setToken(null);
        setChecked(true);
      });
  }, []);

  useEffect(() => {
    if (token) {
      localStorage.setItem(TOKEN_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_KEY);
    }
  }, [token]);

  if (!checked) {
    return null;
  }

  if (!token) {
    return <AuthPage onAuth={setToken} />;
  }

  return <MainPage token={token} onLogout={() => setToken(null)} />;
}
