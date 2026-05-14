import { Navigate, Route, Routes } from "react-router-dom";
import { getToken } from "./api/client";
import { Layout } from "./components/Layout";
import { RequireAuth } from "./components/RequireAuth";
import CookiePolicyPage from "./pages/CookiePolicyPage";
import LoginPage from "./pages/LoginPage";
import NewKohonenPage from "./pages/NewKohonenPage";
import NewPerceptronPage from "./pages/NewPerceptronPage";
import PrivacyPolicyPage from "./pages/PrivacyPolicyPage";
import ProjectDetailPage from "./pages/ProjectDetailPage";
import ProjectsListPage from "./pages/ProjectsListPage";
import RegisterPage from "./pages/RegisterPage";

function HomeRedirect() {
  return <Navigate to={getToken() ? "/projects" : "/login"} replace />;
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<HomeRedirect />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/legal/privacy-policy" element={<PrivacyPolicyPage />} />
        <Route path="/legal/cookie-policy" element={<CookiePolicyPage />} />
        <Route element={<RequireAuth />}>
          <Route path="/projects" element={<ProjectsListPage />} />
          <Route path="/projects/new/perceptron" element={<NewPerceptronPage />} />
          <Route path="/projects/new/kohonen" element={<NewKohonenPage />} />
          <Route path="/projects/:id" element={<ProjectDetailPage />} />
        </Route>
      </Route>
    </Routes>
  );
}
