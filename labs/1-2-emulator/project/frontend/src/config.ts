/** Базовый URL API (без завершающего слэша). В dev через Vite proxy: ``/api``. */
export const API_BASE = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/$/, "");
