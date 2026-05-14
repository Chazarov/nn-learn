import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "../config";
import { fetchPublicConstraints, type PublicConstraints } from "../lib/publicConstraints";

export function getApiBase(): string {
  return API_BASE;
}

export function usePublicConstraints() {
  const [data, setData] = useState<PublicConstraints | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const c = await fetchPublicConstraints(API_BASE);
      setData(c);
    } catch (e) {
      setData(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { constraints: data, error, loading, reload };
}
