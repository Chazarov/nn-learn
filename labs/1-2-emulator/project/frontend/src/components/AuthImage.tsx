import { useEffect, useState } from "react";
import { getToken, imageUrl } from "../api/client";

type Props = {
  imageId: string | null | undefined;
  alt?: string;
};

/** Изображения требуют Bearer — обычный ``<img src>`` не подходит. */
export function AuthImage({ imageId, alt = "Визуализация" }: Props) {
  const [src, setSrc] = useState<string | undefined>();

  useEffect(() => {
    if (!imageId) {
      setSrc((old) => {
        if (old) URL.revokeObjectURL(old);
        return undefined;
      });
      return;
    }
    const token = getToken();
    if (!token) return;

    const ac = new AbortController();

    void (async () => {
      try {
        const res = await fetch(imageUrl(imageId), {
          headers: { Authorization: `Bearer ${token}` },
          signal: ac.signal,
        });
        if (!res.ok) return;
        const blob = await res.blob();
        if (ac.signal.aborted) return;
        const u = URL.createObjectURL(blob);
        setSrc((old) => {
          if (old) URL.revokeObjectURL(old);
          return u;
        });
      } catch {
        /* aborted or network */
      }
    })();

    return () => {
      ac.abort();
      setSrc((old) => {
        if (old) URL.revokeObjectURL(old);
        return undefined;
      });
    };
  }, [imageId]);

  if (!imageId) return null;
  if (!src) return <p>Загрузка изображения…</p>;
  return <img src={src} alt={alt} style={{ maxWidth: "100%", borderRadius: 8 }} />;
}
