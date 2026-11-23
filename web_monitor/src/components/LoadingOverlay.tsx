import type { FC } from 'react';

export type LoadingLanguage = 'es' | 'en' | 'pt-PT';

interface LoadingOverlayProps {
  message: string;
  language: LoadingLanguage;
}

const details: Record<LoadingLanguage, string> = {
  es: 'Espera, por favor. La conexión permanece activa mientras termina la operación.',
  en: 'Please wait. The connection remains active while the operation is completed.',
  'pt-PT': 'Aguarde. A ligação mantém-se ativa enquanto a operação é concluída.',
};

/** Reusable modal loading indicator for long-running web operations. */
const LoadingOverlay: FC<LoadingOverlayProps> = ({ message, language }) => (
  <div className="loading-overlay" role="status" aria-live="polite" aria-busy="true">
    <div className="loading-card">
      <span className="spinner" aria-hidden="true" />
      <div>
        <strong>{message}</strong>
        <small>{details[language]}</small>
      </div>
    </div>
  </div>
);

export default LoadingOverlay;
