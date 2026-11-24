import type { ChangeEvent } from 'react';
import type { Config, ConfigurationProps, FieldProps } from '../types';
import Select from './Select';
import './select.css';

function Field({ label, children }: FieldProps) { return <label className="field">{label}{children}</label>; }

export default function ConfigurationView({ config, setConfig, listenerId, t, log, setLog, onValidate, onConnect, onDisconnect, onTrain, busy }: ConfigurationProps) {
  const change = (key: keyof Config, value: Config[keyof Config]) => setConfig((current) => ({ ...current, [key]: value }));
  const save = () => { const safe = { ...config, rtsp_password: undefined }; localStorage.setItem('rtspTrainingConfig', JSON.stringify(safe)); setLog('Configuración guardada. La contraseña no se persiste.'); };
  const reset = () => setConfig(config => ({ ...config, rtsp_url: 'rtsp://127.0.0.1:8555/audio', rtsp_username: '', rtsp_password: '', probe_timeout_seconds: 30, min_training_seconds: 600 }));
  const seconds = t.language === 'Language' ? 'seconds' : 'segundos';
  return <section className="view"><div className="grid">
    <Field label={t.source}><input value={config.rtsp_url} onChange={(e: ChangeEvent<HTMLInputElement>) => change('rtsp_url', e.target.value)} /></Field>
    <Field label={t.username}><input value={config.rtsp_username} onChange={(e: ChangeEvent<HTMLInputElement>) => change('rtsp_username', e.target.value)} /></Field>
    <Field label={t.password}><input type="password" value={config.rtsp_password} onChange={(e: ChangeEvent<HTMLInputElement>) => change('rtsp_password', e.target.value)} /></Field>
    <Field label={t.timeout}><Select value={String(config.probe_timeout_seconds)} options={[5, 10, 15, 30].map(value => ({ value: String(value), label: `${value} ${seconds}` }))} onChange={value => change('probe_timeout_seconds', Number(value))} ariaLabel={t.timeout} /></Field>
    <Field label={t.sourceAudio}><input disabled value="48 kHz · 2 canales · AAC/RTSP" readOnly /></Field>
    <Field label={t.modelAudio}><input disabled value="44,1 kHz · PCM16LE · sonido + vibración" readOnly /></Field>
    <Field label={t.window}><input disabled value="5 segundos · 220500 muestras" readOnly /></Field>
    <Field label={t.classes}><input disabled value="excellent · good · sufficient · bad · very_bad · broken" readOnly /></Field>
    <Field label={t.policy}><Select value={config.training_policy} options={[{ value: 'continuous_capture_then_batch', label: t.continuous }, { value: 'capture_only', label: t.captureOnly }]} onChange={value => change('training_policy', value)} ariaLabel={t.policy} /></Field>
     <Field label={t.transportLabel}><input disabled value={config.transport.toUpperCase()} readOnly /></Field>
    <Field label={t.minimum}><input type="number" min="5" max="600" step="5" value={config.min_training_seconds} onChange={(e: ChangeEvent<HTMLInputElement>) => change('min_training_seconds', Number(e.target.value))} /><small>{t.recommended}</small></Field>
  </div><div className="box"><h2>{t.recommendedTitle}</h2><p>{t.transport}</p><p>{t.first}</p></div><p className="fixed">{t.modelInput}</p><div className="buttons">
    <button disabled={Boolean(busy)} onClick={save}>{t.save}</button><button disabled={Boolean(busy)} onClick={reset}>{t.reset}</button><button className="primary" disabled={Boolean(busy)} onClick={onValidate}>{t.validate}</button><button className="primary" disabled={Boolean(listenerId) || Boolean(busy)} onClick={onConnect}>{t.connect}</button><button disabled={!listenerId || Boolean(busy)} onClick={onDisconnect}>{t.disconnect}</button><button disabled={!listenerId || Boolean(busy)} onClick={onTrain}>{t.train}</button>
  </div><pre className="log">{log}</pre></section>;
}
