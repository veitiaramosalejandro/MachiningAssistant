import type { Dispatch, ReactNode, SetStateAction } from 'react';

export type Language = 'es' | 'en' | 'pt-PT';
export type Labels = Record<string, string>;
export type JsonObject = Record<string, any>;
export type Handler = () => void | Promise<void>;

export interface Config {
  rtsp_url: string;
  rtsp_username: string;
  rtsp_password: string;
  probe_timeout_seconds: number;
  audio_window_seconds: number;
  label_source: string;
  training_policy: string;
  min_training_seconds: number;
  transport: string;
}

export interface EventRow extends JsonObject { sequence?: number | string }
export interface StreamObservation extends JsonObject {
  current_events?: EventRow[];
  historical_events?: EventRow[];
  recent_audio_predictions?: JsonObject[];
}
export interface ListenerData extends JsonObject {
  listener_id?: string;
  stream_observation?: StreamObservation;
  metadata_stream_status?: JsonObject;
  training_metrics?: JsonObject;
  dataset_report?: JsonObject;
  training_status?: string;
  training_checkpoint?: string;
  training_log_tail?: string;
  stop_received?: boolean;
  stop_received_at?: number;
}

export interface ConfigurationProps {
  config: Config;
  setConfig: Dispatch<SetStateAction<Config>>;
  listenerId: string | null;
  listenerStatus?: string;
  stopReceived?: boolean;
  t: Labels;
  log: string;
  setLog: Dispatch<SetStateAction<string>>;
  onValidate: Handler;
  onConnect: Handler;
  onDisconnect: Handler;
  onTrain: Handler;
  busy: string | null;
}
export interface ViewProps { listener: ListenerData | null; t: Labels }
export interface FieldProps { label: string; children: ReactNode }
