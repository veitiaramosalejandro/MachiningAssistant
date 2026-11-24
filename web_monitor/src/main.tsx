import React, {useCallback, useEffect, useState} from 'react';
import type {Dispatch, ReactNode, SetStateAction} from 'react';
import {createRoot} from 'react-dom/client';
import LoadingOverlay from './components/LoadingOverlay';
import ConfigurationView from './components/ConfigurationView';
import StreamView from './components/StreamView';
import TrainingView from './components/TrainingView';
import Select from './components/Select';
import './styles.css';
import './loading.css';
import './components/select.css';
import './components/language-select.css';

type Language = 'es' | 'en' | 'pt-PT';
type Labels = Record<string, string>;
type JsonObject = Record<string, any>;
type Handler = () => void | Promise<void>;

interface Config {
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

interface EventRow extends JsonObject {
    sequence?: number | string
}

interface StreamObservation extends JsonObject {
    current_events?: EventRow[];
    historical_events?: EventRow[];
    recent_audio_predictions?: JsonObject[]
}

interface ListenerData extends JsonObject {
    listener_id?: string;
    stream_observation?: StreamObservation;
    metadata_stream_status?: JsonObject;
    training_metrics?: JsonObject;
    dataset_report?: JsonObject;
    training_status?: string;
    training_checkpoint?: string;
    training_log_tail?: string
}

interface EventTableProps {
    rows?: EventRow[];
    t: Labels
}

interface FieldProps {
    label: string;
    children: ReactNode
}

interface CardProps {
    label: string;
    value: unknown
}

interface ConfigurationProps {
    config: Config;
    setConfig: Dispatch<SetStateAction<Config>>;
    listenerId: string | null;
    t: Labels;
    log: string;
    setLog: Dispatch<SetStateAction<string>>;
    onValidate: Handler;
    onConnect: Handler;
    onDisconnect: Handler;
    onTrain: Handler
    busy: string | null;
}

interface StreamProps {
    listener: ListenerData | null;
    t: Labels
}

interface TrainingProps {
    listener: ListenerData | null;
    t: Labels
}

const DEFAULT_CONFIG: Config = {
    rtsp_url: 'rtsp://127.0.0.1:8555/audio',
    rtsp_username: '',
    rtsp_password: '',
    probe_timeout_seconds: 30,
    audio_window_seconds: 5,
    label_source: 'synthetic_audio_transform',
    training_policy: 'continuous_capture_then_batch',
    min_training_seconds: 600,
    transport: 'tcp'
};
const TEXT = {
    es: {
        language: 'Idioma / Language / Idioma',
        title: 'Entrenamiento RTSP · CNN1D-VAE',
        intro: 'Recepción en tiempo real y salida del entrenamiento en vistas separadas. El audio y la metadata se observan tal como llegan.',
        config: 'Configuración',
        stream: 'Recepción del streaming',
        training: 'Salida del entrenamiento',
        validate: 'Validar audio + metadata',
        connect: 'Conectar y escuchar RTSP',
        disconnect: 'Desconectar',
        train: 'Generar lote y entrenar CNN1D-VAE',
        source: 'Fuente RTSP local',
        username: 'Usuario RTSP',
        password: 'Contraseña RTSP',
        timeout: 'Tiempo máximo de comprobación',
        sourceAudio: 'Audio de origen',
        modelAudio: 'Audio entregado al modelo',
        window: 'Ventana de audio',
        classes: 'Clases sintéticas',
        policy: 'Política de captura',
        transportLabel: 'Transporte recibido por socket',
        continuous: 'Capturar continuamente y entrenar por lotes',
        captureOnly: 'Solo capturar',
        minimum: 'Duración mínima de captura',
        recommended: 'Recomendado: 10 minutos · aproximadamente 120 ventanas.',
        recommendedTitle: 'Configuración recomendada para entrenamiento',
        transport: 'Transporte TCP · ventana 5 segundos · captura 10 minutos · entrada sonido + vibración + metadata CNC.',
        first: 'Primero valida audio + metadata, después conecta y espera el histórico antes de entrenar.',
        modelInput: 'Entrada: Audio + StreamCncMetadata · `event.quality` no se usa como entrada ni target.',
        save: 'Guardar configuración',
        reset: 'Restablecer RTSP 8555',
        ready: 'Listo para configurar una sesión RTSP.',
        listener: 'Listener',
        audio: 'Audio',
        metadata: 'Metadata',
        messages: 'Mensajes metadata',
        snapshots: 'Snapshots CNC',
        machineEvents: 'Eventos máquina',
        currentEvents: 'Eventos actuales de la sesión',
        historicalEvents: 'Eventos históricos recibidos al conectar',
        historicalHint: 'Eventos disponibles antes de comenzar la transmisión actual.',
        currentHint: 'Eventos que llegan después del histórico inicial.',
        latestContext: 'Último contexto CNC recibido',
        predictions: 'Ventanas/predicciones de audio',
        waitingStream: 'Esperando una nueva transmisión.',
        noHistory: 'Todavía no hay eventos históricos.',
        noSnapshot: 'Sin snapshot todavía.',
        noWindows: 'Sin ventanas completas todavía.',
        trainingTitle: 'Resultado del entrenamiento',
        trainingSubtitle: 'CNN1D-VAE · audio estéreo + contexto CNC · etiquetas sintéticas',
        dataset: 'Dataset',
        samples: 'Muestras',
        classCount: 'Clases',
        trainingStatus: 'Entrenamiento',
        device: 'Dispositivo',
        bestF1: 'Mejor Macro F1',
        checkpoint: 'Checkpoint',
        performance: 'Rendimiento del modelo',
        valAccuracy: 'Accuracy validación',
        valF1: 'Macro F1 validación',
        testAccuracy: 'Accuracy test',
        testF1: 'Macro F1 test',
        fullMetrics: 'Ver métricas completas',
        noMetrics: 'Todavía no hay métricas.',
        composition: 'Composición del dataset',
        noDataset: 'Todavía no se ha generado el lote.',
        fullReport: 'Ver dataset report completo',
        trainingLog: 'Ver log del entrenamiento',
        noLog: 'Sin log todavía.',
        disconnected: 'desconectado',
        waiting: 'en espera',
        error: 'error',
        noData: 'Sin datos todavía.'
    },
    en: {
        language: 'Language',
        title: 'RTSP Training · CNN1D-VAE',
        intro: 'Real-time reception and training output in separate views. Audio and metadata are observed as received.',
        config: 'Configuration',
        stream: 'Stream reception',
        training: 'Training output',
        validate: 'Validate audio + metadata',
        connect: 'Connect and listen RTSP',
        disconnect: 'Disconnect',
        train: 'Build dataset and train CNN1D-VAE',
        source: 'Local RTSP source',
        username: 'RTSP username',
        password: 'RTSP password',
        timeout: 'Maximum probe time',
        sourceAudio: 'Source audio',
        modelAudio: 'Audio delivered to the model',
        window: 'Audio window',
        classes: 'Synthetic classes',
        policy: 'Capture policy',
        transportLabel: 'Transport received by socket',
        continuous: 'Capture continuously and train in batches',
        captureOnly: 'Capture only',
        minimum: 'Minimum capture duration',
        recommended: 'Recommended: 10 minutes · approximately 120 windows.',
        recommendedTitle: 'Recommended training configuration',
        transport: 'TCP transport · 5-second window · 10-minute capture · sound + vibration + CNC metadata input.',
        first: 'Validate audio + metadata first, then connect and wait for history before training.',
        modelInput: 'Input: Audio + StreamCncMetadata · `event.quality` is not used as input or target.',
        save: 'Save configuration',
        reset: 'Reset RTSP 8555',
        ready: 'Ready to configure an RTSP session.',
        listener: 'Listener',
        audio: 'Audio',
        metadata: 'Metadata',
        messages: 'Metadata messages',
        snapshots: 'CNC snapshots',
        machineEvents: 'Machine events',
        currentEvents: 'Current session events',
        historicalEvents: 'Historical events received on connection',
        historicalHint: 'Events available before the current transmission started.',
        currentHint: 'Events received after the initial history.',
        latestContext: 'Latest CNC context received',
        predictions: 'Audio windows/predictions',
        waitingStream: 'Waiting for a new transmission.',
        noHistory: 'No historical events yet.',
        noSnapshot: 'No snapshot yet.',
        noWindows: 'No complete windows yet.',
        trainingTitle: 'Training result',
        trainingSubtitle: 'CNN1D-VAE · stereo audio + CNC context · synthetic labels',
        dataset: 'Dataset',
        samples: 'Samples',
        classCount: 'Classes',
        trainingStatus: 'Training',
        device: 'Device',
        bestF1: 'Best Macro F1',
        checkpoint: 'Checkpoint',
        performance: 'Model performance',
        valAccuracy: 'Validation accuracy',
        valF1: 'Validation Macro F1',
        testAccuracy: 'Test accuracy',
        testF1: 'Test Macro F1',
        fullMetrics: 'View full metrics',
        noMetrics: 'No metrics yet.',
        composition: 'Dataset composition',
        noDataset: 'Dataset has not been generated yet.',
        fullReport: 'View full dataset report',
        trainingLog: 'View training log',
        noLog: 'No log yet.',
        disconnected: 'disconnected',
        waiting: 'waiting',
        error: 'error',
        noData: 'No data yet.'
    },
    'pt-PT': {
        language: 'Idioma',
        title: 'Treino RTSP · CNN1D-VAE',
        intro: 'Receção em tempo real e resultado do treino em vistas separadas. O áudio e os metadados são observados tal como chegam.',
        config: 'Configuração',
        stream: 'Receção do streaming',
        training: 'Resultado do treino',
        validate: 'Validar áudio + metadados',
        connect: 'Ligar e escutar RTSP',
        disconnect: 'Desligar',
        train: 'Gerar lote e treinar CNN1D-VAE',
        source: 'Fonte RTSP local',
        username: 'Utilizador RTSP',
        password: 'Palavra-passe RTSP',
        timeout: 'Tempo máximo de verificação',
        sourceAudio: 'Áudio de origem',
        modelAudio: 'Áudio entregue ao modelo',
        window: 'Janela de áudio',
        classes: 'Classes sintéticas',
        policy: 'Política de captura',
        transportLabel: 'Transporte recebido pelo socket',
        continuous: 'Capturar continuamente e treinar em lotes',
        captureOnly: 'Apenas capturar',
        minimum: 'Duração mínima da captura',
        recommended: 'Recomendado: 10 minutos · aproximadamente 120 janelas.',
        recommendedTitle: 'Configuração recomendada para treino',
        transport: 'Transporte TCP · janela de 5 segundos · captura de 10 minutos · entrada de som + vibração + metadados CNC.',
        first: 'Valide primeiro o áudio + metadados, ligue depois e aguarde o histórico antes do treino.',
        modelInput: 'Entrada: Áudio + StreamCncMetadata · `event.quality` não é usado como entrada nem target.',
        save: 'Guardar configuração',
        reset: 'Repor RTSP 8555',
        ready: 'Pronto para configurar uma sessão RTSP.',
        listener: 'Listener',
        audio: 'Áudio',
        metadata: 'Metadados',
        messages: 'Mensagens de metadados',
        snapshots: 'Snapshots CNC',
        machineEvents: 'Eventos da máquina',
        currentEvents: 'Eventos atuais da sessão',
        historicalEvents: 'Eventos históricos recebidos ao ligar',
        historicalHint: 'Eventos disponíveis antes do início da transmissão atual.',
        currentHint: 'Eventos recebidos depois do histórico inicial.',
        latestContext: 'Último contexto CNC recebido',
        predictions: 'Janelas/predições de áudio',
        waitingStream: 'A aguardar uma nova transmissão.',
        noHistory: 'Ainda não existem eventos históricos.',
        noSnapshot: 'Ainda não existe snapshot.',
        noWindows: 'Ainda não existem janelas completas.',
        trainingTitle: 'Resultado do treino',
        trainingSubtitle: 'CNN1D-VAE · áudio estéreo + contexto CNC · etiquetas sintéticas',
        dataset: 'Lote de dados',
        samples: 'Amostras',
        classCount: 'Classes',
        trainingStatus: 'Treino',
        device: 'Dispositivo',
        bestF1: 'Melhor Macro F1',
        checkpoint: 'Checkpoint',
        performance: 'Desempenho do modelo',
        valAccuracy: 'Accuracy de validação',
        valF1: 'Macro F1 de validação',
        testAccuracy: 'Accuracy de teste',
        testF1: 'Macro F1 de teste',
        fullMetrics: 'Ver métricas completas',
        noMetrics: 'Ainda não existem métricas.',
        composition: 'Composição do lote',
        noDataset: 'O lote ainda não foi gerado.',
        fullReport: 'Ver relatório completo do lote',
        trainingLog: 'Ver registo do treino',
        noLog: 'Ainda não existe registo.',
        disconnected: 'desligado',
        waiting: 'a aguardar',
        error: 'erro',
        noData: 'Sem dados.'
    },
};

async function request<T = JsonObject>(path: string, options: RequestInit = {}): Promise<T> {
    const response = await fetch(path, {headers: {'Content-Type': 'application/json'}, ...options});
    const data: JsonObject = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
    return data as T;
}

const post = <T = JsonObject>(path: string, body: JsonObject): Promise<T> => request<T>(path, {method: 'POST', body: JSON.stringify(body)});
const json = (value: unknown, fallback: string): string => value ? JSON.stringify(value, null, 2) : fallback;
const pct = (value: unknown): string => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '—';

function Field({label, children}: FieldProps) {
    return <label className="field">{label}{children}</label>;
}

function Card({label, value}: CardProps) {
    return <div className="card"><b>{label}</b><span>{String(value ?? '—')}</span></div>;
}

function EventTable({rows, t}: EventTableProps) {
    if (!rows?.length) return <p className="muted">{t.waitingStream}</p>;
    const columns = [['timestamp_seconds', 'Tiempo'], ['occurrence', 'Ocurrencia'], ['value', 'Valor'], ['machine_state', 'Estado'], ['machine_activity', 'Actividad'], ['programmed_rpm', 'RPM programada'], ['adjusted_rpm', 'RPM ajustada'], ['programmed_feed', 'Avance programado'], ['adjusted_feed', 'Avance ajustado'], ['quality', 'quality']] as const;
    const translated: Record<string, string> = {
        Tiempo: t.language === 'Language' ? 'Time' : t.language === 'Idioma' ? 'Tempo' : 'Tiempo',
        Ocurrencia: t.language === 'Language' ? 'Occurrence' : t.language === 'Idioma' ? 'Ocorrência' : 'Ocurrencia',
        Valor: t.language === 'Language' ? 'Value' : 'Valor',
        Estado: t.language === 'Language' ? 'State' : 'Estado',
        Actividad: t.language === 'Language' ? 'Activity' : t.language === 'Idioma' ? 'Atividade' : 'Actividad',
        'RPM programada': t.language === 'Language' ? 'Programmed RPM' : 'RPM programada',
        'RPM ajustada': t.language === 'Language' ? 'Adjusted RPM' : 'RPM ajustada',
        'Avance programado': t.language === 'Language' ? 'Programmed feed' : t.language === 'Idioma' ? 'Avanço programado' : 'Avance programado',
        'Avance ajustado': t.language === 'Language' ? 'Adjusted feed' : t.language === 'Idioma' ? 'Avanço ajustado' : 'Avance ajustado',
        quality: 'quality'
    };
    return <div className="table-wrap">
        <table>
            <thead>
            <tr>{columns.map(([, label]) => <th key={label}>{translated[label]}</th>)}</tr>
            </thead>
            <tbody>{[...rows].reverse().map((row, index) => <tr
                key={`${row.sequence ?? index}-${index}`}>{columns.map(([key]) => <td
                key={key}>{String(row[key] ?? '')}</td>)}</tr>)}</tbody>
        </table>
    </div>;
}

function Configuration({config, setConfig, listenerId, t, log, setLog, onValidate, onConnect, onDisconnect, onTrain, busy}: ConfigurationProps) {
    const change = (key: keyof Config, value: Config[keyof Config]) => setConfig((current) => ({...current, [key]: value}));
    const save = () => {
        const safe = {...config, rtsp_password: undefined};
        localStorage.setItem('rtspTrainingConfig', JSON.stringify(safe));
        setLog('Configuración guardada. La contraseña no se persiste.');
    };
    const reset = () => setConfig(DEFAULT_CONFIG);
    return <section className="view">
        <div className="grid">
            <Field label={t.source}><input value={config.rtsp_url}
                                           onChange={(e) => change('rtsp_url', e.target.value)}/></Field><Field
            label={t.username}><input value={config.rtsp_username}
                                      onChange={(e) => change('rtsp_username', e.target.value)}/></Field><Field
            label={t.password}><input type="password" value={config.rtsp_password}
                                      onChange={(e) => change('rtsp_password', e.target.value)}/></Field><Field
            label={t.timeout}><select value={config.probe_timeout_seconds}
                                      onChange={(e) => change('probe_timeout_seconds', Number(e.target.value))}>
            <option
                value="5">5 {t.language === 'Language' ? 'seconds' : t.language === 'Idioma' ? 'segundos' : 'segundos'}</option>
            <option
                value="10">10 {t.language === 'Language' ? 'seconds' : t.language === 'Idioma' ? 'segundos' : 'segundos'}</option>
            <option
                value="15">15 {t.language === 'Language' ? 'seconds' : t.language === 'Idioma' ? 'segundos' : 'segundos'}</option>
            <option
                value="30">30 {t.language === 'Language' ? 'seconds' : t.language === 'Idioma' ? 'segundos' : 'segundos'}</option>
        </select></Field><Field label={t.sourceAudio}><input disabled value="48 kHz · 2 canales · AAC/RTSP"
                                                             readOnly/></Field><Field label={t.modelAudio}><input
            disabled value="44,1 kHz · PCM16LE · sonido + vibración" readOnly/></Field><Field label={t.window}><input
            disabled value="5 segundos · 220500 muestras" readOnly/></Field><Field label={t.classes}><input disabled
                                                                                                            value="excellent · good · sufficient · bad · very_bad · broken"
                                                                                                            readOnly/></Field><Field
            label={t.policy}><select value={config.training_policy}
                                     onChange={(e) => change('training_policy', e.target.value)}>
            <option value="continuous_capture_then_batch">{t.continuous}</option>
            <option value="capture_only">{t.captureOnly}</option>
        </select></Field><Field label={t.minimum}><input type="number" min="5" max="600" step="5"
                                                         value={config.min_training_seconds}
                                                         onChange={(e) => change('min_training_seconds', Number(e.target.value))}/><small>{t.recommended}</small></Field>
        </div>
        <div className="box"><h2>{t.recommendedTitle}</h2><p>{t.transport}</p><p>{t.first}</p></div>
        <p className="fixed">{t.modelInput}</p>
        <div className="buttons">
            <button disabled={Boolean(busy)} onClick={save}>{t.save}</button>
            <button disabled={Boolean(busy)} onClick={reset}>{t.reset}</button>
            <button className="primary" disabled={Boolean(busy)} onClick={onValidate}>{t.validate}</button>
            <button className="primary" disabled={Boolean(listenerId) || Boolean(busy)} onClick={onConnect}>{t.connect}</button>
            <button disabled={!listenerId || Boolean(busy)} onClick={onDisconnect}>{t.disconnect}</button>
            <button disabled={!listenerId || Boolean(busy)} onClick={onTrain}>{t.train}</button>
        </div>
        <pre className="log">{log}</pre>
    </section>;
}

function Stream({listener, t}: StreamProps) {
    const observation = listener?.stream_observation || {};
    const status = listener?.metadata_stream_status || {};
    return <section className="view">
        <div className="cards"><Card label={t.listener} value={listener?.status || t.disconnected}/><Card
            label={t.audio}
            value={listener?.status === 'failed' ? t.error : listener?.emitted_windows ?? t.waiting}/><Card
            label={t.metadata} value={listener?.metadata_status || status.status || t.waiting}/><Card label={t.messages}
                                                                                                      value={observation.metadata_messages ?? 0}/><Card
            label={t.snapshots} value={observation.metadata_snapshots ?? 0}/><Card label={t.machineEvents}
                                                                                   value={observation.metadata_events ?? 0}/>
        </div>
        <div className="box"><h2>{t.currentEvents}</h2><p className="muted">{t.currentHint}</p><EventTable
            rows={observation.current_events} t={t}/></div>
        <div className="box"><h2>{t.historicalEvents}</h2><p className="muted">{t.historicalHint}</p><EventTable
            rows={observation.historical_events} t={t}/></div>
        <div className="two">
            <div className="box"><h2>{t.latestContext}</h2>
                <pre className="pre">{json(observation.latest_snapshot, t.noSnapshot)}</pre>
            </div>
            <div className="box"><h2>{t.predictions}</h2>
                <pre
                    className="pre">{json(observation.recent_audio_predictions?.length ? observation.recent_audio_predictions : null, t.noWindows)}</pre>
            </div>
        </div>
    </section>;
}

function Training({listener, t}: TrainingProps) {
    const report = listener?.dataset_report;
    const metrics = listener?.training_metrics || {};
    const splits: Record<string, number> = report?.split_counts || {};
    const total = Object.values(splits).reduce((sum: number, value: number) => sum + Number(value || 0), 0);
    return <section className="view">
        <div className="training-head">
            <div><h2>{t.trainingTitle}</h2><p>{t.trainingSubtitle}</p></div>
            <span className="status-pill">{listener?.training_status || t.noData}</span></div>
        <div className="training-kpis"><Kpi label={t.dataset}
                                            value={report ? (report.ready ? 'Listo' : 'Bloqueado') : t.noDataset}/><Kpi
            label={t.samples} value={report?.generated_samples ?? 0}/><Kpi label={t.classCount}
                                                                           value={report?.required_classes?.length ?? 6}/><Kpi
            label={t.trainingStatus} value={listener?.training_status || t.noData}/><Kpi label={t.device}
                                                                                         value={metrics.device || '—'}/><Kpi
            label={t.bestF1} value={pct(metrics.best_validation_macro_f1)}/><Kpi label={t.checkpoint}
                                                                                 value={listener?.training_checkpoint || t.noData}
                                                                                 small/></div>
        <div className="training-layout">
            <div className="box"><h2>{t.performance}</h2>
                <div className="metric-grid"><Metric label={t.valAccuracy}
                                                     value={pct(metrics.validation?.accuracy)}/><Metric label={t.valF1}
                                                                                                        value={pct(metrics.validation?.macro_f1)}/><Metric
                    label={t.testAccuracy} value={pct(metrics.test?.accuracy)}/><Metric label={t.testF1}
                                                                                        value={pct(metrics.test?.macro_f1)}/>
                </div>
                <details>
                    <summary>{t.fullMetrics}</summary>
                    <pre className="pre">{json(metrics, t.noMetrics)}</pre>
                </details>
            </div>
            <div className="box"><h2>{t.composition}</h2>{total ? Object.entries(splits).map(([name, count]) => <div
                className="split-row" key={name}><span>{name}</span>
                <div className="split-track"><i style={{width: `${Math.max(4, Number(count) / total * 100)}%`}}/></div>
                <strong>{count}</strong></div>) : <p className="muted">{t.noDataset}</p>}<p
                className="training-path">{report?.dataset_dir || ''}</p>
                <details>
                    <summary>{t.fullReport}</summary>
                    <pre className="pre">{json(report, t.noDataset)}</pre>
                </details>
            </div>
        </div>
        <details className="training-log">
            <summary>{t.trainingLog}</summary>
            <pre className="pre">{listener?.training_log_tail || t.noLog}</pre>
        </details>
    </section>;
}

function Kpi({label, value, small}: {label: string; value: ReactNode; small?: boolean}) {
    return <div className="kpi"><span>{label}</span><strong className={small ? 'small' : ''}>{value}</strong></div>;
}

function Metric({label, value}: {label: string; value: ReactNode}) {
    return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function App() {
    const storedLanguage = localStorage.getItem('rtspUiLanguage') as Language | null;
    const [language, setLanguage] = useState<Language>(storedLanguage && storedLanguage in TEXT ? storedLanguage : 'es');
    const t = TEXT[language];
    const [tab, setTab] = useState('config');
    const [config, setConfig] = useState<Config>(() => ({...DEFAULT_CONFIG, ...JSON.parse(localStorage.getItem('rtspTrainingConfig') || '{}')}));
    const [listener, setListener] = useState<ListenerData | null>(null);
    const [listenerId, setListenerId] = useState<string | null>(null);
    const [log, setLog] = useState(t.ready);
    const [busy, setBusy] = useState<string | null>(null);
    const refresh = useCallback(async (id: string | null) => {
        if (!id) return;
        try {
            setListener(await request<ListenerData>(`/api/rtsp-training/listener/status?listener_id=${encodeURIComponent(id)}`));
        } catch (error) {
            setLog(`Estado no disponible: ${error instanceof Error ? error.message : String(error)}`);
        }
    }, []);
    useEffect(() => {
        request<ListenerData & {active?: boolean}>('/api/rtsp-training/listener/active').then((data) => {
            if (data.active && data.listener_id) {
                setListenerId(data.listener_id);
                setListener(data);
                setTab('stream');
            }
        }).catch(() => {
        });
    }, []);
    useEffect(() => {
        if (!listenerId) return undefined;
        const timer = setInterval(() => refresh(listenerId), 3000);
        return () => clearInterval(timer);
    }, [listenerId, refresh]);
    const start = async (path: string, body: JsonObject, operation: string): Promise<ListenerData | null> => {
        setBusy(operation);
        try {
            const data = await post<ListenerData>(path, body);
            setListener(data);
            if (data.listener_id) setListenerId(data.listener_id);
            setLog(JSON.stringify(data, null, 2));
            return data;
        } catch (error) {
            setLog(`Operación fallida: ${error instanceof Error ? error.message : String(error)}`);
            return null;
        } finally {
            setBusy(null);
        }
    };
    const operationText = (key: 'validate' | 'connect' | 'disconnect' | 'train'): string => {
        const messages: Record<Language, Record<string, string>> = {
            es: {validate: 'Validando audio y metadata…', connect: 'Conectando y escuchando RTSP…', disconnect: 'Desconectando el listener…', train: 'Generando dataset y entrenando CNN1D-VAE…'},
            en: {validate: 'Validating audio and metadata…', connect: 'Connecting and listening to RTSP…', disconnect: 'Disconnecting listener…', train: 'Building dataset and training CNN1D-VAE…'},
            'pt-PT': {validate: 'A validar áudio e metadados…', connect: 'A ligar e escutar RTSP…', disconnect: 'A desligar o listener…', train: 'A gerar o lote e a treinar CNN1D-VAE…'}
        };
        return messages[language][key];
    };
    const validate = () => start('/api/rtsp-training/validate', {...config, listener_id: listenerId}, operationText('validate')).then((result) => {
        if (!result) return;
        const received = result.received_configuration as JsonObject | undefined;
        if (received) {
            setConfig(current => ({
                ...current,
                rtsp_url: typeof received.rtsp_url === 'string' ? received.rtsp_url : current.rtsp_url,
                transport: typeof received.transport === 'string' ? received.transport : current.transport,
                training_policy: typeof received.training_policy === 'string' ? received.training_policy : current.training_policy,
                min_training_seconds: Number(received.min_training_seconds ?? current.min_training_seconds),
            }));
        }
        setTab('stream');
    });
    const connect = () => start('/api/rtsp-training/listener/start', config, operationText('connect')).then((result) => { if (result) setTab('stream'); });
    const disconnect = async () => {
        await start('/api/rtsp-training/listener/stop', {listener_id: listenerId}, operationText('disconnect'));
        setListenerId(null);
    };
    const train = () => start('/api/rtsp-training/train', {
        listener_id: listenerId,
        epochs: 60,
        device: 'auto'
    }, operationText('train')).then((result) => { if (result) setTab('training'); });
    const changeLanguage = (value: string) => {
        if (!(value in TEXT)) return;
        const selected = value as Language;
        setLanguage(selected);
        localStorage.setItem('rtspUiLanguage', selected);
    };
    return <main>
        <section className="panel">
            <div className="language-bar"><label>{t.language}<Select className="language-select" value={language} options={[{value: 'es', label: 'Español'}, {value: 'en', label: 'English'}, {value: 'pt-PT', label: 'Português'}]} onChange={changeLanguage} ariaLabel={t.language}/></label></div>
            <h1>{t.title}</h1><p className="intro">{t.intro}</p>
            <nav
                className="tabs">{[['config', t.config], ['stream', t.stream], ['training', t.training]].map(([key, label]) =>
                <button className={tab === key ? 'tab active' : 'tab'} onClick={() => setTab(key)}
                        key={key}>{label}</button>)}</nav>
            {tab === 'config' &&
                <ConfigurationView {...{config, setConfig, listenerId, t, log, setLog, busy}} onValidate={validate}
                               onConnect={connect} onDisconnect={disconnect} onTrain={train}/>}{tab === 'stream' &&
            <StreamView listener={listener} t={t}/>}{tab === 'training' && <TrainingView listener={listener} t={t}/>} {busy && <LoadingOverlay message={busy} language={language}/>}</section>
    </main>;
}

createRoot(document.getElementById('root')!).render(<App/>);
