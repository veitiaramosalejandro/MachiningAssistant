# Corrección completa del entrenamiento RTSP multimodal

## Estado

La corrección está implementada en el repositorio, pero el entrenamiento real
solo puede completarse cuando el stream proporcione suficientes ventanas con
las seis clases de calidad y con todos los campos CNC requeridos.

La solución no modifica artificialmente el checkpoint antiguo. Genera un nuevo
dataset y un nuevo checkpoint CNN1D-VAE multimodal.

## Problema corregido

El checkpoint histórico tiene este contrato:

```text
Entrada: audio estéreo solamente
context_dim: 0
Clases: good, sufficient, bad, very_bad
```

Ese checkpoint no puede aceptar contexto CNC ni las nuevas etiquetas de
calidad. El stream necesita otro contrato:

```text
Entradas:
- audio PCM estéreo en ventanas completas de 5 segundos;
- contexto CNC validado y normalizado.

Labels:
- quality 0 -> broken -> score 0.0;
- quality 1 -> very_bad -> score 0.2;
- quality 2 -> bad -> score 0.4;
- quality 3 -> sufficient -> score 0.6;
- quality 4 -> good -> score 0.8;
- quality 5 -> excellent -> score 1.0.
```

Las predicciones del checkpoint histórico nunca se usan como labels.

## Flujo implementado

```text
RTSP
├── pista de audio
│   └── FFmpeg -> PCM16LE 44100 Hz estéreo
│       └── ventanas completas de 220500 frames
│           └── audio_windows/window_*.wav
│
└── pista x-machining-metadata
    ├── snapshots de contexto CNC
    └── eventos con quality 0..5
        │
        ▼
    alineación temporal audio + metadata
        │
        ▼
    dataset multimodal validado
        ├── audio/*.wav
        ├── labels.json
        ├── samples.csv
        ├── contexts.json
        └── dataset_report.json
        │
        ▼
    CNN1D-VAE con audio + contexto CNC
        │
        ▼
    best.pt, metrics.json, history.json y test_predictions.csv
```

## Componentes implementados

### Captura de audio

[`Integration/rtsp_stream_service.py`](D:/proyecto/source/repos/MachiningAssistant/Integration/rtsp_stream_service.py)
guarda cada ventana completa como WAV cuando recibe
`MACHINING_RTSP_AUDIO_WINDOWS_OUTPUT`.

[`classification_streaming.py`](D:/proyecto/source/repos/MachiningAssistant/classification_streaming.py)
expone `window_callback` para capturar la ventana sin alterar la clasificación
ni mezclar ventanas incompletas.

### Labels

[`Integration/stream_quality.py`](D:/proyecto/source/repos/MachiningAssistant/Integration/stream_quality.py)
contiene el contrato único de clases y puntuaciones.

La alineación en [`web_monitor.py`](D:/proyecto/source/repos/MachiningAssistant/web_monitor.py)
usa únicamente:

```text
x-machining-metadata.event.quality
```

Un evento sin `quality` entero entre 0 y 5 se rechaza como label. No se
convierte una confianza softmax en una etiqueta.

### Contexto CNC

Los snapshots se convierten al contrato canónico mediante
[`Integration/rtsp_context.py`](D:/proyecto/source/repos/MachiningAssistant/Integration/rtsp_context.py).

Después se validan con
[`Dataset/Preprocessing/cnc_features.py`](D:/proyecto/source/repos/MachiningAssistant/Dataset/Preprocessing/cnc_features.py).

Las siete características numéricas actuales son:

```text
ActualSpindleRpm
ActualFeedMmPerMin
SPowerPercent
WPowerPercent
ToolDiameterMm
ToolTeethCount
ToolRemainingUsePercent
```

Si falta una característica obligatoria, la muestra se rechaza. No se
rellenan campos faltantes con cero ni con valores inventados.

### Construcción del dataset

[`Integration/rtsp_training_dataset.py`](D:/proyecto/source/repos/MachiningAssistant/Integration/rtsp_training_dataset.py)
materializa el dataset asociado a una sesión:

```text
runs/rtsp/web_sessions/<session_id>/training_dataset/
├── audio/
├── labels.json
├── samples.csv
├── contexts.json
└── dataset_report.json
```

Cada muestra debe tener WAV existente, label de `quality` válido, contexto CNC
válido, `sample_id` único y `label_index` consistente con `labels.json`.

El dataset se considera listo únicamente si cada una de las seis clases tiene
al menos tres muestras: una para `val`, una para `test` y al menos una para
`train`. Esta regla es un mínimo operativo; más muestras por clase son
necesarias para evaluar generalización de forma fiable.

### Entrenamiento

El botón de la interfaz llama a `/api/rtsp-training/train`.

Cuando el dataset está listo, el monitor ejecuta internamente:

```powershell
.\.venv\Scripts\python.exe CNN1D-VAE\train_fourclass.py `
  --data-dir runs\rtsp\web_sessions\<session_id>\training_dataset `
  --csv runs\rtsp\web_sessions\<session_id>\training_dataset\samples.csv `
  --cnc-context runs\rtsp\web_sessions\<session_id>\training_dataset\contexts.json `
  --output-dir runs\rtsp\web_sessions\<session_id>\training_run `
  --epochs 20 `
  --device auto
```

El entrenamiento es una corrida nueva. No se permite continuar el checkpoint
audio-only porque cambiarían las clases y la arquitectura de entrada.

El nuevo checkpoint debe contener, como mínimo:

```json
{
  "task": "classification",
  "model_family": "CNN1D-VAE",
  "context_dim": 7,
  "class_names": [
    "broken",
    "very_bad",
    "bad",
    "sufficient",
    "good",
    "excellent"
  ]
}
```

El checkpoint conserva además el normalizador CNC calculado únicamente con
las muestras de entrenamiento.

### Resultado visible

La interfaz muestra audio recibido, mensajes, snapshots y eventos metadata,
ventanas alineadas, muestras aceptadas y rechazadas, motivos de rechazo,
estado del dataset, estado del entrenamiento, log del entrenador, `best.pt` y
métricas de validation y test.

Los artefactos principales son:

```text
aligned_training.jsonl
training_dataset/dataset_report.json
training_request.json
training.log
training_run/best.pt
training_run/metrics.json
training_run/history.json
training_run/test_predictions.csv
```

## Control de sesiones duplicadas

El monitor comprueba, dentro de su proceso, si ya existe una escucha activa y
rechaza iniciar otra. Audio, metadata y entrenamiento se asocian al mismo
`listener_id`.

Si existen dos procesos antiguos de `web_monitor.py` iniciados por separado,
deben detenerse manualmente y reiniciar una única instancia. La protección
del nuevo código no puede eliminar procesos creados por otra instancia ya
iniciada.

Una sesión antigua tampoco adquiere WAV retroactivamente. Para usar esta
corrección se debe iniciar una sesión nueva después de reiniciar el monitor.

## Criterios de bloqueo correctos

El entrenamiento permanece bloqueado cuando no hay audio, eventos metadata,
duración mínima, WAV, contexto CNC o características CNC obligatorias; cuando
falta alguna de las seis clases; cuando no se pueden crear `train`, `val` y
`test`; o cuando el dataset contiene labels inconsistentes.

Estos bloqueos protegen contra entrenar con datos incompletos o con labels
generados por el propio modelo.

## Lo que no debe hacerse

- No editar manualmente `context_dim` dentro de un checkpoint.
- No cambiar `class_names` de un checkpoint ya entrenado.
- No usar `predicted_label` como target.
- No convertir la confianza softmax en calidad industrial.
- No usar `quality_score` como probabilidad del modelo.
- No entrenar con las dos ventanas antiguas de 10 segundos.
- No mezclar las clases sintéticas históricas con las etiquetas reales del
  stream.
- No aceptar contextos CNC incompletos rellenando valores arbitrarios.
- No ejecutar dos listeners sobre la misma fuente.

## Verificación realizada

```text
py_compile: correcto
git diff --check: correcto
tests de streaming, CNC multimodal y features: 32 passed
```

La validación de código no equivale todavía a un entrenamiento real. El
entrenamiento real requiere una nueva sesión RTSP con audio, contextos CNC
completos y muestras distribuidas entre las seis clases.

## Próximo paso operativo

1. Detener las instancias antiguas de `web_monitor.py` y listeners RTSP.
2. Reiniciar una única instancia del monitor.
3. Iniciar una nueva sesión RTSP.
4. Mantener la captura hasta disponer de las seis clases y contextos válidos.
5. Detener la escucha.
6. Pulsar el botón de entrenamiento.
7. Revisar `dataset_report.json`, `training.log` y `training_run/metrics.json`.

El nuevo `best.pt` debe configurarse como checkpoint activo únicamente después
de revisar sus métricas y confirmar que su contrato contiene audio + CNC y las
seis clases esperadas.
