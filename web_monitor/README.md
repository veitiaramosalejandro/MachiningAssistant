# Machining Assistant Web Monitor

Aplicación web local sin Streamlit. El backend Python mantiene la API key fuera del navegador.

## Configuración

Definir en el entorno del proceso:

```powershell
$env:MACHINING_MONITOR_URL = "http://localhost:8888"
$env:MACHINING_MONITOR_API_KEY = "<clave fuera del repositorio>"
$env:MACHINING_MONITOR_DATA_ROOT = "D:\proyecto\source\repos\REP_MachiningMonitor\machiningmonitor-data"
```

Si se omite `MACHINING_MONITOR_DATA_ROOT`, se usa esa ruta por defecto relativa al workspace padre.

## Arranque

```powershell
& .venv\Scripts\python.exe web_monitor.py
```

Abrir `http://127.0.0.1:8765`.

## Persistencia PostgreSQL + Qdrant

El backend lee `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`,
`POSTGRES_DB` y `QDRANT_URL` desde `.env`. PostgreSQL guarda la fuente relacional
(grabaciones, JSON normalizados, entrenamientos y ventanas); Qdrant guarda los
vectores `latent_mu` del CNN1D-VAE en colecciones separadas por dimensión latente.
La salud se consulta en `GET /api/storage/health` y la interfaz la muestra como
`READY` o `PARTIAL`.

Las importaciones se persisten al usar `/api/import`, `/api/import-local` o
`/api/import-local-recording`. Al terminar correctamente un entrenamiento CNN,
se indexan en Qdrant las ventanas existentes en `window_predictions.csv` y se
guardan sus resultados en PostgreSQL. El entrenamiento LSTM mantiene sus
resultados relacionales, pero todavía no genera embeddings Qdrant.

La interfaz permite consultar el servicio, seleccionar al menos tres grabaciones completas, preparar una copia de entrenamiento a 44,1 kHz y lanzar CNN1D-VAE o LSTM-VAE con progreso y logs.

También incorpora la fuente local `Dataset/Data`. El botón `Preparar Dataset/Data completo` inspecciona todas sus carpetas válidas (`metadata.json` + `audio_*.wav`), convierte el audio a PCM16 estéreo 44,1 kHz y genera una importación reproducible. Las carpetas que pertenecen al mismo `SourceRecording.Folder` se mantienen en el mismo split para evitar fuga entre train, validation y test.

Al finalizar el entrenamiento, la vista de resultados une cada ventana con el contexto temporal disponible en `contexts.json`, eventos coincidentes de `events.json`, offsets de `playback.json` y procedencia de `metadata.json`. Muestra las curvas train/validation/test, baseline, score de referencia del train, operaciones con sus scores y las ventanas con mayor reconstrucción MSE. La referencia p99 se presenta solo como señal estadística; no es todavía un umbral operativo calibrado.

La configuración de entrenamiento se selecciona mediante listas cerradas en la interfaz. Incluye modelo, longitud de ventana, dimensión latente, dispositivo, épocas, ventanas por grabación, batch, learning rate, beta del VAE y frecuencia de evaluación. El backend valida las opciones antes de lanzar el proceso y guarda la configuración elegida en `training_config.json`.
