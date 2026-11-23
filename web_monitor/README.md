# Machining Assistant Web Monitor

Aplicación web local sin Streamlit. El backend Python mantiene la API key fuera del navegador.

La interfaz está implementada con React + Vite. El código fuente está en `src/` y la
compilación de producción se genera en `dist/`.

## Configuración

Definir en el entorno del proceso:

```powershell
$env:MACHINING_MONITOR_URL = "http://localhost:8888"
$env:MACHINING_MONITOR_API_KEY = "<clave fuera del repositorio>"
$env:MACHINING_MONITOR_DATA_ROOT = "D:\proyecto\source\repos\REP_MachiningMonitor\machiningmonitor-data"
```

Si se omite `MACHINING_MONITOR_DATA_ROOT`, se usa esa ruta por defecto relativa al workspace padre.

## Arranque

Instalar dependencias y compilar la interfaz:

```powershell
cd web_monitor
npm install
npm run build
cd ..
```

```powershell
& .venv\Scripts\python.exe web_monitor.py
```

Para desarrollo visual, arrancar el backend en `8765` y, en otra terminal:

```powershell
cd web_monitor
npm run dev
```

Vite reenvía `/api` al backend Python. El backend sirve automáticamente `dist/`
cuando existe; si no hay compilación, usa los assets del directorio fuente.

El monitor escucha por defecto en todas las interfaces (`0.0.0.0:8765`). Desde
otra máquina de la red, abrir `http://IP_DEL_EQUIPO:8765`, sustituyendo la IP
por la dirección IPv4 del equipo donde se ejecuta el proyecto.

Si Windows Firewall bloquea el puerto, hay que permitir explícitamente TCP
8765 para la red privada. No se recomienda publicar este puerto directamente
en Internet: el monitor todavía no tiene autenticación propia para sus
endpoints de importación y entrenamiento.

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

La duración de ventana se selecciona en milisegundos (`50`, `100`, `200` o `500`). El backend la convierte a muestras a 44,1 kHz y redondea al múltiplo de 64 requerido por el CNN1D-VAE; la configuración conserva la duración solicitada, la duración real y el `sequence_length` resultante.
