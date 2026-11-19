La solución real no es quitar el bloqueo: hay que generar un checkpoint nuevo cuyo contrato coincida con los datos del stream. Voy a revisar el modelo, el entrenador y el formato actual de `aligned_training.jsonl` para precisar qué falta y evitar proponer un cambio incompatible.

## Problema real

El problema no está en el RTSP ni en la metadata. El problema es que actualmente hay dos contratos diferentes:

### Datos recibidos del stream

```text
Entrada:
- Audio PCM
- Contexto CNC

Labels esperados:
broken
very_bad
bad
sufficient
good
excellent
```

### Checkpoint actual

```text
Entrada:
- Solo audio
- context_dim = 0

Labels:
good
sufficient
bad
very_bad
```

Por eso no se debe modificar simplemente `training_request.json` para quitar el bloqueo. El checkpoint actual fue entrenado con otra arquitectura lógica, otra cantidad de clases y sin CNC.

Además, su metadata indica que las clases actuales son sintéticas:

```text
good, sufficient, bad, very_bad
```

No representan todavía calidad real de mecanizado.

## Solución correcta

Hay que crear un nuevo dataset multimodal y entrenar un nuevo checkpoint CNN1D-VAE.

El repositorio ya tiene parte de la capacidad necesaria:

- [`CNN1D-VAE/classifier.py`](D:/proyecto/source/repos/MachiningAssistant/CNN1D-VAE/classifier.py) acepta `context_dim > 0`.
- [`fourclass_training.py`](D:/proyecto/source/repos/MachiningAssistant/fourclass_training.py) acepta `--cnc-context`.
- [`cnc_features.py`](D:/proyecto/source/repos/MachiningAssistant/Dataset/Preprocessing/cnc_features.py) convierte el contexto CNC a un vector numérico.
- [`fourclass_prediction.py`](D:/proyecto/source/repos/MachiningAssistant/fourclass_prediction.py) ya sabe inferir con audio + CNC.

Lo que falta es conectar correctamente el stream con el entrenador.

## Flujo que debe implementarse

```text
RTSP
 ├── Audio
 │    └── Ventana WAV de 5 segundos
 │
 └── x-machining-metadata
      ├── Contexto CNC
             │
             ▼
      Alineación temporal
             │
             ▼
      Dataset multimodal
      ├── audio/*.wav
      ├── samples.csv
      ├── labels.json
      └── contexts.json
             │
             ▼
      Entrenamiento CNN1D-VAE
             │
             ▼
      Nuevo best.pt
```

## Qué debe contener el nuevo dataset

### `labels.json`

```json
{
  "0": "broken",
  "1": "very_bad",
  "2": "bad",
  "3": "sufficient",
  "4": "good",
  "5": "excellent"
}
```

### `samples.csv`

Cada ventana debe tener una fila similar a:

```csv
sample_id,split,wav_file,label,label_index,frame_count
stream_000001,train,audio/stream_000001.wav,broken,0,220500
```

El `label_index` es el valor que realmente recibe `CrossEntropyLoss`.

### `contexts.json`

Cada muestra debe tener su contexto CNC asociado:

```json
[
  {
    "sample_id": "stream_000001",
    "context": {
      "SchemaVersion": 1,
      "ContextId": "...",
      "MachineId": "...",
      "DriverSessionId": "...",
      "OccurredAtUtc": "...",
      "ActualSpindleRpm": 2546,
      "ActualFeedMmPerMin": 4962,
      "SPowerPercent": 40,
      "WPowerPercent": 30,
      "ToolDiameterMm": 25,
      "ToolTeethCount": 4,
      "ToolRemainingUsePercent": 80
    }
  }
]
```

El modelo no recibe directamente el JSON completo. Recibe el vector numérico normalizado definido por `FEATURE_NAMES` en [`cnc_features.py`](D:/proyecto/source/repos/MachiningAssistant/Dataset/Preprocessing/cnc_features.py:80).

## Entrenamiento del nuevo modelo

El entrenamiento debe ejecutarse como una nueva corrida, no como continuación del checkpoint actual:

```powershell
.\.venv\Scripts\python.exe fourclass_training.py `
  --data-dir runs\rtsp\datasets\<dataset_id> `
  --csv runs\rtsp\datasets\<dataset_id>\samples.csv `
  --cnc-context runs\rtsp\datasets\<dataset_id>\contexts.json `
  --output-dir runs\train\rtsp_multimodal\<run_id> `
  --epochs 20 `
  --device cpu
```

El nuevo checkpoint debe guardar:

```json
{
  "model_family": "CNN1D-VAE",
  "task": "classification",
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

El valor exacto de `context_dim` debe coincidir con la cantidad de características definidas en `FEATURE_NAMES`.

## Por qué no basta con los 10 segundos actuales

Los 10 segundos actuales representan solamente dos ventanas etiquetadas:

```text
30–35 segundos → broken
65–70 segundos → broken
```

Eso no permite entrenar seis clases. Solo existe evidencia de una clase: `broken`.

Para entrenar un clasificador de seis clases se necesita recopilar ventanas correspondientes a todas las clases. Como recomendación inicial de prueba, cada clase debería tener múltiples ventanas separadas en `train`, `val` y `test`; con una sola ventana por clase no se puede evaluar generalización de forma fiable.

## Qué hacer con los procesos duplicados

No conviene resolverlo matando procesos manualmente cada vez. La solución correcta es que el monitor mantenga una única sesión activa:

1. Antes de iniciar una nueva escucha, comprobar si existe una sesión activa.
2. Reutilizarla o detenerla limpiamente.
3. Asociar audio, metadata y logs al mismo `listener_id`.
4. Impedir dos listeners simultáneos para la misma URL.
5. Al detener la sesión, actualizar correctamente:
   - `session.metadata.json`
   - `metadata.status.json`
   - `training_request.json`

## Qué no debe hacerse

No recomiendo:

- Cambiar manualmente `context_dim` de `0` a `7` dentro del JSON.
- Cambiar las clases del checkpoint sin reentrenar.
- Usar la predicción `good` como label.
- Convertir `quality_score` directamente en una probabilidad.
- Entrenar con solo las dos ventanas actuales.
- Mezclar las clases sintéticas antiguas con las clases reales del stream.
- Lanzar entrenamiento mientras todavía existen listeners duplicados.

## Resultado esperado después de corregirlo

El botón de entrenamiento debería hacer esto:

1. Verificar que existe una única sesión activa.
2. Verificar audio recibido.
3. Verificar metadata CNC válida.
4. Verificar labels `quality` válidos.
5. Verificar que hay datos para todas las clases.
6. Construir el dataset.
7. Iniciar una nueva corrida CNN1D-VAE multimodal.
8. Mostrar métricas:
   - pérdida de entrenamiento;
   - pérdida de validación;
   - accuracy;
   - macro-F1;
   - matriz de confusión;
   - métricas por clase;
   - `context_dim`;
   - checkpoint generado.

La corrección real consiste, por tanto, en implementar el pipeline `stream → dataset multimodal → nuevo entrenamiento → nuevo checkpoint`, no en desbloquear artificialmente el checkpoint actual.

