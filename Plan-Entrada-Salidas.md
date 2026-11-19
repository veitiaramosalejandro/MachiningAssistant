Prepararé un plan de implementación por fases, manteniendo `CNN1D-VAE` como rama de audio y separando claramente el flujo actual de reconstrucción de la futura clasificación con parámetros CNC. Incluiré contratos, datos, entrenamiento, inferencia, pruebas y criterios de aceptación; no modificaré código en esta respuesta.

## Objetivo

Convertir el `CNN1D-VAE` actual en un modelo supervisado que reciba:

```text
audio + parámetros CNC
```

y produzca:

```text
labels + puntuaciones por label
```

Manteniendo la arquitectura CNN1D-VAE como rama principal de audio.

La propuesta asume que los parámetros CNC son numéricos y constantes durante cada ventana de audio.

---

## Arquitectura recomendada

No modificaría los canales de audio de la CNN. Mantendría:

```text
Audio estéreo
(batch, sequence_length, 2)
        ↓
Encoder CNN1D-VAE
        ↓
mu_audio
```

Y añadiría una rama paralela:

```text
Parámetros CNC normalizados
        ↓
ContextEncoder / MLP
        ↓
embedding_context
```

Después se combinan:

```text
concat(mu_audio, embedding_context)
        ↓
capa clasificadora
        ↓
logits
        ↓
softmax
        ↓
labels
```

Esquema:

```text
                    ┌── CNN1D-VAE encoder ── mu_audio ──┐
Audio PCM ──────────┤                                     ├─ Fusión ── Clasificador ── Labels
                    └── Context encoder ─ embedding ────┘
Parámetros CNC ────┘
```

Así la CNN sigue procesando audio y los parámetros CNC no se tratan artificialmente como canales temporales.

---

# Plan de trabajo

## Fase 0: Definir el contrato de negocio

Antes de tocar código hay que documentar exactamente:

### Entradas numéricas

Por ejemplo:

```text
spindle_speed_rpm
feed_rate_mm_per_min
depth_of_cut_mm
spindle_load_percent
temperature_celsius
```

Debe definirse para cada parámetro:

- nombre oficial;
- unidad;
- rango válido;
- origen;
- si es obligatorio;
- si es constante o variable en el tiempo.

### Labels

Debe aclararse si serán:

```text
good
sufficient
bad
very_bad
```

o etiquetas reales como:

```text
normal
tool_wear
chatter
overload
surface_defect
```

También hay que decidir si cada ventana tiene:

```text
una sola clase
```

o varias etiquetas independientes.

La primera versión debería utilizar una sola clase por ventana.

### Criterio de aceptación

Ejemplo:

```text
Cada ventana de audio debe tener exactamente un conjunto de parámetros CNC
y exactamente un label validado.
```

---

## Fase 1: Auditar y preparar los datos

Actualmente `fourclass_data.py` devuelve esencialmente:

```text
signal + label_index
```

La nueva muestra debe tener conceptualmente:

```text
audio + cnc_parameters + label
```

Hay que decidir dónde vivirán los parámetros:

### Alternativa recomendada

Un manifiesto por ventana:

```text
sample_id
wav_file
split
label
label_index
spindle_speed_rpm
feed_rate_mm_per_min
depth_of_cut_mm
...
```

O un JSON relacionado con cada grabación:

```text
recording/
├── audio_000000.wav
├── contexts.json
└── events.json
```

En ambos casos debe poder determinarse qué contexto corresponde a cada ventana.

### Validaciones necesarias

- WAV correcto;
- frecuencia de muestreo correcta;
- número de frames correcto;
- parámetros no nulos cuando sean obligatorios;
- valores dentro de rango;
- unidades consistentes;
- label válido;
- `label` y `label_index` coherentes;
- ausencia de información futura;
- ausencia de fuga entre train, validation y test.

La división debe hacerse por grabaciones o sesiones completas, no por ventanas aleatorias de una misma grabación.

---

## Fase 2: Definir la alineación temporal

Para cada ventana debe conocerse:

```text
start_frame
end_frame
start_time
end_time
```

Y se debe obtener el contexto vigente en ese intervalo.

Hay dos casos:

### Parámetros constantes

Ejemplo:

```text
RPM = 8500 durante toda la ventana
```

Se guarda un único valor por ventana.

### Parámetros variables

Ejemplo:

```text
RPM(t)
feed(t)
load(t)
```

Se requiere una serie temporal alineada con los frames del audio.

Para la primera versión recomiendo trabajar únicamente con parámetros constantes por ventana. La variante temporal puede quedar como segunda fase.

---

## Fase 3: Diseñar la entrada multimodal

La entrada conceptual sería:

```text
audio_tensor:
(batch, sequence_length, 2)

context_tensor:
(batch, num_numeric_features)
```

Ejemplo:

```text
audio_tensor:
(8, 2048, 2)

context_tensor:
(8, 4)
```

con cuatro parámetros CNC.

Los parámetros numéricos deben normalizarse usando únicamente estadísticas del conjunto de entrenamiento:

```text
media_train
desviación_train
```

El checkpoint deberá guardar:

```text
feature_names
feature_units
feature_mean
feature_scale
missing_value_policy
context_schema_version
```

Sin estos datos no se podría reproducir correctamente la inferencia.

---

## Fase 4: Integrar la rama CNC en CNN1D-VAE

El `CNN1D-VAE` actual ya genera:

```text
mu_audio
```

La nueva rama tendría conceptualmente:

```text
context_tensor
    ↓
MLP
    ↓
context_embedding
```

Luego:

```text
fused = concatenate(mu_audio, context_embedding)
```

Y finalmente:

```text
logits = classifier(fused)
```

La salida sería:

```text
(batch, num_classes)
```

Por ejemplo:

```text
(8, 4)
```

No se deberían añadir inicialmente los parámetros CNC como canales de la convolución porque:

- RPM y avance no son audio;
- los parámetros estáticos no son señales temporales;
- se mezclarían escalas y unidades diferentes;
- se rompería innecesariamente el contrato `(batch, time, 2)`.

---

## Fase 5: Definir la función de entrenamiento

Para clasificación, la pérdida principal sería:

```text
CrossEntropyLoss(logits, label_index)
```

Si se conserva la parte VAE de reconstrucción, se podría utilizar:

```text
loss_total =
    loss_classification
    + λ * loss_reconstruction
    + β * loss_KL
```

Conceptualmente:

```text
clasificación → aprender labels
reconstrucción → conservar representación acústica
KL            → regularizar el espacio latente
```

Para empezar recomiendo dos experimentos separados:

### Experimento A: clasificación audio-only

```text
audio → CNN1D-VAE → label
```

### Experimento B: clasificación audio + CNC

```text
audio + parámetros CNC → CNN1D-VAE multimodal → label
```

La comparación indicará si los parámetros CNC realmente aportan información.

---

## Fase 6: Definir la inferencia

El `predict.py` actual recibe:

```text
--checkpoint
--wav
```

Para el nuevo modelo tendría que existir conceptualmente:

```text
--checkpoint
--wav
--context
```

Por ejemplo, un contexto JSON:

```json
{
  "spindle_speed_rpm": 8500,
  "feed_rate_mm_per_min": 1200,
  "depth_of_cut_mm": 0.8,
  "spindle_load_percent": 42
}
```

Durante inferencia:

```text
WAV + contexto
    ↓
validación
    ↓
normalización
    ↓
CNN1D-VAE + rama CNC
    ↓
logits
    ↓
probabilidades
    ↓
predicted_label
```

El label real no se proporciona durante inferencia. Solo se proporciona durante entrenamiento y evaluación.

La salida debería incluir:

```json
{
  "predicted_label": "bad",
  "predicted_class_index": 2,
  "class_probabilities": {
    "good": 0.05,
    "sufficient": 0.10,
    "bad": 0.78,
    "very_bad": 0.07
  },
  "context_used_by_model": true,
  "context_schema_version": "1"
}
```

---

## Fase 7: Versionar checkpoints

Los checkpoints actuales fueron creados para entradas de audio solamente.

No deberían reutilizarse silenciosamente.

El nuevo checkpoint debería distinguirse mediante metadatos:

```text
model_family: CNN1D-VAE
task: classification
input_mode: audio_plus_numeric_context
context_features: [...]
num_classes: [...]
class_names: [...]
checkpoint_format_version: ...
```

Un checkpoint antiguo debería rechazarse si no contiene el esquema de contexto requerido.

---

## Fase 8: Validación técnica

### Pruebas unitarias

Comprobar:

- forma del audio;
- forma del contexto;
- número de parámetros;
- tipos `float32`;
- normalización;
- valores nulos;
- valores infinitos;
- rangos;
- orden de las features;
- número de logits;
- orden de los labels;
- carga y guardado del checkpoint.

### Pruebas de integración

Comprobar:

```text
dataset → dataloader → modelo → pérdida → checkpoint → inferencia
```

También:

```text
WAV + contexto → predicción JSON
```

### Pruebas de fuga

Comparar:

- train;
- validation;
- test.

Y verificar que una misma grabación original no aparezca en varios conjuntos.

---

## Fase 9: Evaluación del modelo

No sería suficiente observar una probabilidad alta.

Se deben medir:

- accuracy;
- precision por clase;
- recall por clase;
- F1 por clase;
- macro F1;
- matriz de confusión;
- soporte por clase;
- rendimiento por máquina;
- rendimiento por operación;
- rendimiento por régimen de RPM;
- rendimiento con parámetros ausentes.

También comparar:

```text
audio-only
context-only
audio + contexto
```

La solución multimodal solo debería aceptarse si mejora de forma consistente frente al modelo audio-only y no solamente en un subconjunto de datos.

---

## Fase 10: Calibración y despliegue

Las probabilidades `softmax` no deben llamarse calibradas automáticamente.

Después de validar el clasificador habría que evaluar:

- calibración;
- Expected Calibration Error;
- Brier score;
- umbrales de abstención;
- comportamiento fuera de distribución.

Si faltan parámetros CNC durante producción, el modelo debe:

```text
rechazar la predicción
```

o devolver:

```text
datos insuficientes
```

No debería reemplazar silenciosamente los valores faltantes por cero sin una política explícita.

---

# Archivos que participarían conceptualmente

| Área | Archivo actual |
|---|---|
| Modelo CNN | `CNN1D-VAE/model.py` |
| Entrada WAV | `CNN1D-VAE/predict.py` |
| Inferencia reconstrucción | `CNN1D-VAE/runtime.py` |
| Datos WAV y normalización | `Dataset/Preprocessing/company_data.py` |
| Dataset clasificado | `Dataset/Preprocessing/fourclass_data.py` |
| Entrenamiento clasificación | `fourclass_training.py` |
| Clasificador actual | `CNN1D-VAE/classifier.py` |
| Etiquetas | `labels.json` y checkpoint |
| Contexto CNC existente | `contexts.json`, `events.json` o contrato gRPC |

## Recomendación final

Implementaría en este orden:

```text
1. Definir parámetros y labels reales.
2. Validar que cada ventana tiene contexto y etiqueta.
3. Crear un modelo de referencia audio-only.
4. Añadir una rama MLP para parámetros CNC.
5. Fusionar audio y contexto en el espacio latente.
6. Entrenar con clasificación supervisada.
7. Comparar audio-only contra audio+CNC.
8. Añadir inferencia con WAV + contexto.
9. Versionar el checkpoint.
10. Validar por sesión, máquina y régimen operativo.
```

La decisión arquitectónica principal sería:

```text
CNN1D-VAE para audio
+
MLP para parámetros CNC
+
clasificador supervisado
```

El `predict.py` actual puede conservarse para reconstrucción, mientras que la nueva inferencia de etiquetas debería identificarse explícitamente como una tarea de clasificación basada en el mismo backbone CNN1D-VAE.