## Pasos recomendados

### Fase 1: Preparar y validar los datos

1. Leer las grabaciones WAV.
2. Asociar cada grabación con sus archivos JSON:
   - `metadata`
   - `contexts`
   - `events`
   - `playback`
3. Validar que audio y contexto correspondan a la misma grabación.
4. Normalizar:
   - frecuencia de muestreo;
   - canales;
   - amplitud;
   - valores nulos;
   - unidades de los datos CNC.
5. Separar las grabaciones completas en:

```text
train/
validation/
test/
```

La separación debe hacerse por grabación completa, no por ventanas mezcladas.

### Fase 2: Entrenar el modelo acústico

Entrenar `CNN1D-VAE` o `LSTM-VAE` usando principalmente grabaciones representativas del comportamiento normal.

Registrar por ejecución:

- configuración del modelo;
- número de grabaciones;
- número de ventanas;
- duración de las grabaciones;
- MSE de entrenamiento;
- MSE de validación;
- MSE de prueba;
- mejor época;
- dispositivo utilizado;
- versión del dataset.

El modelo debe guardar también su arquitectura y configuración exacta.

### Fase 3: Construir la línea base normal

Con el conjunto de entrenamiento calcular:

- MSE medio;
- desviación estándar;
- percentil 90;
- percentil 95;
- percentil 99;
- distribución por máquina;
- distribución por operación;
- distribución por tipo de contexto.

Ejemplo de clasificación:

```text
MSE <= p95       Normal
p95 < MSE <= p99 Revisión
MSE > p99        Anomalía
```

Los umbrales deben calcularse con datos de entrenamiento y validarse posteriormente. No deben definirse arbitrariamente.

### Fase 4: Analizar una grabación nueva

Para cada grabación:

1. Dividir el audio según `sequence_length`.
2. Ejecutar el VAE.
3. Calcular el MSE de cada ventana.
4. Generar el MSE global.
5. Detectar las ventanas anómalas.
6. Calcular el porcentaje de señal afectada.
7. Asociar cada ventana con su posición temporal.
8. Combinar los resultados con los JSON de contexto.

La salida debería indicar:

```text
Estado: revisión
MSE global: 0.91
Umbral p95: 0.74
Ventanas anómalas: 17 de 120
Porcentaje afectado: 14.1 %
Duración estimada: 85 segundos
```

### Fase 5: Crear características de análisis

Por cada grabación y ventana generar características como:

- MSE;
- máximo MSE;
- percentiles;
- duración;
- porcentaje de anomalía;
- frecuencia dominante;
- energía;
- RMS;
- factor de cresta;
- variación temporal;
- parámetros de máquina;
- velocidad;
- carga;
- herramienta;
- evento ocurrido;
- contexto de operación;
- embedding latente del VAE.

Estas características serán la entrada para los análisis posteriores.

### Fase 6: Persistir en PostgreSQL

PostgreSQL debería almacenar:

- grabaciones;
- metadatos;
- contextos;
- eventos;
- ejecuciones de entrenamiento;
- configuraciones;
- resultados globales;
- resultados por ventana;
- umbrales utilizados;
- modelo aplicado;
- versión del dataset.

Así se podrá responder:

- qué modelo analizó la grabación;
- con qué configuración;
- qué umbral se utilizó;
- qué datos produjeron la alerta;
- qué versión del entrenamiento estaba activa.

### Fase 7: Indexar embeddings en Qdrant

Guardar en Qdrant el embedding de cada ventana o grabación junto con sus metadatos:

```text
recording_id
machine_id
operation
tool_id
timestamp
status
mse
context
model_version
```

Después, para una grabación nueva:

1. Obtener su embedding.
2. Buscar grabaciones similares.
3. Filtrar por máquina, operación o herramienta.
4. Comparar el resultado actual con casos anteriores.
5. Mostrar ejemplos similares en la interfaz.

Qdrant debe utilizarse para encontrar casos parecidos, no para sustituir la métrica del VAE.

### Fase 8: Añadir clasificación supervisada

Cuando existan suficientes resultados revisados por expertos, incorporar un modelo como:

- CatBoost;
- LightGBM;
- XGBoost.

Este modelo podrá aprender categorías como:

```text
normal
desgaste
vibración
ruido externo
cambio de herramienta
posible rotura
```

No conviene incorporarlo antes de tener etiquetas consistentes.

### Fase 9: Incorporar el LLM

El LLM debe añadirse al final y recibir únicamente un paquete de evidencias:

- resultados del VAE;
- comparación con los umbrales;
- contexto de la grabación;
- eventos;
- casos similares;
- limitaciones;
- versión del modelo.

Su responsabilidad será:

- explicar;
- resumir;
- ordenar evidencias;
- generar hipótesis;
- recomendar verificaciones.

No debe:

- analizar directamente el WAV como fuente principal;
- decidir por sí solo si existe una avería;
- inventar probabilidades;
- ejecutar comandos;
- cambiar parámetros de la máquina.

## Orden concreto de trabajo

```text
1. Validar datos
2. Entrenar VAE
3. Calcular línea base
4. Analizar grabaciones nuevas
5. Generar estadísticas
6. Guardar resultados en PostgreSQL
7. Indexar embeddings en Qdrant
8. Comparar con grabaciones similares
9. Añadir etiquetas expertas
10. Entrenar clasificador supervisado
11. Incorporar LLM para explicación
```

## Criterios para considerar cada fase terminada

- El VAE reproduce resultados consistentes.
- Los umbrales se calculan automáticamente.
- Una grabación nueva puede analizarse sin reentrenar.
- Cada resultado tiene trazabilidad hacia la grabación y la ventana.
- PostgreSQL conserva el historial completo.
- Qdrant recupera casos similares relevantes.
- La interfaz muestra métricas y evidencias.
- El sistema puede indicar `normal`, `revisión`, `anómalo` o `fuera de distribución`.
- El LLM, cuando se incorpore, solo interpreta datos estructurados.

La prioridad inmediata debería ser completar las fases 1 a 5. Sin una línea base fiable y datos bien asociados, el LLM produciría explicaciones aparentemente razonables pero poco confiables.

No se ha modificado ningún archivo ni configuración.

