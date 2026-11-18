A continuación se detalla lo que falta por implementar en cada fase, tomando como referencia el contenido de `Plan.md` y el estado actual conocido del proyecto.

## Fase 1: Preparar y validar los datos

### Ya disponible

- Lectura de archivos WAV.
- Lectura de `metadata.json`, `contexts.json`, `events.json` y `playback.json`.
- Conversión y normalización básica del audio.
- División en conjuntos `train`, `validation` y `test`.
- Preparación de grabaciones individuales del servicio.

### Falta implementar o completar

- Validación automática de correspondencia entre:
  - grabación;
  - `ExecutionId`;
  - timestamps;
  - eventos;
  - contextos.
- Validación de sincronización temporal entre audio y JSON.
- Detección de:
  - WAV corruptos;
  - canales inconsistentes;
  - frecuencias de muestreo incompatibles;
  - amplitudes extremas;
  - valores nulos o infinitos.
- Validación de unidades y rangos de parámetros CNC.
- División por grabaciones completas independientes para evitar fuga de información.
- Tratamiento específico para grabaciones incompletas o con eventos faltantes.
- Informe de calidad del dataset antes de iniciar el entrenamiento.

### Resultado esperado

Un informe que indique si el dataset es válido, qué grabaciones fueron descartadas y por qué.

---

## Fase 2: Entrenar el modelo acústico

### Ya disponible

- Entrenamiento de `CNN1D-VAE`.
- Entrenamiento de `LSTM-VAE`.
- Uso de CPU o CUDA.
- Checkpoints periódicos.
- Registro de:
  - MSE de entrenamiento;
  - MSE de validación;
  - MSE de prueba;
  - mejor época;
  - dispositivo;
  - configuración.
- Guardado de `best.pt`, `last.pt`, `final.pt` y `last_valid.pt`.
- Límites de `logvar`.
- Validación de valores no finitos.
- Reducción automática del learning rate.

### Falta implementar o completar

- Ejecutar y validar un entrenamiento completo con la nueva configuración.
- Comparar sistemáticamente `CNN1D-VAE` frente a `LSTM-VAE`.
- Añadir evaluación de:
  - estabilidad;
  - convergencia;
  - tiempo por época;
  - uso de memoria GPU;
  - consumo de CPU.
- Validar que el checkpoint seleccionado sea realmente el mejor según validación.
- Guardar explícitamente:
  - versión del código;
  - versión del dataset;
  - hash de la configuración;
  - versión del modelo.
- Añadir parada temprana cuando la validación deje de mejorar.
- Definir una política formal para reanudar entrenamientos fallidos.
- Validar el entrenamiento con varias sesiones diferentes.

### Resultado esperado

Un modelo reproducible, estable y comparable entre ejecuciones.

---

## Fase 3: Construir la línea base normal

### Ya disponible

- Cálculo de MSE por ventana.
- Referencia estadística basada en datos de entrenamiento.
- Uso de `train p99` como referencia inicial.
- Comparación entre entrenamiento, validación y prueba.

### Falta implementar o completar

- Calcular y guardar automáticamente:
  - media;
  - desviación estándar;
  - p90;
  - p95;
  - p99;
  - mínimo;
  - máximo.
- Crear líneas base separadas por:
  - máquina;
  - operación;
  - herramienta;
  - material;
  - programa CNC;
  - tipo de contexto.
- Definir formalmente los estados:

```text
Normal
Revisión
Anómalo
Fuera de distribución
Datos insuficientes
```

- Calibrar los umbrales usando datos de validación.
- Evitar utilizar umbrales globales cuando existan comportamientos normales diferentes.
- Persistir la versión y origen de cada umbral.
- Medir falsos positivos y falsos negativos cuando existan etiquetas expertas.

### Resultado esperado

Un sistema que pueda justificar por qué una ventana se considera normal o anómala.

---

## Fase 4: Analizar una grabación nueva

### Ya disponible

- División del audio en ventanas.
- Inferencia con el VAE.
- Cálculo de MSE por ventana.
- Identificación de ventanas con scores elevados.
- Asociación temporal de las ventanas.
- Visualización de resultados y contexto.

### Falta implementar o completar

- Crear un flujo independiente de inferencia que no requiera reentrenar.
- Calcular automáticamente:
  - MSE global;
  - MSE medio;
  - MSE máximo;
  - porcentaje de ventanas anómalas;
  - duración total de anomalías;
  - primera y última anomalía;
  - número de episodios anómalos.
- Agrupar ventanas consecutivas en incidentes.
- Asociar cada incidente con:
  - evento;
  - contexto;
  - herramienta;
  - operación;
  - parámetros CNC.
- Detectar si la grabación está fuera de distribución.
- Añadir puntuación de confianza calibrada.
- Generar un resultado persistente y consultable posteriormente.
- Permitir comparar una grabación nueva con varias ejecuciones anteriores.

### Resultado esperado

Un análisis completo de una grabación sin necesidad de volver a entrenar el modelo.

---

## Fase 5: Crear características de análisis

### Ya disponible

- MSE por ventana.
- Scores máximos.
- Estadísticas básicas.
- Embeddings latentes del VAE.
- Algunos datos de contexto y eventos.

### Falta implementar o completar

- Extraer características acústicas adicionales:
  - RMS;
  - energía;
  - frecuencia dominante;
  - centroides espectrales;
  - ancho de banda;
  - factor de cresta;
  - kurtosis;
  - skewness;
  - variación temporal;
  - relación señal/ruido.
- Crear características agregadas por grabación.
- Crear características agregadas por evento.
- Asociar características con parámetros de máquina.
- Normalizar características numéricas y categóricas.
- Definir un esquema estable de características.
- Versionar el vector de características.
- Validar qué características realmente ayudan a detectar anomalías.
- Evitar incluir datos futuros o información que produzca fuga de información.

### Resultado esperado

Un conjunto de características preparado para análisis estadístico y futuros modelos supervisados.

---

## Fase 6: Persistir en PostgreSQL

### Ya disponible

- Integración inicial con PostgreSQL.
- Persistencia de importaciones.
- Persistencia de ejecuciones de entrenamiento.
- Persistencia de algunos resultados.

### Falta implementar o completar

- Consolidar el modelo de datos completo para:
  - grabaciones;
  - máquinas;
  - sesiones;
  - contextos;
  - eventos;
  - ventanas;
  - predicciones;
  - checkpoints;
  - umbrales;
  - características.
- Crear relaciones entre grabación, ventana, contexto y evento.
- Guardar el hash o versión del modelo utilizado.
- Guardar el dataset y configuración exacta de cada ejecución.
- Implementar consultas históricas por:
  - máquina;
  - herramienta;
  - operación;
  - fecha;
  - estado;
  - modelo.
- Crear índices adecuados.
- Añadir migraciones versionadas.
- Implementar limpieza y retención de datos.
- Añadir controles de acceso.
- Garantizar idempotencia para no duplicar importaciones.
- Verificar transacciones y recuperación ante errores.

### Resultado esperado

PostgreSQL debe ser la fuente oficial y auditable de los datos y resultados.

---

## Fase 7: Indexar embeddings en Qdrant

### Ya disponible

- Generación de embeddings latentes.
- Integración inicial con Qdrant.
- Indexación de ventanas.
- Asociación de metadatos básicos.

### Falta implementar o completar

- Definir formalmente las colecciones y dimensiones soportadas.
- Establecer una política de nombres y versiones.
- Añadir payloads completos:
  - máquina;
  - operación;
  - herramienta;
  - material;
  - timestamp;
  - MSE;
  - estado;
  - versión del modelo.
- Implementar filtros por contexto.
- Validar la calidad de los resultados similares.
- Evitar mezclar embeddings generados por modelos incompatibles.
- Implementar reindexación al cambiar el modelo.
- Detectar embeddings duplicados.
- Relacionar cada resultado de Qdrant con PostgreSQL.
- Añadir control de consistencia entre ambas bases.
- Medir precisión de las búsquedas de similitud.

### Resultado esperado

Consultar casos históricos similares y utilizarlos como evidencia comparativa.

---

## Fase 8: Añadir clasificación supervisada

### Estado actual

Esta fase todavía está pendiente.

### Falta implementar

- Definir las clases de interés.
- Crear un proceso de etiquetado experto.
- Establecer criterios uniformes de clasificación.
- Construir un dataset etiquetado.
- Separar correctamente entrenamiento, validación y prueba.
- Entrenar modelos como:
  - CatBoost;
  - LightGBM;
  - XGBoost.
- Comparar sus resultados con el VAE.
- Evaluar:
  - precisión;
  - recall;
  - F1;
  - matriz de confusión;
  - sensibilidad por tipo de fallo.
- Gestionar el desbalance entre clases.
- Calibrar las probabilidades.
- Guardar explicaciones de las características más relevantes.
- Validar el modelo con datos de máquinas y sesiones nuevas.

### Resultado esperado

Un modelo capaz de clasificar tipos de anomalía conocidos, no solamente detectar desviaciones.

---

## Fase 9: Incorporar el LLM

### Estado actual

Esta fase no está implementada y debe permanecer posterior a las fases anteriores.

### Falta implementar

- Definir el formato del paquete de evidencias.
- Seleccionar un modelo local o servicio externo.
- Diseñar prompts controlados.
- Exigir salida estructurada en JSON.
- Separar claramente:
  - hechos observados;
  - inferencias;
  - hipótesis;
  - recomendaciones.
- Incorporar referencias a:
  - grabación;
  - ventana;
  - evento;
  - modelo;
  - caso similar.
- Impedir que el LLM ejecute comandos o consultas sin validación.
- Añadir protección contra:
  - alucinaciones;
  - prompt injection;
  - fuga de datos;
  - diagnósticos no respaldados.
- Implementar abstención cuando los datos sean insuficientes.
- Registrar:
  - modelo utilizado;
  - versión del prompt;
  - evidencias enviadas;
  - respuesta generada.
- Evaluar la calidad de las explicaciones con expertos.

### Resultado esperado

Un sistema que explique los resultados técnicos sin sustituir al modelo de detección ni al criterio del especialista.

## Prioridad recomendada

El orden de implementación debería ser:

1. Completar la validación del dataset.
2. Ejecutar y estabilizar el entrenamiento.
3. Completar la línea base estadística.
4. Terminar el flujo de inferencia independiente.
5. Crear el sistema completo de características.
6. Consolidar PostgreSQL.
7. Validar Qdrant.
8. Crear el dataset etiquetado.
9. Entrenar el clasificador supervisado.
10. Incorporar el LLM como capa de explicación.

La principal prioridad actual es terminar las fases 1 a 5. Sin estas fases consolidadas, los resultados de PostgreSQL, Qdrant, el clasificador y el LLM no tendrían una base suficientemente confiable.

