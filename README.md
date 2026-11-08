# 📚 Machine Learning

Repositorio de las entregas de la materia de Machine Learning. Cada entrega se desarrolla en un notebook y puede contar con su propio proceso de ejecución automática mediante GitHub Actions.

## 👩‍💻 Autores

- Katerin Lopez Moros ([@KateLopez05](https://github.com/KateLopez05))
- Bayron Meza Guzman ([@BayronMgDev](https://github.com/BayronMgDev))

## 📌 Entregas

| Entrega | Notebook | Qué hace | Workflow | Resultados |
|---|---|---|---|---|
| 1 | `src/notebook/notebook_1.ipynb` | Carga de un dataset desde Hugging Face, transformaciones con pandas y generación de un reporte en texto. | `.github/workflows/ejecutar.yml` | `src/notebook/reportes/entrega_1` |
| 2 | `src/notebook/notebook_2.ipynb` | Construcción del DataFrame, preprocesamiento, regresión lineal, clasificación y resumen de ambos modelos. | `.github/workflows/ejecutar_ea2.yml` | `src/notebook/reportes/entrega_2/resumen_modelos.txt` |
| 3 | `src/notebook/notebook_3.ipynb` | Pregunta de negocio, preprocesamiento, clasificación supervisada, optimización de hiperparámetros y evaluación del modelo. | Pendiente de implementación. | Resultados documentados en el notebook. |

Cada entrega se desarrolla de manera independiente. La configuración de automatización de la tercera entrega se incorporará posteriormente.

## 🗄️ Estructura del proyecto

```text
.github/
└── workflows/
    ├── ejecutar.yml
    └── ejecutar_ea2.yml

src/
├── clases/
│   ├── loader_hf.py
│   ├── reporte.py
│   ├── regresion_lineal.py
│   └── clasificador_tabular.py
│
└── notebook/
    ├── notebook_1.ipynb
    ├── notebook_2.ipynb
    ├── notebook_3.ipynb
    │
    ├── data/
    │
    ├── modelos/
    │   ├── entrega_2/
    │   └── entrega_3/
    │
    └── reportes/
        ├── entrega_1/
        └── entrega_2/
            └── resumen_modelos.txt

requirements.txt
README.md
```

Los modelos entrenados se guardan en archivos `.pkl` dentro de sus respectivas carpetas. Estos archivos no se publican en el repositorio.

## 📌 Flujo de la entrega 1

```text
Clase de carga → DataFrame → transformaciones mínimas → clase de resumen → archivo .txt
```

1. La clase de carga descarga el dataset de Hugging Face y devuelve un `pandas.DataFrame`.
2. Se aplican cuatro transformaciones mínimas, cada una guardada en una variable nueva.
3. El DataFrame final se pasa a `Reportes`, que escribe el resumen en un archivo `.txt`.

## 🗂️ Dataset

Se utiliza **California Housing**, basado en información censal de California de 1990. Cada fila describe una zona o bloque censal.

| Columna original | Columna renombrada | Significado |
|---|---|---|
| `MedInc` | `Ingreso_mediano` | Ingreso mediano de los hogares. |
| `HouseAge` | `Edad_vivienda` | Antigüedad mediana de las viviendas. |
| `AveRooms` | `Promedio_habitaciones` | Promedio de habitaciones por hogar. |
| `AveBedrms` | `Promedio_dormitorio_hogar` | Promedio de dormitorios por hogar. |
| `Population` | `Poblacion` | Población de la zona. |
| `AveOccup` | `Promedio_ocupacion` | Promedio de ocupantes por hogar. |
| `Latitude` | `Latitud` | Latitud geográfica. |
| `Longitude` | `Longitud` | Longitud geográfica. |
| `MedHouseVal` | `Valor_mediano_vivienda` | Valor mediano de las viviendas. |

## ✅ Entrega 1: Transformaciones mínimas

| # | Operación | Resultado |
|---|---|---|
| 1 | Renombrar columnas con `rename()` | `df_transformado`: mismas filas y columnas, nombres en español. |
| 2 | Filtrar filas con `Edad_vivienda > 20` | `df_t2`: menos filas, mismas columnas. |
| 3 | Seleccionar cinco columnas | `df_t3`: mismas filas, cinco columnas. |
| 4 | Crear `Valor_en_dolares` con `assign()` | `df_t4`: mismas filas, seis columnas. |

```python
df_transformado = df.rename(columns={...})  # 1

df_t2 = df_transformado[
    df_transformado["Edad_vivienda"] > 20
]  # 2

df_t3 = df_t2[
    [
        "Ingreso_mediano",
        "Edad_vivienda",
        "Promedio_habitaciones",
        "Poblacion",
        "Valor_mediano_vivienda"
    ]
]  # 3

df_t4 = df_t3.assign(
    Valor_en_dolares=df_t3["Valor_mediano_vivienda"] * 100000
)  # 4
```

Cada operación devuelve un DataFrame nuevo y no modifica el anterior. El `.txt` final corresponde a `df_t4`, el DataFrame transformado.

## ✅ Entrega 2: Regresión y clasificación

**Dataset:** [`BrianSuToronto/california_housing`](https://huggingface.co/datasets/BrianSuToronto/california_housing), split `train`.

El DataFrame se construye automáticamente mediante `datasets.load_dataset`, sin necesidad de descargar archivos manualmente.

### Flujo del notebook

1. **Construcción del DataFrame:** carga automática, tamaño, columnas, tipos de datos y muestra de registros.
2. **Preprocesamiento:** renombrado de columnas, creación de `Categoria_valor`, enriquecimiento de variables, transformaciones y división estratificada de los datos.
3. **Regresión lineal:** predicción de `Valor_mediano_vivienda` mediante la clase `RegresionLineal`.
4. **Clasificación:** predicción de `Categoria_valor` (Bajo, Medio y Alto) mediante la clase `ClasificadorTabular`.
5. **Resumen automático:** generación de `src/notebook/reportes/entrega_2/resumen_modelos.txt`, con información del dataset, parámetros, métricas, interpretación y limitaciones.

Las clases siguen la estructura de las clases del docente: validar, dividir, entrenar, predecir, evaluar y obtener parámetros.

`ClasificadorTexto` no se utilizó porque está diseñado para datos de texto mediante TF-IDF. Para esta entrega se creó `ClasificadorTabular`, orientado a variables numéricas.

### Resultados de referencia

| Modelo | Métricas | Referencia |
|---|---|---|
| Regresión lineal | MAE 0.497 (≈ 49.653 USD), RMSE 0.691, R² 0.651 | MAE del modelo base: 0.922 |
| Regresión logística | Exactitud 0.730, F1 macro 0.730 | Exactitud del modelo base: 0.334 |

El valor de `Valor_mediano_vivienda` está expresado en cientos de miles de dólares.

En clasificación, la categoría `Medio` presenta mayor dificultad de reconocimiento y la mayoría de los errores se produce entre categorías vecinas.

---

# ✅ Entrega 3: Aprendizaje supervisado – Clasificación de segmentos de vivienda

## 3.1. Pregunta de negocio

¿Qué características socioeconómicas y geográficas permiten identificar si una zona residencial de California pertenece a un segmento de valor de vivienda bajo, medio o alto?

La solución busca apoyar a analistas inmobiliarios, inversionistas y entidades de planificación territorial en la identificación de segmentos residenciales y en el análisis de las características asociadas a cada uno.

La decisión que se busca apoyar es la identificación y comparación de zonas residenciales según su segmento de valor, como insumo para estudios de mercado, inversión y planificación.

## 3.2. Dataset y construcción de la variable objetivo

Se utiliza el dataset California Housing de Hugging Face:

- Identificador: `BrianSuToronto/california_housing`.
- Split: `train`.
- Registros: 20.640.
- Columnas originales: 9.

Cada registro representa una zona o bloque censal de California e incluye características de vivienda, población, ingreso y ubicación geográfica.

A partir de `Valor_mediano_vivienda` se construye la variable objetivo `Categoria_valor` mediante `pandas.qcut()`, dividiendo la distribución en tres grupos de tamaño similar.

| Categoría | Registros |
|---|---:|
| Bajo | 6.884 |
| Medio | 6.876 |
| Alto | 6.880 |
| **Total** | **20.640** |

Los límites observados en esta ejecución fueron:

| Categoría | Rango de `Valor_mediano_vivienda` |
|---|---|
| Bajo | 0,14999 a 1,41400 |
| Medio | 1,41500 a 2,30300 |
| Alto | 2,30400 a 5,00001 |

La variable original `Valor_mediano_vivienda` se utiliza para construir las etiquetas, pero se excluye de las variables predictoras para evitar fuga de información (*data leakage*).

## 3.3. Tipo de aprendizaje y modelo seleccionado

La entrega utiliza aprendizaje supervisado mediante un problema de clasificación multiclase.

Este enfoque corresponde a la pregunta de negocio porque se dispone de una variable objetivo conocida (`Categoria_valor`) con tres clases: Bajo, Medio y Alto. El modelo aprende la relación entre las características de cada zona y su categoría para clasificar nuevas observaciones.

El algoritmo seleccionado es Regresión Logística (`LogisticRegression`) de Scikit-learn. Aunque su nombre contiene la palabra regresión, también se utiliza para clasificación. En este proyecto se configura para distinguir tres categorías.

Se selecciona como modelo inicial porque permite estimar probabilidades por clase, tiene un costo computacional moderado y sus coeficientes permiten analizar la relación aprendida entre las variables y las categorías.

Su capacidad se comprueba mediante métricas calculadas sobre datos que no participaron en el entrenamiento.

## 3.4. Variables del modelo

**Variables predictoras (X):**

| Variable | Descripción |
|---|---|
| `Ingreso_mediano` | Ingreso mediano de los hogares de la zona. |
| `Edad_vivienda` | Antigüedad de las viviendas. |
| `Promedio_habitaciones` | Promedio de habitaciones por hogar. |
| `Promedio_dormitorio_hogar` | Promedio de dormitorios por hogar. |
| `Poblacion` | Población de la zona. |
| `Promedio_ocupacion` | Promedio de ocupantes por hogar. |
| `Latitud` | Coordenada geográfica de latitud. |
| `Longitud` | Coordenada geográfica de longitud. |

**Variable objetivo (y):** `Categoria_valor`, con las clases Bajo, Medio y Alto.

## 3.5. Preprocesamiento realizado

El preprocesamiento se desarrolló sobre el DataFrame `df_modelo`, que contiene 20.640 registros y 10 columnas: las nueve variables originales renombradas y la nueva variable objetivo.

Las comprobaciones y decisiones fueron:

1. **Valores nulos:** no se encontraron valores nulos en las columnas.
2. **Registros duplicados:** no se encontraron registros duplicados.
3. **Tipos de datos:** las variables numéricas son de tipo `float64` y `Categoria_valor` es categórica.
4. **Valores extremos:** se aplicó el método del rango intercuartílico (IQR) a cinco variables. Se identificaron observaciones fuera de los límites estadísticos, pero se conservaron porque el análisis no demuestra que sean errores.
5. **Selección de variables:** se utilizaron ocho predictoras y se excluyó `Valor_mediano_vivienda` para evitar fuga de información.
6. **División de datos:** se separaron los datos en 80 % para entrenamiento y 20 % para prueba, con `random_state=42` y estratificación por la variable objetivo.
7. **Escalado:** se configuró `StandardScaler` dentro del pipeline del modelo. El escalador aprende sus parámetros con los datos de entrenamiento y aplica esa transformación al conjunto de prueba.

La división produjo los siguientes tamaños:

| Conjunto | Registros | Columnas predictoras |
|---|---:|---:|
| Entrenamiento | 16.512 | 8 |
| Prueba | 4.128 | 8 |
| **Total** | **20.640** | **8** |

La distribución estratificada quedó equilibrada:

| Categoría | Entrenamiento | Prueba |
|---|---:|---:|
| Bajo | 5.507 | 1.377 |
| Medio | 5.501 | 1.375 |
| Alto | 5.504 | 1.376 |

## 3.6. Funcionamiento de la Regresión Logística

El algoritmo estima la probabilidad de que cada observación pertenezca a cada una de las categorías.

En esta entrega, el flujo funciona así:

1. **Recibe las variables:** utiliza las ocho características socioeconómicas y geográficas de cada zona residencial.
2. **Estandariza las variables:** `StandardScaler` transforma las predictoras para que sus escalas sean comparables. Sus parámetros se calculan exclusivamente con el conjunto de entrenamiento.
3. **Aprende durante el entrenamiento:** la Regresión Logística ajusta coeficientes e interceptos a partir de los ejemplos etiquetados.
4. **Calcula probabilidades:** para una observación, el modelo estima la probabilidad de pertenecer a cada clase. Scikit-learn utiliza por defecto el esquema multiclase compatible con `lbfgs`.
5. **Asigna una clase:** selecciona la categoría con la probabilidad estimada más alta.
6. **Se evalúa:** compara las predicciones del conjunto de prueba con las etiquetas reales para calcular métricas y analizar los errores.

## 3.7. Parámetros e hiperparámetros utilizados

La configuración inicial utiliza:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

Dentro de un pipeline con `StandardScaler`.

| Parámetro | Valor | Significado |
|---|---|---|
| `max_iter` | 1000 | Número máximo de iteraciones permitidas para la optimización. |
| `random_state` | 42 | Semilla para reproducibilidad cuando el procedimiento utiliza aleatoriedad. |
| `C` | 1.0 (predeterminado) | Inverso de la intensidad de regularización. Un valor menor implica una regularización más fuerte. |
| `solver` | `lbfgs` (predeterminado) | Método de optimización que busca los parámetros del modelo. |
| `fit_intercept` | `True` (predeterminado) | Incluye un término independiente en el modelo. |

Para justificar el hiperparámetro `C`, se realizó una búsqueda mediante `GridSearchCV`, utilizando los siguientes valores:

```python
param_grid = {
    "modelo__C": [0.01, 0.1, 1.0, 10.0]
}
```

La validación se realizó sobre el conjunto de entrenamiento mediante `StratifiedKFold`, con cinco particiones, mezcla de datos y semilla `random_state=42`.

La métrica utilizada para seleccionar la configuración fue `f1_macro`, porque permite evaluar el desempeño de las tres categorías dando el mismo peso a cada una.

### Resultados de la validación cruzada

| Valor de C | F1 macro promedio | Desviación estándar | F1 macro de entrenamiento |
|---:|---:|---:|---:|
| 0.01 | 0.6864 | 0.0087 | 0.6868 |
| 0.10 | 0.7076 | 0.0076 | 0.7084 |
| 1.00 | 0.7125 | 0.0091 | 0.7132 |
| 10.00 | 0.7138 | 0.0090 | 0.7139 |

La mejor configuración encontrada fue:

```python
{
    "modelo__C": 10.0
}
```

Con un F1 macro promedio de validación cruzada de **0.7138**.

El valor `C=10.0` obtuvo el mejor resultado promedio entre las alternativas evaluadas. Sin embargo, la diferencia frente a `C=1.0` fue pequeña (0.0013), menor que la desviación estándar observada en la validación. Por ello, la mejora debe interpretarse como una diferencia leve y no como evidencia de una superioridad estadísticamente significativa.

La búsqueda se realizó exclusivamente con los datos de entrenamiento. El conjunto de prueba se reservó para la evaluación final.

## 3.8. Entrenamiento y evaluación del modelo

Se entrenó inicialmente el modelo con la configuración predeterminada de `C=1.0` y posteriormente se evaluó el modelo optimizado con `C=10.0`.

La evaluación se realizó sobre el conjunto de prueba, compuesto por 4.128 registros que no participaron en el entrenamiento ni en la búsqueda de hiperparámetros.

### 3.8.1. Métricas del modelo inicial

| Métrica | Resultado |
|---|---:|
| Exactitud (Accuracy) | 0.7163 |
| F1 macro | 0.7173 |
| F1 ponderado | 0.7173 |

Resultados por categoría:

| Categoría | Precisión | Recall | F1-score | Casos |
|---|---:|---:|---:|---:|
| Bajo | 0.7730 | 0.7495 | 0.7611 | 1.377 |
| Medio | 0.5902 | 0.6116 | 0.6007 | 1.375 |
| Alto | 0.7924 | 0.7878 | 0.7901 | 1.376 |

### 3.8.2. Matriz de confusión del modelo inicial

Las filas representan las categorías reales y las columnas las categorías predichas.

| Categoría real | Predicho Bajo | Predicho Medio | Predicho Alto |
|---|---:|---:|---:|
| Bajo | 1.032 | 328 | 17 |
| Medio | 267 | 841 | 267 |
| Alto | 36 | 256 | 1.084 |

La matriz muestra que el modelo reconoce mejor las categorías Bajo y Alto. La categoría Medio presenta más confusiones con las otras dos, especialmente porque sus características pueden ser similares a las de zonas cercanas a los límites de los terciles.

### 3.8.3. Métricas del modelo optimizado

Después de la búsqueda de hiperparámetros, se evaluó el mejor estimador encontrado, con `C=10.0`.

| Métrica | Resultado |
|---|---:|
| Exactitud (Accuracy) | 0.7178 |
| F1 macro | 0.7187 |
| F1 ponderado | 0.7188 |

Resultados por categoría:

| Categoría | Precisión | Recall | F1-score | Casos |
|---|---:|---:|---:|---:|
| Bajo | 0.7727 | 0.7480 | 0.7601 | 1.377 |
| Medio | 0.5926 | 0.6145 | 0.6034 | 1.375 |
| Alto | 0.7947 | 0.7907 | 0.7927 | 1.376 |

### 3.8.4. Matriz de confusión del modelo optimizado

| Categoría real | Predicho Bajo | Predicho Medio | Predicho Alto |
|---|---:|---:|---:|
| Bajo | 1.030 | 329 | 18 |
| Medio | 267 | 845 | 263 |
| Alto | 36 | 252 | 1.088 |

La matriz evidencia que el modelo mantiene un mejor reconocimiento de las categorías Bajo y Alto, mientras que la categoría Medio continúa presentando dificultades de clasificación.

### 3.8.5. Comparación de los modelos

| Métrica | Modelo inicial (C=1.0) | Modelo optimizado (C=10.0) |
|---|---:|---:|
| Exactitud | 0.7163 | 0.7178 |
| F1 macro | 0.7173 | 0.7187 |
| F1 ponderado | 0.7173 | 0.7188 |

La optimización produjo una mejora ligera en las métricas generales:

- La exactitud aumentó de 0.7163 a 0.7178.
- El F1 macro aumentó de 0.7173 a 0.7187.
- El F1 ponderado aumentó de 0.7173 a 0.7188.

El desempeño de la categoría Bajo disminuyó ligeramente, mientras que las categorías Medio y Alto presentaron pequeñas mejoras.

En consecuencia, se conserva `C=10.0` como la configuración seleccionada por la búsqueda, teniendo en cuenta que su mejora frente al modelo inicial es reducida.

## 3.9. Interpretación de resultados y relación con la pregunta de negocio

Los resultados muestran que el modelo optimizado alcanza una exactitud de aproximadamente **71,78 %** y un F1 macro de **0,7187** sobre el conjunto de prueba.

Esto indica que el clasificador logra identificar correctamente una parte importante de las zonas residenciales según los tres segmentos de valor construidos.

Sin embargo, el desempeño no es igual para todas las categorías:

- **Bajo:** presenta un F1 de 0.7601. El modelo reconoce adecuadamente la mayoría de las zonas de este segmento.
- **Medio:** presenta un F1 de 0.6034. Es la categoría más difícil de identificar y concentra confusiones con las categorías Bajo y Alto.
- **Alto:** presenta un F1 de 0.7927. Es la categoría con mejor desempeño en la evaluación final.

La mayor parte de los errores se produce entre categorías vecinas. Los errores extremos, en los que una zona de valor Bajo se clasifica como Alto o viceversa, son menos frecuentes.

Desde la perspectiva del negocio, el modelo puede apoyar una primera segmentación de zonas residenciales para estudios de mercado, análisis comparativos y exploración de oportunidades inmobiliarias.

No obstante, sus resultados no deben utilizarse como único criterio para tomar decisiones de inversión. La categoría Medio presenta un nivel de reconocimiento menor y las predicciones deben complementarse con información adicional del mercado.

## 3.10. Ventajas y limitaciones

### Ventajas

- Es apropiada para clasificación supervisada y puede trabajar con más de dos clases.
- Estima probabilidades para cada categoría.
- Sus coeficientes permiten estudiar la relación entre las variables predictoras y la clasificación.
- Tiene un costo computacional moderado.
- Se integra con `Pipeline` para organizar el escalado y el clasificador, reduciendo el riesgo de fuga de información.
- La validación cruzada permite comparar configuraciones de hiperparámetros antes de evaluar el modelo final.

### Limitaciones

- Sus fronteras de decisión son lineales en el espacio de características transformadas; puede no representar relaciones complejas o no lineales.
- Puede verse afectada por valores extremos y por predictores con distribuciones muy asimétricas.
- La interpretación de los coeficientes requiere considerar el escalado y las relaciones entre las variables.
- Las probabilidades estimadas no garantizan que la clasificación sea correcta.
- El rendimiento depende de la calidad de los datos, de las variables seleccionadas y de los parámetros utilizados.
- La categoría Medio presenta menor capacidad de reconocimiento que las otras dos categorías.
- Los segmentos se construyeron a partir de terciles del conjunto de datos, por lo que sus límites son relativos a esta distribución.

## 3.11. Conclusiones y estado de la entrega

La tercera entrega implementa un modelo de aprendizaje supervisado para clasificar zonas residenciales de California en tres segmentos de valor: Bajo, Medio y Alto.

Se realizó el preprocesamiento de los datos, la selección de ocho variables predictoras, la división estratificada de entrenamiento y prueba, y la construcción de un pipeline con `StandardScaler` y `LogisticRegression`.

Posteriormente, se entrenó el modelo inicial y se realizó una búsqueda de hiperparámetros mediante validación cruzada. La configuración seleccionada fue `C=10.0`, con un F1 macro promedio de 0.7138 durante la validación.

En la evaluación final sobre el conjunto de prueba, el modelo optimizado obtuvo:

- Exactitud: **0.7178**.
- F1 macro: **0.7187**.
- F1 ponderado: **0.7188**.

Los resultados indican que el modelo puede servir como herramienta inicial para identificar segmentos residenciales y apoyar análisis comparativos. Su principal dificultad está en la clasificación de la categoría Medio.

La solución responde a la pregunta de negocio como un ejercicio de segmentación relativa. No constituye una tasación individual ni una predicción directa del precio de una vivienda.

**Estado:** el notebook `src/notebook/notebook_3.ipynb` contiene el desarrollo de la pregunta de negocio, la construcción del dataset, el preprocesamiento, el modelo, el entrenamiento, la búsqueda de hiperparámetros, las métricas, las matrices de confusión y la interpretación de resultados.

El modelo optimizado se guarda localmente en `src/notebook/modelos/entrega_3/`. Los archivos generados no se incorporan al repositorio.

La automatización independiente de esta entrega mediante GitHub Actions queda pendiente para una etapa posterior.

---

## 🔧 Cómo ejecutar el proyecto

### Localmente o en GitHub Codespaces

```bash
git clone https://github.com/KateLopez05/Machine-Learning.git

cd Machine-Learning

python -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt
```

Se recomienda **Python 3.12**.

En Windows (PowerShell), el entorno se crea y se activa así:

```powershell
py -3.12 -m venv .venv

.venv\Scripts\Activate.ps1
```

Después, abre el notebook de la entrega que quieras ejecutar, selecciona el kernel del entorno virtual y ejecuta todas sus celdas.

Para ejecutar la Entrega 2 desde la terminal:

```bash
jupyter nbconvert --to notebook --execute \
  --output-dir /tmp --output notebook_2_ejecutado \
  src/notebook/notebook_2.ipynb
```

Para ejecutar la Entrega 3:

```bash
jupyter nbconvert --to notebook --execute \
  --output-dir /tmp --output notebook_3_ejecutado \
  src/notebook/notebook_3.ipynb
```

Los notebooks de las entregas 2 y 3 guardan los modelos entrenados en sus respectivas carpetas dentro de `src/notebook/modelos/`.

### Autenticación en Hugging Face (solo si el dataset lo exige)

Los datasets públicos no necesitan token. Si algún dataset es privado o requiere aceptar condiciones de uso, inicia sesión con el entorno virtual activado:

```bash
hf auth login

hf auth whoami
```

Abre `https://hf.co/oauth/device` en el navegador, escribe el código que muestra la terminal y autoriza el acceso.

No escribas tokens en el código ni los subas a GitHub.

## ✨ Ejecución automática en GitHub Actions

Las entregas 1 y 2 cuentan con workflows independientes.

### Entrega 2 (`ejecutar_ea2.yml`)

1. Instala las dependencias de `requirements.txt`.
2. Ejecuta `notebook_2.ipynb`.
3. Genera el resumen de resultados.
4. Verifica que `src/notebook/reportes/entrega_2/resumen_modelos.txt` exista y no esté vacío.
5. Publica artefactos descargables con el resumen y el notebook ejecutado.
6. Guarda una copia del resumen en el repositorio mediante un commit automático.

El notebook ejecutado se publica como artefacto y no reemplaza el archivo original del repositorio.

**Cómo consultar los resultados:** en la pestaña **Actions**, selecciona la ejecución correspondiente y busca la sección **Artifacts**.

Como los workflows también pueden generar commits automáticos, conviene ejecutar `git pull` antes de empezar a trabajar y antes de cada `git push`.

## Control de versiones y trabajo en equipo

- Los avances se registran con commits de mensajes claros (por ejemplo, `Actualizo documentación de la entrega 3`).
- Antes de editar y antes de subir cambios, se recomienda ejecutar `git pull`.
- Se debe evitar editar el mismo notebook simultáneamente, porque los archivos `.ipynb` pueden generar conflictos con facilidad.
- Los archivos de modelos entrenados (`.pkl`) no se deben subir al repositorio.