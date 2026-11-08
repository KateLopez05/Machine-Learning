# 📚 Machine Learning

Repositorio de las entregas de la materia de Machine Learning. Cada entrega es un notebook que se ejecuta automáticamente en GitHub Actions y deja un resumen en .txt generado por código.

## 👩‍💻 Autores

- Katerin Lopez Moros ([@KateLopez05](https://github.com/KateLopez05))

- Bayron Meza Guzman  ([@BayronMgDev](https://github.com/BayronMgDev))

## 📌 Entregas

| Entrega | Notebook | Qué hace | Workflow | Resumen generado |

|---|---|---|---|---|

| 1 | `src/notebook/notebook_1.ipynb` | Carga de un dataset desde Hugging Face, transformaciones mínimas con pandas y reporte en texto | `.github/workflows/ejecutar.yml` | `src/notebook/reportes/entrega_1` |

| 2 | `src/notebook/notebook_2.ipynb` | Construcción del DataFrame, preprocesamiento, regresión lineal, clasificación y resumen de ambos modelos | `.github/workflows/ejecutar_ea2.yml` | `src/notebook/reportes/entrega_2/resumen_modelos.txt` |
| 3 | `src/notebook/notebook_3.ipynb` | Pregunta de negocio, preprocesamiento y clasificación supervisada de segmentos de vivienda en California | `.github/workflows/ejecutar_ea3.yml` | Resultados de la entrega 3 (por definir) |

Cada entrega tiene su propio workflow, y cada workflow solo se activa cuando cambian los archivos de su entrega, para que trabajar en una no altere la ejecución de la otra. La entrega 2 también se activa con cambios en `src/clases/**` y en `requirements.txt`, porque ejecuta esas clases y esas dependencias.

## 🗄️ Estructura

```

.github/workflows/

  ejecutar.yml             Entrega 1: ejecuta notebook_1.ipynb

  ejecutar_ea2.yml         Entrega 2: ejecuta notebook_2.ipynb y publica el resumen

src/

  clases/                  Clases reutilizables

    loader_hf.py             Carga del dataset desde Hugging Face

    reporte.py               Reportes: resumen .txt de un DataFrame (entrega 1)

    regresion_lineal.py      RegresionLineal: flujo de regresión (clase del docente)

    clasificador_tabular.py  ClasificadorTabular: flujo de clasificación con datos numéricos

  notebook/

    notebook_1.ipynb         Entrega 1

    notebook_2.ipynb         Entrega 2
    notebook_3.ipynb         Entrega 3

    data/                    Datos descargados por la entrega 1

    modelos/                 Modelos .pkl de la entrega 2 (no se suben al repositorio)

    reportes/

      entrega_1/             Resumen automático de la entrega 1

      entrega_2/

        resumen_modelos.txt  Resumen automático de la entrega 2
      entrega_3/              Resultados de la entrega 3 (por definir)

requirements.txt           Dependencias del proyecto

```

## :pushpin: Flujo de la entrega 1

```

Clase de carga → DataFrame → transformaciones mínimas → clase de resumen → archivo .txt

```

1. La clase de carga descarga el dataset de Hugging Face y devuelve un `pandas.DataFrame`.

2. Se aplican cuatro transformaciones mínimas, cada una guardada en una variable nueva.

3. El DataFrame final se pasa a `Reportes`, que escribe el resumen en un archivo `.txt`.

## 🗂️ Dataset

Se usa **California Housing** (censo de California de 1990). Cada fila describe una zona.

| Columna original | Columna renombrada | Significado |

|---|---|---|

| `MedInc` | `Ingreso_mediano` | Ingreso mediano de los hogares |

| `HouseAge` | `Edad_vivienda` | Antigüedad mediana de las viviendas |

| `AveRooms` | `Promedio_habitaciones` | Promedio de habitaciones por hogar |

| `AveBedrms` | `Promedio_dormitorio_hogar` | Promedio de dormitorios por hogar |

| `Population` | `Poblacion` | Población de la zona |

| `AveOccup` | `Promedio_ocupacion` | Promedio de ocupantes por hogar |

| `Latitude` | `Latitud` | Latitud |

| `Longitude` | `Longitud` | Longitud |

| `MedHouseVal` | `Valor_mediano_vivienda` | Valor mediano de las viviendas (variable objetivo) |

## :white_check_mark: Entrega 1: Transformaciones mínimas

| # | Operación | Resultado |

|---|---|---|

| 1 | Renombrar columnas con `rename()` | `df_transformado`: mismas filas y columnas, nombres en español |

| 2 | Filtrar filas con `Edad_vivienda > 20` | `df_t2`: menos filas, mismas columnas |

| 3 | Seleccionar cinco columnas | `df_t3`: mismas filas, 5 columnas |

| 4 | Crear `Valor_en_dolares` con `assign()` | `df_t4`: mismas filas, 6 columnas |

```python

df_transformado = df.rename(columns={...})                       # 1

df_t2 = df_transformado[df_transformado["Edad_vivienda"] > 20]   # 2

df_t3 = df_t2[["Ingreso_mediano", "Edad_vivienda", "Promedio_habitaciones",

               "Poblacion", "Valor_mediano_vivienda"]]           # 3

df_t4 = df_t3.assign(Valor_en_dolares=df_t3["Valor_mediano_vivienda"] * 100000)  # 4

```

Cada operación devuelve un DataFrame nuevo y no modifica el anterior. El `.txt` final corresponde a `df_t4`, el DataFrame ya transformado.

## :white_check_mark: Entrega 2: regresión y clasificación

**Dataset:** [`BrianSuToronto/california_housing`](https://huggingface.co/datasets/BrianSuToronto/california_housing) (censo de California de 1990), split `train`. Cada fila describe una zona. El DataFrame se construye por código con `datasets.load_dataset`, sin descargas manuales.

**Flujo del notebook**

1. **Construcción del DataFrame:** carga automática, tamaño, columnas, tipos y muestra.

2. **Preprocesamiento:** columnas renombradas al español, variable de clasificación `Categoria_valor` (terciles de `Valor_mediano_vivienda`), enriquecimiento con `Habitaciones_por_dormitorio`, transformación logarítmica en cuatro variables sesgadas, división 80/20 estratificada (`random_state=42`) y escalado con `StandardScaler` ajustado solo con entrenamiento.

3. **Regresión lineal:** predice `Valor_mediano_vivienda` con la clase `RegresionLineal` (`src/clases/regresion_lineal.py`), que valida los datos, entrena un `LinearRegression` con parámetros por defecto, guarda el modelo en un `.pkl` y calcula MAE, RMSE y R².

4. **Clasificación:** predice `Categoria_valor` (Bajo, Medio, Alto) con la clase `ClasificadorTabular` (`src/clases/clasificador_tabular.py`), que construye un pipeline `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=42)`, divide con estratificación (las mismas filas que la regresión) y guarda el modelo en un `.pkl`.

5. **Resumen automático:** escribe `src/notebook/reportes/entrega_2/resumen_modelos.txt` con el dataset, el preprocesamiento, los parámetros, las métricas, la interpretación, la fecha de ejecución y las limitaciones. Todo el contenido sale de la ejecución, no está escrito a mano.

Las dos clases siguen la estructura de las clases del docente (validar, dividir, entrenar, predecir, evaluar y obtener parámetros). `ClasificadorTexto` no se usó porque está pensada para texto (TF-IDF), y esta entrega trabaja con variables numéricas, por lo que se creó `ClasificadorTabular` con la misma estructura. Los modelos entrenados se guardan en `src/notebook/modelos/entrega_2/` y no se suben al repositorio.

**Resultados de referencia** (última ejecución, datos de prueba)

| Modelo | Métricas | Referencia |

|---|---|---|

| Regresión lineal | MAE 0.497 (≈ 49.653 USD), RMSE 0.691, R² 0.651 | MAE del modelo base: 0.922 |

| Regresión logística | Exactitud 0.730, F1 macro 0.730 | Exactitud del modelo base: 0.334 |

El valor de `Valor_mediano_vivienda` está en cientos de miles de dólares. En clasificación, la clase `Medio` es la más difícil de reconocer, y cerca del 95 % de los errores ocurre entre clases vecinas.
✅ Entrega 3: aprendizaje supervisado y clasificación de segmentos de vivienda
3.1. Pregunta de negocio
¿Qué características socioeconómicas y geográficas permiten identificar si una zona residencial de California pertenece a un segmento de valor de vivienda bajo, medio o alto?

La solución busca apoyar a analistas inmobiliarios, inversionistas y entidades de planificación territorial en la identificación de segmentos residenciales y en el análisis de las características asociadas a cada uno.

La decisión que se busca apoyar es la identificación y comparación de zonas residenciales según su segmento de valor, como insumo para estudios de mercado, inversión y planificación.
3.2. Dataset y construcción de la variable objetivo
Se utiliza el dataset California Housing de Hugging Face, identificado como BrianSuToronto/california_housing, en el split train.

El conjunto contiene 20.640 registros y 9 columnas originales. Cada registro representa una zona o bloque censal de California e incluye características de vivienda, población, ingreso y ubicación geográfica.

A partir de Valor_mediano_vivienda se construye la variable objetivo Categoria_valor mediante pandas.qcut(), dividiendo la distribución en tres grupos de tamaño similar:
Clase	Registros
Bajo	6.884
Medio	6.876
Alto	6.880
Total	20.640


Los límites observados en esta ejecución fueron:
Categoría	Rango de Valor_mediano_vivienda
Bajo	0,14999 a 1,41400
Medio	1,41500 a 2,30300
Alto	2,30400 a 5,00001


La variable original Valor_mediano_vivienda se utiliza para construir las etiquetas, pero se excluye de las variables predictoras para evitar fuga de información (data leakage).
3.3. Tipo de aprendizaje y modelo seleccionado
La entrega utiliza aprendizaje supervisado, mediante un problema de clasificación multiclase. Este enfoque corresponde a la pregunta porque se dispone de una variable objetivo conocida (Categoria_valor) con tres clases: Bajo, Medio y Alto. El modelo aprende la relación entre las características de cada zona y su categoría para clasificar nuevas observaciones.

El algoritmo seleccionado es Regresión Logística (LogisticRegression) de Scikit-learn. Aunque su nombre contiene la palabra regresión, también se utiliza para clasificación. En este proyecto se configura para distinguir tres categorías.

Se elige como modelo inicial porque permite estimar probabilidades por clase, tiene un costo computacional moderado y sus coeficientes permiten analizar la relación aprendida entre las variables y las categorías. Su capacidad debe comprobarse con las métricas del conjunto de prueba.
3.4. Variables del modelo
Variables predictoras (X):
Variable	Descripción
Ingreso_mediano	Ingreso mediano de los hogares de la zona.
Edad_vivienda	Antigüedad de las viviendas.
Promedio_habitaciones	Promedio de habitaciones por hogar.
Promedio_dormitorio_hogar	Promedio de dormitorios por hogar.
Poblacion	Población de la zona.
Promedio_ocupacion	Promedio de ocupantes por hogar.
Latitud	Coordenada geográfica de latitud.
Longitud	Coordenada geográfica de longitud.


Variable objetivo (y): Categoria_valor, con las clases Bajo, Medio y Alto.
3.5. Preprocesamiento realizado
El preprocesamiento se desarrolló sobre el DataFrame df_modelo, que contiene 20.640 registros y 10 columnas: las nueve variables originales renombradas y la nueva variable objetivo.

Las comprobaciones y decisiones fueron:
1. Valores nulos: no se encontraron valores nulos en las columnas.
2. Registros duplicados: no se encontraron registros duplicados.
3. Tipos de datos: las variables numéricas son de tipo float64 y Categoria_valor es categórica.
4. Valores extremos: se aplicó el método del rango intercuartílico (IQR) a cinco variables. Se identificaron observaciones fuera de los límites estadísticos, pero se conservaron porque el análisis no demuestra que sean errores.
5. Selección de variables: se utilizaron ocho predictoras y se excluyó Valor_mediano_vivienda para evitar fuga de información.
6. División de datos: se separaron los datos en 80 % para entrenamiento y 20 % para prueba, con random_state=42 y estratificación por la variable objetivo.
7. Escalado: se configuró StandardScaler dentro del pipeline del modelo. El escalador aprende sus parámetros con los datos de entrenamiento y aplica esa transformación al conjunto de prueba.
La división produjo los siguientes tamaños:
Conjunto	Registros	Columnas predictoras
Entrenamiento	16.512	8
Prueba	4.128	8
Total	20.640	8


La distribución estratificada quedó equilibrada:
Categoría	Entrenamiento	Prueba
Bajo	5.507	1.377
Medio	5.501	1.375
Alto	5.504	1.376


3.6. Funcionamiento de la Regresión Logística
El algoritmo estima la probabilidad de que cada observación pertenezca a cada una de las categorías. En esta entrega, el flujo funciona así:
1. Recibe las variables: utiliza las ocho características socioeconómicas y geográficas de cada zona residencial.
2. Estandariza las variables: StandardScaler transforma las predictoras para que sus escalas sean comparables. Sus parámetros se calculan exclusivamente con el conjunto de entrenamiento.
3. Aprende durante el entrenamiento: la Regresión Logística ajusta coeficientes e interceptos a partir de los ejemplos etiquetados. Estos parámetros representan la relación aprendida entre las características y las clases.
4. Calcula probabilidades: para una observación, el modelo estima la probabilidad de pertenecer a cada clase. En la clasificación multiclase, Scikit-learn utiliza por defecto el esquema multiclase compatible con lbfgs.
5. Asigna una clase: selecciona la categoría con la probabilidad estimada más alta.
6. Se evalúa: compara las predicciones del conjunto de prueba con las etiquetas reales para calcular métricas y analizar los errores.
3.7. Parámetros principales
La configuración inicial utiliza LogisticRegression(max_iter=1000, random_state=42) dentro de un pipeline con StandardScaler.
Parámetro	Valor	Significado
max_iter	1000	Número máximo de iteraciones permitidas para la optimización.
random_state	42	Semilla para reproducibilidad cuando el procedimiento utiliza aleatoriedad.
C	1.0 (predeterminado)	Inverso de la intensidad de regularización. Un valor menor implica una regularización más fuerte.
solver	lbfgs (predeterminado)	Método de optimización que busca los parámetros del modelo.
fit_intercept	True (predeterminado)	Incluye un término independiente en el modelo.


StandardScaler calcula la media y la desviación estándar de cada predictor con los datos de entrenamiento y utiliza esos parámetros para transformar los datos de prueba.
3.8. Ventajas y limitaciones
Ventajas:
- Es apropiada para clasificación supervisada y puede trabajar con más de dos clases.
- Estima probabilidades para cada categoría.
- Sus coeficientes permiten estudiar la relación entre las variables predictoras y la clasificación.
- Tiene un costo computacional moderado y es una buena referencia inicial.
- Se integra con Pipeline para organizar el escalado y el clasificador, reduciendo el riesgo de fuga de información.
Limitaciones:
- Sus fronteras de decisión son lineales en el espacio de características transformadas; puede no representar relaciones complejas o no lineales.
- Puede verse afectada por valores extremos y por predictores con distribuciones muy asimétricas.
- La interpretación de los coeficientes requiere considerar el escalado y las relaciones entre las variables.
- Las probabilidades estimadas no garantizan que la clasificación sea correcta.
- El rendimiento depende de la calidad de los datos, de las variables seleccionadas y de los parámetros utilizados.
3.9. Justificación frente a la pregunta de negocio
La Regresión Logística permite abordar la pregunta de negocio como un problema de clasificación: con las características de una zona residencial, estima si pertenece al segmento Bajo, Medio o Alto.

Se seleccionó como modelo inicial porque es interpretable, permite obtener probabilidades por categoría y puede entrenarse con las ocho variables disponibles. La evaluación con datos de prueba permitirá determinar si su capacidad de clasificación es suficiente para apoyar el análisis de segmentos residenciales.

La clasificación representa una segmentación relativa construida con terciles del valor mediano de vivienda del dataset; no constituye una tasación individual ni una predicción directa del precio de una vivienda.
3.10. Estado de la entrega
En el notebook src/notebook/notebook_3.ipynb se documentan la pregunta de negocio, la construcción del DataFrame, la variable objetivo, el diagnóstico y el preprocesamiento. La selección y explicación del modelo se documentan en esta sección del README.

La siguiente etapa consiste en completar el entrenamiento y la evaluación del clasificador. Por tanto, esta documentación no anticipa métricas ni resultados de rendimiento de la Entrega 3.

El workflow independiente .github/workflows/ejecutar_ea3.yml está previsto para automatizar la ejecución de esta entrega. Su implementación y los artefactos que publicará deberán documentarse cuando estén definidos.

##  🔧 Cómo ejecutar el proyecto

### Localmente (o en Codespaces)

```bash

git clone https://github.com/KateLopez05/Machine-Learning.git

cd Machine-Learning

python -m venv .venv

source .venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

```

Se recomienda **Python 3.12**, la misma versión que usa el workflow de GitHub Actions.

En Windows (PowerShell), el entorno se crea y se activa así:

```powershell

py -3.12 -m venv .venv

.venv\Scripts\Activate.ps1

```

En Bash, el entorno virtual debe activarse con `source` (o con un punto). No se debe ejecutar `.venv/bin/activate` directamente. Cuando se activa bien, el prompt muestra `(.venv)`.

Después, abre el notebook de la entrega que quieras ejecutar, selecciona el kernel de .venv y ejecuta todas sus celdas. Para la Entrega 2:

```bash

jupyter nbconvert --to notebook --execute \

  --output-dir /tmp --output notebook_2_ejecutado \

  src/notebook/notebook_2.ipynb

```

El resumen queda en `src/notebook/reportes/entrega_2/resumen_modelos.txt`.

Para ejecutar manualmente la Entrega 3 desde la raíz del repositorio, cuando el notebook esté incorporado:
jupyter nbconvert --to notebook --execute \
  --output-dir /tmp --output notebook_3_ejecutado \
  src/notebook/notebook_3.ipynb
El workflow independiente de la Entrega 3 se implementará en .github/workflows/ejecutar_ea3.yml. Su configuración de publicación se completará cuando se defina el archivo de resultados que generará el notebook.

### Autenticación en Hugging Face (solo si el dataset lo exige)

Los datasets públicos no necesitan token. Si algún dataset es privado o pide aceptar condiciones de uso, inicia sesión con el entorno activado:

```bash

hf auth login

hf auth whoami

```

Abre `https://hf.co/oauth/device` en el navegador, escribe el código que muestra la terminal y autoriza el acceso. No escribas tokens en el código ni los subas a GitHub.

## :sparkles: Ejecución automática en GitHub Actions

Cada workflow se activa con un `push` a `main` que cambie los archivos de su entrega, e instala las dependencias en un entorno virtual con Python 3.12.

**Entrega 2 (`ejecutar_ea2.yml`)**

1. Instala las dependencias de `requirements.txt`.

2. Borra el resumen anterior, para que el archivo final corresponda a esta ejecución.

3. Ejecuta `notebook_2.ipynb` completo: dataset, DataFrame, preprocesamiento, ambos modelos y resumen.

4. Verifica que `src/notebook/reportes/entrega_2/resumen_modelos.txt` exista y no esté vacío.

5. Publica dos artefactos descargables: `entrega-2-resumen` (el `.txt`) y `entrega-2-notebook-ejecutado`.

6. Guarda una copia del `.txt` en el repositorio con un commit automático `[skip ci]`.

El notebook ejecutado se publica solo como artefacto y no reemplaza al `notebook_2.ipynb` del repositorio.

**Cómo ver y descargar los resultados:** en la pestaña **Actions**, elige **Ejecutar EA2**, abre una ejecución y baja hasta la sección **Artifacts**.

**Ejecución exitosa de la entrega 2:** [ver ejecución](https://github.com/KateLopez05/Machine-Learning/actions/runs/39890508964)

Como los workflows también suben commits, conviene hacer `git pull` antes de empezar a trabajar y antes de cada `git push`.

## Control de versiones y trabajo en equipo

- Los avances se registran con commits de mensaje claro (por ejemplo, `Agrego workflow de la entrega 2`), y los automáticos del workflow empiezan con `chore:`.

- Antes de editar y antes de subir, se hace `git pull`.

- Se evita editar el mismo notebook al mismo tiempo, porque los archivos `.ipynb` generan conflictos con facilidad.