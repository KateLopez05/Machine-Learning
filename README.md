# 📚 Machine Learning

Repositorio de las entregas de la materia de Machine Learning. Cada entrega es un notebook que se ejecuta automáticamente en GitHub Actions y deja un resumen en .txt generado por código.

## 👩‍💻 Autores

- Katerin Lopez Moros ([@KateLopez05](https://github.com/KateLopez05))
- Bayron Meza Guzman  ([@BayronMgDev](https://github.com/BayronMgDev))

## 📌 Entregas

| Entrega | Notebook | Qué hace | Workflow | Resumen generado |
|---|---|---|---|---|
| 1 | `src/notebook/notebook_1.ipynb` | Carga de un dataset desde Hugging Face, transformaciones mínimas con pandas y reporte en texto | `.github/workflows/ejecutar.yml` | `src/notebook/reportes/entrega_1` |
| 2 | `src/notebook/notebook_2.ipynb` | Construcción del DataFrame, preprocesamiento, regresión lineal, clasificación y resumen de ambos modelos | `.github/workflows/ejecutar_ea2.yml` | `src/reportes/entrega_2/resumen_modelos.txt` |

Cada entrega tiene su propio workflow, y cada workflow solo se activa cuando cambian los archivos de su entrega, para que trabajar en una no altere la ejecución de la otra.

## 🗄️ Estructura

```
.github/workflows/
  ejecutar.yml             Entrega 1: ejecuta notebook_1.ipynb
  ejecutar_ea2.yml         Entrega 2: ejecuta notebook_2.ipynb y publica el resumen
src/
  clases/                  Clases reutilizables (carga desde Hugging Face y reportes)
  notebook/
    notebook_1.ipynb       Entrega 1
    notebook_2.ipynb       Entrega 2
  reportes/
    entrega_1/
      california_housing.txt Resumen automático de la entrega 1
    entrega_2/
      resumen_modelos.txt   Resumen automático de la entrega 2
requirements.txt           Dependencias del proyecto
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
df_transformado = df.rename(columns={...})                       # 1
df_t2 = df_transformado[df_transformado["Edad_vivienda"] > 20]   # 2
df_t3 = df_t2[["Ingreso_mediano", "Edad_vivienda", "Promedio_habitaciones",
               "Poblacion", "Valor_mediano_vivienda"]]           # 3
df_t4 = df_t3.assign(Valor_en_dolares=df_t3["Valor_mediano_vivienda"] * 100000)  # 4
```

Cada operación devuelve un DataFrame nuevo y no modifica el anterior. El `.txt` final corresponde a `df_t4`, el DataFrame ya transformado.

## :white_check_mark: Entrega 2: regresión y clasificación

**Dataset:** [`BrianSuToronto/california_housing`](https://huggingface.co/datasets/BrianSuToronto/california_housing) (censo de California de 1990), split `train`. Cada fila describe una zona. El DataFrame se construye por código con `datasets.load_dataset`, sin descargas manuales.

**Flujo del notebook**

1. **Construcción del DataFrame:** carga automática, tamaño, columnas, tipos y muestra.
2. **Preprocesamiento:** columnas renombradas al español, variable de clasificación `Categoria_valor` (terciles de `Valor_mediano_vivienda`), enriquecimiento con `Habitaciones_por_dormitorio`, transformación logarítmica en cuatro variables sesgadas, división 80/20 estratificada (`random_state=42`) y escalado con `StandardScaler` ajustado solo con entrenamiento.
3. **Regresión lineal:** predice `Valor_mediano_vivienda` con `LinearRegression` (parámetros por defecto).
4. **Clasificación:** predice `Categoria_valor` (Bajo, Medio, Alto) con `LogisticRegression(max_iter=1000, random_state=42)`.
5. **Resumen automático:** escribe `src/reportes/entrega_2/resumen_modelos.txt` con el dataset, el preprocesamiento, los parámetros, las métricas, la interpretación, la fecha de ejecución y las limitaciones. Todo el contenido sale de la ejecución, no está escrito a mano.

**Resultados de referencia** (última ejecución, datos de prueba)

| Modelo | Métricas | Referencia |
|---|---|---|
| Regresión lineal | MAE 0.497 (≈ 49.653 USD), RMSE 0.691, R² 0.651 | MAE del modelo base: 0.922 |
| Regresión logística | Exactitud 0.730, F1 macro 0.730 | Exactitud del modelo base: 0.334 |

El valor de `Valor_mediano_vivienda` está en cientos de miles de dólares. En clasificación, la clase `Medio` es la más difícil de reconocer, y cerca del 95 % de los errores ocurre entre clases vecinas.

##  🔧 Cómo ejecutar el proyecto

### Localmente (o en Codespaces)

```bash
git clone https://github.com/KateLopez05/Machine-Learning.git
cd Machine-Learning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Bash, el entorno virtual debe activarse con `source` (o con un punto). No se debe ejecutar `.venv/bin/activate` directamente. Cuando se activa bien, el prompt muestra `(.venv)`.

Después, abre `src/notebook/notebook_2.ipynb`, selecciona el kernel de `.venv` y ejecuta todas las celdas, o ejecútalo desde la terminal:

```bash
jupyter nbconvert --to notebook --execute \
  --output-dir /tmp --output notebook_2_ejecutado \
  src/notebook/notebook_2.ipynb
```

El resumen queda en `src/reportes/entrega_2/resumen_modelos.txt`.

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
4. Verifica que `src/reportes/entrega_2/resumen_modelos.txt` exista y no esté vacío.
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