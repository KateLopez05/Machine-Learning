# 📚 Machine Learning

Repositorio de las entregas de la materia de **Machine Learning**: un notebook ejecutable que carga un dataset desde Hugging Face, le aplica transformaciones mínimas con pandas y deja un resumen `.txt`.

## 👩‍💻 Autores

- Katerin Lopez Moros ([@KateLopez05](https://github.com/KateLopez05))
- Bayron Meza Guzman  ([@BayronMgDev](https://github.com/BayronMgDev))

## 🗄️ Estructura

```
.github/workflows/               Workflow de GitHub Actions que ejecuta el notebook
src/
  clases/                        Clases reutilizables
    __init__.py
    loader_hf.py                  Carga del dataset desde Hugging Face a un DataFrame
    reporte.py                    Reportes: genera y guarda el resumen .txt de un DataFrame
  notebook/
    data/
        california_housing.csv    Dataset con formato .csv
    reportes/
        california_housing.txt    Reporte generado en .txt
    notebook_1.ipynb              Notebook de la entrega (carga, transformaciones y reporte)
requirements.txt                  Dependencias del proyecto
```

## :pushpin: Flujo de la entrega

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

## :white_check_mark: Transformaciones mínimas

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

## :pencil2: Uso

```bash
cd /workspaces/Machine-Learning
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Bash, el entorno virtual debe activarse con `source` (o con un punto):

```bash
source .venv/bin/activate
# equivalente:
. .venv/bin/activate
```

No se debe ejecutar `.venv/bin/activate` directamente, porque `activate` es un script que modifica la sesión actual de la terminal. Cuando se activa correctamente, el prompt muestra `(.venv)`.

Después abre `src/notebook/notebook_1.ipynb`, selecciona el kernel de `.venv` y ejecuta todas las celdas de arriba hacia abajo.

## 📐 Autenticación en Hugging Face

Para descargar datasets públicos normalmente no se necesita un token. Si el dataset es privado o requiere aceptar condiciones de uso, inicia sesión con el entorno virtual activado:

```bash
hf auth login
```

La terminal muestra una URL (`https://hf.co/oauth/device`) y un código: ábrela en el navegador, escribe el código y autoriza el acceso. Comprueba que funcionó:

```bash
hf auth whoami
```

La sesión queda guardada localmente. No escribas tokens en archivos Python ni los subas a GitHub; si uno se expone, revócalo desde [Access Tokens](https://huggingface.co/settings/tokens) y crea uno nuevo.

## ⚙️ Carga de un dataset como DataFrame

El dataset se descarga con la librería `datasets` y se convierte a un `pandas.DataFrame`:

```python
from datasets import load_dataset

ds = load_dataset("gvlassis/california_housing")
df = ds["train"].to_pandas()

print(df.shape)
print(df.columns.tolist())
```

La clase de carga del proyecto está en `src/clases/loader_hf.py`.

## :page_facing_up: Reporte en texto

`Reportes` recibe el DataFrame en cada método, así que se crea sin argumentos:

```python
from clases.reporte import Reportes

reportes = Reportes()
reportes.imprimir_reporte(df_t4)                              # muestra el resumen en consola
ruta = reportes.guardar_reporte(df_t4, "salidas/resumen.txt")  # escribe el .txt
print(f"Reporte guardado en: {ruta}")
```

El reporte incluye filas, columnas, duplicados, tipos de dato, valores nulos por columna y estadísticas de las columnas numéricas. Para ejecutarlo fuera del notebook, desde la raíz del proyecto:

```bash
PYTHONPATH=src python -c '
from clases.reporte import Reportes
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3]})
Reportes().imprimir_reporte(df)
'
```

## :sparkles: Automatización

El workflow de `.github/workflows/` ejecuta el notebook y sube los resultados a `main` con un commit automático. Antes de hacer `git push`, el workflow trae los cambios remotos con `git pull --rebase origin main`, para evitar que `main` lo rechace por haber avanzado mientras corría.

Como el workflow también sube commits, haz `git pull` antes de empezar a trabajar y antes de cada `git push`.

## :busts_in_silhouette: Trabajo en equipo

- Haz `git pull` antes de editar y antes de subir cambios.
- Evita editar el mismo notebook al mismo tiempo: los archivos `.ipynb` generan conflictos con facilidad.
- Si quieres aislar tus cambios, trabaja en una rama propia y combínala con un Pull Request.
