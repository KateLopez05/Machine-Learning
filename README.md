# 📚 Machine Learning

Repositorio académico de las actividades y proyectos de la asignatura **Machine Learning**. Contiene notebooks de análisis de datos, preprocesamiento, construcción de modelos de aprendizaje automático, evaluación de resultados y automatización de ejecuciones mediante GitHub Actions.

## 👩‍💻 Autores

- Katerin Lopez Moros — [@KateLopez05](https://github.com/KateLopez05)
- Bayron Meza Guzman — [@BayronMgDev](https://github.com/BayronMgDev)

## 🧰 Tecnologías utilizadas

- Python 3.12
- Jupyter Notebook
- Pandas y NumPy
- Matplotlib
- Scikit-learn
- Hugging Face Datasets
- Git y GitHub
- GitHub Actions

Las dependencias del proyecto se centralizan en `requirements.txt`.

## 📌 Contenido del proyecto

| Entrega | Notebook | Descripción | Automatización |
|---|---|---|---|
| 1 | `notebook_1.ipynb` | Carga de datasets, transformación de DataFrames y generación de reportes. | GitHub Actions |
| 2 | `notebook_2.ipynb` | Preprocesamiento, regresión lineal, clasificación y evaluación de modelos. | GitHub Actions |
| 3 | `notebook_3.ipynb` | Clasificación supervisada de segmentos de valor de vivienda y optimización de hiperparámetros. | GitHub Actions |

## 🗂️ Estructura del repositorio

```text
Machine-Learning/
│
├── .github/
│   └── workflows/
│       ├── ejecutar.yml
│       ├── ejecutar_ea2.yml
│       └── ejecutar_ea3.yml
│
├── src/
│   ├── clases/
│   │   ├── loader_hf.py
│   │   ├── reporte.py
│   │   ├── regresion_lineal.py
│   │   └── clasificador_tabular.py
│   │
│   └── notebook/
│       ├── notebook_1.ipynb
│       ├── notebook_2.ipynb
│       ├── notebook_3.ipynb
│       │
│       ├── data/
│       │
│       ├── modelos/
│       │   ├── entrega_2/
│       │   └── entrega_3/
│       │
│       └── reportes/
│           ├── entrega_1/
│           ├── entrega_2/
│           │   └── resumen_modelos.txt
│           └── entrega_3/
│               ├── resumen_modelo.txt
│               ├── hallazgos_actividad_3.md
│               └── figuras/
│                   ├── 01_distribucion_categorias.png
│                   ├── 02_matriz_confusion.png
│                   └── 03_comparacion_hiperparametros.png
│
├── requirements.txt
└── README.md
```

Los archivos generados pueden variar según la ejecución de cada actividad. Los modelos entrenados se almacenan en las carpetas correspondientes y no son necesarios para instalar o ejecutar inicialmente el proyecto.

---

# 🧪 Entrega 1: Carga y transformación de datos

Esta actividad desarrolla un flujo de trabajo para descargar datasets desde Hugging Face, convertirlos en DataFrames y aplicar transformaciones utilizando Pandas.

### Flujo de trabajo

1. Carga automática de un dataset mediante `HFDatasetLoader`.
2. Conversión de los datos a un DataFrame.
3. Renombrado de columnas.
4. Filtrado de registros.
5. Selección de columnas.
6. Creación de nuevas variables.
7. Generación de un reporte mediante la clase `Reportes`.

### Transformaciones principales

| Operación | Método |
|---|---|
| Renombrar columnas | `rename()` |
| Filtrar registros | Condiciones booleanas |
| Seleccionar columnas | Indexación de Pandas |
| Crear variables | `assign()` |

El reporte generado contiene información sobre las dimensiones del DataFrame, tipos de datos, valores nulos, duplicados y estadísticas descriptivas.

---

# 📊 Entrega 2: Regresión y clasificación

**Dataset:** [California Housing - Hugging Face](https://huggingface.co/datasets/BrianSuToronto/california_housing)

Esta actividad implementa modelos de aprendizaje automático para estudiar y analizar los datos de vivienda de California.

### Flujo de trabajo

1. Construcción automática del DataFrame desde Hugging Face.
2. Exploración y análisis de los datos.
3. Preprocesamiento y transformación de variables.
4. División de los datos en entrenamiento y prueba.
5. Entrenamiento de un modelo de regresión lineal.
6. Construcción de un modelo de clasificación.
7. Evaluación mediante métricas.
8. Generación automática de un resumen de resultados.

### Clases reutilizables

El proyecto incorpora clases propias para organizar y reutilizar los procesos:

- `HFDatasetLoader`: descarga datasets y los convierte en DataFrames.
- `Reportes`: genera reportes descriptivos.
- `RegresionLineal`: encapsula el entrenamiento y evaluación de modelos de regresión.
- `ClasificadorTabular`: implementa flujos de clasificación con variables numéricas.

### Resultados de referencia

| Modelo | Métrica | Resultado |
|---|---|---:|
| Regresión lineal | MAE | 0.497 |
| Regresión lineal | RMSE | 0.691 |
| Regresión lineal | R² | 0.651 |
| Regresión logística | Accuracy | 0.730 |
| Regresión logística | F1 macro | 0.730 |

El MAE de regresión corresponde aproximadamente a 49.653 USD, teniendo en cuenta que el valor objetivo está expresado en cientos de miles de dólares.

**Reporte generado:**

`src/notebook/reportes/entrega_2/resumen_modelos.txt`

---

# 🤖 Entrega 3: Aprendizaje supervisado y clasificación de viviendas

Esta actividad desarrolla un modelo de clasificación multiclase para identificar si una zona residencial de California pertenece a un segmento de valor bajo, medio o alto.

## Pregunta de negocio

¿Qué características socioeconómicas y geográficas permiten identificar si una zona residencial de California pertenece a un segmento de valor de vivienda bajo, medio o alto?

### Dataset

[BrianSuToronto/california_housing](https://huggingface.co/datasets/BrianSuToronto/california_housing)

- Registros: 20.640.
- Columnas originales: 9.
- Variable objetivo: `Categoria_valor`.
- Categorías: Bajo, Medio y Alto.

La variable objetivo se construye mediante terciles de `Valor_mediano_vivienda`.

| Categoría | Registros |
|---|---:|
| Bajo | 6.884 |
| Medio | 6.876 |
| Alto | 6.880 |
| **Total** | **20.640** |

### Variables predictoras

El modelo utiliza ocho características:

- `Ingreso_mediano`
- `Edad_vivienda`
- `Promedio_habitaciones`
- `Promedio_dormitorio_hogar`
- `Poblacion`
- `Promedio_ocupacion`
- `Latitud`
- `Longitud`

La variable original `Valor_mediano_vivienda` se excluye de las predictoras para evitar fuga de información.

### Preprocesamiento

- Validación de valores nulos y duplicados.
- Revisión de tipos de datos.
- Análisis de valores extremos mediante IQR.
- División estratificada de entrenamiento y prueba (80/20).
- Estandarización mediante `StandardScaler`.

### Modelo utilizado

Se implementó un pipeline con:

- `StandardScaler`
- `LogisticRegression`

La optimización de hiperparámetros se realizó con `GridSearchCV`, utilizando validación cruzada estratificada de cinco particiones y F1 macro como métrica de selección.

La mejor configuración encontrada fue:

```python
{
    "modelo__C": 10.0
}
```

### Resultados finales

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.7178 |
| F1 macro | 0.7187 |
| F1 ponderado | 0.7188 |
| F1 categoría Bajo | 0.7601 |
| F1 categoría Medio | 0.6034 |
| F1 categoría Alto | 0.7927 |

La categoría Medio representa la mayor dificultad de clasificación. La mayoría de los errores se presenta entre categorías vecinas.

### Visualizaciones generadas

**Distribución de categorías**

![Distribución de categorías](src/notebook/reportes/entrega_3/figuras/01_distribucion_categorias.png)

**Matriz de confusión**

![Matriz de confusión](src/notebook/reportes/entrega_3/figuras/02_matriz_confusion.png)

**Comparación de hiperparámetros**

![Comparación de hiperparámetros](src/notebook/reportes/entrega_3/figuras/03_comparacion_hiperparametros.png)

### Archivos de resultados

- `src/notebook/reportes/entrega_3/resumen_modelo.txt`
- `src/notebook/reportes/entrega_3/hallazgos_actividad_3.md`
- `src/notebook/reportes/entrega_3/figuras/`

El resumen registra la fecha de ejecución utilizando la zona horaria `America/Bogota`.

El modelo optimizado se guarda localmente en `src/notebook/modelos/entrega_3/`.

---

# ⚙️ Instalación y configuración

## Requisitos previos

- Python 3.12.
- Git.
- Acceso a internet para descargar los datasets desde Hugging Face.

## Clonar el repositorio

```bash
git clone https://github.com/KateLopez05/Machine-Learning.git

cd Machine-Learning
```

## Crear entorno virtual

```bash
python -m venv .venv
```

Activar el entorno:

**Linux, macOS y GitHub Codespaces**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

## Instalar dependencias

```bash
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

## Verificar dependencias

```bash
python -m pip check
```

---

# ▶️ Ejecución de los notebooks

Se puede ejecutar cada notebook desde Jupyter o Visual Studio Code, seleccionando el kernel correspondiente al entorno virtual `.venv`.

Para iniciar Jupyter:

```bash
jupyter notebook
```

Para ejecutar los notebooks desde la terminal:

**Entrega 1**

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  src/notebook/notebook_1.ipynb
```

**Entrega 2**

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  src/notebook/notebook_2.ipynb
```

**Entrega 3**

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  src/notebook/notebook_3.ipynb
```

Los notebooks pueden requerir acceso a internet para descargar los datasets.

---

# 🔄 Automatización con GitHub Actions

El repositorio dispone de tres workflows independientes para ejecutar las actividades.

| Workflow | Actividad | Archivo |
|---|---|---|
| Ejecutar EA1 | Entrega 1 | `ejecutar.yml` |
| Ejecutar EA2 | Entrega 2 | `ejecutar_ea2.yml` |
| Ejecutar EA3 | Entrega 3 | `ejecutar_ea3.yml` |

### Funcionamiento general

Los workflows permiten:

1. Obtener el código del repositorio.
2. Configurar Python 3.12.
3. Instalar las dependencias.
4. Ejecutar el notebook correspondiente.
5. Generar y verificar los resultados.
6. Publicar los archivos configurados como artefactos descargables.
7. Actualizar los resultados correspondientes en GitHub cuando el workflow lo tiene configurado.

Las ejecuciones pueden iniciarse automáticamente mediante los eventos `push` configurados en cada workflow o manualmente desde GitHub Actions cuando está habilitado `workflow_dispatch`.

### Consultar ejecuciones

1. Ingresar al repositorio en GitHub.
2. Abrir la pestaña **Actions**.
3. Seleccionar el workflow correspondiente.
4. Consultar el estado y los registros de ejecución.
5. Descargar los artefactos disponibles en la sección **Artifacts** de la ejecución.

Los artefactos permiten recuperar resultados generados sin necesidad de ejecutar nuevamente los notebooks.

---

# 📝 Control de versiones

Para mantener sincronizado el entorno local con GitHub:

```bash
git pull origin main
```

Después de realizar cambios:

```bash
git add .

git commit -m "docs: actualizar documentación del proyecto"

git push origin main
```

**Importante:** algunos workflows generan commits automáticos al actualizar los reportes. Por esta razón, se recomienda sincronizar la rama local con `origin/main` antes de enviar nuevos cambios.

Se recomienda utilizar mensajes de commit descriptivos, evitar modificar simultáneamente el mismo notebook y no subir credenciales, tokens ni información sensible.

---

## 📌 Estado del proyecto

Las tres entregas cuentan con notebooks ejecutables y procesos automatizados mediante GitHub Actions.

El repositorio centraliza el código, las clases reutilizables, las dependencias, los reportes y la documentación necesaria para reproducir los análisis.

**Repositorio:** [KateLopez05/Machine-Learning](https://github.com/KateLopez05/Machine-Learning)
