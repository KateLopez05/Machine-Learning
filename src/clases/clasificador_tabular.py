"""Flujo reutilizable para entrenar y evaluar un clasificador con datos tabulares.

Sigue la misma estructura que ``ClasificadorTexto`` y ``RegresionLineal``:

    load_dataset   -> (el DataFrame ya viene cargado)
    make_pipeline  -> construir_modelo
    fit            -> entrenar
    score          -> evaluar
    predict        -> predecir

y agrega lo que el código mínimo no tenía: validar los datos, dividir de forma
estratificada, guardar el modelo con fecha, extraer sus parámetros y ejecutar
todo el flujo en orden.

La diferencia con ``ClasificadorTexto`` es la entrada: aquí las predictoras son
columnas numéricas (no texto), por lo que el pipeline usa ``StandardScaler`` en
lugar de ``TfidfVectorizer``.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ClasificadorTabular:
    """Clasifica filas de un DataFrame numérico en categorías.

    Args:
        columnas_X: nombre o lista de nombres de las columnas predictoras.
        columna_objetivo: columna con la clase (y).
        clases: orden en que se muestran las clases en las métricas, por
            ejemplo ``["Bajo", "Medio", "Alto"]``. Si es ``None`` se usa el
            orden alfabético de scikit-learn.
        modelo: clasificador de scikit-learn sin entrenar. Por defecto,
            ``LogisticRegression(max_iter=1000, random_state=random_state)``.
            Se copia con ``clone`` para poder reutilizar el mismo objeto.
        escalar: si es ``True``, el pipeline incluye ``StandardScaler``, que
            aprende media y desviación solo con los datos de entrenamiento.
            Usa ``False`` si las predictoras ya llegan escaladas.
        test_size: proporción reservada para prueba, entre 0 y 1.
        random_state: semilla para repetir la misma partición.
        directorio_modelos: carpeta donde se guardan los ``.pkl``.
    """

    def __init__(
        self,
        columnas_X: str | list[str] | tuple[str, ...],
        columna_objetivo: str,
        clases: list[str] | None = None,
        modelo: Any | None = None,
        escalar: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
        directorio_modelos: str | Path = "modelos",
    ) -> None:
        self.columnas_X = self._normalizar_nombres_X(columnas_X)
        if not isinstance(columna_objetivo, str) or not columna_objetivo.strip():
            raise ValueError("columna_objetivo debe ser un nombre de columna no vacío.")
        if columna_objetivo in self.columnas_X:
            raise ValueError("La variable objetivo no puede ser también una predictora.")
        if not 0 < test_size < 1:
            raise ValueError("test_size debe estar entre 0 y 1, sin incluirlos.")
        if not isinstance(random_state, int) or isinstance(random_state, bool) or random_state < 0:
            raise ValueError("random_state debe ser un entero mayor o igual que cero.")

        self.columna_objetivo = columna_objetivo
        self.clases = list(clases) if clases is not None else None
        self.modelo = modelo if modelo is not None else LogisticRegression(
            max_iter=1000, random_state=random_state
        )
        self.escalar = escalar
        self.test_size = test_size
        self.random_state = random_state
        self.directorio_modelos = Path(directorio_modelos)

        self.pipeline: Pipeline | None = None
        self.ruta_modelo: Path | None = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.predicciones: np.ndarray | None = None
        self.metricas: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # preparar y dividir
    # ------------------------------------------------------------------ #
    def preparar_datos(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Valida el DataFrame y devuelve X (numérico) e y (etiquetas como texto).

        - Las columnas predictoras se convierten a número; las filas con
          valores no numéricos, nulos o infinitos se eliminan.
        - Las filas sin etiqueta se eliminan.
        - Exige al menos 2 clases y 2 ejemplos por clase (para estratificar).
        - Conserva el índice original del DataFrame, para poder comparar con
          otros modelos que usen las mismas filas.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un pandas.DataFrame.")
        faltantes = set(self.columnas_X + [self.columna_objetivo]).difference(df.columns)
        if faltantes:
            raise KeyError(f"Faltan columnas requeridas: {sorted(faltantes)}.")

        X = df[self.columnas_X].apply(pd.to_numeric, errors="coerce")
        X = X.where(np.isfinite(X), np.nan)
        etiquetas = df[self.columna_objetivo]
        etiqueta_valida = etiquetas.notna() & (etiquetas.astype(str).str.strip() != "")
        validas = X.notna().all(axis=1) & etiqueta_valida

        X = X[validas]
        y = etiquetas[validas].astype(str).str.strip()
        eliminadas = len(df) - len(X)

        conteo = y.value_counts()
        if len(conteo) < 2:
            raise ValueError("Se necesitan al menos dos clases distintas.")
        if (conteo < 2).any():
            raise ValueError(f"Cada clase necesita al menos 2 ejemplos: {conteo.to_dict()}.")
        if self.clases is not None:
            ausentes = set(self.clases).difference(conteo.index)
            if ausentes:
                raise ValueError(f"Las clases {sorted(ausentes)} no aparecen en los datos.")
        print(
            f"[preparar_datos] {len(X)} filas válidas; {eliminadas} filas eliminadas; "
            f"X={self.columnas_X}; clases: {conteo.to_dict()}."
        )
        return X, y

    def dividir_datos(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide en entrenamiento y prueba conservando la proporción de cada clase."""
        self._validar_pares(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        print(
            f"[dividir_datos] stratify=y; entrenamiento={len(X_train)}; prueba={len(X_test)}; "
            f"random_state={self.random_state}."
        )
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------ #
    # construir
    # ------------------------------------------------------------------ #
    def construir_modelo(self) -> Pipeline:
        """Encadena el escalado (opcional) y el clasificador."""
        pasos = []
        if self.escalar:
            pasos.append(("escalado", StandardScaler()))
        pasos.append(("modelo", clone(self.modelo)))
        self.pipeline = Pipeline(pasos)
        descripcion = " -> ".join(type(paso).__name__ for _, paso in pasos)
        print(f"[construir_modelo] {descripcion}")
        return self.pipeline

    # ------------------------------------------------------------------ #
    # entrenar
    # ------------------------------------------------------------------ #
    def entrenar(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
        """Ajusta el pipeline y lo guarda como .pkl con fecha y hora.

        El escalador aprende media y desviación solo con ``X_train``; a los
        datos de prueba solo los transforma al predecir.
        """
        self._validar_pares(X_train, y_train)
        if self.pipeline is None:
            self.construir_modelo()
        self.pipeline.fit(X_train[self.columnas_X], y_train)

        self.directorio_modelos.mkdir(parents=True, exist_ok=True)
        fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.ruta_modelo = self.directorio_modelos / f"{fecha}_{type(self.modelo).__name__}.pkl"
        with self.ruta_modelo.open("wb") as archivo:
            pickle.dump(self.pipeline, archivo)
        print(f"[entrenar] {len(X_train)} filas; modelo guardado en {self.ruta_modelo}")
        return {"pipeline": self.pipeline, "ruta_modelo": self.ruta_modelo, "fecha_modelo": fecha}

    def cargar_modelo(self, ruta_modelo: str | Path) -> Pipeline:
        """Carga un .pkl guardado por ``entrenar`` para volver a usarlo."""
        ruta = Path(ruta_modelo)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el modelo: {ruta}.")
        with ruta.open("rb") as archivo:
            self.pipeline = pickle.load(archivo)
        self.ruta_modelo = ruta
        return self.pipeline

    # ------------------------------------------------------------------ #
    # predecir
    # ------------------------------------------------------------------ #
    def _modelo_listo(self) -> Pipeline:
        if self.pipeline is None or not hasattr(self.pipeline, "classes_"):
            raise RuntimeError("Primero entrena el modelo (entrenar o ejecutar) o cárgalo (cargar_modelo).")
        return self.pipeline

    def _como_dataframe(self, X: pd.DataFrame | list[list[float]]) -> pd.DataFrame:
        """Acepta un DataFrame o una lista de filas, y las ordena como en el entrenamiento."""
        if isinstance(X, pd.DataFrame):
            faltantes = set(self.columnas_X).difference(X.columns)
            if faltantes:
                raise KeyError(f"Faltan columnas requeridas: {sorted(faltantes)}.")
            return X[self.columnas_X]
        datos = pd.DataFrame(X)
        if datos.shape[1] != len(self.columnas_X):
            raise ValueError(f"Cada fila debe tener {len(self.columnas_X)} valores.")
        datos.columns = self.columnas_X
        return datos

    def predecir_proba(self, X: pd.DataFrame | list[list[float]]) -> pd.DataFrame:
        """Probabilidad de cada clase por fila (una columna por clase)."""
        pipe = self._modelo_listo()
        if not hasattr(pipe, "predict_proba"):
            raise TypeError(f"{type(self.modelo).__name__} no entrega probabilidades.")
        return pd.DataFrame(pipe.predict_proba(self._como_dataframe(X)), columns=pipe.classes_)

    def predecir(self, X: pd.DataFrame | list[list[float]]) -> np.ndarray:
        """Devuelve la clase de cada fila."""
        return self._modelo_listo().predict(self._como_dataframe(X))

    # ------------------------------------------------------------------ #
    # evaluar
    # ------------------------------------------------------------------ #
    def evaluar(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        """Calcula exactitud, precisión, recall y F1 por clase, macro y ponderado, y la matriz de confusión."""
        self._validar_pares(X_test, y_test)
        pred = self.predecir(X_test)
        self.predicciones = pred
        clases = self.clases if self.clases is not None else list(self._modelo_listo().classes_)

        p, r, f, n = precision_recall_fscore_support(y_test, pred, labels=clases, zero_division=0)
        por_clase = pd.DataFrame({"precision": p, "recall": r, "f1": f, "casos": n}, index=clases)
        matriz = pd.DataFrame(
            confusion_matrix(y_test, pred, labels=clases),
            index=[f"real {c}" for c in clases],
            columns=[f"pred {c}" for c in clases],
        )
        reporte = classification_report(y_test, pred, labels=clases, zero_division=0, digits=3)
        self.metricas = {
            "exactitud": float(accuracy_score(y_test, pred)),
            "f1_macro": float(por_clase["f1"].mean()),
            "f1_ponderado": float(np.average(por_clase["f1"], weights=por_clase["casos"])),
            "por_clase": por_clase,
            "matriz": matriz,
            "reporte": reporte,
        }
        print(
            f"[evaluar] exactitud={self.metricas['exactitud']:.3f}; "
            f"F1 macro={self.metricas['f1_macro']:.3f}; F1 ponderado={self.metricas['f1_ponderado']:.3f}."
        )
        return self.metricas

    # ------------------------------------------------------------------ #
    # parámetros
    # ------------------------------------------------------------------ #
    def obtener_parametros(self) -> dict[str, Any]:
        """Devuelve los hiperparámetros, los coeficientes por clase y los interceptos.

        Los coeficientes están en la escala de las variables que recibe el
        clasificador: si ``escalar=True``, son coeficientes estandarizados.
        """
        pipe = self._modelo_listo()
        modelo = pipe.named_steps["modelo"]
        parametros: dict[str, Any] = {"hiperparametros": modelo.get_params(), "coeficientes": None, "interceptos": None}
        if hasattr(modelo, "coef_"):
            coef = np.atleast_2d(modelo.coef_)
            clases = list(modelo.classes_)
            columnas = clases if coef.shape[0] == len(clases) else clases[1:]
            parametros["coeficientes"] = pd.DataFrame(coef.T, index=self.columnas_X, columns=columnas)
            parametros["interceptos"] = pd.Series(np.atleast_1d(modelo.intercept_), index=columnas)
        print(f"[obtener_parametros] {type(modelo).__name__}; {len(self.columnas_X)} predictoras.")
        return parametros

    # ------------------------------------------------------------------ #
    # todo el flujo
    # ------------------------------------------------------------------ #
    def ejecutar(self, df: pd.DataFrame) -> dict[str, Any]:
        """Ejecuta en orden: preparar, dividir, construir, entrenar, evaluar y obtener parámetros."""
        print("[ejecutar][1/6] preparar_datos")
        X, y = self.preparar_datos(df)
        print("[ejecutar][2/6] dividir_datos")
        self.X_train, self.X_test, self.y_train, self.y_test = self.dividir_datos(X, y)
        print("[ejecutar][3/6] construir_modelo")
        self.construir_modelo()
        print("[ejecutar][4/6] entrenar")
        entrenamiento = self.entrenar(self.X_train, self.y_train)
        print("[ejecutar][5/6] evaluar")
        metricas = self.evaluar(self.X_test, self.y_test)
        print("[ejecutar][6/6] obtener_parametros")
        parametros = self.obtener_parametros()
        print("[ejecutar] flujo completo terminado.")
        return {
            "ruta_modelo": entrenamiento["ruta_modelo"],
            "metricas": metricas,
            "parametros": parametros,
            "X_test": self.X_test,
            "y_test": self.y_test,
            "predicciones": self.predicciones,
        }

    # ------------------------------------------------------------------ #
    # utilidades
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalizar_nombres_X(columnas_X: str | list[str] | tuple[str, ...]) -> list[str]:
        """Convierte uno o varios nombres de columnas en una lista validada."""
        if isinstance(columnas_X, str):
            nombres = [columnas_X]
        elif isinstance(columnas_X, (list, tuple)):
            nombres = list(columnas_X)
        else:
            raise TypeError("columnas_X debe ser un nombre o una lista de nombres.")
        if not nombres or any(not isinstance(n, str) or not n.strip() for n in nombres):
            raise ValueError("Los nombres de las columnas no pueden estar vacíos.")
        if len(set(nombres)) != len(nombres):
            raise ValueError("columnas_X no puede tener nombres repetidos.")
        return nombres

    @staticmethod
    def _validar_pares(X: pd.DataFrame, y: pd.Series) -> None:
        """Comprueba que X e y forman un par válido: mismos tamaños, sin vacíos ni inválidos."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas.DataFrame.")
        if not isinstance(y, pd.Series):
            raise TypeError("y debe ser una pandas.Series.")
        if X.empty or y.empty or len(X) != len(y):
            raise ValueError("X e y deben tener filas y longitudes compatibles.")
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("X contiene valores no numéricos o inválidos.")
        if y.isna().any():
            raise ValueError("y contiene valores nulos.")
