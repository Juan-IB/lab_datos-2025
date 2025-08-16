# -*-coding:utf-8 -*-
"""
@File    :   clasificacion_multiclase.py
@Time    :   2025/03/01 21:55:08
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Entrenamiento de modelo no supervisado de clasificación multiclase.
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from funciones import mostrar_matriz_de_confusion, crop_imagenes
from sklearn import tree, metrics, model_selection
from random import sample
import seaborn as sns
from itertools import product
from derivados_mnist import mnist_values_n, mnist_labels

# %%
print("arbol_multiclase.py: Cargando...", end="")
# mnist_values_n = pd.read_csv(r"Resources/mnist_values_n.csv")
# mnist_labels = pd.read_csv(r"Resources/mnist_labels.csv")["labels"]

atributos = list(mnist_values_n.columns)
clases = [str(i) for i in sorted(list(mnist_labels.unique()))]

"""
╔═════════════════════════════════════════════════════════════════════════════╗
║Separación del conjunto de datos original                                    ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
X_dev, X_heldout, y_dev, y_heldout = model_selection.train_test_split(
    mnist_values_n,
    mnist_labels,
    test_size=0.1,
    random_state=0,
    shuffle=True,
    stratify=mnist_labels,
)
# datasets croppeados, para evaluar el desempeño del croppeo después.
X_dev_cropped = crop_imagenes(X_dev, 5)
X_heldout_cropped = crop_imagenes(X_heldout, 5)


# Crear arbol de decisión y variar la profundidad para maximizar la precisión
def probar_depths(depths: list[int] | range) -> pd.DataFrame:
    """Evalúa el desempeño de un arbol de decisiones de diferente profundidad, entrenando con el conjunto dev y evaluando sobre el heldout (no KFold por ahora).
    Args:
        depths (list[int] | range): Lista de profundidades a evaluar
    Returns:
        pd.DataFrame: Resultados de la evaluación, con columnas 'depth' y 'score'.
    """
    results_k = {"depth": [], "score": []}
    print(f"Probando distintos depths (criterio gini), sin KFolding:")
    for i, depth in enumerate(depths):
        print(f"modelo {i+1}/{len(depths)} (depth={depth}):")
        modelo = tree.DecisionTreeClassifier(max_depth=depth, random_state=0)
        modelo.fit(X_dev, y_dev)
        results_k["depth"].append(depth)
        score = modelo.score(X_heldout, y_heldout)
        results_k["score"].append(score)
        print(f"\tscore: {score:.2%}")

    results_k = pd.DataFrame(results_k)
    return results_k


# Evolución de la precisión según el aumento de k
def grafico_depths() -> tuple[Figure, Axes]:
    """Genera un gráfico de la exactitud en función de la profundidad.
    NO GRAFICA! Sólo configura el gráfico.

    Returns:
        tuple[Figure, Axes]: Figura y Ejes producidos. No grafica!
    """
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    # ax.set_xlim(left=min(depths))
    # ax.set_ylim(bottom=min(accuracies['score']))
    ax.yaxis.set_major_formatter("{:.2%}".format)
    ax.minorticks_on()
    ax.grid()
    ax.set_xlabel("Profundidad (k)")
    ax.set_ylabel("Exactitud")
    return fig, ax


def probar_depths_kfolding(
    depths: list[int] | range,
    nsplits: int,
    cropped: bool = False,
    criterio: str = "gini",
) -> pd.DataFrame:
    """Evalúa el desempeño de un arbol de decisiones de diferente profundidad, entrenando con el conjunto dev y evaluando sobre el heldout (no KFold por ahora).
    Args:
        depths (list[int] | range): Lista de profundidades a evaluar
    Returns:
        pd.DataFrame: Resultados de la evaluación, con columnas 'depth' y 'score'.
    """
    print(
        f"Probando distintos depths",
        "sobre dataset croppeado" if cropped else "",
        f"(criterio {criterio}):",
    )
    # STRATIFIED Kfold para conservar la misma proporción entre dígitos para cada fold!
    kf = model_selection.StratifiedKFold(
        nsplits,
        shuffle=True,
        random_state=0,
    )

    results = []

    total_time: float = 0
    n: int = 0

    for i, depth in enumerate(depths):
        print(f"modelo {i+1}/{len(depths)} ({criterio}, depth={depth}):")

        modelo = tree.DecisionTreeClassifier(
            criterion=criterio,
            max_depth=depth,
            random_state=0,
        )
        scores: dict[str, np.ndarray] = model_selection.cross_validate(
            modelo,
            X_dev_cropped if cropped else X_dev,
            y_dev,
            cv=kf,
            n_jobs=-1,
        )
        time = scores["fit_time"].sum() + scores["score_time"].sum()
        score = scores["test_score"].mean()
        total_time += time

        results.append(score)
        print(f"\ttime: {time:.1f}s, mean_score: {score:.2%}")

    results_df = pd.DataFrame(
        {"depth": depths, "score": results},
    )
    print(f"Total processor time: {total_time:.0f}s")
    return results_df


# Impacto en el rendimiento del hiperparámetro max_features
def probar_mfs_kfolding(
    mfs: np.ndarray[np.float64] | range,
    nsplits: int,
    cropped: bool = False,
    criterio: str = "gini",
) -> pd.DataFrame:
    """Evalúa el desempeño de un arbol de decisión de profundidad 10 en función de distintos max_features (tomar un menor número de columnas que el total de 784).
    Args:
        mfs (np.ndarray[np.float64] | range): rango de valores max_features a probar. Pueden ser fraccionarios (en intervalo (0;1] ) o enteros (número de atributos a tomar).
        nsplits (int): Cantidad de folds que promediar.
        cropped (bool): Usar o no el dataset croppeado.
        criterio (str): Criterio del modelo; 'gini', 'entropy' o 'log_loss'.
    Returns:
        pd.DataFrame: Resultados; tabla con columnas 'mf' y 'score'
    """
    print(
        f"Probando distintos mfs",
        "sobre dataset croppeado" if cropped else "",
        f"(depth=10, criterio {criterio}):",
    )
    # STRATIFIED Kfold para conservar la misma proporción entre dígitos para cada fold!
    kf = model_selection.StratifiedKFold(
        nsplits,
        shuffle=True,
        random_state=0,
    )

    results = []

    total_time: float = 0
    n: int = 0

    for i, mf in enumerate(mfs):
        print(f"modelo {i+1}/{len(mfs)} ({criterio}, depth=10, mf={mf}):")

        modelo = tree.DecisionTreeClassifier(
            criterion=criterio,
            max_depth=10,
            max_features=mf,
            random_state=0,
        )
        scores: dict[str, np.ndarray] = model_selection.cross_validate(
            modelo,
            X_dev_cropped if cropped else X_dev,
            y_dev,
            cv=kf,
            n_jobs=-1,
        )
        time = scores["fit_time"].sum() + scores["score_time"].sum()
        score = scores["test_score"].mean()
        total_time += time

        results.append(score)
        print(f"\ttime: {time:.1f}s, mean_score: {score:.2%}")

    results_df = pd.DataFrame(
        {"mf": mfs, "score": results},
    )
    print(f"Total processor time: {total_time:.0f}s")
    return results_df


# Evolución de la precisión según el aumento de mf
def grafico_mfs() -> tuple[Figure, Axes]:

    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")

    ax.yaxis.set_major_formatter("{:.2%}".format)
    ax.minorticks_on()

    ax.tick_params(axis="x", labelrotation=-60)

    ax.set_xlabel("Max_features (Atributos usados)")
    ax.set_ylabel("Exactitud")
    ax.grid()
    return fig, ax


def probar_depths_mfs_kfolding(
    rango_depths: list[int],
    rango_mfs: list[int],
    nsplits: int,
    cropped: bool = False,
    criterio: str = "gini",
) -> np.ndarray[np.float64]:
    """Evalúa el desempeño de un modelo variando los hiperparámetros max_depth y max_features. Realiza validación cruzada mediante k-Folding.
    Args:
        rango_depths: rango de profundidades a probar.
        rango_mfs: rango de valores max_features a probar.
        nsplits (int): Cantidad de folds que promediar.
        cropped (bool): Usar o no el dataset croppeado.
        criterio (str): Criterio del modelo; 'gini', 'entropy' o 'log_loss'.

    Returns:
        np.ndarray[np.float64]: Matriz de las exactitudes promediadas entre los Folds.
    """

    print(
        f"Probando distintas combinaciones depths-mfs",
        "sobre dataset croppeado" if cropped else "",
        f"(criterio {criterio}):",
    )
    kf = model_selection.StratifiedKFold(
        nsplits,
        shuffle=True,
        random_state=0,
    )

    resultados = np.zeros((len(rango_depths), len(rango_mfs)))
    total_time: float = 0
    n: int = 1

    for (i, depth), (j, mf) in product(
        enumerate(rango_depths),
        enumerate(rango_mfs),
    ):
        modelo = tree.DecisionTreeClassifier(
            criterion=criterio,
            max_depth=depth,
            max_features=mf,
            random_state=0,
        )
        print(f"modelo {n}/{resultados.size} ({criterio}, depth={depth}, mf={mf}):")
        # En vez de necesitar otro loop más, hago la validación cruzada mediante esta operación, donde el KFolder se pasa como parámetro cv (cross-validator).
        # n_jobs=-1 hace que utilice todos los procesadores disponibles (sino tarda un huevo).
        scores: dict[str, np.ndarray] = model_selection.cross_validate(
            modelo, X_dev_cropped if cropped else X_dev, y_dev, cv=kf, n_jobs=-1
        )
        # Tiempo de procesador total usado para este modelo:
        time = scores["fit_time"].sum() + scores["score_time"].sum()
        # promedio del desempeño en los 5 Folds:
        score = scores["test_score"].mean()

        total_time += time
        n += 1

        print(f"\ttime: {time:.1f} s, mean_score: {score:.1%}")
        resultados[i, j] = score
    print(f"Total processor time: {total_time:.0f}s")
    return resultados


def mostrar_heatmap_resultados_depth_mf(
    resultados: np.ndarray[np.float64], rango_depths: list[int], rango_mfs: list[int]
) -> Figure:
    """Genera una imagen mostrando el desempeño de los distintos modelos en función de los hiperparámetros max_depth y max_features.

    Args:
        resultados (np.ndarray[np.float64]): Resultados de la función probar_depths_mfs_kfolding()
        rango_depths: rango de profundidades probados.
        rango_mfs: rango de valores max_features probados.

    Returns:
        Figure: Imagen generada.
    """
    fig, ax = plt.subplots()
    sns.heatmap(
        resultados,
        annot=True,
        fmt=".1%",
        cmap="plasma",
        ax=ax,
        annot_kws={"size": "x-small"},
    )
    # Quizás no es la mejor práctica pero por simpleza copio los valores usados acá, en vez de transportarlos de alguna manera desde la función anterior hasta acá.

    ax.tick_params("x", rotation=90)
    ax.tick_params("y", rotation=0)
    ax.set_xlabel("Max_features")
    ax.set_xticklabels(rango_mfs)
    ax.set_ylabel("Max_depth")
    ax.set_yticklabels(rango_depths)
    ax.xaxis.tick_top()

    return fig


# Entrenar múltiples arboles variando sus hiperparámetros
def probar_mss_msl_kfolding(
    rango_mss: list[int],
    rango_msl: list[int],
    max_depth: int,
    max_features: int | float,
    nsplits: int,
    cropped: bool = False,
    criterio: str = "gini",
) -> np.ndarray[np.float64]:
    """Evalúa el desempeño de un modelo variando los hiperparámetros min_samples_split y min_samples_leaf (mss y msl respectivamente). Realiza validación cruzada mediante k-Folding.
    Args:
        rango_mss: rango de valores min_samples_split a probar.
        rango_msl: rango de valores min_samples_leaf a probar.
        max_depth (int): parámetro max_depth para el modelo.
        max_features (int | float): parámetro max_features para el modelo.
        nsplits (int): Cantidad de folds que promediar.
        cropped (bool): Usar o no el dataset croppeado.
        criterio (str): Criterio del modelo; 'gini', 'entropy' o 'log_loss'.

    Returns:
        np.ndarray[np.float64]: Matriz de las exactitudes promediadas entre los 5 Folds
    """

    kf = model_selection.StratifiedKFold(
        nsplits,
        shuffle=True,
        random_state=0,
    )

    resultados = np.zeros((len(rango_mss), len(rango_msl)))
    total_time: float = 0
    n: int = 0

    for (i, mss), (j, msl) in product(
        enumerate(rango_mss),
        enumerate(rango_msl),
    ):
        print(
            f"modelo {n}/{resultados.size} ({criterio}, max_depth= {max_depth}, mf= {max_features}, mss={mss}, msl={msl}):"
        )
        modelo = tree.DecisionTreeClassifier(
            criterion=criterio,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=mss,
            min_samples_leaf=msl,
            random_state=0,
        )
        # En vez de necesitar otro loop más, hace la validación cruzada mediante esta operación, donde el KFolder se pasa como parámetro cv (cross-validator).
        # n_jobs=-1 hace que utilice todos los procesadores disponibles (sino tarda un huevo).
        scores: dict[str, np.ndarray] = model_selection.cross_validate(
            modelo, X_dev_cropped if cropped else X_dev, y_dev, cv=kf, n_jobs=-1
        )
        # Tiempo de procesador total usado para este modelo:
        time = scores["fit_time"].sum() + scores["score_time"].sum()
        # promedio del desempeño en los 5 Folds:
        score = scores["test_score"].mean()

        total_time += time
        n += 1

        print(f"\ttime: {time:.1f} s, mean_score: {score:.1%}")
        resultados[i, j] = score
    print(f"Total processor time: {total_time:.0f}s")
    return resultados


# Seleccionar mejor modelo
def mostrar_heatmap_resultados_mss_msl(
    resultados: np.ndarray[np.float64],
    rango_mss: list[int],
    rango_msl: list[int],
) -> Figure:
    """Genera una imagen mostrando el desempeño de los distintos modelos en función de los hiperparámetros mss y msl.

    Args:
        resultados (np.ndarray[np.float64]): Resultados de la función probar_mss_msl()
        rango_mss: rango de valores min_samples_split probados.
        rango_msl: rango de valores min_samples_leaf probados.

    Returns:
        Figure: Imagen generada.
    """
    fig, ax = plt.subplots()
    sns.heatmap(
        resultados,
        annot=True,
        fmt=".1%",
        cmap="plasma",
        ax=ax,
        annot_kws={"size": "x-small"},
    )
    # Quizás no es la mejor práctica pero por simpleza copio los valores usados acá, en vez de transportarlos de alguna manera desde la función anterior hasta acá.
    ax.set_xticklabels(rango_msl)
    ax.set_yticklabels(rango_mss)
    ax.tick_params("x", rotation=90)
    ax.tick_params("y", rotation=0)
    ax.set_xlabel("Min_samples_leaf")
    ax.set_ylabel("Min_samples_split")
    ax.xaxis.tick_top()

    # Mejor modelo: depth=10, max_features=0.16, min_samples_split=8,min_samples_leaf=11, min_impurity_decrease=0
    return fig


# Entrenamiento del mejor arbol
def entrenar_arbol(
    max_depth: int = None,
    max_features: int = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    criterio: str = "gini",

):
    print(f"Creando modelo ({criterio}, max_depth={max_depth}, max_features= {max_features}, min_samples_split= {min_samples_split}, min_samples_leaf= {min_samples_leaf})...", end="")
    modelo = tree.DecisionTreeClassifier(
        criterion=criterio,  # Los otros criterios tardan más del doble.
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=0,  # Cualquier cambio a este param decrece el score.
        random_state=0,
    )
    print("✓")
    print("Entrenando Modelo...", end="")
    modelo.fit(X_dev, y_dev)
    print("✓")
    return modelo


def evaluar_mejor_arbol(modelo: tree.DecisionTreeClassifier) -> np.ndarray:
    """Evalúa el desempeño del mejor árbol, imprimiendo su exactitud y generando su matriz de confusión.

    Args:
        modelo (tree.DecisionTreeClassifier): Mejor Modelo de clasificador encontrado.

    Returns:
        np.ndarray: Matriz de confusión del modelo.
    """
    y_pred = modelo.predict(X_heldout)
    acc = metrics.accuracy_score(y_heldout, y_pred)
    print(f"Accuracy: {acc:.2%}")
    M = metrics.confusion_matrix(y_heldout, y_pred)
    return M



print("✓")
