# -*-coding:utf-8 -*-
"""
@File    :   knn_binario.py
@Time    :   2025/03/02 16:28:57
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Ejercicio 2 del TP02. Elaboración de un modelo de tipo KNN para distinguir ceros y unos. Búsqueda de píxeles que mejor distingan estos dígitos.
"""
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import colors
from functools import cache
from random import sample
import random

random.seed(0)
# Imports from sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Import de funciones
from funciones import mostrar_digito, mostrar_matriz_de_confusion
from derivados_mnist import mnist_labels, mnist_values_n, medianas_n_mnist_n

print("knn_binario.py: Cargando...", end="")
# mnist_values_n = pd.read_csv(r"Resources/mnist_values_n.csv")
# mnist_labels = pd.read_csv(r"Resources/mnist_labels.csv")["labels"]
# medianas_n_mnist_n = pd.read_csv(r"Resources/medianas_n_mnist_n.csv")

ceros_unos_values = mnist_values_n[mnist_labels.isin([0, 1])]
ceros_unos_labels = mnist_labels[mnist_labels.isin([0, 1])]

# Splitting dataset (80% train, 20% test)
train_test: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = train_test_split(
    ceros_unos_values,
    ceros_unos_labels,
    test_size=0.2,
    random_state=0,
    stratify=ceros_unos_labels,
)
X_train, X_test, y_train, y_test = train_test


def probar_k_vecinos(ks: list[int] | range) -> pd.DataFrame:
    """Obtiene la exactitud de un modelo KNN con diferentes valores de k, entre 1 y 20.
    Args:
        ks (list[int] | range)
    Returns:
        pd.DataFrame: valores de la exactitud (columna 'score') en función de la columna 'k'.
    """
    results = np.zeros((len(ks), 2))
    for i, k in enumerate(ks):
        print(f"Probando k={k}...", end="")
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        results[i] = k, acc
        print("✓")

    return pd.DataFrame(results, columns=["k", "score"])


def grafico_exactitud_vs_k() -> tuple[Figure, Axes]:
    """Preconfigura un gráfico para representar la dependencia de la exactitud en función de k. No agrega Datos.

    Returns:
        tuple[Figure, Axes]: Figura y Ejes del gráfico, para luego continuar graficando.
    """
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter("{:.2%}".format)
    ax.set_ylabel("Exactitud")
    ax.set_xlabel("Cantidad de Vecinos (k)")
    ax.minorticks_on()
    ax.grid()
    return fig, ax
    # Mejor k: 2, acc= 99.83%


# ¿Qué columnas son las más decisivas?
@cache  # Siempre devuelve lo mismo. No necesita recomputar cada vez que la llaman. Cache hace que guarde la respuesta y recuerde si ya fue llamada antes.
def pixeles_discriminantes(criterio: str = "default") -> pd.Series:
    """Esta función busca encontrar los píxeles que mejor discriminan entre el cero y el uno. Buscamos los píxeles que más frecuentemente sean diferentes entre el cero y el uno. Para eso calculamos la diferencia entre las medianas de estos dígitos y la elevamos al cuadrado. Este valor luego dividido por el producto de las varianzas de los unos y los ceros, para tener en cuenta cuánto varía cada píxel dentro de cada dígito. El píxel con el valor final más alto es el que más consistentemente discrimina entre el cero y el uno.
    Args:
        criterio (str): esquema para decidir cuál píxel es más discriminante que otro. Opciones son:
            'default': el esquema predeterminado toma la diferencia de las medianas para cada píxel y las divide por el producto de las varianzas de los ceros y los unos.
            'varianza_total': elige los píxeles de mayor varianza en el total del dataset.
            'varianza_producto_inv': elige los píxeles menos variantes en los unos y los ceros (constantes dentro del dígito, diferentes entre los dos dígitos)
    Returns:
        pd.Series: Valores finales para cada píxel.
    """
    if criterio == "default":
        # La mediana normalizada de los unos normalizados menos la mediana normalizada de los ceros normalizados. (Trabalenguas!)
        amplitud_1_0 = medianas_n_mnist_n.iloc[1, :] - medianas_n_mnist_n.iloc[0, :]
        # Los valores más altos aparecen en el 1 pero no el 0.
        # Los valores más bajos aparecen en el 0 pero no el 1.

        # Elevo al cuadrado para considerar ambos signos como buenos discriminantes
        discriminantes = amplitud_1_0**2
        discriminantes /= (
            mnist_values_n[mnist_labels == 1].var()
            * mnist_values_n[mnist_labels == 0].var()
        )
    elif criterio == "varianza_total":
        discriminantes = ceros_unos_values.var()
    elif criterio == "varianza_producto_inv":
        discriminantes = 1 / (
            mnist_values_n[mnist_labels == 1].var()
            * mnist_values_n[mnist_labels == 0].var()
        )
    return discriminantes


def mostrar_puntajes_discriminantes(
    criterio: str = "default", norm=None, vmin: float = None, vmax: float = None
) -> Figure:
    """Toma los puntajes producidos en pixeles_discriminantes() y los muestra en una imagen, con escala de colores logarítmica para mejor visualizar el resultado.
    Args:
        criterio (str): esquema para decidir cuál píxel es más discriminante que otro. Opciones son:
            'default': el esquema predeterminado toma la diferencia de las medianas para cada píxel y las divide por el producto de las varianzas de los ceros y los unos.
            'varianza_total': elige los píxeles de mayor varianza en el total del dataset.
            'varianza_producto_inv': elige los píxeles menos variantes en los unos y los ceros (constantes dentro del dígito, diferentes entre los dos dígitos)
    Returns:
        Figure: Imagen producida.
    """
    fig, ax = plt.subplots()
    mostrar_digito(
        pixeles_discriminantes(criterio),
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )
    # Escala logarítmica para ver mejor
    return fig


@cache
def n_pixeles_mas_disc(n: int, criterio: str = "default") -> list[int]:
    """Obtiene los n píxeles más discriminantes a partir del retorno de la función pixeles_discriminantes().

    Args:
        n (int): Cantidad de atributos discriminantes a devolver.
        criterio (str): esquema para decidir cuál píxel es más discriminante que otro. Opciones son:
            'default': el esquema predeterminado toma la diferencia de las medianas para cada píxel y las divide por el producto de las varianzas de los ceros y los unos.
            'varianza_total': elige los píxeles de mayor varianza en el total del dataset.
            'varianza_producto_inv': elige los píxeles menos variantes en los unos y los ceros (constantes dentro del dígito, diferentes entre los dos dígitos)
    Returns:
        list[int]: Lista de números de columna correspondientes a los n atributos más discriminantes.
    """
    top_n_pixels = (
        pixeles_discriminantes(criterio=criterio).nlargest(n).index.astype("int64")
    )
    return list(top_n_pixels)


def mostrar_n_px_disc(n: int, criterio: str = "default") -> None:
    """Muestra cuáles son los n píxeles más discriminantes gráficamente, marcándolos en una imagen de 28x28.

    Args:
        n (int): Número de píxeles discriminantes a mostrar.
        criterio (str): esquema para decidir cuál píxel es más discriminante que otro. Opciones son:
            'default': el esquema predeterminado toma la diferencia de las medianas para cada píxel y las divide por el producto de las varianzas de los ceros y los unos.
            'varianza_total': elige los píxeles de mayor varianza en el total del dataset.
            'varianza_producto_inv': elige los píxeles menos variantes en los unos y los ceros (constantes dentro del dígito, diferentes entre los dos dígitos)
    """
    image = pd.Series([0] * 784)
    image[n_pixeles_mas_disc(n, criterio)] = 1
    # Son los píxeles centrales los que más distinguen entre el 1 y 0.
    mostrar_digito(image)


# Mejor k para estos 3 mejores pixeles
def probar_k_vecinos_top3(criterio: str = "default") -> pd.DataFrame:
    """Evalúa cómo se desempeña un modelo utilizando sólo los 3 mejores atributos, en función de diferentes k.
    Args:
        criterio (str): esquema para decidir cuál píxel es más discriminante que otro. Opciones son:
            'default': el esquema predeterminado toma la diferencia de las medianas para cada píxel y las divide por el producto de las varianzas de los ceros y los unos.
            'varianza_total': elige los píxeles de mayor varianza en el total del dataset.
            'varianza_producto_inv': elige los píxeles menos variantes en los unos y los ceros (constantes dentro del dígito, diferentes entre los dos dígitos)
    Returns:
        pd.DataFrame: Tabla con valores de exactitud ("score") en función de k.
    """
    print(f"Probando con 3 mejores píxeles.")
    top_3_pixels = n_pixeles_mas_disc(3, criterio)

    X_train_subset = X_train.iloc[:, top_3_pixels]
    X_test_subset = X_test.iloc[:, top_3_pixels]

    ks = range(1, 21)
    results = np.zeros((len(ks), 2))

    for i, k in enumerate(ks):
        print(f"\tk={k}...", end="")
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train_subset, y_train)
        acc = modelo.score(X_test_subset, y_test)
        results[i] = k, acc
        print("✓")

    return pd.DataFrame(results, columns=["k", "score"])
    # Mejor k: 4, acc = 98.91%


def probar_k_vecinos_top_n(
    n: int, ks: list[int] | range, criterio: str = "default"
) -> pd.DataFrame:
    """Evalúa cómo se desempeña un modelo utilizando sólo los 3 mejores atributos, en función de diferentes k.
    Args:
        n (str): Número de mejores píxeles a usar.
        ks (list[int] | range): valores de k a probar.
        criterio (str): esquema para decidir cuál píxel es más discriminante que otro. Opciones son:
            'default': el esquema predeterminado toma la diferencia de las medianas para cada píxel y las divide por el producto de las varianzas de los ceros y los unos.
            'varianza_total': elige los píxeles de mayor varianza en el total del dataset.
            'varianza_producto_inv': elige los píxeles menos variantes en los unos y los ceros (constantes dentro del dígito, diferentes entre los dos dígitos)
    Returns:
        pd.DataFrame: Tabla con valores de exactitud ("score") en función de k.
    """
    print(f"Probando con {n} mejores píxeles (criterio= {criterio}).")

    top_n_pixels = n_pixeles_mas_disc(n, criterio)

    X_train_subset = X_train.iloc[:, top_n_pixels]
    X_test_subset = X_test.iloc[:, top_n_pixels]

    results = np.zeros((len(ks), 2))

    for i, k in enumerate(ks):
        print(f"\tk={k}...", end="")
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train_subset, y_train)
        acc = modelo.score(X_test_subset, y_test)
        results[i] = k, acc
        print("✓")

    return pd.DataFrame(results, columns=["k", "score"])


def probar_k_vecinos_rand_n_m_veces(
    n: int, m: int, ks: list[int] | range
) -> pd.DataFrame:
    """Evalúa cómo se desempeña un modelo utilizando sólo n atributos elegidos al azar, en función de diferentes k. Promedia el desempeño de m selecciones diferentes.

    Args:
        n (int): Número de atributos a usar
        m (int): cantidad de intentos a promediar.
        ks (list[int] | range): valores de k a probar.

    Returns:
        pd.DataFrame: Tabla con valores de exactitud ("score") en función de k.
    """
    results = np.zeros((m, len(ks), 2))
    print(f"Probando con {n} píxeles al azar {m} veces.")

    for i in range(m):
        print(f"Probando {i+1}/{m}:")
        some_n_pixels = sample(range(784), n)

        X_train_subset = X_train.iloc[:, some_n_pixels]
        X_test_subset = X_test.iloc[:, some_n_pixels]

        for j, k in enumerate(ks):
            print(f"\tk={k}...", end="")
            modelo = KNeighborsClassifier(n_neighbors=k)
            modelo.fit(X_train_subset, y_train)
            acc = modelo.score(X_test_subset, y_test)
            results[i][j] = k, acc
            print(f"{acc:.2%}")

    return pd.DataFrame(results.mean(0), columns=["k", "score"])


# ¿Qué cantidad de columnas obtiene mejor desempeño?
def probar_k_y_n(
    ns: list[int] | range,
    ks: list[int] | range,
    criterio: str = "default",
) -> np.ndarray[np.float64]:
    """Evalúa cómo se desempeña un modelo variando el número de mejores atributos (n) y el número de vecinos (k).
    Args:
        criterio (str): esquema para decidir cuál píxel es más discriminante que otro. Opciones son:
            'default': el esquema predeterminado toma la diferencia de las medianas para cada píxel y las divide por el producto de las varianzas de los ceros y los unos.
            'varianza_total': elige los píxeles de mayor varianza en el total del dataset.
            'varianza_producto_inv': elige los píxeles menos variantes en los unos y los ceros (constantes dentro del dígito, diferentes entre los dos dígitos)
    Returns:
        pd.DataFrame: Matriz de 20x20 con valores de exactitud ("score") para cada combinación de n y k.
    """

    best_acc = 0
    M_results = np.zeros((len(ns), len(ks)))  # 360 items
    for i, n in enumerate(ns):

        top_n_pixels = n_pixeles_mas_disc(n, criterio)
        X_train_subset = X_train.iloc[:, top_n_pixels]
        X_test_subset = X_test.iloc[:, top_n_pixels]

        # Mejor k para estos n pixeles discriminantes
        for j, k in enumerate(ks):
            modelo = KNeighborsClassifier(n_neighbors=k)
            modelo.fit(X_train_subset, y_train)
            y_pred = modelo.predict(X_test_subset)
            acc = accuracy_score(y_test, y_pred)
            M_results[i][j] = acc
            if acc > best_acc:
                best_acc = acc
                print(f"{acc:.2%}, n={n}, k={k}")
    return M_results


def mostrar_heatmap_n_y_k(
    M_results: np.ndarray[np.float64],
    ns: list[int] | range,
    ks: list[int] | range,
) -> Figure:
    """Muestra un heatmap representando los resultados de la función probar_k_vecinos_top_n().

    Args:
        M_results (np.ndarray[np.float64]): Matriz con los resultados de la función mencionada

    Returns:
        Figure: Imagen del heatmap producido.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(
        M_results,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        norm=colors.TwoSlopeNorm(0.99),
        ax=ax,
        annot_kws={
            "rotation": 30,
        },
    )
    ax.set_xlabel("k")
    ax.set_ylabel("n")
    ax.set_xticklabels(ks)
    ax.set_yticklabels(ns)
    ax.set_title(
        "Exactitudes del modelo KNN con $k$ vecinos,\n tomando las $n$ columnas más discriminantes."
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    return fig


def entrenar_modelo(k: int, n: int, criterio: str = "default") -> np.ndarray[np.int64]:
    """Entrena y evalúa un modelo con los parámetros dados.

    Args:
        k (int): k vecinos más cercanos a promediar
        n (int): n atributos más discriminantes a usar
        criterio (str): esquema para decidir cuál píxel es más discriminante que otro. Opciones son:
            'default': el esquema predeterminado toma la diferencia de las medianas para cada píxel y las divide por el producto de las varianzas de los ceros y los unos.
            'varianza_total': elige los píxeles de mayor varianza en el total del dataset.
            'varianza_producto_inv': elige los píxeles menos variantes en los unos y los ceros (constantes dentro del dígito, diferentes entre los dos dígitos)
    Returns:
        np.ndarray[np.int64]: Matriz de confusión del modelo.
    """
    top_n_pixels = n_pixeles_mas_disc(n, criterio)

    X_train_subset = X_train.iloc[:, top_n_pixels]
    X_test_subset = X_test.iloc[:, top_n_pixels]
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train_subset, y_train)

    return confusion_matrix(y_test, modelo.predict(X_test_subset))


print("✓")
