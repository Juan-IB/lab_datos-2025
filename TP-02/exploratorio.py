# -*-coding:utf-8 -*-
"""
@File    :   exploratorio.py
@Time    :   2025/02/26 09:32:53
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Análisis exploratorio de los datos de MNIST-C 'Fog'.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from itertools import product
from funciones import mostrar_digito
from derivados_mnist import (
    mnist_values,
    mnist_values_n,
    mnist_labels,
    medianas_mnist_n,
    medianas_n_mnist_n,
)

print("exploratorio.py: Cargando...", end="")
# mnist_values: pd.DataFrame = pd.read_csv(r"Resources/mnist_values.csv")
# mnist_values_n: pd.DataFrame = pd.read_csv(r"Resources/mnist_values_n.csv")
# mnist_labels: pd.Series = pd.read_csv(r"Resources/mnist_labels.csv")
# medianas_mnist_n: pd.DataFrame = pd.read_csv(r"Resources/medianas_mnist_n.csv")
# medianas_n_mnist_n = pd.read_csv(r"Resources/medianas_n_mnist_n.csv")


# %%
def clases_presentes() -> list[int]:
    """Clases presentes en el dataset. Encuentra los valores distintos de los labels y los ordena.

    Returns:
        list[int]: Lista de dígitos presentes en el dataset.
    """
    digitos = sorted(mnist_labels.unique())
    return digitos


def descripcion_dataset() -> dict[str, int]:
    """Descripción del Dataset a manejar, en cuestión de tamaño.

    Returns:
        dict[str, int]: Diccionario conteniendo valores de:
            'entradas': cantidad total de filas del dataset,
            'datos_por_entrada': columnas del dataset,
            'dataset_size': cantidad total de datos (filas * columnas)
    """
    cant_entradas = len(mnist_values_n)
    cant_datos_por_entrada = len(mnist_values_n.columns)
    cant_datos = len(mnist_values_n) * len(mnist_values_n.columns)
    return {
        "entradas": cant_entradas,
        "datos_por_entrada": cant_datos_por_entrada,
        "dataset_size": cant_datos,
    }


# %% Distribucion de digitos
def distribucion_digitos() -> dict[str, pd.DataFrame]:
    """Análisis de cómo están distribuidos los dígitos en el dataset.

    Returns:
        dict[str, pd.DataFrame]: Diccionario con entradas:
            'counts': número de ocurrencias para cada dígito.
            'rel_freqs': frecuencias relativas para cada dígito (dividir 'counts' por 70000)
            'dist_mean': error porcetual entre la frecuencia de cada dígito y el promedio.
    """
    frecuencias_digitos: pd.Series = mnist_labels.value_counts().sort_index()

    # Frecuencias relativas (idealmente 0.10 cada dígito)
    frecuencias_relativas_digitos = mnist_labels.value_counts(
        normalize=True
    ).sort_index()

    # Cuánto más (o menos) frecuente es cada dígito que el promedio.
    media = frecuencias_relativas_digitos.mean()
    variacion_desde_el_promedio = (
        (frecuencias_relativas_digitos - media) / media
    ).apply(lambda x: f"{x:+.2%}")

    return {
        "counts": frecuencias_digitos,
        "rel_freqs": frecuencias_relativas_digitos,
        "dist_mean": variacion_desde_el_promedio,
    }


# %% Valores máximos para los píxeles
def mostrar_maximos_por_px() -> None:
    """Muestra una imagen compuesta por los valores más altos en todo el dataset para cada píxel."""
    maximos_por_px = mnist_values.max(axis=0)
    maximo_total = maximos_por_px.max()  # 255.

    mostrar_digito(maximos_por_px)


# %% Valores mínimos para los píxeles
def mostrar_minimos_por_px() -> None:
    """Muestra una imagen compuesta por los valores más bajos en todo el dataset para cada píxel."""
    minimos_por_px = mnist_values.min(axis=0)
    minimo_total = minimos_por_px.min()  # 0.

    mostrar_digito(minimos_por_px)


# %% Amplitud de cada píxel
def mostrar_amplitud_por_px() -> None:
    """Muestra una imagen compuesta por las amplitudes (valor más alto menos el valor más bajo) para cada píxel."""
    amplitudes_por_px = mnist_values.max(axis=0) - mnist_values.min(axis=0)
    mostrar_digito(amplitudes_por_px)


# %% Media de cada píxel
def mostrar_medias_por_px() -> None:
    """Muestra una imagen compuesta por la media para cada píxel en todo el dataset."""
    medias_por_px = mnist_values.mean(axis=0)
    mostrar_digito(medias_por_px)


# %% Mediana de cada píxel
def mostrar_medianas_por_px() -> None:
    """Muestra una imagen compuesta por la mediana para cada píxel en todo el dataset."""
    medianas_por_px = mnist_values.median(axis=0)
    mostrar_digito(medianas_por_px)


# %% Valor total de cada píxel
def mostrar_valor_total_por_px() -> None:
    """Muestra una imagen compuesta por la suma de todos los valores para cada píxel en el dataset (Imagen igual que la media pero en otra escala)."""
    valor_total_por_px = mnist_values.sum(axis=0)
    mostrar_digito(valor_total_por_px)  # Lo mismo, pues la media es la suma / 70.000.


# %% Análisis por dígito
def mostrar_medianas_de_cada_digito() -> Figure:
    """Muestra una Figura compuesta por 10 imágenes, una por dígito del dataset, donde cada imagen es la mediana de los píxeles de todas las entradas de ese dígito."""
    fig, ax = plt.subplots(1, 10, figsize=(18, 1))

    for d, mediana in medianas_mnist_n.iterrows():
        mostrar_digito(
            mediana,
            ax=ax[d],
        )
    return fig


# %% Distribuciones de valores de las medianas de cada dígito.
def guardar_distribucion_horizontal_vertical_medianas() -> None:
    """Analiza la distribución vertical y horizontal de los valores de las medianas de cada dígito. Muestra la suma de cada columna a lo largo de la imagen, y la suma de cada fila.
    Guarda las figuras, una por dígito, que muestran la mediana, junto con gráficos de barras representando cómo se distribuyen los valores a lo largo y ancho de la imagen.
    """
    for d, mediana_n in medianas_n_mnist_n.iterrows():
        array_mediana_n = np.array(mediana_n).reshape((28, 28))

        figax: tuple[Figure, list[list[Axes]]] = plt.subplots(2, 2, figsize=(5, 5))
        fig, ax = figax
        mostrar_digito(
            mediana_n,
            ax=ax[0][0],
            cbar_ax=ax[1][1],
        )

        row_sum = array_mediana_n.sum(axis=1)
        ax[0][1].barh(
            y=range(28),
            width=row_sum,
        )
        ax[0][1].sharey(ax[0][0])
        ax[0][1].set_xticks([])

        column_sum = array_mediana_n.sum(axis=0)
        ax[1][0].bar(
            x=range(28),
            height=column_sum,
        )
        ax[1][0].sharex(ax[0][0])
        ax[1][0].set_yticks([])

        ax[1][1].set_aspect(6)
        fig.savefig(f"Images/dist_hv_{d}.png")


# %% Diferencias entre las medianas de cada dígito.
def mostrar_diferencias_cruzadas_medianas() -> Figure:
    """
    Returns:
        Figure: Gráfico de 10x10 imágenes mostrando la diferencia entre cada par de medianas para todas las combinaciones de dos dígitos del 0 al 9.
    """
    figax: tuple[Figure, list[list[Axes]]] = plt.subplots(10, 10)
    fig, ax = figax
    cax = fig.add_axes((0.95, 0.15, 0.05, 0.7))
    for (i, mediana_i), (j, mediana_j) in product(
        medianas_n_mnist_n.iterrows(), repeat=2
    ):
        dif = mediana_i - mediana_j
        mostrar_digito(dif, ax=ax[i][j], cmap="RdBu", cbar_ax=cax)

    # fig.savefig(f"Images/dif_medianas_10x10.png")
    return fig



# %% Buscar los casos que más se diferencia de las medianas para cada dígito.
def n_mas_disimiles(digito: int, n: int) -> list[int]:
    """Busca los índices de las entradas más disímiles a la mediana para un dado dígito.

    Args:
        digito (int): Dígito de interés.
        n (int): número de disímiles a encontrar.

    Returns:
        list[int]: Lista de índices de las n entradas más diferentes a la mediana para el dígito dado.
    """
    digito_n = mnist_values_n[mnist_labels == digito]
    median = medianas_n_mnist_n.iloc[digito]

    distances: list[tuple[int, float]] = []

    for i, test_row in digito_n.iterrows():
        distances.append((((test_row - median) ** 2).sum(), i))
        # distances.append((i, ((test_row - median) ** 2).sum()))
    n_mas_disimiles_index: list[int] = [
        tup[1] for tup in sorted(distances, reverse=True)[:n]
        # tup[0] for tup in sorted(distances, key=lambda x: x[1], reverse=True)[:n]:
    ]

    return n_mas_disimiles_index


# %% Ejemplos de caracteres disímiles para cada dígito
def mostrar_10x10_mas_disimiles() -> Figure:
    """Encuentra para cada dígito, las 10 entradas más disímiles a la mediana.

    Returns:
        Figure: Figura de 10x10 imágenes, con una fila por dígito, la cual contiene las imágenes de las 10 entradas más diferentes a la mediana.
    """
    fig_ax: tuple[Figure, list[list[Axes]]] = plt.subplots(
        10, 10, figsize=(10, 10), layout="constrained"
    )
    fig, ax = fig_ax
    fig.suptitle("Dígitos más disímiles (mnist)", size=30)
    # fig.set_facecolor("#ffffff00")
    for i in range(10):
        print(f"Calculando disimiles en {i}")
        pic_indexs = n_mas_disimiles(i, 10)
        for j, pic_index in enumerate(pic_indexs):

            if j == 0:
                ax[i][j].set_ylabel(
                    mnist_labels.iloc[int(pic_index)],
                    size=20,
                    rotation=0,
                    labelpad=16,
                )

            mostrar_digito(
                mnist_values_n.iloc[int(pic_index)],
                ax=ax[i][j],
                cbar_ax=fig.add_axes((0, 0, 0, 0), visible=False),
            )
    
    fig.savefig(f"Images/mas_disimiles_10x10.png")
    return fig


print("✓")
