# -*-coding:utf-8 -*-
"""
@File    :   funciones.py
@Time    :   2025/03/01 12:43:14
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Archivo para tener por separado las funciones generales que definamos. Esperamos usarlas varias veces a lo largo del código.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from matplotlib.figure import Figure
from matplotlib.axes import Axes

print("funciones.py: Cargando...", end="")


def mostrar_digito(
    fila: pd.Series,
    *,
    ax: Axes = None,
    cbar_ax: Axes = None,
    cmap: str = None,
    **kwargs,
) -> Figure | None:
    """Genera una imagen de tipo Heatmap mostrando el dígito pasado en el parámetro fila.

    Args:
        fila (pd.Series): serie de 784 (u otro cuadrado perfecto) valores.
        ax (Axes, optional): Objeto Axes sobre el cual dibujar la imagen. Si no se pasa uno, se generan automáticamente objetos fig y ax. Defaults to None.
        cbar_ax (Axes, optional): objeto Axes sobre el cual dibujar el colorbar. Por predeterminado roba espacio del objeto ejes donde se dibuja la imagen. Defaults to None.

    Returns:
        Figure | None: Si no se pasó un eje, devuelve la figura que generó. Si se pasó un eje, la función devuelve None.
    """

    standalone: bool = False
    if ax is None:
        standalone = True  # crear figura y ejes
        fig, ax = plt.subplots()

    # Toma una variable size para calcular el tamaño del cuadrado
    size = int(math.sqrt(len(fila)))

    axesimage = ax.imshow(
        np.array(fila).reshape((size, size)),
        cmap="binary" if cmap is None else cmap,
        **kwargs,
    )

    cbar = plt.colorbar(
        axesimage,
        ax=ax,
        cax=cbar_ax,
        location="right",
        anchor=(0, 0.3),
        shrink=0.7,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if standalone:
        return fig
    else:
        return None


def normalizar(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Normaliza un Dataframe o Series.

    Args:
        df (pd.DataFrame | pd.Series): Datos a normalizar. Para normalizar sólo un eje de un dataframe, usar su método .apply(normalizar, axis=).

    Returns:
        pd.DataFrame | pd.Series: Dataframe o Serie normalizadas.
    """
    min = df.min()
    max = df.max()
    return (df - min) / (max - min)


def mostrar_matriz_de_confusion(
    M: np.ndarray, fmt: str | None = "d", cmap: str = None, **kwargs
) -> Figure:
    """Visualiza la matriz de confusión pasada como parámetro.

    Args:
        M (np.ndarray): Matriz de confusión
        fmt (str | None, optional): format string para los valores numéricos en cada celda de la matriz. Por predeterminado formatea como entero. Defaults to "d".

    Returns:
        Figure: Imagen generada.
    """
    fig, ax = plt.subplots()
    sns.heatmap(
        M,
        annot=True,
        fmt=fmt,
        cmap="plasma" if cmap is None else cmap,
        ax=ax,
        **kwargs,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    return fig

    # Lo que se necesita para hacer lo mismo con matplotlib:
    fig, ax = plt.subplots()
    axesimage = ax.imshow(M, cmap="plasma", vmin=0)
    cbar = plt.colorbar(axesimage, ax=ax)
    # cbar.set_ticks(range(0, int(round(M.max(), -2)) + 100, ))
    for (j, i), label in np.ndenumerate(M):
        ax.text(i, j, round(label), ha="center", va="center")
    ax.set_xticks(range(len(M)))
    ax.set_xlabel("Valor predicho")
    ax.set_yticks(range(len(M)))
    ax.set_ylabel("Valor real")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    return fig


def crop_imagenes(data_imagen: pd.DataFrame, n: int) -> pd.DataFrame:
    """Quita los n píxeles externos del borde de la imagen. Dada una imagen de AxA, devuelve una de (A-2n)x(A-2n).

    Args:
        data_imagen (pd.DataFrame): Dataframe con imágenes a cropear, una por una.
        n (int): Número de píxeles a eliminar de los bordes

    Returns:
        pd.DataFrame: Resultado de cropear las imágenes.
    """
    cropped_data = []

    for row in data_imagen.values:
        img = row.reshape(28, 28)  # Reshape into 28x28 image
        cropped_img = img[n:-n, n:-n]  # Crop n pixels from each side
        cropped_data.append(cropped_img.flatten())  # Flatten back to a row

    return pd.DataFrame(cropped_data)


print("✓")
