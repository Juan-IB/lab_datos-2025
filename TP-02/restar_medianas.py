# -*-coding:utf-8 -*-
"""
@File    :   restar_medias.py
@Time    :   2025/02/27 14:24:51
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   
"""
# %%
import pandas as pd
import numpy as np
from sklearn import model_selection
from funciones import mostrar_matriz_de_confusion, normalizar
from sklearn.metrics import confusion_matrix, accuracy_score

mnist_values_n: pd.DataFrame = pd.read_csv(r"Resources/mnist_values_n.csv")
mnist_labels: pd.Series = pd.read_csv(r"Resources/mnist_labels.csv")


# %%
def mostrar_modelo_restar_medianas() -> None:
    """'''Entrena''' y evalúa un modelo primitivo que decide la identidad del dígito basado en cuál mediana es la más parecida. Muestra la Exactitud, la matriz de confusión y una versión relativa para ver qué porcentaje de los valores reales son mal categorizados."""
    t: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] = (
        model_selection.train_test_split(
            mnist_values_n,
            mnist_labels,
            test_size=0.15,
            stratify=mnist_labels,
            random_state=0,
        )
    )
    X_train, X_test, y_train, y_test = t

    medianas_n_train: pd.DataFrame = (
        X_train.groupby(y_train.squeeze()).median().apply(normalizar, axis=1)
    )

    y_pred = []
    for row in X_test.values:
        distances = [
            ((row - mediana_digito) ** 2).sum()
            # Suma de diferencias cuadradas a la mediana de entrenamiento.
            for mediana_digito in medianas_n_train.values
        ]
        index_de_menor_distancia = distances.index(min(distances))
        y_pred.append(index_de_menor_distancia)

    M = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print("Modelo restar_medianas:")
    print(f"Accuracy: {acc:.2%}\nMatriz de confusión:")
    mostrar_matriz_de_confusion(M)
    print("Matriz de confusión Relativa (las filas suman 1):")
    mostrar_matriz_de_confusion(
        np.apply_along_axis((lambda x: x / x.sum()), 1, M),
        fmt=".1%",
        annot_kws={"size": "x-small"},
    )

# %%
