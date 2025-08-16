# %%
import pandas as pd
import os
from collections import defaultdict
from funciones import normalizar


print("Módulo derivados_mnist.py - Cargando datos y Guardando derivados")
# %% Crear DefaultDict
# Crea un default dict para establecer el dtype de todos los datos como float, excepto el label.
dtypesdict = defaultdict(lambda: "float")
dtypesdict["labels"] = "uint8"
# %% Leer Mnist original
# Lee los datos. Requiere que exista el archivo mnist-c-fog.csv en el directorio del código.

# %% Crear derivados y guardarlos para que todos los módulos puedan a acceder a ellos.
# Aseguramos que exista la carpeta
if os.path.isdir(r"Resources") and all(
    list(
        map(
            os.path.isfile,
            [
                (r"Resources/mnist_labels.csv"),
                (r"Resources/mnist_values.csv"),
                (r"Resources/mnist_values_n.csv"),
                (r"Resources/medianas_mnist_n.csv"),
                (r"Resources/medianas_n_mnist_n.csv"),
            ],
        )
    )
):
    print("Leyendo Derivados preexistentes:")

    print("\tmnist_labels.csv...", end="")
    mnist_labels = pd.read_csv(r"Resources/mnist_labels.csv")["labels"]
    print("✓")

    print("\tmnist_values.csv...", end="")
    mnist_values = pd.read_csv(r"Resources/mnist_values.csv")
    print("✓")

    print("\tmnist_values_n.csv...", end="")
    mnist_values_n = pd.read_csv(r"Resources/mnist_values_n.csv")
    print("✓")

    print("\tmedianas_mnist_n.csv...", end="")
    medianas_mnist_n = pd.read_csv(r"Resources/medianas_mnist_n.csv")
    print("✓")

    print("\tmedianas_n_mnist_n.csv...", end="")
    medianas_n_mnist_n = pd.read_csv(r"Resources/medianas_n_mnist_n.csv")
    print("✓")
else:
    print("Leyendo mnist_c_fog.csv...", end="")
    mnist = pd.read_csv(
        r"mnist_c_fog_tp.csv",
        index_col=0,
        dtype=dtypesdict,
    )
    print("✓")
    print("Creando Derivados:")
    os.makedirs(r"Resources")

    print("\tmnist_labels.csv...", end="")
    mnist_labels: pd.Series = mnist["labels"]
    mnist_labels.to_csv(r"Resources/mnist_labels.csv", index=False)
    print("✓")
    print("\tmnist_values.csv...", end="")
    mnist_values: pd.DataFrame = mnist.drop(columns="labels")
    mnist_values.to_csv(r"Resources/mnist_values.csv", index=False)

    print("✓")

    # Valores normalizados (Cada uno por separado)
    print("\tmnist_values_n.csv...", end="")
    mnist_values_n: pd.DataFrame = mnist_values.apply(normalizar, axis=1)
    mnist_values_n.to_csv(r"Resources/mnist_values_n.csv", index=False)
    print("✓")

    # Medianas de los valores normalizados
    # Uso los normalizados para tener en cuenta que puede haber variacion en la intensidad del trazo entre una imagen y otra.
    # Así, tendríamos en igual consideracion el máximo de una imagen y el máximo de otra, sin importar que una alcance un valor menor que la otra.
    print("\tmedianas_mnist_n.csv...", end="")
    medianas_mnist_n: pd.DataFrame = mnist_values_n.groupby(mnist_labels).median()
    medianas_mnist_n.to_csv(r"Resources/medianas_mnist_n.csv", index=False)
    print("✓")

    # Medianas normalizadas de los valores normalizados
    # Así cuando luego comparemos con un dígito normalizado, el máximo de la mediana (que será un valor relativamente bajo) igual será tomado con el peso que corresponde (1).
    print("\tmedianas_n_mnist_n.csv...", end="")
    medianas_n_mnist_n: pd.DataFrame = medianas_mnist_n.apply(normalizar, axis=1)
    medianas_n_mnist_n.to_csv(r"Resources/medianas_n_mnist_n.csv", index=False)
    print("✓")


print("\tDone!")
