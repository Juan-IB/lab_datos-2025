# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Imports from sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Import de funciones
from funciones import mostrar_matriz_de_confusion

# %% 
# Import del df y lectura de los datos.
mnist_values = pd.read_csv(r"Resources/mnist_values_n.csv")
mnist_labels = pd.read_csv(r"Resources/mnist_labels.csv")["labels"]

# Clases presentes (Dígitos)
digitos = sorted(mnist_labels.unique())

# %% 
# Finalmente no se usa, pero dejamos el codigo como parte del analisis
# exploratorio
# Cropping del df original
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ CROPPING - No se usa en este archivo, quedó como analisis exploratorio      ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
def crop_df(df, n):
    cropped_data = []

    for row in df.values:
        img = row.reshape(28, 28)  # Hacemos un reshape a una imagen de 28x28
        cropped_img = img[n:-n, n:-n]  # Cropea n pixels de cada lado
        cropped_data.append(cropped_img.flatten()) # Aplanamos a una fila nuevamente

    return pd.DataFrame(cropped_data)

# cropped_mnist_values = crop_df(mnist_values, 5)

# %%
# Obtengo los ceros y unos, hay 6903 ceros y 7877 unos.
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ Obtenemos los cero y uno en subsets                                         ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

mnist_cero = mnist_values[mnist_labels == 0]
mnist_uno = mnist_values[mnist_labels == 1]

print(f'cantidad de ceros: {len(mnist_cero)}, cantidad de unos: {len(mnist_uno)}')

# %%
# Selecciono solo aquellas filas donde tengo ceros y unos
ceros_unos_values = mnist_values[mnist_labels.isin([0, 1])]
ceros_unos_labels = mnist_labels[mnist_labels.isin([0, 1])]

# %% 
# Split del dataset (90% train, 10% test)
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ Hacemos split del modelo usando con train y test                            ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
X_train, X_test, y_train, y_test = train_test_split(
    ceros_unos_values,
    ceros_unos_labels,
    test_size=0.2,
    random_state=1,
)

# %%
# Creacion y entrenamiento del usando todos los atributos
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ Entrenamiento de modelo usando todos los atributos                          ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# %% 
# Accuracy test y matriz de confusion

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
mostrar_matriz_de_confusion(cm)

# %%
# Entrenamiento con la varianza
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ Entrenamiento con 3 atributos de mayor varianza                             ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# Calculo la varianza de cada columna de pixeles
pixel_variances = X_train.var()

# Obtengo los 3 pixeles con mayor varianza
top_3_pixels = pixel_variances.nlargest(3).index 

# Obtengo el subset del entrenamiento con los 3 de mayor varianza
X_train_subset = X_train[top_3_pixels]
X_test_subset = X_test[top_3_pixels]

# %% 
# Entreno el modelo con el subset de 3 atributos
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_subset, y_train)
y_pred = knn.predict(X_test_subset)

# %% 
# Accuracy test con 3 atributos
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
mostrar_matriz_de_confusion(cm)
# %%
