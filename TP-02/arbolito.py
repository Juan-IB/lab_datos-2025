# %% 
# Imports
import pandas as pd

# Imports from sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Import de funciones
from funciones import mostrar_matriz_de_confusion

# %% 
# Import del dataset
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║  CARGA DEL DF                                                               ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
mnist_values_n = pd.read_csv(r"Resources/mnist_values_n.csv")
mnist_labels = pd.read_csv(r"Resources/mnist_labels.csv")["labels"]

"""
╔═════════════════════════════════════════════════════════════════════════════╗
║  CROP DEL DF                                                                ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
def crop_df(df, n):
    cropped_data = []

    for row in df.values:
        img = row.reshape(28, 28)  # Reshape into 28x28 image
        cropped_img = img[n:-n, n:-n]  # Crop n pixels from each side
        cropped_data.append(cropped_img.flatten())  # Flatten back to a row

    return pd.DataFrame(cropped_data)

X = crop_df(mnist_values_n, 5)
y = mnist_labels


# %%
# Split del modelo
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ SPLIT DEL MODELO CON ESTRATIFICACION                                        ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)

# %%
# Entrenamiento del modelo
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ ENTRENAMIENTO DEL MODELO                                                    ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
arbol = DecisionTreeClassifier(max_depth=10,
                               random_state=42, 
                               criterion='entropy',
                               min_samples_leaf=1
                               )

# Entreno
arbol.fit(X_train, y_train)

# Predict
y_pred = arbol.predict(X_test)
# %%
# Evaluacion y prediccion
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ EVALUACION DE PREDICCION Y MATRIZ DE CONFUSION                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_pred,y_test)
# mostrar_matriz_de_confusion(cm)

# Normalizada (row-wise)
cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
mostrar_matriz_de_confusion(cm_normalized,fmt='.2f')
# %%
# K-folding accuracy
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ K-Folding con estratificacion                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# Define Stratified K-Fold
# Con n_splits=5 da 0.8203, con 10 da 0.8229, no vale la pena la espera 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(arbol, X, y, cv=skf, scoring='accuracy')

# Print results
print(f"Accuracy per fold: {scores}")
print(f"Average accuracy: {scores.mean():.4f}")