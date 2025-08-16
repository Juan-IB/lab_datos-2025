# -*-coding:utf-8 -*-
"""
@File    :   main.py
@Time    :   2025/03/09 11:08:43
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Archivo principal, donde se puede ejecutar todo el código paso a paso.
"""
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║LECTURA DE LOS DATOS                                                         ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# %% Descargar dataset
import descargar_mnist

# %% Cargar dataset
import derivados_mnist  # Crea derivados de datos para los demás módulos.

# %% Realizar otros imports internos
import exploratorio
import knn_binario
import arbol_multiclase
from funciones import mostrar_matriz_de_confusion

# %%
import os
from matplotlib.colors import TwoSlopeNorm
from matplotlib import pyplot as plt
import numpy as np

# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ANÁLISIS EXPLORATORIO DE LOS DATOS - GENERALIDADES                           ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

if not os.path.isdir("Images"): 
    os.makedirs("Images")  # Aseguramos que exista la carpeta

print("Dígitos presentes en el Dataset:")
print(exploratorio.clases_presentes())

# %%
descripcion_dataset = exploratorio.descripcion_dataset()
print(f"Cardinalidad del dataset: {descripcion_dataset['entradas']}")
print(f"Cantidad de datos por entrada: {descripcion_dataset['datos_por_entrada']}")
print(f"Cantidad de datos: {descripcion_dataset['dataset_size']:,}")

# %%
distrib_digitos = exploratorio.distribucion_digitos()
print("Ocurrencias de cada dígito:")
print(distrib_digitos["counts"])
# %%
print("Frecuencias relativas de cada dígito:")
print(distrib_digitos["rel_freqs"])
# %%
print(
    "Error porcentual de la frecuencia de cada dígito con respecto a la frecuencia promedio:"
)
print(distrib_digitos["dist_mean"])

"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ANÁLISIS EXPLORATORIO DE LOS DATOS - PÍXELES                                 ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# %%
print("Máximo en todo el dataset para cada píxel:")
# Esta imagen muestra qué píxeles son realmente usados en el dataset, pues los más oscuros no alcanzan ningún valor alto.
exploratorio.mostrar_maximos_por_px()

# %%
print("Mínimo en todo el dataset para cada píxel:")
# Esta imagen muestra qué píxeles siempre tienen valor alto, por lo que no variarán tanto entre instancias.
exploratorio.mostrar_minimos_por_px()

# %%
print("Amplitud para cada píxel:")
exploratorio.mostrar_amplitud_por_px()

# %%
print("Media de cada píxel:")
exploratorio.mostrar_medias_por_px()
# se ve un 3/5/8/9 !

# %%
print("Mediana de cada píxel:")
exploratorio.mostrar_medianas_por_px()
# Casi igual que la media

# %%
print("Valor total de cada píxel:")
exploratorio.mostrar_valor_total_por_px()
# Igual a la media, pues la media es la suma total / 70.000.

"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ANÁLISIS EXPLORATORIO DE LOS DATOS - PÍXELES POR DÍGITO                      ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""


# %%
print("Medianas de cada dígito:")
medianas_de_cada_digito = exploratorio.mostrar_medianas_de_cada_digito()

# %%
print("Distribuciones horizontal y vertical de las medianas de los dígitos:")
exploratorio.guardar_distribucion_horizontal_vertical_medianas()
for d in range(10):
    img = plt.imread(f"Images/dist_hv_{d}.png")
    plt.imshow(img)

# %%
print("Diferencias entre las medianas de cada par de dígitos:")
diferencias_cruzadas_medianas = exploratorio.mostrar_diferencias_cruzadas_medianas()

# %%
print("Dígitos más diferentes a su mediana:")
mas_disimiles = exploratorio.mostrar_10x10_mas_disimiles()
# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO BINARIO - MODELO K-NEAREST-NEIGHBORS                                    ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# %%
# Primero, Evaluemos cómo afecta la profundidad en el desempeño del modelo:
ks = range(1, 21)
results1 = knn_binario.probar_k_vecinos(ks)
# %%
# Mostrar gráfico exactitud vs k
fig4, ax4 = knn_binario.grafico_exactitud_vs_k()

ax4.plot(results1["k"], results1["score"], marker="o")
ax4.set_xticks(ks)
print(f"Best Accuracy: {results1['score'].max():.2%}")
# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO BINARIO - PÍXELES DISCRIMINANTES                                        ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

# Cuáles son los 3 píxeles más discriminantes?
print(
    "Criterio 1 (diferencia de medianas cuadrada, dividida por producto de las varianzas de los ceros y de los unos):"
)
top3_a = knn_binario.n_pixeles_mas_disc(3, criterio="default")
print("\t", top3_a)

print("Criterio 2 (varianza en la totalidad del dataset):")
top3_b = knn_binario.n_pixeles_mas_disc(3, criterio="varianza_total")
print("\t", top3_b)

print(
    "Criterio 3 (inversa del producto de las varianzas de los píxeles del cero y de los del uno):"
)
top3_c = knn_binario.n_pixeles_mas_disc(3, criterio="varianza_producto_inv")
print("\t", top3_c)
# Vemos que según los tres criterios, la columna 406 y la 434 distinguen bien.

# %%
# Mostrar los píxeles seleccionados:
fig5 = knn_binario.mostrar_n_px_disc(3, criterio="default")
fig6 = knn_binario.mostrar_n_px_disc(3, criterio="varianza_total")
fig7 = knn_binario.mostrar_n_px_disc(3, criterio="varianza_producto_inv")

# %%
# Cómo lucen estos 3 criterios?
fig8 = knn_binario.mostrar_puntajes_discriminantes(criterio="default")
fig9 = knn_binario.mostrar_puntajes_discriminantes(criterio="varianza_total")
fig10 = knn_binario.mostrar_puntajes_discriminantes(criterio="varianza_producto_inv")

# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO BINARIO - COMPARACIÓN DE CRITERIOS                                      ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# Comparación de desempeño de los distintos criterios
ks = range(1, 21)
results2 = knn_binario.probar_k_vecinos_top_n(15, ks, criterio="default")
results3 = knn_binario.probar_k_vecinos_top_n(15, ks, criterio="varianza_total")
results4 = knn_binario.probar_k_vecinos_top_n(15, ks, criterio="varianza_producto_inv")
# Usamos más de 3 dígitos para resaltar la diferencia en la selección de mejores píxeles.
results5 = knn_binario.probar_k_vecinos_rand_n_m_veces(10, 6, ks)
# %%
# Mostrar Comparación criterios
fig11, ax11 = knn_binario.grafico_exactitud_vs_k()
ax11.plot(
    ks,
    results2["score"],
    marker="o",
    c="blue",
    label="default",
)
ax11.plot(
    ks,
    results3["score"],
    marker="o",
    c="green",
    label="varianza_total",
)
ax11.plot(
    ks,
    results4["score"],
    marker="o",
    c="red",
    label="varianza_producto_inv",
)
fig11.legend()
# default performa mejor.
# %%
# Desempeño de elegir atributos al azar
fig12, ax12 = knn_binario.grafico_exactitud_vs_k()
ax12.plot(
    ks,
    results5["score"],
    marker="o",
    c="red",
    label="random",
)
# Curioso, tiene forma de serrucho...
# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO BINARIO - BÚSQUEDA MEJOR MODELO                                         ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

rango_n = range(1, 21)
rango_k = range(1, 21)
print("Best_accuracy, n, k (cada vez que se rompe el récord):")
results6 = knn_binario.probar_k_y_n(rango_n, rango_k, criterio="default")
# %%
# Mostrar heatmap resultados
# Usa 99% como punto medio de la escala de colores!
# Para mejor distinguir los mejores modelos.
fig13 = knn_binario.mostrar_heatmap_n_y_k(results6, rango_n, rango_k)
# Uno de los mejores: n=15, k=8
#%%
# Mejores atributos utilizados por el modelo n=15
top15_a = knn_binario.n_pixeles_mas_disc(15, criterio="default")
print("\t", top15_a)
fig13b = knn_binario.mostrar_n_px_disc(15, criterio="default")

# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO BINARIO - ENTRENAMIENTO Y EVALUACIÓN MEJOR MODELO                       ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

# Entrenar modelo con los mejores parámetros
mejor_knn = knn_binario.entrenar_modelo(k=8, n=15)
# %%
# Matriz de confusión
mostrar_matriz_de_confusion(mejor_knn, annot_kws={"size": "x-large"}, cmap="binary")
print(f"Accuracy: {mejor_knn.trace()/mejor_knn.sum():.2%}")


# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO MULTICLASE - MODELO DECISION TREE CLASSIFIER                            ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# %%
# Desempeño del modelo con distintas profundidades (k) Sin K-Folding.
depths = list(range(1, 10, 3)) + [19]
results7 = arbol_multiclase.probar_depths(depths)


# %%
# Mostrar gráfico
fig14, ax14 = arbol_multiclase.grafico_depths()
ax14.plot(
    depths,
    results7["score"],
    marker="o",
    color="green",
)
# Profundidad= 10 es la mejor, el resto es lujo.

# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO MULTICLASE - MAX_FEATURES Y CROPPEO                                     ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

# Desempeño del modelo eligiendo una fracción de los atributos totales.
# Probamos también si desempeña (y performa) mejor manejar el dataset con sólo los valores del centro (crop).
mfs = range(10, 150, 20)
results8 = arbol_multiclase.probar_mfs_kfolding(mfs, nsplits=5)
results9 = arbol_multiclase.probar_mfs_kfolding(mfs, nsplits=5, cropped=True)


# %%
# Mostrar gráfico
fig15, ax15 = arbol_multiclase.grafico_mfs()
ax15.plot(
    mfs,
    results8["score"],
    marker="o",
    color="red",
    label="uncropped",
)
ax15.plot(
    mfs,
    results9["score"],
    marker="o",
    color="green",
    label="cropped",
)
ax15.set_xticks(mfs)
fig15.legend(loc="center right")
# Vemos que croppeando performa mejor que sin croppear, y que un max_features = 70 es un valor con buen performance y decente exactitud.
# Vamos a necesitar performance más adelante, pues vamos a analizar dos hiperparámetros más: min_samples_split y min_samples_leaf.


# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO MULTICLASE - COMPARACIÓN DE CRITERIOS EN DEPTHS                         ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# ¿Cuál criterio tiene mejor desempeño?
depths = range(1, 10, 2)
results10 = arbol_multiclase.probar_depths_kfolding(
    depths, nsplits=5, cropped=True, criterio="gini"
)
results10b = arbol_multiclase.probar_depths_kfolding(
    depths, nsplits=5, cropped=True, criterio="entropy"
)
# %%
fig15b, ax15b = arbol_multiclase.grafico_depths()
ax15b.plot(
    depths,
    results10["score"],
    marker="o",
    color="red",
    label="gini",
)
ax15b.plot(
    depths,
    results10b["score"],
    marker="o",
    color="green",
    label="entropy",
)
fig15b.legend(loc="center right")
# Desempeño muy similar por el doble del tiempo de procesamiento :(

# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO MULTICLASE - COMPARACIÓN DE CRITERIOS EN MAX_FEATURES                   ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""

mfs = range(10, 150, 40)
results11 = arbol_multiclase.probar_mfs_kfolding(
    mfs, nsplits=5, cropped=True, criterio="gini"
)
results12 = arbol_multiclase.probar_mfs_kfolding(
    mfs, nsplits=5, cropped=True, criterio="entropy"
)
# Entropy performa casi un punto encima de gini, pero toma el doble del tiempo.
# Cuando se cuenta con un set limitado de atributos, entropy juzga mejor!

# %%
fig16, ax16 = arbol_multiclase.grafico_mfs()
ax16.plot(
    mfs,
    results11["score"],
    marker="o",
    color="red",
    label="gini",
)
ax16.plot(
    mfs,
    results12["score"],
    marker="o",
    color="green",
    label="entropy",
)
fig16.legend(loc="center right")
# Acá se nota que entropy es más exacto, pero al costo del performance
# Es grave la situación de performance. Estas computaciones son costosas.

# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO MULTICLASE - HEATMAP DEPTHS Y MAX_FEATURES                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# Para verificar que vamos bien y elegimos una buena combinación depths-max_features, verifiquemos con algunos pares de valores:
depths = range(2, 12, 2)
mfs = list(range(10, 150, 40)) + [300]
results13 = arbol_multiclase.probar_depths_mfs_kfolding(
    depths, mfs, nsplits=5, cropped=True, criterio="gini"
)
# %%
# Heatmap para mfs y depths con criterio gini.
fig18 = arbol_multiclase.mostrar_heatmap_resultados_depth_mf(results13, depths, mfs)
# En efecto, vemos que los efectos de los hiperparámetros depth y max_features son independientes.

# %%

"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO MULTICLASE - HEATMAP MIN_SAMPLES_SPLIT Y MIN_SAMPLES_LEAF               ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# Usando gini, buscar los mejores parámetros min_samples_split y min_samples_leaf.
# Usamos max_features=30 para acelerar el cálculo.

rango_mss = list(range(2, 15, 3)) + [30, 80, 150, 300]
rango_msl = list(range(1, 15, 3)) + [30, 50]

results13 = arbol_multiclase.probar_mss_msl_kfolding(
    rango_mss,
    rango_msl,
    max_depth=10,
    max_features=30,
    nsplits=5,
    cropped=True,
    criterio="gini",
)
# %%
# Mostrar resultados como heatmap
fig19 = arbol_multiclase.mostrar_heatmap_resultados_mss_msl(
    results13, rango_mss, rango_msl
)
# Ok, el mejor valor parece rondar los mss=5 y msl=4. Veamos más de cerca, y por las dudas aumentando el max_features:
# %%
rango_mss = list(range(2, 6, 1))
rango_msl = list(range(1, 8, 2))
results14 = arbol_multiclase.probar_mss_msl_kfolding(
    rango_mss,
    rango_msl,
    max_depth=10,
    max_features=130,
    nsplits=5,
    cropped=True,
    criterio="gini",
)
# %%
fig20 = arbol_multiclase.mostrar_heatmap_resultados_mss_msl(
    results14, rango_mss, rango_msl
)
# Al aumentar el max_features, ¡le deja de importar el mss y msl! Todos los modelos terminan con exactitud de ~80%. Habíamos planteado max_features como medida de mejorar la velocidad de ejecución de los entrenamientos, pero el modelo final va a tomar la totalidad de los atributos. Así que concluimos que max_depth = 10 es el único hiperparámetro importante para este modelaje.
# Mencionamos también el min_impurity_increase, que en pruebas preliminares demostró que cualquier valor diferente de cero empeoraba el desempeño del modelo.
# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CASO MULTICLASE - ENTRENAMIENTO MEJOR ARBOL                                  ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
modelo = arbol_multiclase.entrenar_arbol(max_depth=10)
M_confusion = arbol_multiclase.evaluar_mejor_arbol(modelo)

# %%
print("Matriz de confusión:")
fig21 = mostrar_matriz_de_confusion(
    M_confusion,
    cmap="binary",
    norm=TwoSlopeNorm(50),
)
# %%
print("Matriz de confusión 'Relativa' (las filas suman 1):")
fig22 = mostrar_matriz_de_confusion(
    np.apply_along_axis((lambda x: x / x.sum()), 1, M_confusion),
    fmt=".2%",
    cmap="binary",
    annot_kws={"size": "x-small"},
    norm=TwoSlopeNorm(0.07),
)

# %%
