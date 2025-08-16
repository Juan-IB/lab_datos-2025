# -*-coding:utf-8 -*-
"""
@File    :   obtener_datos.py
@Time    :   2025/08/07 12:07:23
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.
@License :   CC BY-NC-SA 4.0
@Desc    :   (nuevo) Descarga el dataset desde la fuente en línea, y lo adapta al formato necesario para la ejecución del resto del código. Elimina la necesidad de tener almacenado localmente el dataset inicial.
"""
# %%
import requests
import json
import os
import zipfile
from shutil import rmtree
from tqdm import tqdm
import math

import numpy as np
import pandas as pd

print("Módulo derivados_mnist.py - Cargando datos y Guardando derivados")


# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ Descargar dataset                                                           ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# %%
def download_url(
          url: str, 
          file_path: str, 
          chunk_size: int = 128):
    """
    Descarga y almacena un archivo desde una fuente en linea, en partes (chunks).

    Args:
        url (str): Dirección del archivo en linea.
        file_path (str): Dirección local en la que se almacenará el archivo.
        chunk_size (int): Tamaño de las partes en las que se dividirá el archivo (en Byte/s). Por defecto, 128.

    Returns:
        None
    """
    
    source_raw = requests.get(url, stream=True)
    total_size: int = int(source_raw.headers['content-length'])
    
    # Descargar y almacenar archivo, dividiendolo en paquetes más pequeños, de tamaño chunk_size.
    with open(file_path, 'wb') as file, tqdm(desc='Parte', total=total_size, unit='B', unit_scale=True, unit_divisor=chunk_size, colour='blue') as bar:
        for chunk in source_raw.iter_content(chunk_size=chunk_size):
            size = file.write(chunk)
            bar.update(size) # Actualizar barra de progreso
        file.close()

# %%
if not os.path.isdir('.temp'):
            os.mkdir('.temp')

# %%
print("Descargando datos:")
# Descarga los datasts MNIST_C
try:
    url: str = "https://zenodo.org/api/records/3239543/files/mnist_c.zip/content"

    download_url(url, ".temp/mnist_c.zip", chunk_size=1024)

except:
    if os.path.isfile('.temp/mnist_c.zip'):
        os.remove('.temp/mnist_c.zip')
    
    print('Error al descargar el archivo. Compruebe la fuente.')

# %%
# Descomprime el archivo descargado. Conserva solo el dataset fog.
try:
    with zipfile.ZipFile('.temp/mnist_c.zip') as zpfile:
        zpfile.extractall('.temp')
        zpfile.close()

    files: list[str] = os.listdir('.temp/mnist_c/fog')

    # Mueve solo los archivos del dataset fog
    if (not all(map(os.path.isfile, ['.temp/' + file for file in files]))):
        list(map(os.rename, ['.temp/mnist_c/fog/' + file for file in files], ['.temp/' + file for file in files]))

    rmtree('.temp/mnist_c')
    os.remove('.temp/mnist_c.zip')

except:
    print('Error al descomprimir. Vuelva a ejecutar el archivo')

# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║ Adaptar formato                                                             ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
train_images = np.lib.format.open_memmap(".temp/train_images.npy")
train_labels = np.lib.format.open_memmap(".temp/train_labels.npy")

test_images = np.lib.format.open_memmap(".temp/test_images.npy")
test_labels = np.lib.format.open_memmap(".temp/test_labels.npy")

train = np.reshape(train_images.flat,(60000,784))
test = np.reshape(test_images.flat, (10000, 784))
images = np.concatenate((train,test))

labels = np.concatenate((train_labels, test_labels))

data = pd.DataFrame(images)
data['labels'] = labels

data.to_csv(r'mnist_c_fog_tp.csv', index=True)

# %%
# Elimina los archivos temporales

train_images._mmap.close()
train_labels._mmap.close()
test_images._mmap.close()
test_labels._mmap.close()

rmtree('.temp')