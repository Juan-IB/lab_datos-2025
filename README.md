# lab_datos-2025 - es

Este proyecto se realizó en el contexto del laboratorio de datos de la UBA (Universidad de Buenos Aires).
Su objetivo era implementar conocimientos y realizar actividades propias de proyectos de ciencia de datos.

# Usage

Primero, necesitas instalar las bibliotecas necesarias. Puedes:
* **Option 1:** Ejecutar `install-lib.bat` para instalar las dependencias:
    ```powershell
    .\install-lib.bat
    ```
* **Option 2:** O instalar las librerias desde el archivo `requirements.txt`:
    ```powershell
    pip install -r requirements.txt
    ```
A continuación, establezca la ruta actual en una de las dos carpetas TP-0* y ejecute main.py desde allí. Esto puede tardar varios minutos. A continuación, se ofrecen instrucciones específicas:

## TP - 01
Antes de ejecutar, asegúrese de estar conectado a internet. El código fuente contiene un script que llamará a la API de StreetWeb y realizará una solicitud web.
> [!IMPORTANT]
> Debido a las limitaciones de velocidad de la API, esto tomará algún tiempo (aproximadamente 15 minutos).
***No interrumpas*** este proceso.

```powershell
cd '.\TP-01'
py .\src\main.py
```

## TP - 02
> [!IMPORTANT]
> El código fuente contiene un script que generará varias tablas derivadas, ocupando un total de 1,21 GB de espacio de almacenamiento.

Si no hay suficiente espacio, pandas puede generar un error, pero el programa mantiene una copia de las tablas en la memoria, por lo que no es crítico para la ejecución (es solo para acelerar la carga de datos en ejecuciones posteriores).

```powershell
cd '.\TP-02'
py .\src\main.py
```


# License

Este proyecto está licenciado bajo la [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Datasets

Este proyecto utiliza los siguiente conjuntos de datos (datasets):

### [Padrón Oficial de Establecimientos Educativos](https://www.argentina.gob.ar/educacion/evaluacion-e-informacion-educativa/padron-oficial-de-establecimientos-educativos)

- **Fuente:** Ministerio de Educación de la Nación - República Argentina
- **Nombre:** Padrón Oficial de Establecimientos Educativos 2022
- **Licencia:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Descripción:** Un nomenclador federal unificado que incluye todos los establecimientos educativos del país, con datos sobre su ubicación, contacto, nivel(es) educativos ofrecidos, planes y programas, carreras, títulos, entre otras variables.
- **Enlace:** [[Enlace al dataset](https://www.argentina.gob.ar/educacion/evaluacion-e-informacion-educativa/padron-oficial-de-establecimientos-educativos)]
- **Enlace alternativo** [[Enlace al dataset]](https://www.datos.gob.ar/es/dataset/educacion-padron-oficial-establecimientos-educativos)

### [Mapa Cultural: Espacios Culturales - Centros Culturales](https://datos.gob.ar/dataset/cultura_37305de4-3cce-4d4b-9d9a-fec3ca61d09f)

- **Fuente:** Ministerio de Cultura de la Nación - República Argentina
- **Nombre:** Centros Culturales
- **Licencia:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Descripción:** Padrón Centros Culturales (2022) en Argentina, hasta el año 2022.
- **Enlace:** [[Enlace al dataset](https://datos.gob.ar/dataset/cultura-mapa-cultural-espacios-culturales/archivo/cultura_0e9a431c-b4f7-455b-aa1a-f419b5740900)]
- **Enlace alternativo** [[Enlace al dataset]](https://datos.gob.ar/dataset/cultura_37305de4-3cce-4d4b-9d9a-fec3ca61d09f)

### [Censo 2022](https://www.indec.gob.ar/indec/web/Nivel4-Tema-2-41-165)

- **Fuente:** INDEC - República Argentina
- **Nombre:** Censo 2022
- **Licencia:** [Creative Commons Attribution‑ShareAlike 2.5 Argentina (CC BY‑SA 2.5 AR)](https://creativecommons.org/licenses/by-nc-sa/2.5/ar/deed.es)
- **Descripción:** Censo Nacional de Población, Hogares y Viviendas 2022 por Departamento – Estructura por edad de la población.
- **Enlace:** [[Enlace al database](https://redatam.indec.gob.ar/binarg/RpWebEngine.exe/Portal?BASE=CPV2022&lang=ESP)]

### [mnist_c - Fog version](https://zenodo.org/records/3239543)

- **Fuente:** Mu, N., & Gilmer, J. (2019). MNIST-C: A Robustness Benchmark for Computer Vision. Zenodo
- **Nombre:** mnist_c_fog
- **Licencia:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Descripción:** Una versión corrupta (con niebla) del conjunto de datos MNIST clásico.
- **Enlace:** [[Enlace al dataset](https://zenodo.org/records/3239543/preview/mnist_c.zip?include_deleted=0#tree_item31)]

# Team Corgis :dog:

![Corgi](https://github.com/Juan-IB/lab_datos-2025/blob/11ef120cdff54ae8a3d6595c5c462c85628ad73c/corgi.png "Team Corgis")

---

# lab_datos-2025 - en

This project was made in the context of the UBA (Universidad de Buenos Aires) data lab subject.
It was intended to implement knowledge and carry out activities that belong to data science projects.

# Usage

First, you need to install the required libraries. You can:
* **Option 1:** Run `install-lib.bat` to install dependencies:
    ```powershell
    .\install-lib.bat
    ```
* **Option 2:** Or install libraries from the `requirements.txt` file:
    ```powershell
    pip install -r requirements.txt
    ```
Next, set current path to one of two TP-0* folders and run main.py from there. This may take a while. Specific instructions follow:

## TP - 01
Before running, make sure you are connected to the internet. Source code contains a script that will call StreetWeb API and make a web request.
> [!IMPORTANT]
> Due to API speed limitations, this will take some time (approximately 15 minutes).
***Don't interrupt*** this process.

```powershell
cd '.\TP-01'
py .\src\main.py
```

## TP - 02
> [!IMPORTANT]
> Source code contains a script that will generates several derived tables, taking up a total of 1.21 GB of storage space.

If there isn't enough space, pandas might throw an error, but the program keeps a copy of the tables in memory, so it's not critical for execution (it's just to speed up data loading in subsequent runs).

```powershell
cd '.\TP-02'
py .\src\main.py
```


# License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Datasets

This project uses the following dataset:

### [Padrón Oficial de Establecimientos Educativos](https://www.argentina.gob.ar/educacion/evaluacion-e-informacion-educativa/padron-oficial-de-establecimientos-educativos)

- **Source:** Ministerio de Educación de la Nación - República Argentina
- **Name:** Padrón Oficial de Establecimientos Educativos 2022
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Description:** Un nomenclador federal unificado que incluye todos los establecimientos educativos del país, con datos sobre su ubicación, contacto, nivel(es) educativos ofrecidos, planes y programas, carreras, títulos, entre otras variables.
- **Link:** [[Link to dataset](https://www.argentina.gob.ar/educacion/evaluacion-e-informacion-educativa/padron-oficial-de-establecimientos-educativos)]
- **Alternate link** [[Link to dataset]](https://www.datos.gob.ar/es/dataset/educacion-padron-oficial-establecimientos-educativos)

### [Mapa Cultural: Espacios Culturales - Centros Culturales](https://datos.gob.ar/dataset/cultura_37305de4-3cce-4d4b-9d9a-fec3ca61d09f)

- **Source:** Ministerio de Cultura de la Nación - República Argentina
- **Name:** Centros Culturales
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Description:** Padrón Centros Culturales (2022) en Argentina, hasta el año 2022.
- **Link:** [[Link to dataset](https://datos.gob.ar/dataset/cultura-mapa-cultural-espacios-culturales/archivo/cultura_0e9a431c-b4f7-455b-aa1a-f419b5740900)]
- **Alternate link** [[Link to dataset]](https://datos.gob.ar/dataset/cultura_37305de4-3cce-4d4b-9d9a-fec3ca61d09f)

### [Censo 2022](https://www.indec.gob.ar/indec/web/Nivel4-Tema-2-41-165)

- **Source:** INDEC - República Argentina
- **Name:** Censo 2022
- **License:** [Creative Commons Attribution‑ShareAlike 2.5 Argentina (CC BY‑SA 2.5 AR)](https://creativecommons.org/licenses/by-nc-sa/2.5/ar/deed.es)
- **Description:** Censo Nacional de Población, Hogares y Viviendas 2022 por Departamento – Estructura por edad de la población.
- **Link:** [[Link to database](https://redatam.indec.gob.ar/binarg/RpWebEngine.exe/Portal?BASE=CPV2022&lang=ESP)]

### [mnist_c - Fog version](https://zenodo.org/records/3239543)

- **Source:** Mu, N., & Gilmer, J. (2019). MNIST-C: A Robustness Benchmark for Computer Vision. Zenodo
- **Name:** mnist_c_fog
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Description:** A corrupted version of classic MNIST dataset with fog.
- **Link:** [[Link to dataset](https://zenodo.org/records/3239543/preview/mnist_c.zip?include_deleted=0#tree_item31)]

# Team Corgis :dog:

![Corgi](https://github.com/Juan-IB/lab_datos-2025/blob/11ef120cdff54ae8a3d6595c5c462c85628ad73c/corgi.png "Team Corgis")

