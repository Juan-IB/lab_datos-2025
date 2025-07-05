# -*-coding:utf-8 -*-
"""
@File    :   establecimientos_educativos.py
@Time    :   2025/02/19 16:01:57
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Procesamiento de los datos de establecimientos_educativos.
"""
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║IMPORTS                                                                      ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
import pandas as pd
def main():
    # Lectura de datos
    ee = pd.read_excel(
        "TablasOriginales/establecimientos_educativos.xlsx",
        header=6,
        dtype=str,
    )
    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║PRE-PROCESAMIENTO: Selección de Columnas a Conservar                     ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """
    # Ordenamos el dataset por Cueanexo y resetamos el index
    ee = ee.sort_values(by="Cueanexo").reset_index(drop=True)

    # Filtramos por las columnas que queremos comenzar a trabajar
    ee = ee[
        [
            "Cueanexo",
            "Código de localidad",
            "Departamento",
            "Común",
            "Nivel inicial - Jardín maternal",
            "Nivel inicial - Jardín de infantes",
            "Primario",
            "Secundario",
            "Secundario - INET",
            "SNU",
            "SNU - INET",
        ]
    ]

    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║PROCESAMIENTO: Selección de Filas a Conservar                            ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """
    # Falta filtrar por aquellas que son comunes.
    # Filtro por escuelas comunes
    ee_comun = ee[ee["Común"] == "1"]

    # Extraemos las columas necesarias para la nueva tabla
    # El copy es porque el slicing sigue apuntando al df original, y para
    # evitar problemas y warnings, creo uno nuevo a partir del slicing
    cueanexo_depto = ee_comun[["Cueanexo", "Código de localidad", "Departamento"]].copy()

    # Extraemos los 5 primeros digitos de cod localidad que es el id depto
    cueanexo_depto["Id_depto"] = (
        cueanexo_depto["Código de localidad"].astype(str).apply(lambda x: x[:5])
    )

    # Trae la tabla modelo de departamento para asignarle a CABA los valores
    # correspondientes, dado que son diferentes en la tabla que los que usa
    # el indec desde 2019
    # departamento.csv debe ya existir (es creada en padron_población.py)
    departamento = pd.read_csv("TablasModelo/departamento.csv", dtype=str)
    comunas = departamento[departamento["Descripcion"].str.startswith("Comuna")]

    # Mapea cueanexo cruzandola con departamento para obtener los id_depto
    # correctos
    cueanexo_depto["Id_depto"] = (
        cueanexo_depto["Departamento"]
        .map(comunas.set_index("Descripcion")["Id_depto"])
        .fillna(cueanexo_depto["Id_depto"])
    )
    cueanexo_depto = cueanexo_depto.drop(["Departamento", "Código de localidad"], axis=1)

    # Arreglar los códigos de ees en Ushuaia
    cueanexo_depto["Id_depto"] = cueanexo_depto["Id_depto"].replace(
        {"94007": 94008, "94014": 94015}
    )

    # Exporta a tablas modelo
    cueanexo_depto.to_csv("TablasModelo/establecimientos_educativos.csv", index=False)


    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║TABLAS: Creación de Tabla nivel_educativo                                ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """

    # Obtenemos el nombre de las columnas que se encuentra la descripcion del nivel
    columns = ee.columns

    # Creamos un diccionario vacío para cargar los diferentes valores
    # Establecemos los valores manualmente, para agruparlos.

    nivel_educativo_a_etapa = {
        "Nivel inicial - Jardín maternal": 0,
        "Nivel inicial - Jardín de infantes": 0,
        "Primario": 1,
        "Secundario": 2,
        "Secundario - INET": 2,
        "SNU": 3,
        "SNU - INET": 3,
    }
    # Cargamos el dataframe con orientacion indice, para que tome las keys :o filas,
    df_nivel_etapa = pd.DataFrame.from_dict(
        nivel_educativo_a_etapa,
        orient="index",
        columns=["Descripcion"],
    )

    # Agregamos el indice como columna para renombrarlo apropiadamente
    df_nivel_etapa = (
        df_nivel_etapa.reset_index()
        .rename(
            columns={
                "index": "Descripcion",
                "Descripcion": "Id_nivel",
            }
        )
        .iloc[:, [1, 0]]
    )


    # Creamos otro dataframe con los Id_nivel que asignamos previamente, y cómo se
    # relacionan con las etapas educativas (Sin distinguir entre técnicos etc.)

    etapa = {
        0: "Jardín",
        1: "Primario",
        2: "Secundario",
        3: "SNU",
    }
    df_etapa = pd.DataFrame.from_dict(
        etapa,
        orient="index",
        columns=["Descripcion"],
    )
    df_etapa = df_etapa.reset_index().rename(columns={"index": "Id_nivel"})

    # Añadimos edad minima y máxima para las etapas educativas a través de una columna
    edad_min_max = {
        0: (0, 5),
        1: (6, 12),
        2: (13, 18),
        3: (19, 999),
    }

    df_etapa["Edad_min"] = df_etapa["Id_nivel"].map(lambda x: edad_min_max[x][0])

    df_etapa["Edad_max"] = df_etapa["Id_nivel"].map(lambda x: edad_min_max[x][1])

    # Exportamos a tabla modelo nivel educativo
    df_etapa.to_csv("TablasModelo/nivel_educativo.csv", index=False)

    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║TABLAS: Creación de Tabla ee-nivel_educativo                             ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """
    # Hacemos slicing sobre el df de est_edu para quedarnos con cueanexo y niveles
    cueanexo_niveles = ee_comun.drop(
        ["Departamento", "Código de localidad", "Común"], axis=1
    ).copy()

    # Hacemos un melt. Es un cross-join entre "id_vars" y todas las columnas restantes
    # Nos queda una tabla de 3 columnas:
    # Cueanexo - Nivel (nombre_excolumna) - Tiene_nivel (valor de la columna)
    df_long = cueanexo_niveles.melt(
        id_vars=["Cueanexo"], var_name="Nivel", value_name="Tiene_nivel"
    )

    # Por precaucion, en caso de que algun 1 tiene un espacio extra
    df_long["Tiene_nivel"] = df_long["Tiene_nivel"].str.strip()

    # Filtra aquellos que tienen_nivel == "1" dado que quedan resultados null por el join
    df_long = df_long[df_long["Tiene_nivel"] == "1"]

    # Hacemos un merge de ambas tablas (como un join) y les coloca el id
    # correspondiente segun la tabla de niveles
    df_long = df_long.merge(
        df_nivel_etapa, left_on="Nivel", right_on="Descripcion", how="left"
    )

    # Nos quedamos solo con las columnas que requerimos para el esquema relacional
    ee_nivel = df_long[["Cueanexo", "Id_nivel"]]

    # Ordenamos el df por Cueanexo -> Id_Nivel, ambos ASC
    ee_nivel = ee_nivel.sort_values(["Cueanexo", "Id_nivel"]).reset_index(drop=True)

    # Exportamos a tablas modelo
    ee_nivel.to_csv(
        "TablasModelo/ee_nivel.csv",
        index=False,
        # float_format=int,
    )
