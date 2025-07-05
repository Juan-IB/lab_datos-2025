# -*-coding:utf-8 -*-
"""
@File    :   padron_poblacion.py
@Time    :   2025/02/19 15:57:49
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Este archivo realiza el pre-procesamiento del padrón poblacional
del censo 2022. Los datos de este vienen desagregados por "área" (jurisdicción/
departamento), es decir, hay una tabla por área. Es entonces necesario un 
script que recorra el archivo, detecte de qué área se está hablando y agregue 
como columnas dicha información a cada entrada de las ~50.000 entradas 
presentes.
"""
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║IMPORTS                                                                      ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
import pandas as pd
def main():
    # Importa los datos raw del excel (requiere openpyxl)
    # Instalar con pip install openpyxl
    censo_raw = pd.read_excel("TablasOriginales/padron_poblacion.xlsX")

    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║PRE-PROCESAMIENTO: Obtención del Id_departamento y Nombre_depto          ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """
    # Dropea la primer columa que no tiene datos de tablas
    censo = censo_raw.iloc[:, 1:]

    # Dropeo todas las filas que tienen Null en todas las columnas
    censo = censo.dropna(how="all").reset_index(drop=True)

    # Renombro las columnas para trabajar mas comodo
    censo = censo.rename(
        columns={
            censo.columns[0]: "1",
            censo.columns[1]: "2",
            censo.columns[2]: "3",
            censo.columns[3]: "4",
        }
    )

    # Creo una nueva columna donde almaceno los numeros de area,
    # que se extraen del string con un regex
    censo["Id_depto"] = censo["1"].str.extract(r"AREA # (\d+)")
    censo["Nombre_depto"] = censo["2"].where(censo["1"].str.startswith("AREA"))

    # Busco el indice de la fila donde dice "Resumen" para dropear todo desupues
    #  de eso
    idx = censo[censo.iloc[:, 0].str.contains("RESUMEN", na=False)].index.min()

    # Chequeo que ese indice existe, por precaucion sobre todo
    if not pd.isna(idx):  # Solo hace slicing si idx no es na
        censo = censo.iloc[:idx, :].reset_index(drop=True)

    # Uso Forward fill en las columna Id_depto para propagar
    # el valor hacia abajo
    censo["Id_depto"] = censo["Id_depto"].ffill()
    censo["Nombre_depto"] = censo["Nombre_depto"].ffill()

    # Remuevo las filas que dicen area, Edad o Total
    # Es un regex y el pipe '|' funciona como un OR
    censo = censo[~censo["1"].str.contains("AREA|Total|Edad", na=False)].reset_index(
        drop=True
    )

    # Renombro las columnas con sus respectivos nombres
    censo = censo.rename(
        columns={
            censo.columns[0]: "Edad",
            censo.columns[1]: "Casos",
            censo.columns[2]: "Porcentaje",
            censo.columns[3]: "Acumulado",
        }
    )
    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║PRE-PROCESAMIENTO: Selección de Columnas a Conservar                     ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """
    # Seleccionamos las columnas del esquema
    grupo_poblacional = censo[["Id_depto", "Edad", "Casos"]]

    # Ordenamos por depto y edad
    grupo_poblacional = grupo_poblacional.sort_values(["Id_depto", "Edad"]).reset_index(
        drop=True
    )

    # Exportamos a tablas modelo
    grupo_poblacional.to_csv("TablasModelo/grupo_poblacional.csv", index=False)

    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║TABLAS: Creación de Tabla departamento                                   ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """
    # Toma el Id_depto y Nombre_depto y elimina los duplicados
    deptos = censo[["Id_depto", "Nombre_depto"]].drop_duplicates()

    # Renombro la columna para que corresponda con el MR.
    deptos.rename(columns={"Nombre_depto": "Descripcion"}, inplace=True)

    # Extrae del Id_depto el código de provincia.
    deptos["Id_prov"] = deptos["Id_depto"].apply(lambda id: id[0:2])

    # Ordena por ids de provincia y de depto.
    deptos.sort_values(["Id_prov", "Id_depto"], inplace=True)

    # Reordeno columnas por preferencia.
    deptos = deptos.iloc[:, [2, 0, 1]]

    deptos.to_csv("TablasModelo/departamento.csv", index=False)
