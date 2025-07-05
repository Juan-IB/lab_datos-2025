# -*-coding:utf-8 -*-
"""
@File    :   centros_culturales.py
@Time    :   2025/02/19 16:26:44
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Procesamiento de los datos de centros_culturales
"""

"""
╔═════════════════════════════════════════════════════════════════════════════╗
║IMPORTS                                                                      ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
import pandas as pd
from get_comuna import get_comuna


def main():
    # Lectura de datos
    cc_raw = pd.read_csv("TablasOriginales/centros_culturales.csv", dtype=str)

    # departamento.csv debe ya existir (es creada en padron_población.py)
    dd = pd.read_csv("TablasModelo/departamento.csv", dtype=str)

    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║PRE-PROCESAMIENTO: Selección de Columnas a Conservar                     ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """
    # Elimina espacios en blanco de nom de columas, por ej Mail era 'Mail '
    cc_raw.columns = cc_raw.columns.str.strip()

    # Carga las columnas para el procesado
    cc = cc_raw[
        ["ID_DEPTO", "Nombre", "Domicilio", "Mail", "Latitud", "Longitud", "Capacidad"]
    ]

    # Selecciono solo las comunas en dd para no tener duplicados, como 'Capital'
    # o el map no funciona
    dd = dd[dd["Descripcion"].str.startswith("Comuna")]

    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║PROCESAMIENTO: Limpieza de Datos                                         ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """

    # Modificamos manualmente la latitud de Acassuso 6930 que estaba incorrecta
    # (Figura que está en CABA pero las coordenadas indican Vte. López)
    # (Coordenadas correctas extraídas de google maps)
    cc.loc[
        (cc["Latitud"] == "-34.53719000") & (cc["Domicilio"] == "Acassuso 6930"),
        "Latitud",
    ] = "-34.65192535"

    # Eliminar filas que no contienen valores que indican un domicilio,
    # para mantener la integridad de la superclave
    cc = cc.drop(list(cc.loc[cc["Domicilio"].apply(lambda m: len(m) < 4)].index))

    # Loc hace que aquellas filas de la columna 'Comuna' (La crea si no existe)
    # que cumplen con ID_DEPTO == '02000' se les aplique la funcion lambda
    # Esta toma latitud y longitud y nos devuelve el distrito, en este caso comunas de CABA.
    # IMPORTANTE: Utiliza un Web Request. Puede tardar ~15 min en completar.
    cc.loc[cc["ID_DEPTO"] == "02000", "Comuna"] = cc.loc[
        cc["ID_DEPTO"] == "02000"
    ].apply(lambda row: get_comuna(row["Latitud"], row["Longitud"]), axis=1)

    # Mapeamos Id_depto de la tabla de departamentos con la columna de comuna
    cc["ID_DEPTO"] = (
        cc["Comuna"].map(dd.set_index("Descripcion")["Id_depto"]).fillna(cc["ID_DEPTO"])
    )

    # Reemplazamos todo mail incorrecto (vacío, erróneo o duplicado) con NULL.
    # Los mails correctos sufren una limpieza.
    cc["Mail"] = cc["Mail"].apply(
        lambda mail: str(mail).strip().lower() if (str(mail).count("@") == 1) else pd.NA
    )

    # Reemplazar todas las capacidades que sean cero por NULL
    cc["Capacidad"] = cc["Capacidad"].apply(lambda c: c if c != "0" else pd.NA)

    # Arreglar los códigos de ccs en Ushuaia
    cc["ID_DEPTO"] = cc["ID_DEPTO"].replace({"94007": 94008, "94014": 94015})

    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║TABLAS: Creación de Tabla centros_culturales                             ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """
    # Renombre de Id_depto por esquema
    cc = cc.rename(columns={"ID_DEPTO": "Id_depto"})

    # Nos quedamos con columnas del esquema
    # longitud y latitud ya no son necesarias
    cc = cc[["Id_depto", "Nombre", "Domicilio", "Mail", "Capacidad"]]

    # Ordenamos por Id_depto y Nombre
    cc = cc.sort_values(["Id_depto", "Nombre"]).reset_index(drop=True)

    # Añadimos un indice arbitrario
    cc = cc.rename(columns={"Mail": "Email"})

    # Exportamos a csv tabla modelo
    cc.to_csv("TablasModelo/centros_culturales.csv", index=False)

    """
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║TABLAS: Creación de Tabla provincia                                      ║
    ╚═════════════════════════════════════════════════════════════════════════╝
    """
    # Selecciono solo los atributos relevantes.
    provincias = cc_raw[["ID_PROV", "Provincia"]]

    # Remuevo duplicados.
    provincias.drop_duplicates(inplace=True)

    # Renombro columnas como en el MR.
    provincias.rename(
        columns={"ID_PROV": "Id_prov", "Provincia": "Descripcion"},
        inplace=True,
    )
    # Exportar a TablasModelo.
    provincias.to_csv("TablasModelo/provincia.csv", index=False)
