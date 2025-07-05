# -*-coding:utf-8 -*-
"""
@File    :   pruebas_calidad.py
@Time    :   2025/02/14 12:32:23
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Procesamiento de métricas para la calidad de datos, siguiendo el método GQM.
"""
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║IMPORTS                                                                      ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
# %%
import pandas as pd
import duckdb as dd

# %% Lectura de datos
ee_raw = pd.read_excel(
    r"TablasOriginales/establecimientos_educativos.xlsx", header=6, dtype=str
)
cc_raw = pd.read_csv(r"TablasOriginales/centros_culturales.csv", dtype=str)
cc = pd.read_csv(r"TablasModelo/centros_culturales.csv", dtype={'Id_depto':'str'})
ee = pd.read_csv(r"TablasModelo/establecimientos_educativos.csv", dtype={'Id_depto':'str'})
ee_nivel = pd.read_csv(r"TablasModelo/ee_nivel.csv")
nivel_educativo = pd.read_csv(r"TablasModelo/nivel_educativo.csv")
gp = pd.read_csv(r"TablasModelo/grupo_poblacional.csv", dtype={'Id_depto':'str'})
departamento = pd.read_csv(r"TablasModelo/departamento.csv", dtype={'Id_depto':'str'})
provincia = pd.read_csv(r"TablasModelo/provincia.csv", dtype={'Id_prov':'str'})



# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CALIDAD: CENTROS CULTURALES                                                  ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""


def calidad_mails_cc_raw():
    # Proporción de Mails duplicados sobre entradas con al menos un mail
    duplicados = cc_raw.loc[
        cc_raw["Mail "].apply(lambda m: str(m).count("@") > 1), "Mail "
    ]
    con_mail = cc_raw.loc[
        cc_raw["Mail "].apply(lambda m: str(m).count("@") >= 1), "Mail "
    ]
    invalidos = cc_raw.loc[
        cc_raw["Mail "].apply(lambda m: str(m).count("@") < 1), "Mail "
    ]

    tasa_de_duplicados = len(duplicados) / len(con_mail)
    tasa_de_invalidos = len(invalidos) / len(cc_raw)

    return f"Tasa de mails duplicados en CC original: {tasa_de_duplicados:.1%}\nTasa de mails inválidos en CC original: {tasa_de_invalidos:.1%}"
def calidad_mails_cc():
    # Proporción de Mails duplicados sobre entradas con al menos un mail
    duplicados = cc.loc[cc["Email"].apply(lambda m: str(m).count("@") > 1), "Email"]
    con_mail = cc.loc[cc["Email"].apply(lambda m: str(m).count("@") >= 1), "Email"]
    invalidos = cc.loc[cc["Email"].apply(lambda m: str(m).count("@") < 1), "Email"]

    tasa_de_duplicados = len(duplicados) / len(con_mail)
    tasa_de_invalidos = len(invalidos) / len(cc)

    return f"Tasa de mails duplicados en CC: {tasa_de_duplicados:.1%}\nTasa de mails inválidos en CC: {tasa_de_invalidos:.1%}"

def calidad_capacidad_cc_raw():
    # Proporción de valores null's o con 0 en capacidad
    capacidad_null = cc_raw[cc_raw['Capacidad'].isna()]
    capacidad_zero = cc_raw[cc_raw['Capacidad'] == '0']

    tasa_sin_capacidad = (len(capacidad_null) + len(capacidad_zero)) / len(cc_raw)

    return f"Tasa de entradas sin capacidad en CC original: {tasa_sin_capacidad:.1%}"
def calidad_capacidad_cc():
    # Proporción de valores null's (convertimos capacidad = 0 a NULLs en procesamiento)
    capacidad_null = cc[cc['Capacidad'].isna()]

    tasa_sin_capacidad = len(capacidad_null) / len(cc_raw)

    return f"Tasa de entradas sin capacidad en CC: {tasa_sin_capacidad:.1%}"

def calidad_domicilio_cc_raw():
    # Proporción de valores que no indican un domicilio
    sin_domicilio = cc_raw.loc[cc_raw["Domicilio"].apply(lambda m: len(m) < 4), "Domicilio"]

    tasa_sin_domicilio = len(sin_domicilio) / len(cc_raw)

    return f"Tasa de entradas sin domicilio en CC original: {tasa_sin_domicilio:.1%}"
def calidad_domicilio_cc():
    # Proporción de valores que no indican un domicilio
    sin_domicilio = cc.loc[cc["Domicilio"].apply(lambda m: len(m) < 4), "Domicilio"]

    tasa_sin_domicilio = len(sin_domicilio) / len(cc)

    return f"Tasa de entradas sin domicilio en CC: {tasa_sin_domicilio:.1%}"

# %% Calidad del join entre departamento y cc por IDs_Depto


def calidad_join_cc_raw():
    discrepancias = dd.sql(
            """--sql
    SELECT 
        d.Id_depto as En_departamento,
        cc_raw.ID_DEPTO as En_CC_raw,
    FROM departamento d
        FULL OUTER JOIN cc_raw
            ON cc_raw.ID_DEPTO = d.Id_depto
    WHERE d.Id_depto IS NULL OR cc_raw.ID_DEPTO IS NULL
    GROUP BY 
        d.Id_depto,
        cc_raw.ID_DEPTO,
    """
        ).df()
    deptos_cc_no_en_departamento = len(discrepancias[discrepancias.isna()['En_departamento']]             
    )


    deptos_departamento_no_en_cc = len(discrepancias[discrepancias.isna()['En_CC_raw']]
    )
    # Hay un montón de departamentos en d que no tienen CC!!
    return f"Id_deptos de CC original ausentes en departamento: {deptos_cc_no_en_departamento}\nId_deptos de departamento ausentes en CC original: {deptos_departamento_no_en_cc}"
def calidad_join_cc():
    discrepancias = dd.sql(
            """--sql
    SELECT 
        d.Id_depto as En_departamento,
        cc.ID_DEPTO as En_CC,
    FROM departamento d
        FULL OUTER JOIN cc
            ON cc.ID_DEPTO = d.Id_depto
    WHERE d.Id_depto IS NULL OR cc.ID_DEPTO IS NULL
    GROUP BY 
        d.Id_depto,
        cc.ID_DEPTO,
    """
        ).df()
    
    deptos_cc_no_en_departamento = len(discrepancias[discrepancias.isna()['En_departamento']]             
    )
    deptos_departamento_no_en_cc = len(discrepancias[discrepancias.isna()['En_CC']]
    )
    # Hay un montón de departamentos en gp que no tienen CC!!
    return f"Id_deptos de CC ausentes en departamento: {deptos_cc_no_en_departamento}\nId_deptos de departamento ausentes en CC: {deptos_departamento_no_en_cc}"
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CALIDAD: ESTABLECIMIENTOS EDUCATIVOS                                         ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
def calidad_join_ee_raw():
    discrepancias = dd.sql(
        """--sql
    SELECT 
        d.Id_depto as En_departamento,
        SUBSTRING(ee_raw['Código de localidad'], 0, 6) as En_EE,
    FROM ee_raw
        FULL OUTER JOIN departamento d
            ON SUBSTRING(ee_raw['Código de localidad'], 0, 6) = d.Id_depto
    WHERE En_EE IS NULL OR En_departamento IS NULL
    GROUP BY En_EE, En_departamento
    ORDER BY En_EE, En_departamento
        """
    ).df()
    ee_no_en_gp = len(discrepancias[discrepancias.isna()['En_departamento']] )
    # 15 comunas, río grande y Ushuaia (por mal código);
    # + Antártida (Hay una escuela pero ese depto no figuró en el censo)
    # Total: 17
    gp_no_en_ee = len(discrepancias[discrepancias.isna()['En_EE']])
    # 14 comunas (02105 está en ambas pero representa comunas diferentes).
    # + 94011: Tolhuin (No tiene EEs aparentemente)
    # Total: 16
    return f"Id_deptos de EE Original ausentes en departamento: {ee_no_en_gp}\nId_deptos de departamento ausentes en EE Original: {gp_no_en_ee}"
def calidad_join_ee():
    discrepancias = dd.sql(
        """--sql
    SELECT 
        d.Id_depto as En_departamento,
        ee.Id_depto as En_EE,
    FROM ee
        FULL OUTER JOIN departamento d
            ON ee.Id_depto = d.Id_depto
    WHERE En_EE IS NULL OR En_departamento IS NULL
    GROUP BY En_EE, En_departamento
    ORDER BY En_EE, En_departamento
        """
    ).df()
    ee_no_en_gp = len(discrepancias[discrepancias.isna()['En_departamento']] )
    # Sólo 94028: Antártida (Hay una escuela pero no figuró en el censo)
    gp_no_en_ee = len(discrepancias[discrepancias.isna()['En_EE']])
    # Sólo 94011: Tolhuin (No tiene EEs aparentemente)
    return f"Id_deptos de EE ausentes en departamento: {ee_no_en_gp}\nId_deptos de departamento ausentes en EE: {gp_no_en_ee}"

def calidad_niveles_ee_raw():
    # Proporción de valores vacios
    niveles_null = ee_raw[list(ee_raw.columns)[21:28]].apply(lambda m: m != "1").sum().sum()
    niveles_num_celdas = len(ee_raw)*int(len(list(ee_raw.columns)[21:28]))

    tasa_niveles_vacios = niveles_null / niveles_num_celdas

    return f"Número de entradas vacias de niveles (EE original): {niveles_null}\nTasa de entradas vacias de niveles (EE original): {tasa_niveles_vacios:.1%}"

if __name__ == "__main__":
    # Calidad CC
    print("---------Centros Culturales (CC)---------")
    print(calidad_mails_cc_raw())
    print(calidad_mails_cc())
    print(calidad_capacidad_cc_raw())
    print(calidad_capacidad_cc())
    print(calidad_domicilio_cc_raw())
    print(calidad_domicilio_cc())

    print("\n---------Uniones---------")
    #Calidad del join entre departamento y cc por 
    print(calidad_join_cc_raw())
    print(calidad_join_cc())

    print("\n---------Establecimientos Educativos (EE)---------")
    #Calidad EE
    print(calidad_niveles_ee_raw())
    print(calidad_join_ee_raw())
    print(calidad_join_ee())