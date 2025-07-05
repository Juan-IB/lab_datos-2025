# -*-coding:utf-8 -*-
"""
@File    :   queries.py
@Time    :   2025/02/23 18:05:35
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Realización de las consultas requeridas, junto con las visualizaciones para el trabajo.
"""

# %% Imports
import duckdb as dd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cc = pd.read_csv(r"TablasModelo/centros_culturales.csv", dtype={"Id_depto": "str"})
ee = pd.read_csv(
    r"TablasModelo/establecimientos_educativos.csv", dtype={"Id_depto": "str"}
)
ee_nivel = pd.read_csv(r"TablasModelo/ee_nivel.csv")
nivel_educativo = pd.read_csv(r"TablasModelo/nivel_educativo.csv")
gp = pd.read_csv(r"TablasModelo/grupo_poblacional.csv", dtype={"Id_depto": "str"})
departamento = pd.read_csv(r"TablasModelo/departamento.csv", dtype={"Id_depto": "str"})
provincia = pd.read_csv(r"TablasModelo/provincia.csv", dtype={"Id_prov": "str"})


# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CONSULTA i)                                                                  ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""


def consulta1():
    cantidad_de_ee_por_depto_por_nivel = dd.sql(
        """--sql
        SELECT ee.Id_depto, ee_nivel.Id_nivel, COUNT(*) AS Cantidad
        FROM 
            ee
            INNER JOIN ee_nivel ON ee_nivel.Cueanexo = ee.Cueanexo
        GROUP BY ee.Id_depto, ee_nivel.Id_nivel
        ORDER BY ee.Id_depto , ee_nivel.Id_nivel, Cantidad DESC
        """
    ).df()

    gp_con_nivel = dd.sql(
        """--sql
    SELECT 
        gp.*,
        n.Id_nivel 
    FROM
        gp
        INNER JOIN nivel_educativo n 
            ON gp.Edad BETWEEN n.Edad_min AND n.Edad_max
    ORDER BY gp.Id_depto, gp.Edad

    """
    ).df()

    poblacion_por_depto_por_nivel = dd.sql(
        """--sql
        SELECT 
            Id_depto,
            Id_nivel,
            SUM(Casos) AS Poblacion ,
        FROM gp_con_nivel
        GROUP BY
            Id_depto,
            Id_nivel,
        ORDER BY
            Id_depto,
            Id_nivel
        """
    ).df()

    data_jardines = dd.sql(
        """--sql
    SELECT 
        p.Id_depto,
        e.Cantidad AS 'Jardines',
        p.Poblacion AS 'Población Jardín'
    FROM
        poblacion_por_depto_por_nivel p
        INNER JOIN cantidad_de_ee_por_depto_por_nivel e
            ON (p.Id_depto = e.Id_depto AND p.Id_nivel = e.Id_nivel)
    WHERE 
        p.Id_nivel = 0
        """
    ).df()

    data_primarias = dd.sql(
        """--sql
    SELECT 
        p.Id_depto,
        e.Cantidad AS 'Primarias',
        p.Poblacion AS 'Población Primaria'
    FROM
        poblacion_por_depto_por_nivel p
        INNER JOIN cantidad_de_ee_por_depto_por_nivel e
            ON (p.Id_depto = e.Id_depto AND p.Id_nivel = e.Id_nivel)
    WHERE 
        p.Id_nivel = 1
        """
    ).df()

    data_secundarios = dd.sql(
        """--sql
    SELECT 
        p.Id_depto,
        e.Cantidad AS 'Secundarios',
        p.Poblacion AS 'Población Secundaria'
    FROM
        poblacion_por_depto_por_nivel p
        INNER JOIN cantidad_de_ee_por_depto_por_nivel e
            ON (p.Id_depto = e.Id_depto AND p.Id_nivel = e.Id_nivel)
    WHERE 
        p.Id_nivel = 2
        """
    ).df()

    consulta1 = dd.sql(
        """--sql
    SELECT 
        p.Descripcion AS Provincia,
        d.Descripcion AS Departamento,
        CAST(jar['Jardines'] AS INTEGER) AS 'Jardines',
        CAST(jar['Población Jardín'] AS INTEGER) AS 'Población Jardín',
        CAST(pri['Primarias'] AS INTEGER) AS 'Primarias',
        CAST(pri['Población Primaria'] AS INTEGER) AS 'Población Primaria',
        CAST(sec['Secundarios'] AS INTEGER) AS 'Secundarios',
        CAST(sec['Población Secundaria'] AS INTEGER) AS 'Población Secundaria',
    FROM 
        departamento as d
        INNER JOIN provincia p
            ON d.Id_prov = p.Id_prov
        INNER JOIN data_jardines jar
            ON jar.Id_depto = d.Id_depto
        INNER JOIN data_primarias pri
            ON pri.Id_depto = d.Id_depto
        INNER JOIN data_secundarios sec
            ON sec.Id_depto = d.Id_depto
    ORDER BY 
        Provincia ASC,
        pri['Primarias'] DESC
        """
    ).df()
    return consulta1


# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CONSULTA ii)                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""


def consulta2():
    consulta2 = dd.sql(
        """--sql
    SELECT 
        p.Descripcion AS Provincia,
        d.Descripcion AS Departamento,
        (
            SELECT Count(*) FROM cc
            WHERE Capacidad > 100 AND cc.Id_depto = d.Id_depto
            ) 
        AS Cantidad,
    FROM departamento AS d
        INNER JOIN provincia AS p
            ON p.Id_prov = d.Id_prov
    GROUP BY 
        Departamento,
        d.Id_depto,
        Provincia,
    ORDER BY 
        Provincia ASC,
        Cantidad DESC,
        Departamento ASC
    """
    ).df()
    return consulta2


# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CONSULTA iii)                                                                ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""


def consulta3():
    consulta3 = dd.sql(
        """--sql
    SELECT 
        d.Descripcion AS Departamento,
        p.Descripcion AS Provincia,
        (
            SELECT Count(*)
            FROM ee
            WHERE ee.Id_depto = d.Id_depto
            ) AS Cant_EE,
        (
            SELECT Count(*)
            FROM cc
            WHERE cc.Id_depto = d.Id_depto
            ) AS Cant_CC,
        CAST((
            SELECT SUM(Casos)
            FROM gp
            WHERE gp.Id_depto = d.Id_depto
            ) AS INTEGER) AS Poblacion_total
    FROM 
        departamento d
        INNER JOIN provincia p
            ON d.Id_prov = p.Id_prov
    ORDER BY 
        Cant_EE DESC,
        Cant_CC DESC,
        Provincia ASC,
        Departamento ASC
    """
    ).df()
    return consulta3


# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CONSULTA iv)                                                                  ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""


def consulta4():
    dominios_frecuencias_por_depto = dd.sql(
        """--sql
    SELECT 
        Id_depto,
        SUBSTRING(Email, POSITION('@' IN Email) + 1) AS dominio,
        count(*) AS freq
    FROM
        cc
    WHERE 
        Email IS NOT NULL
    GROUP BY dominio, Id_depto
    ORDER BY Id_depto ASC, freq DESC
    """
    ).df()

    consulta4 = dd.sql(
        """--sql
    SELECT DISTINCT
        p.Descripcion as Provincia,
        d.Descripcion AS Departamento,
        dfd.dominio AS 'Dominio más frecuente en CC'
    FROM dominios_frecuencias_por_depto AS dfd
        INNER JOIN departamento AS d
            ON d.Id_depto = dfd.Id_depto
        INNER JOIN provincia AS p 
            ON p.Id_prov = d.Id_prov
        WHERE dfd.dominio = 
            (
            SELECT dfd1.dominio
            FROM dominios_frecuencias_por_depto AS dfd1
            INNER JOIN departamento as d1
            ON dfd1.Id_depto = d.Id_depto
            WHERE dfd1.freq = 
                (
                SELECT MAX(dfd2.freq)
                FROM dominios_frecuencias_por_depto AS dfd2
                WHERE dfd2.Id_depto = d.Id_depto
                )
            ORDER BY dfd1.dominio ASC
            LIMIT 1
            )
        ORDER BY p.Id_prov ASC, d.Id_depto ASC
    """
    ).df()
    return consulta4


# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CONSULTA VISUALIZACIÓN i)                                                    ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""


def visualizacion1():
    c3 = consulta3()
    v1 = dd.sql(
        """--sql
    SELECT Provincia, sum(Cant_CC) as Cant_CC
    FROM c3
    GROUP BY Provincia
    ORDER BY Cant_CC DESC, Provincia ASC
    """
    ).df()

    v1.loc[
        v1["Provincia"] == "Tierra del Fuego, Antártida e Islas del Atlántico Sur",
        "Provincia",
    ] = "Tierra del Fuego"
    v1.loc[
        v1["Provincia"] == "Ciudad Autónoma de Buenos Aires",
        "Provincia",
    ] = "CABA"

    # barplot para mostrar de forma descendiente los
    fig, ax = plt.subplots()
    sns.barplot(
        data=v1,
        y="Provincia",
        x="Cant_CC",
        ax=ax,
        color="#a8dfd1",
        width=0.85,
    )
    ax.set_xlabel("Cantidad de centros culturales")
    fig
    fig.set_size_inches(7, 4)
    plt.xticks(rotation=90)
    
    return fig


# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CONSULTA VISUALIZACIÓN ii)                                                   ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""


def visualizacion2():
    fig2, ax2 = plt.subplots()
    c1 = consulta1()
    ax2.scatter(
        x=c1["Población Jardín"],
        y=c1["Jardines"],
        c="#c44933",
        alpha=0.8,
        label="Inicial",
    )
    ax2.scatter(
        x=c1["Población Primaria"],
        y=c1["Primarias"],
        c="#55b528",
        alpha=0.8,
        label="Primario",
    )
    ax2.scatter(
        x=c1["Población Secundaria"],
        y=c1["Secundarios"],
        c="#282ab5",
        alpha=0.8,
        label="Secundario",
    )

    ax2.set_ylabel("Cantidad de EE")
    ax2.set_xlabel("Cantidad de habitantes")  
    ax2.set_ylim(top=300)  
    ax2.set_xlim(-5000, 80000)
    fig2.set_size_inches(7, 3)
    ax2.legend(loc="lower right")
    return fig2


# %%
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CONSULTA VISUALIZACIÓN iii)                                                  ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""


def visualizacion3():
    visualizacion3 = dd.sql(
        """
                            
        SELECT d.Id_prov, d.Id_depto, d.Provincia,
        COUNT(d.ID_depto) AS Cantidad_EE, 
        FROM 
            (SELECT DISTINCT p.Id_prov,
            p.Descripcion as Provincia, d.Id_depto
            FROM departamento AS d
            INNER JOIN provincia AS p
            ON p.Id_prov = d.Id_prov
            ORDER BY d.Id_depto) as d
        INNER JOIN ee
        ON d.Id_depto = ee.Id_depto
        GROUP BY d.id_prov, d.Provincia, d.Id_depto
        ORDER BY d.id_prov, d.Id_depto
                            
    """
    ).to_df()
    # Renombramos Tierra del Fuego y CABA para que los labels no aporten carga cognitiva
    visualizacion3.loc[
        visualizacion3["Provincia"]
        == "Tierra del Fuego, Antártida e Islas del Atlántico Sur",
        "Provincia",
    ] = "Tierra del Fuego"
    visualizacion3.loc[
        visualizacion3["Provincia"] == "Ciudad Autónoma de Buenos Aires", "Provincia"
    ] = "CABA"

    # Calculamos las medianas agrupando por provincia
    medianas = visualizacion3.groupby("Provincia")["Cantidad_EE"].median()

    # Ordenamos por la mediana y obtenemos el orden por el indice
    sorted_categories = medianas.sort_values(ascending=False).index

    # Creamos la figura 3 con su ax correspondiente
    fig3, ax3 = plt.subplots()

    # Agregamos titulo y labels
    ax3.set_xlabel("Provincia")
    ax3.set_ylabel("Cantidad de ee por departamento")
    ax3.set_ylim((0, 410))
    sns.boxplot(
        data=visualizacion3,
        x=visualizacion3["Provincia"],
        y=visualizacion3["Cantidad_EE"],
        order=sorted_categories,
        color="#a8dfd1",
        showmeans=True,
    )
    fig3.set_size_inches(10, 4)
    plt.xticks(rotation=90)
    return fig3


# %%

"""
╔═════════════════════════════════════════════════════════════════════════════╗
║CONSULTA VISUALIZACIÓN iv)                                                   ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""


def visualizacion4():
    c3 = consulta3()
    visualizacion4 = dd.sql(
        """--sql
    SELECT c.*,
    (c.Cant_CC * 1000 / c.Poblacion_total) AS cc_cada_mil,
    (c.Cant_EE * 1000 / c.Poblacion_total) AS ee_cada_mil,
    FROM c3 AS c
    """
    ).df()

    fig4, ax4 = plt.subplots()

    ax4.scatter(
        y=visualizacion4["cc_cada_mil"], x=visualizacion4["ee_cada_mil"], alpha=0.5,
        c="#55b528",
        edgecolors='none',
    )

    ax4.set_xlabel("Cantidad de ee cada 1000 habitantes")
    ax4.set_ylabel("Cantidad de cc cada 1000 habitantes")
    ax4.set_xlim((0, 4))
    ax4.set_ylim((0, 0.1))
    fig4.set_size_inches(7, 4)
    
    return fig4


# %%
def visualizacion4bis():
    c3 = consulta3()
    v4bis = dd.sql(
        """--sql
    SELECT c.*,
    (c.Poblacion_total/c.Cant_CC) AS pob_cada_cc,
    (c.Cant_EE * 1000 / c.Poblacion_total) AS ee_cada_mil,
    FROM c3 AS c
    WHERE NOT (c.Cant_EE = 0 OR c.Cant_CC = 0) 
    """
    ).df()

    fig4bis, ax4bis = plt.subplots()

    sns.boxplot(
        y=v4bis["pob_cada_cc"],
        x=(v4bis["ee_cada_mil"] * 2 - 0.5).round() / 2 + 0.25,
        ax=ax4bis,
        native_scale=True,
        color="#a8dfd1",
        showmeans=True,
    )

    ax4bis.set_xlabel("Cantidad de ee cada 1000 habitantes")
    ax4bis.set_ylabel("habitantes por cc")
    ax4bis.set_xlim((0, 4.5))
    ax4bis.set_ylim((0, 400000))
    fig4bis.set_size_inches(4, 3)
    return fig4bis


# %%
def visualizacion4tris():

    c3 = consulta3()
    v4tris = dd.sql(
        """--sql
    SELECT 
        c.*,
        (c.Cant_CC * 1000 / c.Poblacion_total) AS cc_cada_mil,
        (c.Cant_EE * 1000 / c.Poblacion_total) AS ee_cada_mil,
    FROM c3 AS c
    WHERE NOT (c.Cant_EE = 0 OR c.Cant_CC = 0) 
        """
    ).df()

    fig4tris, ax4tris = plt.subplots()
    sns.boxplot(
        x=(v4tris["ee_cada_mil"] * 2 - 0.5).round() / 2 + 0.25,
        y=v4tris["cc_cada_mil"],
        native_scale=True,
        ax=ax4tris,
        color="#a8dfd1",
        showmeans=True,
    )

    ax4tris.set_xlabel("Cantidad de ee cada 1000 habitantes")
    ax4tris.set_ylabel("Cantidad de cc cada 1000 habitantes")
    ax4tris.set_xlim((0, 4.5))
    fig4tris.set_size_inches(4, 3)
    return fig4tris



# %%
def visualizacion1bis():
    c3 = consulta3()
    v1bis = dd.sql(
        """--sql
    SELECT Provincia, 100000*sum(Cant_CC)/sum(Poblacion_total) as cc_por_cien_mil_hab
    FROM c3
    GROUP BY Provincia
    ORDER BY cc_por_cien_mil_hab DESC
    """
    ).df()

    v1bis.loc[
        v1bis["Provincia"] == "Tierra del Fuego, Antártida e Islas del Atlántico Sur",
        "Provincia",
    ] = "Tierra del Fuego"
    v1bis.loc[
        v1bis["Provincia"] == "Ciudad Autónoma de Buenos Aires",
        "Provincia",
    ] = "CABA"

    # barplot para mostrar de forma descendiente los
    fig1bis, ax1bis = plt.subplots()
    sns.barplot(
        y=v1bis["Provincia"],
        x=v1bis["cc_por_cien_mil_hab"],
        ax=ax1bis,
        color="#a8dfd1",
        width=0.85,
    )
    fig1bis.set_size_inches(7, 4)
    
    ax1bis.set_xlabel("CC cada 100.000 habitantes")
    return fig1bis



# %%
