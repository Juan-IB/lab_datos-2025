# -*-coding:utf-8 -*-
"""
@File    :   main.py
@Time    :   2025/02/12 10:50:54
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Este archivo centraliza la ejecución de los módulos que componen los pasos del TP. Cada archivo posee una función main() que ejecuta la totalidad del correspondiente archivo. Las respuestas a las consultas y las visalizaciones son guardadas en la carpeta ResultadosQueries, la cual es creada si no existe.
"""
import os
import padron_poblacion
import centros_culturales
import establecimientos_educativos


"""
╔═════════════════════════════════════════════════════════════════════════════╗
║Importación de los datos, preprocesamiento y limpieza.                       ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
if not os.path.isdir("TablasModelo"): 
    os.makedirs("TablasModelo")  # Aseguramos que exista la carpeta
# Los procedimientos de cada paso están descriptos en cada archivo.
padron_poblacion.main()
establecimientos_educativos.main()

print("[IMPORTANTE] Este archivo realiza un web request. Por ende, tardará aproximadamente 15 minutos, con lo cual se recomienda explorar el contenido de los archivos MIENTRAS se ejecuta.")
centros_culturales.main()
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║Pruebas de Calidad                                                           ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
import pruebas_calidad

# Calidad CC
print("---------Centros Culturales (CC)---------")
print(pruebas_calidad.calidad_mails_cc_raw())
print(pruebas_calidad.calidad_mails_cc())
print(pruebas_calidad.calidad_capacidad_cc_raw())
print(pruebas_calidad.calidad_capacidad_cc())
print(pruebas_calidad.calidad_domicilio_cc_raw())
print(pruebas_calidad.calidad_domicilio_cc())

print("\n---------Uniones---------")
# Calidad del join entre departamento y cc por
print(pruebas_calidad.calidad_join_cc_raw())
print(pruebas_calidad.calidad_join_cc())

print("\n---------Establecimientos Educativos (EE)---------")
# Calidad EE
print(pruebas_calidad.calidad_niveles_ee_raw())
print(pruebas_calidad.calidad_join_ee_raw())
print(pruebas_calidad.calidad_join_ee())
"""
╔═════════════════════════════════════════════════════════════════════════════╗
║Consultas SQL                                                                ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
import queries
if not os.path.isdir("ResultadosQueries"): 
    os.makedirs("ResultadosQueries")  # Aseguramos que exista la carpeta
queries.consulta1().to_csv(r"ResultadosQueries/consulta1.csv", index=False)
queries.consulta2().to_csv(r"ResultadosQueries/consulta2.csv", index=False)
queries.consulta3().to_csv(r"ResultadosQueries/consulta3.csv", index=False)
queries.consulta4().to_csv(r"ResultadosQueries/consulta4.csv", index=False)

"""
╔═════════════════════════════════════════════════════════════════════════════╗
║Visualizaciones                                                              ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
queries.visualizacion1().savefig(
    r"ResultadosQueries/visualizacion1.png",
    bbox_inches="tight",
)
queries.visualizacion1bis().savefig(
    r"ResultadosQueries/visualizacion1bis.png",
    bbox_inches="tight",
)
queries.visualizacion2().savefig(
    r"ResultadosQueries/visualizacion2.png",
    bbox_inches="tight",
)
queries.visualizacion3().savefig(
    r"ResultadosQueries/visualizacion3.png",
    bbox_inches="tight",
)
queries.visualizacion4().savefig(
    r"ResultadosQueries/visualizacion4.png",
    bbox_inches="tight",
)
queries.visualizacion4bis().savefig(
    r"ResultadosQueries/visualizacion4bis.png",
    bbox_inches="tight",
)
queries.visualizacion4tris().savefig(
    r"ResultadosQueries/visualizacion4tris.png",
    bbox_inches="tight",
)
