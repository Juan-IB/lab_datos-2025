# -*-coding:utf-8 -*-
"""
@File    :   get_comuna.py
@Time    :   2025/02/14 12:33:54
@Author  :   Máximo Mele; Diego Horacio Hermida; Juan Ignacio Bianchini
@Version :   1.0
@License :   CC BY-NC-SA 4.0
@Desc    :   Obtención de la comuna para centros educativos de CABA mediante web requests. Diseñado en colaboración con un modelo de inteligencia artficial; "OpenAI o1", que propuso el API (OpenStreetMap) a utilizar y el método de hacer interfaz con dicho API. Se 
"""

# %%
import requests
import pandas as pd
import time


def get_comuna(lat, lon):
    """
    Given a latitude and longitude in Buenos Aires,
    use Nominatim's reverse-geocoding to find the 'Comuna n'
    string. Returns None if not found.
    """
    # URL de la base de datos de OpenStreetMap
    url = "https://nominatim.openstreetmap.org/reverse"

    # Parámetros para pasar al web request
    params = {
        "format": "jsonv2",
        "lat": lat,
        "lon": lon,
        "addressdetails": 1,
    }

    # Headers necesarios para que la API no bloquee la consulta
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Proyecto TP01/1.0 (contact: mm.mele.maximo@gmail.com)",
    }
    time.sleep(1)
    """
    Entre este Timeout para limitar la frecuencia de consultas y el tiempo de respuesta, cada call de la función tarda entre 2 y 3s, si no hace timeout.
    En total, corriendo el script para los 296 CCs de CABA tomará aprox. 10-15 minutos.
    """
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        geodata = response.json()
        # The 'state_district' field often contains "Comuna N" in Buenos Aires -IA
        address = geodata.get("address", {})
        city_district = address.get("state_district", "")

        if city_district.startswith("Comuna"):
            print(city_district)
            return city_district  # e.g. "Comuna 14" -IA
        else:
            # If the address does not specify the Comuna directly you might try other fields or a fallback like None -IA
            return city_district
    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching Comuna for lat={lat}, lon={lon}: {e}")
        return None
