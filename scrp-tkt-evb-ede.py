import io
import base64
from email.message import EmailMessage
from googleapiclient.discovery import build
import os
import joblib
import pandas as pd
import hashlib
import string

def _numero_a_sufijo(n: int) -> str:
    """
    Convierte un entero >= 0 a sufijo alfabético estilo Excel:
    0→A, 1→B, …, 25→Z, 26→AA, 27→AB, …, 701→ZZ, 702→AAA, …
    Escala infinitamente sin límite de dígitos.
    """
    letras = string.ascii_uppercase
    resultado = []
    n += 1  # Hacemos 1-indexed internamente
    while n > 0:
        n, resto = divmod(n - 1, 26)
        resultado.append(letras[resto])
    return "".join(reversed(resultado))

def generar_id(url: str, prefijo: str = "EVT") -> str:
    """Genera un ID único y estable basado en la URL del evento."""
    hash_corto = hashlib.md5(str(url).encode()).hexdigest()[:8].upper()
    return f"{prefijo}-{hash_corto}"

def generar_ids_con_sufijo(df: pd.DataFrame, col_url: str, prefijo: str) -> pd.Series:
    """
    Para scrapers donde todos los eventos comparten la misma URL
    (ej: Ferias y Congresos), genera IDs con sufijo A, B, C... AA, AB...
    El hash se toma de la URL base, el sufijo diferencia cada fila.
    """
    url_base = str(df[col_url].iloc[0]) if not df.empty else "sin-url"
    hash_corto = hashlib.md5(url_base.encode()).hexdigest()[:8].upper()
    return pd.Series(
        [f"{prefijo}-{hash_corto}-{_numero_a_sufijo(i)}" for i in range(len(df))],
        index=df.index
    )

def log(mensaje):
    timestamp = datetime.now().strftime('%H:%M:%S')
    linea = f"[{timestamp}] {mensaje}"
    print(linea)
    log_buffer.write(linea + "\n")
# Buffer para acumular los prints
log_buffer = io.StringIO()

_modelo_clasificador = None
def cargar_modelo_clasificador():
    global _modelo_clasificador
    if _modelo_clasificador is not None:
        return _modelo_clasificador

    rutas_posibles = [
        "modelo_clasificador_eventos.pkl",
        "__modelo_clasificador_eventos.pkl",
        os.path.join(os.path.dirname(__file__), "modelo_clasificador_eventos.pkl"),
    ]

    for ruta in rutas_posibles:
        if os.path.exists(ruta):
            # Verificación extra: ¿El archivo tiene contenido?
            if os.path.getsize(ruta) < 100: 
                log(f"❌ El archivo en {ruta} es demasiado pequeño para ser un modelo válido.")
                continue
                
            try:
                with open(ruta, "rb") as f:
                    _modelo_clasificador = joblib.load(ruta)
                log(f"✅ Modelo clasificador cargado desde: {ruta}")
                return _modelo_clasificador
            except Exception as e:
                log(f"❌ Error crítico al leer el archivo pkl en {ruta}: {e}")
                # Si un archivo está mal, probamos la siguiente ruta en lugar de romper todo
                continue

    log("⚠️ ADVERTENCIA: No se pudo cargar ningún modelo válido. Se omitirá la clasificación.")
    return None

def aplicar_clasificador(df, col_nombre, col_lugar, col_tipo_evento, col_confianza="confianza_clasificacion"):
    """
    Aplica el modelo logístico a las filas donde 'col_tipo_evento' está vacía.

    Args:
        df: DataFrame a procesar.
        col_nombre: Nombre de la columna con el nombre del evento (columna A).
        col_lugar: Nombre de la columna con el lugar (columna B).
        col_tipo_evento: Nombre de la columna de tipo de evento (columna E).
        col_confianza: Nombre de la columna donde se registrará el nivel de confianza.

    Returns:
        DataFrame con predicciones y confianzas completadas, y un dict de métricas.
    """
    modelo = cargar_modelo_clasificador()
    metricas = {"predicciones": 0, "confianza_promedio": None, "fuente": ""}

    if modelo is None or df.empty:
        return df, metricas

    # Aseguramos que la columna de confianza exista
    if col_confianza not in df.columns:
        df[col_confianza] = None

    # Identificamos filas a predecir: tipo de evento vacío/nulo
    mask_vacio = df[col_tipo_evento].isna() | (df[col_tipo_evento].astype(str).str.strip() == "")

    if not mask_vacio.any():
        return df, metricas

    df_a_predecir = df[mask_vacio].copy()

    # Construimos el vector de features igual que en el entrenamiento
    X = (
        df_a_predecir[col_nombre].astype(str).fillna("")
        + " "
        + df_a_predecir[col_lugar].astype(str).fillna("")
    )

    try:
        predicciones = modelo.predict(X)
        # predict_proba devuelve la prob de cada clase; tomamos el máximo como confianza
        probs = modelo.predict_proba(X)
        confianzas = np.max(probs, axis=1).round(4)

        df.loc[mask_vacio, col_tipo_evento] = predicciones
        df.loc[mask_vacio, col_confianza] = confianzas

        metricas["predicciones"] = int(mask_vacio.sum())
        metricas["confianza_promedio"] = round(float(np.mean(confianzas)), 4)

    except Exception as e:
        log(f"⚠️ Error al ejecutar el clasificador: {e}")

    return df, metricas

def click_load_more_until_disappears(driver):
    """
    Hace clic en el botón 'Cargar más' repetidamente hasta que desaparece.

    Args:
        driver (webdriver): El objeto webdriver de Selenium.
    """
    try:
        while True:  # Bucle infinito, se rompe cuando el botón desaparece
            try:
                # Espera hasta que el botón esté presente y sea clickeable (máximo 10 segundos)
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//div[@class='infinite-scroll refresh col-xs-10 text-center padding-vertical-hard margin-top']/span[@class='text-uppercase bg-light-blue padding-vertical padding-horizontal-hard']"))
                )

                # Hace clic en el botón
                load_more_button.click()
                time.sleep(5)  # Espera un poco para que se carguen más elementos
                print("Botón 'Cargar más' clickeado.")

            except NoSuchElementException:
                # Si el botón ya no existe, salimos del bucle
                print("El botón 'Cargar más' ya no está presente.")
                break  # Sale del bucle while

            except Exception as e:
                # Captura otras excepciones (por ejemplo, TimeoutException si el botón tarda en aparecer)
                print(f"Error al hacer clic en el botón 'Cargar más': {e}")
                break  # Sale del bucle while
    except Exception as e:
        log(f"Error general: {e}")


def extract_artist_data(soup):
    """
    Extrae el título y el href de los elementos 'tkt-artist-list-image-item' y los almacena en un DataFrame.

    Args:
        soup (BeautifulSoup): El objeto BeautifulSoup que contiene el HTML analizado.

    Returns:
        pandas.DataFrame: Un DataFrame con las columnas 'title' y 'href'.
    """
    artist_elements = soup.find_all('div', class_='tkt-artist-list-image-item relative col-xs-10 col-sm-25 margin-bottom')
    artist_data = []

    for artist_element in artist_elements:
        a_tag = artist_element.find('a', class_='info-container absolute')
        if a_tag:
            title = a_tag.get('title')
            href = 'ticketek.com.ar/' + a_tag.get('href', '')  # Agrega el prefijo y maneja hrefs faltantes
            artist_data.append({'title': title, 'href': href})

    return pd.DataFrame(artist_data)
import time
from bs4 import BeautifulSoup
from urllib.parse import quote
from selenium.common.exceptions import WebDriverException  # Importa la excepción
import numpy as np

def extract_details_from_page(driver, href):
    try:
        driver.get(href)
        time.sleep(2) 
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        details = {'price': None, 'lugar': None, 'description': None}

        # --- LUGAR (Estrategia Multicapa) ---
        # 1. Intentar por el widget de compra (lo que ya hacíamos)
        lugar_element = soup.find('div', class_='padding-vertical pull-left')
        if lugar_element:
            details['lugar'] = lugar_element.get_text(strip=True)
        
        # 2. Si falló, buscar en los atributos 'data-venue' del header (Caso Sin Bandera)
        if not details['lugar']:
            header = soup.find('div', attrs={'data-tkt-show-header': True})
            if header and header.has_attr('data-venue'):
                details['lugar'] = header['data-venue']

        # --- PRECIO (Sección lateral) ---
        left_sidebar = soup.find('section', id='left-sidebar')
        if left_sidebar:
            details['price'] = left_sidebar.get_text(separator=" ", strip=True)

        # --- DESCRIPCIÓN (Barrido Total) ---
        # Buscamos en 'top' y 'main-content'. 
        # En Sin Bandera, la info está en un div dentro de 'top'.
        textos = []
        for section_id in ['top', 'main-content', 'left-sidebar']:
            section = soup.find('section', id=section_id)
            if section:
                # Buscamos todos los bloques de texto
                bloques = section.find_all('div', class_='tkt-content-content')
                for b in bloques:
                    textos.append(b.get_text(separator=" ", strip=True))
        
        if textos:
            details['description'] = " . ".join(textos)

        return details
    except Exception as e:
        return {'error': str(e), 'price': None, 'lugar': None, 'description': None}
    
import re

def extract_details_from_location(driver, href):
    try:
        driver.get(href)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        details = {'price': None, 'lugar': None, 'description': None}

        # --- 1. LUGAR (Enfoque por URL Robusto) ---
        print(f"Procesando URL: {href}")
        
        # Eliminamos posibles barras al final y dividimos
        # Esto funciona tanto para URLs completas como para paths
        parts = [p for p in href.strip('/').split('/') if p]
        
        # Si la URL tiene la estructura de Ticketek (/artista/lugar)
        # El lugar siempre será el último segmento
        if len(parts) >= 2:
            last_part = parts[-1]
            # Limpiamos guiones y capitalizamos (ej: quality-arena -> Quality Arena)
            lugar_limpio = last_part.replace('-', ' ').title()
            details['lugar'] = lugar_limpio
        else:
            # Fallback: Intentar extraer del HTML si la URL es muy corta
            header_widget = soup.find(attrs={'data-tkt-show-header': True})
            if header_widget and header_widget.get('data-venue'):
                details['lugar'] = header_widget['data-venue'].strip()

        # --- 2. PRECIO ---
        left_sidebar = soup.find('section', id='left-sidebar')
        if left_sidebar:
            details['price'] = left_sidebar.get_text(separator=" ", strip=True)

        # --- 3. DESCRIPCIÓN ---
        textos_acumulados = []
        # Añadimos 'footer' por si acaso, pero mantenemos tu estructura
        for sec_id in ['top', 'main-content', 'left-sidebar']:
            seccion = soup.find('section', id=sec_id)
            if seccion:
                bloques = seccion.find_all('div', class_='tkt-content-content')
                for bloque in bloques:
                    txt = bloque.get_text(separator=" ", strip=True)
                    if txt:
                        textos_acumulados.append(txt)

        if textos_acumulados:
            details['description'] = " . ".join(textos_acumulados)

        return details

    except Exception as e:
        log(f"Error en extracción: {e}")
        return {'error': str(e), 'price': None, 'lugar': None, 'description': None}
import time
from bs4 import BeautifulSoup
from urllib.parse import quote
from selenium.common.exceptions import WebDriverException  # Importa la excepción

import pandas as pd
import re
from datetime import datetime

def clean_data(df):
    """
    Limpia los datos en el DataFrame df, extrayendo precios y fechas.
    """

    def extract_and_sum_prices(text):
        """
        Extrae y suma precios del formato "$NNNN + $NNNN".
        """
        if text is None or not isinstance(text, str):
            return None
        
        # Eliminar todos los puntos de la cadena de texto
        text_sin_puntos = text.replace('.', '')
        # Captura formatos de precios
        prices = re.findall(r'(?:Desde )?\$(\d{1,6})(?: \+ \$(\d{1,6}))?', text_sin_puntos)
        
        if prices:
            total_prices = []
            for match in prices:
                price1 = match[0]
                price2 = match[1]
                price1_int = int(price1)
                if price2:
                    price2_int = int(price2)
                    total_price = price1_int + price2_int
                    #log(f"Precio 1: {price1_int}, Precio 2: {price2_int}, Suma: {total_price}")
                    total_prices.append(total_price)
                else:
                    print(f"Precio: {price1_int}")
                    total_prices.append(price1_int)
            return total_prices
        return None

    def calculate_average(price_list):
        """
        Calcula el promedio de una lista de precios.
        """
        if price_list:
            avg = sum(price_list) / len(price_list)
            #log(f"Lista de precios: {price_list}, Promedio: {avg}")
            return avg
        return None

    def extraer_fecha(texto):
        """
        Extrae la fecha priorizando el segmento de Córdoba/Quality.
        """
        if not texto or not isinstance(texto, str):
            return None

        # 1. Segmentación por ciudad
        segmento_interes = texto
        if "córdoba" in texto.lower() or "quality" in texto.lower():
            partes = texto.split('.')
            for parte in partes:
                if "córdoba" in parte.lower() or "quality" in parte.lower():
                    segmento_interes = parte
                    break

        # 2. Diccionario de meses
        meses = {
            'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
            'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
        }
        
        # Regex mejorado para capturar día, mes y opcionalmente el año
        patron = re.compile(r'(\d{1,2})\s+de\s+([a-z]+)(?:\s+de\s+)?(\d{4})?', re.IGNORECASE)
        
        match = patron.search(segmento_interes)
        if match:
            dia, mes_str, anio_str = match.groups()
            mes = meses.get(mes_str.lower())
            
            if mes:
                if anio_str:
                    anio = int(anio_str)
                else:
                    ahora = datetime.now()
                    anio = ahora.year + (1 if mes < ahora.month else 0)
                
                return f"{anio:04}-{mes:02}-{int(dia):02}"
        return None

    # --- Ejecución de las transformaciones ---
    df['price_list'] = df['price'].apply(extract_and_sum_prices)
    df['price_avg'] = df['price_list'].apply(calculate_average)
    df.drop(columns=['price_list'], inplace=True)
    
    df['date'] = df['description'].apply(extraer_fecha)

    return df
from urllib.parse import quote

def process_hrefs(driver, df):
    prices = []
    lugares = []
    descriptions = []
    errors = []  # Lista para almacenar los errores

    for href in df['href']:
        if href:  # verifica si el href existe.
            full_href = "https://" + quote(href)  # agrega el esquema y codifica la url.
            try:
                if href.count('/') == 2:
                    details = extract_details_from_page(driver, full_href)
                elif href.count('/') == 1:
                    driver.get(full_href)
                    time.sleep(2)

                    soup = BeautifulSoup(driver.page_source, 'html.parser')
                    location_links = soup.find_all('a', class_='artist-shows-item')
                    details = {'price': None, 'lugar': None, 'description': None}  # Inicializa details

                    for link in location_links:
                        location_data = link.get('data-venue-locality')
                        if location_data and 'Córdoba' in location_data:
                            location_href = "https://ticketek.com.ar/" + quote(link.get('data-link'))
                            details = extract_details_from_location(driver, location_href)
                            break
                else:
                    details = {'price': None, 'lugar': None, 'description': None}

                prices.append(details.get('price'))
                lugares.append(details.get('lugar'))
                descriptions.append(details.get('description'))
                errors.append(details.get('error') if 'error' in details else None)  # Almacena el error, si existe

            except Exception as e:
                log(f"Error processing {full_href}: {e}")
                prices.append(None)
                lugares.append(None)
                descriptions.append(None)
                errors.append(str(e))  # Almacena el error

        else:
            prices.append(None)
            lugares.append(None)
            descriptions.append(None)
            errors.append(None)  # No hay error si no hay href

    df['price'] = prices
    df['lugar'] = lugares
    df['description'] = descriptions
    df['error'] = errors  # Agrega la columna de errores
    return df

def reordenar_y_agregar_columnas(df):
    """
    Reordena las columnas del DataFrame y agrega nuevas columnas según las especificaciones.

    Args:
        df (pandas.DataFrame): El DataFrame original con las columnas:
                              ['title', 'href', 'price', 'lugar', 'description', 'price_avg', 'date'].

    Returns:
        pandas.DataFrame: El DataFrame modificado con las columnas reordenadas y las nuevas columnas agregadas:
                          ['title', 'lugar', 'date', 'finaliza', 'tipo de evento', 'detalle', 'alcance',
                           'price_avg', 'fuente', 'href', 'price', 'description'].
                          Las columnas 'finaliza', 'tipo de evento', 'detalle' y 'alcance' estarán vacías (None).
                          La columna 'fuente' contendrá el valor 'Ticketek' en todas las filas.
    """

    # Crear un nuevo orden de columnas
    nuevo_orden_columnas = ['title', 'lugar', 'date', 'finaliza', 'tipo de evento', 'detalle', 'alcance',
                           'price_avg', 'fuente', 'href', 'price', 'description']

    # Crear las nuevas columnas vacías
    df['finaliza'] = None
    df['tipo de evento'] = None
    df['detalle'] = None
    df['alcance'] = None

    # Crear la columna 'fuente' con el valor 'Ticketek'
    df['fuente'] = 'Ticketek'

    # Reordenar las columnas
    df = df[nuevo_orden_columnas]

    return df
def limpiar_lugar(nombre):
    # Convertimos a string por seguridad y verificamos
    nombre_str = str(nombre)
    if 'Quality Espacio' in nombre_str:
        return 'Quality Espacio'
    elif 'Quality Arena' in nombre_str:
        return 'Quality Arena'
    elif 'Quality Teatro' in nombre_str:
        return 'Quality Teatro'
    elif 'Teatro Comedia' in nombre_str:
        return 'Teatro Comedia'
    else:
        return nombre



import pandas as pd
import time
import re
import gspread
from datetime import datetime
from urllib.parse import quote
from bs4 import BeautifulSoup
from oauth2client.service_account import ServiceAccountCredentials
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, WebDriverException

# --- CONFIGURACIÓN INICIAL ---
def iniciar_driver():
    chrome_options = Options()
    # Ocultamos los logs de errores de Google que mencionaste antes
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_argument('--log-level=3')
    
    # --- Cambio solicitado: Modo Headless ---
    chrome_options.add_argument("--headless=new") 
    
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    chrome_options.add_argument(f"user-agent={user_agent}")
    
    # He añadido el retorno del driver para que la función sea operativa
    driver = webdriver.Chrome(options=chrome_options)
    return driver
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def subir_a_google_sheets(df, nombre_tabla, nombre_hoja="sheet1", retries=3):
    import numpy as np
    import time
    import pandas as pd
    import os
    import json
    import gspread
    from google.oauth2 import service_account
    from datetime import datetime

    secreto_json = os.environ.get('GCP_SERVICE_ACCOUNT_JSON')
    if secreto_json is None:
        print("🔴 DIAGNÓSTICO: La variable os.environ no encuentra 'GCP_SERVICE_ACCOUNT_JSON'. Revisa el YAML.")
        return False
    
    intentos = 0
    while intentos < retries:
        try:
            # --- CONEXIÓN ---
            info_claves = json.loads(secreto_json)
            creds = service_account.Credentials.from_service_account_info(
                info_claves, 
                scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            )
            client = gspread.authorize(creds)
            sheet = client.open(nombre_tabla).worksheet(nombre_hoja)
            
            # Obtener datos existentes
            existing_data = sheet.get_all_values()
            
            # --- 1. PREPARACIÓN DE DATOS ENTRANTES ---
            df_entrada = df.copy()
            conteo_reales = 0
            
            # Todas las tablas se tratan como acumulativas/históricas, con un tratamiento especial
            # solo para Ferias y Congresos (Auto) en la eliminación de duplicados por 'Eventos'.
            es_ferias_auto = nombre_tabla == 'Ferias y Congresos (Auto)'

            # --- 2. LÓGICA DE DETECCIÓN DE FILAS NUEVAS ---
            if len(existing_data) > 1:
                existing_df = pd.DataFrame(existing_data[1:], columns=existing_data[0])
                df_nuevas_reales = df_entrada.copy()
                print(f"ℹ️ Modo Históricamente Acumulativo: Agregando todas las filas a '{nombre_tabla}'.")
                combined_df = pd.concat([existing_df, df_nuevas_reales], ignore_index=True)
                conteo_reales = len(df_nuevas_reales)
            else:
                # Caso Hoja Vacía
                combined_df = df_entrada
                df_nuevas_reales = df_entrada.copy()
                conteo_reales = len(df_entrada)

            # --- 3. LIMPIEZA Y ELIMINACIÓN DE DUPLICADOS INTERNOS ---
            if not combined_df.empty:
                if es_ferias_auto:
                    # Para Ferias y Congresos, eliminamos duplicados por evento.
                    if 'Eventos' in combined_df.columns:
                        combined_df = combined_df.drop_duplicates(subset=['Eventos'], keep='first')
                else:
                    # Para el resto de las hojas históricas, mantenemos el historial y eliminamos
                    # duplicados exactos según la mejor clave disponible.
                    if set(['href', 'title', 'date']).issubset(combined_df.columns):
                        combined_df = combined_df.drop_duplicates(subset=['href', 'title', 'date'], keep='first')
                    elif set(['title', 'date']).issubset(combined_df.columns):
                        combined_df = combined_df.drop_duplicates(subset=['title', 'date'], keep='first')
                    else:
                        columnas_id = ['ID','Origen', 'href', 'Link', 'URL']
                        id_col_final = next((c for c in columnas_id if c in combined_df.columns), None)
                        if id_col_final:
                            combined_df = combined_df.drop_duplicates(subset=[id_col_final], keep='first')

                # --- 4. ORDENAMIENTO (Lo más nuevo arriba) ---
                col_fecha_carga = next((c for c in ['fecha de carga', 'Fecha Scrp'] if c in combined_df.columns), None)
                if col_fecha_carga:
                    combined_df[col_fecha_carga] = pd.to_datetime(combined_df[col_fecha_carga], errors='coerce')
                    combined_df = combined_df.sort_values(by=col_fecha_carga, ascending=False)

                # --- 5. FORMATEO ANTI-ERROR (JSON Serializing) ---
                def serializar_datos(val):
                    if pd.isna(val) or val is pd.NaT: return ""
                    if isinstance(val, (datetime, pd.Timestamp)):
                        return val.strftime('%Y-%m-%d %H:%M:%S')
                    return str(val) if isinstance(val, (dict, list)) else val

                combined_df = combined_df.replace([np.inf, -np.inf], np.nan).fillna("")
                data_final = combined_df.applymap(serializar_datos)
                
                # --- 6. SUBIDA FINAL ---
                # sheet.clear()
                # valores_a_subir = [data_final.columns.values.tolist()] + data_final.values.tolist()
                # sheet.update(valores_a_subir, value_input_option='USER_ENTERED')
                
                log(f"✅ Hoja '{nombre_tabla}' actualizada.")
                log(f"📊 Se agregaron {conteo_reales} filas nuevas.")
                return True 
            else:
                print(f"⚠️ DataFrame vacío para {nombre_tabla}")
                return False
        
        except Exception as e:
            intentos += 1
            print(f"⚠️ Error al subir a Sheets (Intento {intentos}/{retries}): {e}")
            if intentos < retries:
                time.sleep(5)
            
    return False
    
def ejecutar_scraper_ticketek():
    """
    Ejecuta el scraper y devuelve un reporte del estado.
    """
    driver = None
    reporte = {
        "nombre": "Ticketek",
        "estado": "Pendiente",
        "filas_procesadas": 0,
        "error": None,
        "inicio": datetime.now().strftime('%H:%M:%S')
    }
    
    # Este DataFrame vive en el ámbito de raper_ticketek
    df_rechazados = pd.DataFrame(columns=['Nombre', 'Locación', 'Fecha', 'Motivo', 'Linea', 'Fuente'])

    # Esta función DEBE estar aquí adentro (un nivel de tabulación más)
    def registrar_rechazo(nombre, loc, fecha, motivo, linea, fuente, href, col_href="Link"):
        nonlocal df_rechazados # Ahora sí puede encontrar la variable de arriba
        nuevo = pd.DataFrame([{
            'Nombre': nombre, 
            'Locación': loc, 
            'Fecha': fecha,
            'Motivo': motivo, 
            'Linea': str(linea), 
            'Fuente': fuente,
            col_href: href
        }])
        df_rechazados = pd.concat([df_rechazados, nuevo], ignore_index=True)
    
    try:
        driver = iniciar_driver()

        # 1. Cargar página y expandir
        url = "https://www.ticketek.com.ar/buscar/?f%5B0%5D=field_artist_node_eb%253Afield_show_venue%253Afield_city:C%C3%B3rdoba"
        driver.get(url)
        time.sleep(10)
        
        # Usamos tus funciones
        click_load_more_until_disappears(driver)
        
        # 2. Extraer lista base
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        df_artists = extract_artist_data(soup)
        
        if df_artists.empty:
            log("No se encontraron artistas. Finalizando tarea Ticketek.")
            return

        # 3. Procesar detalles de cada link
        df_artists2 = process_hrefs(driver, df_artists)

        #3.1 Auditoría
        df_con_errores = df_artists2[df_artists2['error'].notna()]
        
        for _, row in df_con_errores.iterrows():
            motivo_error = f"Error de carga/navegación: {row['error']}"
            registrar_rechazo(
                nombre=row['title'], 
                loc="N/A", 
                fecha="N/A", 
                motivo=motivo_error, 
                linea="570",
                fuente='Ticketek',
                href=row['href']
            )
        
        # 4. Limpieza y Reordenamiento
        df_artists2_cleaned = clean_data(df_artists2.copy())
        #4.1 Auditoría
        mask_sin_fecha = (df_artists2_cleaned['date'].isna()) & (df_artists2_cleaned['error'].isna())
        df_fallos_fecha = df_artists2_cleaned[mask_sin_fecha]
        
        for _, row in df_fallos_fecha.iterrows():
            # Guardamos la descripción completa para auditoría técnica
            descripcion_completa = row['description'] if row['description'] else "SIN DESCRIPCIÓN DISPONIBLE"
            
            registrar_rechazo(
                nombre=row['title'], 
                loc=row['lugar'] if row['lugar'] else "No detectado", 
                fecha="No encontrada", 
                motivo=f"FALLO DE EXTRACCIÓN de fecha. Texto analizado: {descripcion_completa}", 
                linea="586",
                fuente='Ticketek',
                href=row['href']
            )
        df_artists2_cleaned['lugar'] = df_artists2_cleaned['lugar'].apply(limpiar_lugar)
        
        # --- PASO 2: Registro de Auditoría para Lugares Inválidos (Línea 244) ---
        # Solo registramos los que llegaron aquí con fecha pero el lugar resultó None/Vacio
        sin_lugar = df_artists2_cleaned[df_artists2_cleaned['lugar'].isna()]
        for _, row in sin_lugar.iterrows():
            registrar_rechazo(
                nombre=row['title'], 
                loc="No detectado", 
                fecha=row['date'], 
                motivo="Se descarta por falta de lugar (lugar es None después de limpiar_lugar)", 
                linea="620",
                fuente='Ticketek',
                href=row['href']
            )

        # --- PASO 3: El descarte (Dropna) ---
        # Eliminamos filas que no tengan Fecha (vienen de la 239) o Lugar (de la 244)
        df_artists2_cleaned = df_artists2_cleaned.dropna(subset=['date', 'lugar'])
        
        df_final = reordenar_y_agregar_columnas(df_artists2_cleaned.copy())
        df_final['finaliza'] = df_final['date']
        df_final['fecha de carga'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        df_final['price_avg'] = df_final['price_avg'].astype(str).str.replace("'", "", regex=False).astype(float)
        df_final, metricas_tkt = aplicar_clasificador(
            df=df_final,
            col_nombre='title',
            col_lugar='lugar',
            col_tipo_evento='tipo de evento',
            col_confianza='confianza_clasificacion'
            )
        log(f"🤖 Ticketek — Predicciones: {metricas_tkt['predicciones']} | Confianza promedio: {metricas_tkt['confianza_promedio']}")

        subir_a_google_sheets(df_final,'Ticketek historico (Auto)','Hoja 1')
        reporte["estado"] = "Exitoso"
        reporte["filas_procesadas"] = len(df_final)
        print(f"⚠️ Se registraron {len(df_con_errores)} fallos de carga en la auditoría.")
        if not df_rechazados.empty:
            subir_a_google_sheets(df_rechazados, 'Rechazados', 'Eventos')
            print("Rechazados Ticketek subidos exitosamente")
    except Exception as e:
        reporte["estado"] = "Fallido"
        reporte["error"] = str(e)
        log(f"❌ Error en Ticketek: {e}")
    finally:
        if driver:
            driver.quit()
        reporte["fin"] = datetime.now().strftime('%H:%M:%S')
        return reporte
log('TICKETEK')
#ejecutar_scraper_ticketek()

###########################################################################
################### EDEN ##################################################
###########################################################################
import pandas as pd
import time
import re
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- FUNCIONES DE APOYO PARA EDEN ---
df_final=[]
def extraer_promedio_precios_formato2(soup):
    precios_totales = []
    sector_divs = soup.find_all('div', class_='item sectorOption animated fadeInUp')
    for sector_div in sector_divs:
        precio_span = sector_div.find('span')
        if precio_span:
            precio_texto = precio_span.text.replace('Desde $', '').replace('.', '').replace(',', '.').strip()
            match = re.match(r'(\d+\.?\d*)\s*\+\s*.*?\$?\s*(\d+\.?\d*)', precio_texto)
            if match:
                precio_total = float(match.group(1)) + float(match.group(2))
                precios_totales.append(precio_total)
    return sum(precios_totales) / len(precios_totales) if precios_totales else None

def extraer_promedio_precios(soup):
    precios_f1 = []
    for sector in soup.find_all('div', class_='festival-shows sectors'):
        for price_div in sector.find_all('div', class_='additional-price'):
            try:
                precios_f1.append(float(price_div.text.replace('$', '').replace('.', '').replace(',', '.').strip()))
            except: pass
    
    prom_f1 = sum(precios_f1) / len(precios_f1) if precios_f1 else None
    prom_f2 = extraer_promedio_precios_formato2(soup)
    return prom_f2 if prom_f2 is not None else prom_f1


def lugar_excluido_eden(texto):
    """Detecta si el texto del lugar corresponde a una ubicación que debe rechazarse."""
    if not texto or not isinstance(texto, str):
        return False
    texto_normalizado = texto.lower()
    texto_normalizado = texto_normalizado.replace('í', 'i').replace('ó', 'o').replace('á', 'a').replace('é', 'e').replace('ú', 'u')
    return 'rio cuarto' in texto_normalizado or 'oncativo' in texto_normalizado


def normalizar_fecha_complejo(fecha_str):
    """Normaliza una cadena de fecha y hora con múltiples formatos, incluyendo el estándar de Edén."""
    if not fecha_str: return []
    fechas_normalizadas = []
    año_actual = pd.Timestamp.now().year
    
    # 1. Limpieza y diccionarios (Añadido 'maio/junho' y abreviaturas por seguridad)
    fecha_str = re.sub(r'<.*?>', '', str(fecha_str)) # Elimina tags como <screens.event...>
    
    dias_semana_esp = {
        'Lunes': 'Monday', 'Martes': 'Tuesday', 'Miércoles': 'Wednesday',
        'Miercoles': 'Wednesday', 'Jueves': 'Thursday', 'Viernes': 'Friday',
        'Sábado': 'Saturday', 'Sabado': 'Saturday', 'Domingo': 'Sunday'
    }
    meses_esp = {
        'Enero': 'January', 'Febrero': 'February', 'Marzo': 'March',
        'Abril': 'April', 'Mayo': 'May', 'Maio': 'May', 'Junio': 'June', 'Junho': 'June',
        'Julio': 'July', 'Agosto': 'August', 'Septiembre': 'September', 'Setiembre': 'September',
        'Octubre': 'October', 'Noviembre': 'November', 'Diciembre': 'December',
        'Ene': 'January', 'Feb': 'February', 'Mar': 'March', 'Abr': 'April',
        'May': 'May', 'Jun': 'June', 'Jul': 'July', 'Ago': 'August',
        'Sep': 'September', 'Oct': 'October', 'Nov': 'November', 'Dic': 'December'
    }

    # Reemplazar nombres de días y meses al inglés (Case insensitive para mayor seguridad)
    for esp, eng in dias_semana_esp.items():
        fecha_str = re.sub(r'\b' + esp + r'\b', eng, fecha_str, flags=re.IGNORECASE)
    for esp, eng in meses_esp.items():
        fecha_str = re.sub(r'\b' + esp + r'\b', eng, fecha_str, flags=re.IGNORECASE)

    # --- NUEVO CASO EDÉN: Formato multilínea o con "a las" y año explícito ---
    # Esto soluciona lo visto en el debug: "22 de abril de 2026 a las 21:00"
    lineas = fecha_str.split('\n')
    for linea in lineas:
        linea = linea.strip()
        if not linea or 'Próximas' in linea: continue

        # Patrón: "22 de April de 2026 a las 21:00"
        match_eden = re.search(r'(\d+)\s+de\s+(\w+)\s+de\s+(\d{4})\s+a\s+las\s+(\d+[:.]\d+)', linea, re.IGNORECASE)
        if match_eden:
            dia, mes, año, hora = match_eden.groups()
            hora = hora.replace('.', ':')
            try:
                fechas_normalizadas.append(pd.to_datetime(f'{dia} {mes} {año} {hora}', format='%d %B %Y %H:%M'))
                continue 
            except: pass

    # Si ya capturamos fechas por el formato Edén, las devolvemos para no duplicar procesos
    if fechas_normalizadas: return fechas_normalizadas

    # --- TUS REDUNDANCIAS ORIGINALES ---
    
    # Caso de rangos de fechas (ej: Viernes 23 y Sábado 24 de Mayo 16hs)
    match_rango = re.search(r'(\w+) (\d+) y (\w+) (\d+) de (\w+) (\d+)(?:hs|)', fecha_str)
    if match_rango:
        dia1_sem, dia1_num, dia2_sem, dia2_num, mes, hora = match_rango.groups()
        try:
            fecha1_dt = pd.to_datetime(f'{dia1_num} {mes} {año_actual} {hora[:-2]}:00', format='%d %B %Y %H:%M')
            fecha2_dt = pd.to_datetime(f'{dia2_num} {mes} {año_actual} {hora[:-2]}:00', format='%d %B %Y %H:%M')
            fechas_normalizadas.extend([fecha1_dt, fecha2_dt])
            return fechas_normalizadas
        except: pass

    # Caso de rangos de fechas con tres días
    match_rango_tres = re.search(r'(\w+) (\d+), (\w+) (\d+) y (\w+) (\d+) de (\w+)', fecha_str)
    if match_rango_tres:
        _, d1, _, d2, _, d3, mes = match_rango_tres.groups()
        try:
            fechas_normalizadas.extend([
                pd.to_datetime(f'{d1} {mes} {año_actual}', format='%d %B %Y'),
                pd.to_datetime(f'{d2} {mes} {año_actual}', format='%d %B %Y'),
                pd.to_datetime(f'{d3} {mes} {año_actual}', format='%d %B %Y')
            ])
            return fechas_normalizadas
        except: pass

    # Caso de fecha y hora simple (tu lógica original de hs y . )
    match_simple_hs = re.search(r'(?:\w+ )?(\d+) de (\w+) (\d+(?:\.\d+)?)hs', fecha_str)
    if match_simple_hs:
        dia, mes, hora_str = match_simple_hs.groups()
        parts = hora_str.split('.')
        h = int(parts[0])
        m = int(parts[1]) * 60 // 100 if len(parts) > 1 else 0
        try:
            fechas_normalizadas.append(pd.to_datetime(f'{dia} {mes} {año_actual} {h}:{m}:00', format='%d %B %Y %H:%M:%S'))
            return fechas_normalizadas
        except: pass

    # Redundancia para "22 de April 21hs" o "22 de April 21:00"
    elif match_simple := re.search(r'(?:\w+ )?(\d+) de (\w+) (\d+)hs', fecha_str):
        dia, mes, hora = match_simple.groups()
        try:
            fechas_normalizadas.append(pd.to_datetime(f'{dia} {mes} {año_actual} {hora}:00:00', format='%d %B %Y %H:%M:%S'))
            return fechas_normalizadas
        except: pass

    elif match_simple := re.search(r'(?:\w+ )?(\d+) de (\w+) (\d+:\d+)(?:hs|)', fecha_str):
        dia, mes, hora = match_simple.groups()
        try:
            fechas_normalizadas.append(pd.to_datetime(f'{dia} {mes} {año_actual} {hora}', format='%d %B %Y %H:%M'))
            return fechas_normalizadas
        except: pass

    elif match_simple := re.search(r'(?:\w+ )?(\d+) de (\w+)', fecha_str):
        dia, mes = match_simple.groups()
        try:
            fechas_normalizadas.append(pd.to_datetime(f'{dia} {mes} {año_actual}', format='%d %B %Y'))
            return fechas_normalizadas
        except: pass

    return fechas_normalizadas

def procesar_dataframe_complejo(df, columna_fecha='Fecha'):
    """Procesa el DataFrame para normalizar la columna de fecha con la función compleja."""
    filas_nuevas = []
    for index, row in df.iterrows():
        fecha_str = str(row[columna_fecha]).strip()
        fechas_normalizadas = normalizar_fecha_complejo(fecha_str)
        for fecha in fechas_normalizadas:
            nueva_fila = row.copy()
            nueva_fila[columna_fecha] = fecha
            filas_nuevas.append(nueva_fila)

    if not filas_nuevas:
        return pd.DataFrame()

    df_normalizado = pd.DataFrame(filas_nuevas)
    
    if columna_fecha in df_normalizado.columns:
        # 1. Forzamos la conversión a datetime (asegura que .dt funcione)
        fechas_dt = pd.to_datetime(df_normalizado[columna_fecha], errors='coerce')
        
        # 2. Convertimos a string formateado y lo guardamos en una variable limpia
        fechas_str = fechas_dt.dt.strftime('%Y-%m-%d %H:%M:%S').fillna("")
        
        # 3. Reemplazamos la columna entera forzando el tipo de dato final a object (string)
        df_normalizado[columna_fecha] = fechas_str.values
        df_normalizado[columna_fecha] = df_normalizado[columna_fecha].astype(str)
    
    return df_normalizado

def ejecutar_scraper_eden():
    from google.oauth2 import service_account
    import json
    import os
    import pandas as pd
    import time
    from datetime import datetime
    from bs4 import BeautifulSoup
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    # 1. INICIALIZACIÓN DE VARIABLES
    driver = None
    df_final = pd.DataFrame()
    df_norm = pd.DataFrame()
    data_df = pd.DataFrame()
    fallos_fecha = set()
    df_rechazados = pd.DataFrame(columns=['Nombre', 'Locación', 'Fecha', 'Motivo', 'Linea', 'Fuente', 'Link','Fecha Scrp'])

    reporte = {
        "nombre": "Eden Entradas",
        "estado": "Pendiente",
        "filas_procesadas": 0,
        "error": None,
        "inicio": datetime.now().strftime('%H:%M:%S')
    }

    def registrar_rechazo(nombre, loc, fecha, motivo, linea, fuente, href, col_href="Link"):
        nonlocal df_rechazados
        nuevo = pd.DataFrame([{
            'Nombre': nombre, 'Locación': loc, 'Fecha': fecha,
            'Motivo': motivo, 'Linea': str(linea), 'Fuente': fuente,
            col_href: href, 'Fecha Scrp': datetime.now().strftime('%Y-%m-%d')
        }])
        df_rechazados = pd.concat([df_rechazados, nuevo], ignore_index=True)

    try:
        print("Eden: iniciar ejecutar_scraper_eden")
        driver = iniciar_driver()
        print("Eden: iniciar_driver completado")
        BASE_URL = "https://www.edenentradas.ar"
        print(f"Eden: navegando a {BASE_URL}/")
        driver.get(BASE_URL + "/")
        print("Eden: Driver iniciado y página principal solicitada")
        time.sleep(10)

        # 2. Hacer scroll para cargar elementos dinámicamente
        print("Eden: iniciando scroll para cargar elementos dinámicamente...")
        from selenium.webdriver.common.keys import Keys
        
        ultima_altura = driver.execute_script("return document.body.scrollHeight")
        scrolls = 0
        max_scrolls = 10
        print(f'Ultima altura {ultima_altura}')
        while scrolls < max_scrolls:
            # Scroll hacia abajo
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            scrolls += 1
            
            # Calcular nueva altura después del scroll
            nueva_altura = driver.execute_script("return document.body.scrollHeight")
            print(f'Nueva Altura: {nueva_altura}')
            
            # Si no hay más contenido para cargar, salir del loop
            if nueva_altura == ultima_altura:
                print(f"Eden: no hay más contenido después de {scrolls} scrolls")
                break
            
            ultima_altura = nueva_altura
            print(f"Eden: scroll {scrolls}/{max_scrolls} completado")

        # 2. Scrapeo de lista principal
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        print(soup)
        eventos_html = soup.find_all('div', class_='grid_element')
        print(f"Eden: elementos 'grid_element' encontrados: {len(eventos_html)}")
        
        if not eventos_html:
            print("Eden: no se detectaron eventos en la página principal")
            registrar_rechazo("Página Principal", "N/A", "N/A", "No se detectaron elementos grid_element", "46", "Eden", BASE_URL)
            return reporte

        data = []
        print("EDEN: Iniciando Loopeo de grilla...")
        for evento in eventos_html:
            data.append({
                'Nombre': evento.find('div', class_='item_title').text.strip() if evento.find('div', class_='item_title') else None,
                'Locación': evento.find('strong').text.strip() if evento.find('strong') else None,
                'Fecha': evento.find('span').text.strip() if evento.find('span') else None,
                'href': evento.find('a')['href'] if evento.find('a') else None
            })

        data_df = pd.DataFrame(data)
        data_df = data_df.dropna(subset=['Locación', 'href']).drop_duplicates(subset=['href']).reset_index(drop=True)
        log(f"📊 Eden: {len(data_df)} eventos únicos detectados")
        print(f"Eden: Iniciando procesamiento de {len(data_df)} eventos")

        # 3. Recorrido de detalles
        for index, row in data_df.iterrows():
            full_href = f"{BASE_URL}{row['href'].replace('..', '')}"
            print(f"Eden: Procesando evento {index+1}/{len(data_df)}: {row['Nombre']} - {full_href}")
            try:
                driver.get(full_href)
                time.sleep(3)
                soup_det = BeautifulSoup(driver.page_source, 'html.parser')
                
                cols = soup_det.find_all('div', class_='col-xs-7')
                raw_text = " | ".join([e.text.strip() for e in cols])
                
                data_df.loc[index, 'filtro_ciudad'] = raw_text
                data_df.loc[index, 'Fecha'] = raw_text 

                if not any(x in raw_text for x in ['Córdoba', 'Cordoba', 'Cba', 'CBA']):
                    continue
                
                lugar_limpio = raw_text.split('|')[-1].split('Cordoba')[0].strip() if '|' in raw_text else row['Locación']
                data_df.loc[index, 'Locación'] = lugar_limpio

                try:
                    btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "div.picker-full button.next, #buyButton")))
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(2)
                    data_df.loc[index, 'precio_promedio'] = extraer_promedio_precios(BeautifulSoup(driver.page_source, 'html.parser'))
                    print(f"Eden: precio promedio extraído para evento {index+1}")
                except Exception as e:
                    data_df.loc[index, 'precio_promedio'] = None
                    print(f"Eden: no se pudo extraer precio promedio para evento {index+1} ({row['Nombre']}): {e}")

            except Exception as e:
                registrar_rechazo(row['Nombre'], row['Locación'], row['Fecha'], f"Error detalle: {str(e)}", "102", "Eden", full_href)
                print(f"Eden: excepción detalle evento {index+1} ({row['Nombre']}): {e}")
                continue

        # 4. Filtrado y Normalización
        print("Eden: aplicando filtro de ciudad Córdoba a los eventos")
        df_filtrado = data_df[data_df['filtro_ciudad'].str.contains('Córdoba|Cordoba|Cba', case=False, na=False)].copy()
        df_filtrado = df_filtrado[~df_filtrado.apply(lambda row: lugar_excluido_eden(row['Locación']) or lugar_excluido_eden(row.get('filtro_ciudad', '')), axis=1)].copy()
        print(f"Eden: eventos después de filtro de ciudad: {len(df_filtrado)}")
        
        if not df_filtrado.empty:
            print(f"⚙️ Normalizando {len(df_filtrado)} eventos...")
            # === DEBUG INICIO ===
            print('Debug previo a procesar_datafreme_complejo')
            print("\n📋 df_norm.dtypes:")
            print(df_norm.dtypes)
            
            print("\n🔍 Tipos reales por columna (primeras 3 filas):")
            for col in df_norm.columns:
                print(f"\n  Columna: '{col}'")
                for i, val in enumerate(df_norm[col].head(3)):
                    print(f"    [{i}] tipo={type(val).__name__!r}  valor={repr(val)}")
            
            print("\n🧪 Buscando Timestamps sueltos en columnas object...")
            for col in df_norm.select_dtypes(include='object').columns:
                ts_mask = df_norm[col].apply(lambda x: isinstance(x, pd.Timestamp))
                if ts_mask.any():
                    print(f"  ⚠️  Columna '{col}' tiene {ts_mask.sum()} Timestamp(s):")
                    print(df_norm.loc[ts_mask, col].head(3))
            
            print("\n✅ Fin diagnóstico df_norm")
            # === DEBUG FIN ===
            df_norm = procesar_dataframe_complejo(df_filtrado, columna_fecha='Fecha')
            print(f"Eden: df_norm procesado con {len(df_norm)} filas")
            
            # === DEBUG INICIO ===
            print("\n📋 df_norm.dtypes:")
            print(df_norm.dtypes)
            
            print("\n🔍 Tipos reales por columna (primeras 3 filas):")
            for col in df_norm.columns:
                print(f"\n  Columna: '{col}'")
                for i, val in enumerate(df_norm[col].head(3)):
                    print(f"    [{i}] tipo={type(val).__name__!r}  valor={repr(val)}")
            
            print("\n🧪 Buscando Timestamps sueltos en columnas object...")
            for col in df_norm.select_dtypes(include='object').columns:
                ts_mask = df_norm[col].apply(lambda x: isinstance(x, pd.Timestamp))
                if ts_mask.any():
                    print(f"  ⚠️  Columna '{col}' tiene {ts_mask.sum()} Timestamp(s):")
                    print(df_norm.loc[ts_mask, col].head(3))
            
            print("\n✅ Fin diagnóstico df_norm")
            # === DEBUG FIN ===
            
            # Auditoría de fallos
            eventos_antes = set(df_filtrado['Nombre'])
            eventos_despues = set(df_norm['Nombre']) if not df_norm.empty else set()
            fallos_fecha = eventos_antes - eventos_despues
            
            for nombre in fallos_fecha:
                orig = df_filtrado[df_filtrado['Nombre'] == nombre].iloc[0]
                registrar_rechazo(nombre, orig['Locación'], orig['Fecha'], "Fallo Regex Normalizador", "121", "Eden", orig['href'])

        # 5. Formateo Final
        if not df_norm.empty:
            print(f"Eden: df_norm contiene {len(df_norm)} filas y se empieza a formatear a df_final")
            df_norm = df_norm.apply(lambda col: 
                col.dt.strftime('%Y-%m-%d %H:%M:%S') 
                if pd.api.types.is_datetime64_any_dtype(col) 
                else col.astype(str)
            )

            df_final = pd.DataFrame({
                'Eventos': df_norm['Nombre'],
                'Lugar': df_norm['Locación'],
                'Comienza': df_norm['Fecha'],
                'Finaliza': df_norm['Fecha'],
                'Tipo de evento': None,
                'Detalle': None,
                'Alcance': None,
                'Costo de entrada': df_norm.get('precio_promedio', ""),
                'Fuente': 'Eden Entradas',
                'Origen': df_norm['href'].apply(lambda x: f"https://www.edenentradas.ar{str(x).replace('..', '')}"),
                'fecha de carga': datetime.today().strftime('%Y-%m-%d %H:%M:%S') 
            })
            print(f"Eden: df_final creado con {len(df_final)} filas")
            # BLINDAJE FINAL: Convertimos todo el DF a string para evitar errores en el Clasificador
            df_final = df_final.astype(str).replace('None', '').replace('nan', '')
            df_final = df_final.drop_duplicates(subset=['Origen'])
            
            # Clasificador
            df_final, metricas = aplicar_clasificador(df_final, 'Eventos', 'Lugar', 'Tipo de evento', 'confianza_clasificacion')
            
            log(f"🤖 Eden — Predicciones: {metricas['predicciones']} | Confianza: {metricas['confianza_promedio']}")
            print(f"Eden: Subiendo {len(df_final)} eventos a Sheets")
            subir_a_google_sheets(df_final, 'Eden historico (Auto)', 'Hoja 1')
            
            reporte["filas_procesadas"] = len(df_final)
            reporte["estado"] = "Exitoso"
        else:
            print("❌ df_norm quedó vacío tras el proceso.")
            reporte["estado"] = "Advertencia: Sin datos normalizados"

        # 6. Subida de Rechazados (también forzando strings)
        if not df_rechazados.empty:
            subir_a_google_sheets(df_rechazados.astype(str), 'Rechazados', 'Eventos')

    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN EDÉN: {type(e).__name__}: {e}")
        reporte["estado"] = "Fallido"
        reporte["error"] = str(e)
        
    finally:
        if driver:
            print("Eden: cerrando driver")
            driver.quit()
        print(f"Eden: finalizar ejecutar_scraper_eden con estado {reporte['estado']}")
        return reporte
log('')
log('EDÉN')
#ejecutar_scraper_eden()

##################################################################################################################
####################################### EVENTBRITE ###############################################################
##################################################################################################################
import pandas as pd
import time
import re
import requests
import numpy as np
from datetime import datetime, timedelta
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- FUNCIONES DE APOYO ---

def limpiar_fecha_texto(fecha):
    """Limpia el texto de Eventbrite antes de procesarlo."""
    if not fecha or fecha == 'N/A': return "Formato desconocido"
    fecha = re.sub(r"\+.*", "", fecha).strip()
    return fecha

def convertir_fechas(fecha):
    if not fecha or fecha == "N/A": return "Formato desconocido"
    fecha_low = fecha.lower()
    ahora = datetime.now()
    
    try:
        # 1. HOY
        if "hoy" in fecha_low:
            match = re.search(r'(\d{1,2}:\d{2})', fecha_low)
            if match:
                hora, minuto = map(int, match.group(1).split(":"))
                return ahora.replace(hour=hora, minute=minuto, second=0, microsecond=0)
        
        # 2. MAÑANA
        elif "mañana" in fecha_low:
            match = re.search(r'(\d{1,2}:\d{2})', fecha_low)
            if match:
                hora, minuto = map(int, match.group(1).split(":"))
                tomorrow = ahora + timedelta(days=1)
                return tomorrow.replace(hour=hora, minute=minuto, second=0, microsecond=0)
        
        # 3. DÍA DE LA SEMANA
        dias = {"lunes":0, "martes":1, "miércoles":2, "jueves":3, "viernes":4, "sábado":5, "domingo":6}
        for nombre, cod in dias.items():
            if nombre in fecha_low:
                match = re.search(r'(\d{1,2}:\d{2})', fecha_low)
                if match:
                    hora, minuto = map(int, match.group(1).split(":"))
                    dias_adelante = (cod - ahora.weekday()) % 7
                    if dias_adelante == 0: dias_adelante = 7
                    target = ahora + timedelta(days=dias_adelante)
                    return target.replace(hour=hora, minute=minuto, second=0, microsecond=0)

        # 4. FECHA ESPECÍFICA (ej: "31 oct, 19:00")
        meses = {
            "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
            "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12
        }
        match_esp = re.search(r'(\d{1,2})\s([a-z]{3}).*?(\d{1,2}:\d{2})', fecha_low)
        if match_esp:
            dia = int(match_esp.group(1))
            mes_txt = match_esp.group(2)
            hora_str = match_esp.group(3)
            if mes_txt in meses:
                mes = meses[mes_txt]
                año = ahora.year
                if mes < ahora.month: año += 1
                h, m = map(int, hora_str.split(":"))
                return datetime(año, mes, dia, h, m)

        return fecha 
    except Exception as e:
        return "Error formato"

# --- FUNCIÓN PRINCIPAL ---

def ejecutar_scraper_eventbrite():
    driver = None
    reporte = {
        "nombre": "Eventbrite",
        "estado": "Pendiente",
        "filas_procesadas": 0,
        "error": None,
        "inicio": datetime.now().strftime('%H:%M:%S')
    }
    
    # --- CONFIGURACIÓN AUDITORÍA ---
    df_rechazados = pd.DataFrame(columns=['Nombre', 'Locación', 'Fecha', 'Motivo', 'Linea', 'Fuente', 'Link'])

    def registrar_rechazo(nombre, loc, fecha, motivo, linea, fuente, href):
        nonlocal df_rechazados
        nuevo = pd.DataFrame([{
            'Nombre': nombre, 'Locación': loc, 'Fecha': fecha,
            'Motivo': motivo, 'Linea': str(linea), 'Fuente': fuente,
            'Link': href
        }])
        df_rechazados = pd.concat([df_rechazados, nuevo], ignore_index=True)

    date_keywords = ['lun', 'mar', 'mié', 'jue', 'vie', 'sáb', 'dom', 'mañana', 'hoy', 'ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
    
    try:
        driver = iniciar_driver()
        base_url = 'https://www.eventbrite.com.ar/d/argentina--c%C3%B3rdoba/all-events/'
        event_data = []
        seen_links = set()

        for page in range(1, 6):
            print(f"📄 Eventbrite: Procesando página {page}...")
            driver.get(f'{base_url}?page={page}')
            
            try:
                # Esperamos a que aparezca el contenedor de las cards, no solo el h3
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'section.event-card-details'))
                )
                # Scroll lento para disparar el lazy loading de Eventbrite
                for _ in range(3):
                    driver.execute_script("window.scrollBy(0, 400);")
                    time.sleep(0.5)
            except Exception as e: 
                print(f"⚠️ No se detectaron cards en página {page}. Posible cambio de diseño o fin.")
                break

            events = driver.find_elements(By.CSS_SELECTOR, 'article, section.discover-horizontal-event-card, div[class*="Stack_root"]')
            
            for event in events:
                try:
                    # 1. Extracción Básica
                    try:
                        name_el = event.find_element(By.TAG_NAME, 'h3')
                        name = name_el.text.strip()
                        link = event.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    except:
                        continue

                    if not name or link in seen_links: 
                        continue
                    
                    # 2. Extracción de Fecha y Locación vía párrafos
                    paragraphs = event.find_elements(By.TAG_NAME, 'p')
                    date_info, location = 'N/A', 'N/A'

                    if paragraphs:
                        idx_fecha = -1
                        for i, p in enumerate(paragraphs):
                            txt = p.text.strip().lower()
                            if any(kw in txt for kw in date_keywords):
                                idx_fecha = i
                                break
                        
                        if idx_fecha != -1:
                            date_info = paragraphs[idx_fecha].text.strip()
                            if len(paragraphs) > idx_fecha + 1:
                                location = paragraphs[idx_fecha + 1].text.strip()
                        else:
                            location = paragraphs[0].text.strip()

                    # 3. Auditoría inicial: Datos incompletos
                    if date_info == 'N/A' or location == 'N/A':
                        registrar_rechazo(name, location, date_info, "Card con datos insuficientes (Fecha/Locación N/A)", "125", "Eventbrite", link)
                        continue

                    event_data.append({
                        'Nombre': name, 'Fecha': date_info, 'Locación': location,
                        'Precio': "Consultar", 'Origen': link
                    })
                    seen_links.add(link)
                except: 
                    continue

        # --- PROCESAMIENTO ---
        if not event_data:
            reporte["estado"] = "Primera página vacía. Reintentando."
            raise ValueError("No se encontraron datos en Eventbrite")

        df_crudo = pd.DataFrame(event_data)
        
        # 4. Auditoría: Filtrado de Locación (Hoteles MICE)
        keywords_locacion = ['quinto centenario', 'blas pascal', 'quorum', 'sheraton', 'holiday inn']
        mask_locacion = df_crudo['Locación'].str.lower().str.contains('|'.join(keywords_locacion), na=False)
        
        df_rechazados_loc = df_crudo[~mask_locacion]
        for _, row in df_rechazados_loc.iterrows():
            registrar_rechazo(row['Nombre'], row['Locación'], row['Fecha'], "Locación no coincide con Hoteles MICE", "150", "Eventbrite", row['Origen'])

        df_filtrado = df_crudo[mask_locacion].copy()

        # 5. Auditoría: Conversión de Fecha
        if not df_filtrado.empty:
            df_filtrado['Fecha Convertida'] = df_filtrado['Fecha'].apply(convertir_fechas)
            
            # Identificamos fallos (si devuelve string en lugar de datetime o "Error formato")
            mask_fecha_ok = df_filtrado['Fecha Convertida'].apply(lambda x: isinstance(x, datetime))
            
            df_rechazados_fecha = df_filtrado[~mask_fecha_ok]
            for _, row in df_rechazados_fecha.iterrows():
                registrar_rechazo(row['Nombre'], row['Locación'], row['Fecha'], f"Fallo en conversión de fecha: {row['Fecha']}", "165", "Eventbrite", row['Origen'])
            
            df_final_data = df_filtrado[mask_fecha_ok].copy()
            if 'tipo de evento' not in df_final_data.columns:
                df_final_data['tipo de evento'] = ''
            if 'confianza_clasificacion' not in df_final_data.columns:
                df_final_data['confianza_clasificacion'] = None

            df_final_data, metricas_eb = aplicar_clasificador(
                df=df_final_data,
                col_nombre='Nombre',
                col_lugar='Locación',
                col_tipo_evento='tipo de evento',
                col_confianza='confianza_clasificacion'
            )
            log(f"🤖 Eventbrite — Predicciones: {metricas_eb['predicciones']} | Confianza promedio: {metricas_eb['confianza_promedio']}")
            
            if not df_final_data.empty:
                df_final = pd.DataFrame({
                    'Nombre': df_final_data['Nombre'],
                    'Locación': df_final_data['Locación'],
                    'Fecha Convertida': df_final_data['Fecha Convertida'].astype(str),
                    'termina': "",
                    'tipo de evento': df_final_data['tipo de evento'].fillna(''),
                    'detalle': "",
                    'alcance': "",
                    'Precio': 0.0,
                    'fuente': 'eventbrite',
                    'Origen': df_final_data['Origen'],
                    'Fecha Scrp': datetime.today().strftime('%Y-%m-%d')
                })
                
                subir_a_google_sheets(df_final, 'base_h_scrp_eventbrite', 'Hoja 1')
                reporte["filas_procesadas"] = len(df_final)
                reporte["estado"] = "Exitoso"
            else:
                reporte["estado"] = "Exitoso (Sin eventos válidos tras filtros)"

        # --- SUBIDA FINAL DE AUDITORÍA ---
        if not df_rechazados.empty:
            # Subimos a la pestaña 'Eventbrite' del documento 'Rechazados'
            subir_a_google_sheets(df_rechazados, 'Rechazados', 'Eventos')
            print(f"✅ Auditoría Eventbrite: {len(df_rechazados)} registros subidos.")

    except Exception as e:
        print(f"❌ Error Crítico Eventbrite: {e}")
        reporte["estado"] = "Fallido"
        reporte["error"] = str(e)
        if driver:
            driver.quit()
        raise e
    finally:
        if driver:
            driver.quit()
        reporte["fin"] = datetime.now().strftime('%H:%M:%S')
    return reporte

intentos_maximos = 0
resultado_final = None
log('')
log('EVENTBRITE')
for i in range(1, intentos_maximos + 1):
    try:
        print(f"🚀 Iniciando Eventbrite - Intento {i} de {intentos_maximos}...")
        resultado_final = ejecutar_scraper_eventbrite()
        
        # Si llega aquí, es que funcionó (no hubo raise)
        print(f"✅ Intento {i} completado con éxito.")
        break 

    except Exception as e:
        print(f"❌ Error en intento {i}: {e}")
        
        # Guardamos un reporte provisional por si este es el último fallo
        resultado_final = {
            "nombre": "Eventbrite",
            "estado": "Fallido definitivamente",
            "error": str(e),
            "filas_procesadas": 0,
            "inicio": datetime.now().strftime('%H:%M:%S') # O la hora que prefieras
        }

        if i < intentos_maximos:
            print(f"⚠️ Reintentando en 10 segundos...")
            time.sleep(10)
        else:
            log("🛑 Fallo en eventbrite (Intentos agotados)")

# Ahora, pase lo que pase, resultado_final contiene el diccionario
#print(f"Estado final registrado: {resultado_final['estado']}")
# Aquí puedes usar resultado_final para subirlo a otro lado o mostrarlo
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
def enviar_log_smtp(cuerpo_log, lista_destinatarios):
    """Envía el log acumulado a múltiples correos usando SMTP (reemplaza Gmail API)."""
    try:
        # Configuración desde variables de entorno para seguridad
        remitente = "rmansilla@cordobaacelera.com.ar"  # El mail que generó la App Password
        password = os.environ.get('EMAIL_APP_PASSWORD')
        
        if not password:
            log("🔴 Error: No se encontró EMAIL_APP_PASSWORD en los secretos.")
            return
        else:
            log(f"🔑 Contraseña detectada. Largo: {len(password)} caracteres.")
            # ESTA LÍNEA DE CONTROL:
            log(f"¿Tiene espacios o saltos de línea?: {password.strip() != password}")
          

        # Iniciamos la conexión con el servidor SMTP de Gmail
        log("🔗 Conectando al servidor de correo...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Cifrado de seguridad
        server.login(remitente, password)

        for destinatario in lista_destinatarios:
            # Creamos el contenedor del mensaje
            message = MIMEMultipart()
            message['To'] = destinatario
            message['From'] = f"Scraper Automático <{remitente}>"
            message['Subject'] = "📊 REPORTE SCRP AGENDA"
            
            # Agregamos el cuerpo del log
            message.attach(MIMEText(cuerpo_log, 'plain'))

            # Envío del correo
            server.send_message(message)
            log(f"📧 Mail enviado a {destinatario}")

        # Cerramos la conexión después de enviar a todos
        server.quit()
        log("✅ Proceso de envío finalizado.")

    except Exception as e:
        log(f"🔴 Error al enviar mail vía SMTP: {e}")



# Llamamos a la función con la lista de correos

log('')
log('Ferias y Congresos')
def ejecutar_scraper_ferias_y_congresos():
    """
    Scraper integral para Ferias y Congresos con auditoría de rechazos
    y gestión de fechas de rango.
    """
    driver = None
    reporte = {
        "nombre": "Ferias y Congresos",
        "estado": "Pendiente",
        "filas_procesadas": 0,
        "error": None,
        "inicio": datetime.now().strftime('%H:%M:%S')
    }
    
    df_rechazados = pd.DataFrame(columns=['Nombre', 'Locación', 'Fecha', 'Motivo', 'Linea', 'Fuente', 'Link'])

    def registrar_rechazo(nombre, loc, fecha, motivo, linea, fuente, href):
        nonlocal df_rechazados
        nuevo = pd.DataFrame([{
            'Nombre': nombre, 
            'Locación': loc, 
            'Fecha': fecha,
            'Motivo': motivo, 
            'Linea': str(linea), 
            'Fuente': fuente,
            'Link': href
        }])
        df_rechazados = pd.concat([df_rechazados, nuevo], ignore_index=True)

    def parsear_rango_fechas(texto_fecha):
        meses = {
            'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
            'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
        }
        ahora = datetime.now()
        
        try:
            # Captura formatos: "07 al 09 de Febrero" o "19 Enero al 06 de Abril"
            match = re.search(r'(\d+)\s*([a-zA-Z]+)?\s*al\s*(\d+)\s*de?\s*([a-zA-Z]+)', texto_fecha)
            if not match: return None, None
            
            d1, m1_str, d2, m2_str = match.groups()
            m2 = meses[m2_str.capitalize()]
            m1 = meses[m1_str.capitalize()] if m1_str else m2
            
            year = ahora.year
            # Creamos fechas tentativas para este año
            f_ini_temp = ahora.replace(year=year, month=m1, day=int(d1))
            f_fin_temp = ahora.replace(year=year, month=m2, day=int(d2))
            
            # Si el evento terminó antes de hoy, lo pasamos al año siguiente
            if f_fin_temp < ahora:
                year += 1
                f_ini_temp = f_ini_temp.replace(year=year)
                f_fin_temp = f_fin_temp.replace(year=year)
            
            return f_ini_temp.strftime('%Y-%m-%d'), f_fin_temp.strftime('%Y-%m-%d')
        except:
            return None, None

    try:
        driver = iniciar_driver() 
        url_fuente = "https://www.feriasycongresos.com/calendario-de-eventos?busqueda=C%C3%B3rdoba"
        driver.get(url_fuente)
        
        # Espera dinámica (Vue.js)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "mod-evento")))
        time.sleep(5)

        bloques = driver.find_elements(By.CLASS_NAME, "mod-evento")
        raw_data = []

        for bloque in bloques:
            try:
                nombre = bloque.find_element(By.TAG_NAME, "h1").text
                fecha_raw = bloque.find_element(By.CSS_SELECTOR, ".txt2 .bold").text
                
                try:
                    recinto_raw = bloque.find_element(By.XPATH, ".//span[contains(text(), 'Recinto:')]").find_element(By.XPATH, "parent::*").text
                except:
                    recinto_raw = "No detectado"

                # 1. Filtro Córdoba Capital (Requisito Punto 1)
                lugares_validos = ["Capital, Córdoba", "Arguello, Córdoba"]
                if not any(lugar in recinto_raw for lugar in lugares_validos):
                    registrar_rechazo(
                        nombre=nombre, loc=recinto_raw, fecha=fecha_raw,
                        motivo="El evento no se encuentra en córdoba capital",
                        linea="1367", fuente="Ferias y Congresos", href=url_fuente
                    )
                    print(f'El evento {nombre} no se encuentra en córdoba capital')
                    continue

                # 2. Procesamiento de Rango de Fechas (Requisito Punto 3)
                f_ini, f_fin = parsear_rango_fechas(fecha_raw)
                
                if not f_ini:
                    registrar_rechazo(
                        nombre=nombre, loc=recinto_raw, fecha=fecha_raw,
                        motivo="FALLO DE EXTRACCIÓN de fecha (formato no reconocido)",
                        linea="1381", fuente="Ferias y Congresos", href=url_fuente
                    )
                    continue

                # 3. Construcción de fila válida
                raw_data.append({
                'Eventos': nombre,          # Cambiado a 'Eventos' para consistencia con otros scrapers
                'Lugar': recinto_raw.replace("Recinto:", "").strip(),
                'Comienza': f_ini,          # Usamos nombres estándar para evitar problemas de duplicados
                'Finaliza': f_fin,
                'Tipo de evento': None,
                'Detalle': '',
                'Alcance': '',
                'Costo de entrada': '',
                'Fuente': 'Ferias y Congresos',
                'Origen': url_fuente        # Usamos 'Origen' como ID único
                })
            except Exception:
                continue

        df_final = pd.DataFrame(raw_data)
        # --- 4. Formateo y Orden Final ---
        if raw_data:
            print('RAW DATA ENCONTRADA')
            df_final = pd.DataFrame(raw_data)
            print(f"Longitud df_final post raw_data:{len(df_final)}")
            
            # Aseguramos el orden exacto de las columnas antes de enviar
            columnas_ordenadas = [
                'Eventos', 'Lugar', 'Comienza', 'Finaliza', 
                'Tipo de evento', 'Detalle', 'Alcance', 
                'Costo de entrada', 'Fuente', 'Origen'
            ]
            
            # Reindexamos para asegurar el orden y agregamos la fecha de carga
            df_final = df_final[columnas_ordenadas]
            print(f"Longitud df_final post columnas ordenadas{len(df_final)}")
            df_final['fecha de carga'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

            df_final, metricas_fc = aplicar_clasificador(
                df=df_final,
                col_nombre='Eventos',
                col_lugar='Lugar',
                col_tipo_evento='Tipo de evento',
                col_confianza='confianza_clasificacion'
            )
            log(f"🤖 Ferias y Congresos — Predicciones: {metricas_fc['predicciones']} | Confianza promedio: {metricas_fc['confianza_promedio']}")
            
            # Subida a Sheets
            subir_a_google_sheets(df_final, 'Ferias y Congresos (Auto)', 'Hoja 1')
            
            reporte["estado"] = "Exitoso"
            reporte["filas_procesadas"] = len(df_final)

        # Subida de auditoría si hay rechazados
        if not df_rechazados.empty:
            subir_a_google_sheets(df_rechazados, 'Rechazados', 'Eventos')
            print(f"✅ Auditoría: {len(df_rechazados)} eventos rechazados subidos.")

    except Exception as e:
        reporte["estado"] = "Fallido"
        reporte["error"] = str(e)
    finally:
        if driver:
            driver.quit()
        reporte["fin"] = datetime.now().strftime('%H:%M:%S')
        return reporte

# Ejecución
print("Iniciando Ferias y Congresos...")
#ejecutar_scraper_ferias_y_congresos()


log('')
log('Secretaría de Turismo Municipal')
import time
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def ejecutar_scraper_turismo_cba():
    """
    Scraper para Agencia Córdoba Turismo adaptado al formato estándar.
    """
    driver = None
    reporte = {
        "nombre": "Agencia Turismo Cba",
        "estado": "Pendiente",
        "filas_procesadas": 0,
        "error": None,
        "inicio": datetime.now().strftime('%H:%M:%S')
    }
    
    # DataFrame para auditoría de descartes
    df_rechazados = pd.DataFrame(columns=['Nombre', 'Locación', 'Fecha', 'Motivo', 'Linea', 'Fuente', 'Link'])

    def registrar_rechazo(nombre, loc, fecha, motivo, linea, fuente, href):
        nonlocal df_rechazados
        nuevo = pd.DataFrame([{
            'Nombre': nombre, 'Locación': loc, 'Fecha': fecha,
            'Motivo': motivo, 'Linea': str(linea), 'Fuente': fuente, 'Link': href
        }])
        df_rechazados = pd.concat([df_rechazados, nuevo], ignore_index=True)

    def formatear_fecha(fecha_str):
        try:
            if not fecha_str or "N/A" in str(fecha_str): return None
            fecha_str = " ".join(fecha_str.split())
            dt = datetime.strptime(fecha_str, "%d/%m/%Y %H:%M")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return None

    try:
        driver = iniciar_driver() # Usa tu función global
        url_agenda = "https://turismo.cordoba.gob.ar/agenda/agenda-turistica"
        exclusiones = ["edenentradas", "ticketek", "quality","autoentrada"]
        
        driver.get(url_agenda)
        print('Turismo Driver iniciado')
        print(f"🚀 {reporte['nombre']}: Cargando y expandiendo agenda...")
        clicks=0

        # 1. Expandir contenido "Cargar Más"
        while True:
            try:
                boton = WebDriverWait(driver, 8).until(
                    EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Cargar Más')]"))
                )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", boton)
                time.sleep(1)
                driver.execute_script("arguments[0].click();", boton)
                clicks += 1
                print(f"🔄 Click #{clicks} en 'Cargar Más' — esperando contenido...") 
                time.sleep(3)
            except:
                break # No hay más botón

        # 2. Parsear contenido
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        cards = soup.find_all('div', class_='card')
        eventos_lista = []
        cantidad_cards=1
        for card in cards:
            try:
                # --- Link y Filtros ---
                link_tag = card.find('a', href=True)
                fuente_link = link_tag['href'].lower() if link_tag else url_agenda
                
                # --- Nombre y Locación ---
                nombre = card.find('h4', class_='card-title').get_text(strip=True) if card.find('h4') else "Sin Nombre"
                locacion = card.find('p', class_='lugar').get_text(strip=True) if card.find('p', class_='lugar') else ""

                print(f"\n  [{cantidad_cards}/{len(cards)}] 🎭 Procesando: '{nombre}'")
                print(f"            📍 Lugar : {locacion or '(sin lugar)'}")
                print(f"            🔗 Link  : {fuente_link}")
                cantidad_cards=cantidad_cards + 1
                
                # Filtrado por plataformas ya cubiertas
                if any(p in fuente_link for p in exclusiones):
                    registrar_rechazo(nombre, locacion, "N/A", f"Exclusión: Plataforma externa ({fuente_link})", "67", "Turismo Cba", fuente_link)
                    continue

                # --- Fechas ---
                fechas_p = card.find_all('p', class_='fs-4')
                inicio_raw = fechas_p[0].get_text(" ", strip=True).replace("hs", "").strip() if len(fechas_p) > 0 else ""
                fin_raw = fechas_p[1].get_text(" ", strip=True).replace("hs", "").strip() if len(fechas_p) > 1 else ""
                print(f"            📅 Inicio raw : '{inicio_raw}'")
                print(f"            📅 Fin raw    : '{fin_raw}'")
                
                fecha_inicio = formatear_fecha(inicio_raw)
                fecha_fin = formatear_fecha(fin_raw)

                if not fecha_inicio:
                    registrar_rechazo(nombre, locacion, inicio_raw, "Error de formato de fecha o fecha vacía", "78", "Turismo Cba", fuente_link)
                    continue

                # --- Precio ---
                footer_txt = card.find('div', class_='footer').get_text(strip=True) if card.find('div', class_='footer') else ""
                precio = "0" if "Gratuito" in footer_txt else footer_txt.replace("Precio", "").replace("$", "").strip()

                # --- Append al formato final ---
                eventos_lista.append({
                    "Eventos": nombre,
                    "Lugar": locacion,
                    "Comienza": fecha_inicio,
                    "Finaliza": fecha_fin if fecha_fin else fecha_inicio,
                    "Tipo de evento": "",
                    "Detalle": "",
                    "Alcance": "",
                    "Costo de entrada": precio,
                    "Fuente": "Agencia Turismo Cba",
                    "Origen": fuente_link,
                    "Fecha Scrp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

            except Exception as e:
                continue

        # 3. Procesamiento final y subida
        df_final = pd.DataFrame(eventos_lista)

        df_final, metricas_fc = aplicar_clasificador(
            df=df_final,
            col_nombre='Eventos',
            col_lugar='Lugar',
            col_tipo_evento='Tipo de evento',
            col_confianza='confianza_clasificacion'
        )
        log(f"🤖 Ferias y Congresos — Predicciones: {metricas_fc['predicciones']} | Confianza promedio: {metricas_fc['confianza_promedio']}")
                
        if not df_final.empty:
            # Subir a Google Sheets (usando tu función global)
            # Nota: Asegúrate de que el nombre del archivo en Sheets sea el correcto
            subir_a_google_sheets(df_final, 'Turismo CBA (Auto)', 'Hoja 1')
            
            reporte["estado"] = "Exitoso"
            reporte["filas_procesadas"] = len(df_final)
        else:
            reporte["estado"] = "Sin datos"

        # Subir rechazados si existen
        if not df_rechazados.empty:
            subir_a_google_sheets(df_rechazados, 'Rechazados', 'Eventos')

    except Exception as e:
        reporte["estado"] = "Fallido"
        reporte["error"] = str(e)
        print(f"❌ Error en Agencia Turismo Cba: {e}")
    
    finally:
        if driver:
            driver.quit()
        reporte["fin"] = datetime.now().strftime('%H:%M:%S')
        return reporte

#ejecutar_scraper_turismo_cba()
def ejecutar_scraper_autoentrada():
    import pandas as pd
    import time
    import re
    import unicodedata
    from datetime import datetime
    from bs4 import BeautifulSoup
    from selenium.webdriver.common.by import By
    # --- VARIABLES ---
    driver = None
    df_final = pd.DataFrame()
    df_rechazados = pd.DataFrame(columns=['Nombre', 'Locación', 'Fecha', 'Motivo', 'Linea', 'Fuente', 'Link', 'Fecha Scrp'])
    reporte = {
        "nombre": "Autoentrada",
        "estado": "Pendiente",
        "filas_procesadas": 0,
        "error": None,
        "inicio": datetime.now().strftime('%H:%M:%S')
    }
    def registrar_rechazo(nombre, loc, fecha, motivo, linea, href):
        nonlocal df_rechazados
        nuevo = pd.DataFrame([{
            'Nombre': nombre, 'Locación': loc, 'Fecha': fecha,
            'Motivo': motivo, 'Linea': str(linea), 'Fuente': 'Autoentrada',
            'Link': href, 'Fecha Scrp': datetime.now().strftime('%Y-%m-%d')
        }])
        df_rechazados = pd.concat([df_rechazados, nuevo], ignore_index=True)
    def normalizar_texto(texto):
        if not texto: return ""
        texto = texto.lower()
        texto = ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
        return texto.strip()
    def extraer_componentes(t, mes_respaldo, anio_respaldo):
        meses_map = {
            "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
            "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
            "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"
        }
        partes = re.findall(r'[a-z]+|\d+', t)
        numeros = [p for p in partes if p.isdigit()]
        palabras = [p for p in partes if not p.isdigit()]
        dia = "01"
        for n in numeros:
            if len(n) <= 2:
                dia = n.zfill(2)
                break
        mes = mes_respaldo
        for p in palabras:
            if p in meses_map:
                mes = meses_map[p]
                break
        anio = anio_respaldo
        for n in numeros:
            if len(n) == 4:
                anio = n
                break
        return dia, mes, anio
    def procesar_rango_fechas(fecha_full):
        hoy = datetime.now()
        mes_hoy = str(hoy.month).zfill(2)
        anio_hoy = str(hoy.year)
        t_full = normalizar_texto(fecha_full)
        t_full = re.sub(r'lunes|martes|miercoles|jueves|viernes|sabado|domingo', '', t_full).strip()
        if " al " in t_full:
            partes = t_full.split(" al ")
            dia_fin, mes_fin, anio_fin = extraer_componentes(partes[1], mes_hoy, anio_hoy)
            dia_ini, mes_ini, anio_ini = extraer_componentes(partes[0], mes_fin, anio_fin)
            return f"{dia_ini}/{mes_ini}/{anio_ini}", f"{dia_fin}/{mes_fin}/{anio_fin}"
        else:
            dia, mes, anio = extraer_componentes(t_full, mes_hoy, anio_hoy)
            return f"{dia}/{mes}/{anio}", f"{dia}/{mes}/{anio}"
    interior = [
        'carlos paz', 'cosquin', 'jesus maria', 'alta gracia',
        'rio cuarto', 'villa maria', 'mina clavero', 'san francisco',
        'rio ceballos', 'colonia caroya', 'villa allende', 'la cumbre',
        'embalse', 'oncativo', 'cafayate', 'jujuy'
    ]
    try:
        BASE_URL = "https://ventas.autoentrada.com/"
        driver = iniciar_driver()
        driver.get(BASE_URL)
        print("Autoentrada: Driver iniciado")
        print("Autoentrada: Cargando todos los eventos (scroll)...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2.5)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        elementos = driver.find_elements(By.CLASS_NAME, "evento")
        print(f"📊 Autoentrada: {len(elementos)} eventos totales detectados")
        eventos_procesados = []
        for el in elementos:
            try:
                try:
                    url_evento = el.find_element(By.TAG_NAME, "a").get_attribute("href")
                except:
                    url_evento = BASE_URL
                nombre = el.find_element(By.TAG_NAME, "h2").text
                info_p = el.find_element(By.TAG_NAME, "p").text
                lineas = info_p.split('\n')
                fecha_raw = lineas[0] if len(lineas) > 0 else ""
                ubi_raw = lineas[1] if len(lineas) > 1 else ""
                ubi_norm = normalizar_texto(ubi_raw)
                es_cordoba = any(p in ubi_norm for p in ['cordoba', 'cba'])
                es_interior = any(loc in ubi_norm for loc in interior)
                if not es_cordoba or es_interior:
                    continue
                f_inicio, f_termina = procesar_rango_fechas(fecha_raw)
                if not f_inicio:
                    registrar_rechazo(nombre, ubi_raw, fecha_raw, "Fallo parseo de fecha", "scraper", url_evento)
                    continue
                partes_ubi = [p.strip() for p in ubi_raw.split(',')]
                eventos_procesados.append({
                    'Eventos': nombre,
                    'Lugar': partes_ubi[0] if partes_ubi else "",
                    'Comienza': f_inicio,
                    'Finaliza': f_termina,
                    'Tipo de evento': "",
                    'Detalle': "",
                    'Alcance': "",
                    'Costo de entrada': "",
                    'Fuente': 'Autoentrada',
                    'Origen': url_evento,
                    'fecha de carga': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            except Exception as e:
                try:
                    nombre_err = el.find_element(By.TAG_NAME, "h2").text
                except:
                    nombre_err = "Desconocido"
                registrar_rechazo(nombre_err, "", "", f"Error extracción: {str(e)}", "loop", BASE_URL)
                continue
        print(f"📊 Autoentrada: {len(eventos_procesados)} eventos de Córdoba Capital")
        if eventos_procesados:
            df_final = pd.DataFrame(eventos_procesados)
            df_final = df_final.drop_duplicates(subset=['Eventos', 'Comienza'])
            df_final, metricas = aplicar_clasificador(df_final, 'Eventos', 'Lugar', 'Tipo de evento', 'confianza_clasificacion')
            log(f"🤖 Autoentrada — Predicciones: {metricas['predicciones']} | Confianza: {metricas['confianza_promedio']}")
            df_final = df_final.fillna('').astype(str).replace(['None', 'nan', 'NaN'], '')
            subir_a_google_sheets(df_final, 'Autoentrada historico (Auto)', 'Hoja 1')
            reporte["filas_procesadas"] = len(df_final)
            reporte["estado"] = "Exitoso"
        else:
            log("❌ Autoentrada: Sin eventos válidos tras el filtrado.")
            reporte["estado"] = "Advertencia: Sin datos"
        if not df_rechazados.empty:
            subir_a_google_sheets(df_rechazados.astype(str), 'Rechazados', 'Eventos')
    except Exception as e:
        log(f"❌ ERROR CRÍTICO EN AUTOENTRADA: {e}")
        reporte["estado"] = "Fallido"
        reporte["error"] = str(e)
    finally:
        if driver: driver.quit()
        return reporte

# --- LLAMADO (comentar para desactivar) ---
log('')
log('AUTOENTRADA')
#ejecutar_scraper_autoentrada()

#ENTE METROPOLITANO
def ejecutar_scraper_metropolitano():
    import pandas as pd
    import re
    import time
    from datetime import datetime
    from bs4 import BeautifulSoup
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver = None
    df_rechazados = pd.DataFrame(columns=['Nombre', 'Locación', 'Fecha', 'Motivo', 'Linea', 'Fuente', 'Link', 'Fecha Scrp'])

    reporte = {
        "nombre": "Ente Metropolitano",
        "estado": "Pendiente",
        "filas_procesadas": 0,
        "error": None,
        "inicio": datetime.now().strftime('%H:%M:%S')
    }

    def registrar_rechazo(nombre, loc, fecha, motivo, linea, href):
        nonlocal df_rechazados
        nuevo = pd.DataFrame([{
            'Nombre': nombre, 'Locación': loc, 'Fecha': fecha,
            'Motivo': motivo, 'Linea': str(linea), 'Fuente': 'Ente Metropolitano',
            'Link': href, 'Fecha Scrp': datetime.now().strftime('%Y-%m-%d')
        }])
        df_rechazados = pd.concat([df_rechazados, nuevo], ignore_index=True)

    def parsear_fecha_metropolitano(fecha_raw):
        texto = fecha_raw.replace('\n', ' ').strip()
        texto = re.sub(r'\s+', ' ', texto)

        RE_FECHA = r'\d{2}/\d{2}/\d{4}'
        RE_HORA  = r'\d{2}:\d{2}'

        def fecha_a_iso(f, h="00:00"):
            try:
                return datetime.strptime(f"{f} {h}", "%d/%m/%Y %H:%M").strftime("%Y-%m-%d %H:%M:%S")
            except:
                return None

        fechas = re.findall(RE_FECHA, texto)
        horas  = re.findall(RE_HORA,  texto)

        # CASO A: Dos fechas — "09/01/2026 - 11/02/2026"
        if len(fechas) == 2:
            comienza = fecha_a_iso(fechas[0], horas[0] if horas else "00:00")
            finaliza = fecha_a_iso(fechas[1], horas[1] if len(horas) > 1 else (horas[0] if horas else "00:00"))
            return comienza, finaliza

        # CASO B: Una fecha
        if len(fechas) == 1:
            f = fechas[0]
            if len(horas) == 2:
                return fecha_a_iso(f, horas[0]), fecha_a_iso(f, horas[1])
            if len(horas) == 1:
                iso = fecha_a_iso(f, horas[0])
                return iso, iso
            iso = fecha_a_iso(f)
            return iso, iso

        # CASO C: Sin fecha reconocible
        return None, None

    try:
        BASE_URL = "https://entemetropolitano.gob.ar/agenda-de-eventos/"
        driver = iniciar_driver()
        driver.get(BASE_URL)
        print("Metropolitano: Driver iniciado")
        time.sleep(3)

        # --- CARGAR TODOS LOS EVENTOS ---
        print("Metropolitano: Cargando todos los eventos...")
        clicks = 0
        while True:
            try:
                fin = driver.find_element(By.ID, "mensaje-fin-eventos")
                if fin.is_displayed():
                    print(f"Metropolitano: Fin de eventos tras {clicks} clicks")
                    break

                btn = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "btn-cargar-mas-eventos"))
                )
                driver.execute_script("arguments[0].click();", btn)
                clicks += 1
                print(f"Metropolitano: Click #{clicks} en 'Cargar más'")
                time.sleep(2)

            except Exception as e:
                print(f"Metropolitano: Loop de carga finalizado — {e}")
                break

        # --- PARSEO ---
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        eventos_html = soup.select('div#eventos-container a.evento')
        print(f"📊 Metropolitano: {len(eventos_html)} eventos detectados")

        eventos_procesados = []
        for ev in eventos_html:
            try:
                nombre_el = ev.select_one('h3.evento-titulo')
                fecha_el  = ev.select_one('div.evento-fecha p')
                lugar_el  = ev.select_one('div.evento-ubicacion p')
                href      = ev.get('href', BASE_URL)

                nombre    = nombre_el.text.strip() if nombre_el else None
                fecha_raw = fecha_el.text.strip()  if fecha_el  else ""
                lugar_raw = lugar_el.text.strip()  if lugar_el  else ""

                if not nombre:
                    continue

                comienza, finaliza = parsear_fecha_metropolitano(fecha_raw)

                if not comienza:
                    registrar_rechazo(nombre, lugar_raw, fecha_raw, "Sin fecha reconocible", "parseo", href)
                    continue

                eventos_procesados.append({
                    'Eventos':          nombre,
                    'Lugar':            lugar_raw,
                    'Comienza':         comienza,
                    'Finaliza':         finaliza,
                    'Tipo de evento':   "",
                    'Detalle':          "",
                    'Alcance':          "",
                    'Costo de entrada': "",
                    'Fuente':           'Ente Metropolitano',
                    'Origen':           href,
                    'fecha de carga':   datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

            except Exception as e:
                registrar_rechazo(
                    ev.select_one('h3.evento-titulo').text.strip() if ev.select_one('h3.evento-titulo') else "Desconocido",
                    "", "", f"Error extracción: {str(e)}", "loop", BASE_URL
                )
                continue

        print(f"📊 Metropolitano: {len(eventos_procesados)} eventos válidos")

        # --- ARMADO, CLASIFICACIÓN Y SUBIDA ---
        if eventos_procesados:
            df_final = pd.DataFrame(eventos_procesados)
            df_final = df_final.astype(str).replace('None', '').replace('nan', '')
            df_final = df_final.drop_duplicates(subset=['Origen'])

            # Preview local
            print(f"\n📋 df_final — {len(df_final)} filas x {len(df_final.columns)} columnas")
            print(df_final.to_string())

            # Clasificador
            df_final, metricas = aplicar_clasificador(df_final, 'Eventos', 'Lugar', 'Tipo de evento', 'confianza_clasificacion')
            log(f"🤖 Metropolitano — Predicciones: {metricas['predicciones']} | Confianza: {metricas['confianza_promedio']}")

            subir_a_google_sheets(df_final, 'Metropolitano historico (Auto)', 'Hoja 1')

            reporte["filas_procesadas"] = len(df_final)
            reporte["estado"] = "Exitoso"
        else:
            log("❌ Metropolitano: Sin eventos válidos.")
            reporte["estado"] = "Advertencia: Sin datos"

        if not df_rechazados.empty:
            subir_a_google_sheets(df_rechazados.astype(str), 'Rechazados', 'Eventos')

    except Exception as e:
        log(f"❌ ERROR CRÍTICO EN METROPOLITANO: {e}")
        reporte["estado"] = "Fallido"
        reporte["error"] = str(e)

    finally:
        if driver: driver.quit()
        return reporte


# --- LLAMADO (comentar para desactivar) ---
#log('')
#log('METROPOLITANO')
#ejecutar_scraper_metropolitano()

def ejecutar_scraper_fcefyn():
    import pandas as pd
    import re
    import time
    import json
    import os
    from datetime import datetime
    from bs4 import BeautifulSoup
    from selenium.webdriver.common.by import By

    driver = None
    df_rechazados = pd.DataFrame(columns=['Nombre', 'Locación', 'Fecha', 'Motivo', 'Linea', 'Fuente', 'Link', 'Fecha Scrp'])

    reporte = {
        "nombre": "FCEFyN",
        "estado": "Pendiente",
        "filas_procesadas": 0,
        "error": None,
        "inicio": datetime.now().strftime('%H:%M:%S')
    }

    MEMORIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memoria")
    os.makedirs(MEMORIA_DIR, exist_ok=True)  # Crea la carpeta si no existe
    MEMORIA_PATH = os.path.join(MEMORIA_DIR, "fcefyn_memoria.json")
    print(f"📁 Ruta memoria: {MEMORIA_PATH}")
    print(f"📁 Archivo existe: {os.path.exists(MEMORIA_PATH)}")

    def cargar_memoria():
        if os.path.exists(MEMORIA_PATH):
            try:
                with open(MEMORIA_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {"ultimo_href": None, "ultima_fecha_scrp": None}

    def guardar_memoria(href, fecha_scrp):
        try:
            with open(MEMORIA_PATH, "w", encoding="utf-8") as f:
                json.dump({"ultimo_href": href, "ultima_fecha_scrp": fecha_scrp}, f, ensure_ascii=False, indent=2)
            log(f"💾 FCEFyN: Memoria guardada — {href}")
        except Exception as e:
            log(f"⚠️ FCEFyN: No se pudo guardar memoria — {e}")

    def registrar_rechazo(nombre, loc, fecha, motivo, linea, href):
        nonlocal df_rechazados
        nuevo = pd.DataFrame([{
            'Nombre': nombre, 'Locación': loc, 'Fecha': fecha,
            'Motivo': motivo, 'Linea': str(linea), 'Fuente': 'FCEFyN',
            'Link': href, 'Fecha Scrp': datetime.now().strftime('%Y-%m-%d')
        }])
        df_rechazados = pd.concat([df_rechazados, nuevo], ignore_index=True)

    def iso_desde_datetime_attr(time_tag):
        title = time_tag.get('title', '')
        meses_map = {
            "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
            "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
            "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"
        }
        try:
            title_lower = title.lower()
            match = re.search(
                r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\s+a\s+las\s+(\d{2}:\d{2})',
                title_lower
            )
            if match:
                dia, mes_txt, anio, hora = match.groups()
                mes = meses_map.get(mes_txt)
                if mes:
                    return datetime.strptime(
                        f"{dia.zfill(2)}/{mes}/{anio} {hora}",
                        "%d/%m/%Y %H:%M"
                    ).strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
        return None

    def parsear_fechas_fcefyn(soup_card):
        times = soup_card.select('div.clock time')
        if not times:
            return None, None
        comienza = iso_desde_datetime_attr(times[0])
        finaliza = iso_desde_datetime_attr(times[1]) if len(times) > 1 else None
        if not finaliza:
            finaliza = comienza
        return comienza, finaliza

    # --- PATRONES DE FILTRO ---
    PATRON_EVENTO = (
        r'colaci[oó]n|'
        r'sesi[oó]n\s+hcd|'
        r'posgrados?|'
        r'defensas?\s+de\s+tesis'
    )
    PATRON_VIRTUAL = (
        r'virtual|online|videoconferencia|conferencia\s+virtual|'
        r'meet\.google\.com|youtube|v[íi]a\s+meet|meet|zoom|'
        r'sincr[óo]nica|asincr[óo]nica'
    )

    try:
        BASE_URL  = "https://fcefyn.unc.edu.ar"
        LISTA_URL = f"{BASE_URL}/archivo/eventos/en/home"

        memoria = cargar_memoria()
        print(f"📁 Contenido memoria: {memoria}")
        ultimo_href_conocido = memoria.get("ultimo_href")
        print(f"📁 Último href conocido: {ultimo_href_conocido}")
        print(f"FCEFyN: Último href en memoria → {ultimo_href_conocido or 'ninguno (primera ejecución)'}")

        driver = iniciar_driver()
        print("FCEFyN: Driver iniciado")

        eventos_procesados = []
        pagina = 1
        detener = False
        primer_href_nueva_ejecucion = None

        while not detener:
            url_pagina = f"{LISTA_URL}?page={pagina}"
            driver.get(url_pagina)
            time.sleep(2)

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            if soup.find('h1', string=lambda t: t and 'Contenido no encontrado' in t):
                print(f"FCEFyN: Página {pagina} sin contenido — fin del scraping")
                break

            cards = soup.select('div.card.event-teaser')
            if not cards:
                print(f"FCEFyN: Página {pagina} sin cards — fin del scraping")
                break

            print(f"FCEFyN: Página {pagina} — {len(cards)} eventos")

            for card in cards:
                try:
                    a_tag  = card.select_one('div.card-content a')
                    href   = BASE_URL + a_tag['href'] if a_tag else None
                    nombre = card.select_one('h4').text.strip() if card.select_one('h4') else None
                    lugar  = card.select_one('div.place').text.strip() if card.select_one('div.place') else ""

                    if not nombre or not href:
                        continue

                    if primer_href_nueva_ejecucion is None:
                        primer_href_nueva_ejecucion = href

                    if href == ultimo_href_conocido:
                        print(f"FCEFyN: Alcanzado último evento conocido en página {pagina} — deteniendo")
                        detener = True
                        print(f"  Comparando: {href[:60]}... == {str(ultimo_href_conocido)[:60]}...")
                        break

                    comienza, finaliza = parsear_fechas_fcefyn(card)

                    if not comienza:
                        registrar_rechazo(nombre, lugar, "", "Sin fecha parseable", "loop", href)
                        continue

                    eventos_procesados.append({
                        'Eventos':          nombre,
                        'Lugar':            lugar,
                        'Comienza':         comienza,
                        'Finaliza':         finaliza,
                        'Tipo de evento':   "",
                        'Detalle':          "",
                        'Alcance':          "",
                        'Costo de entrada': "",
                        'Fuente':           'FCEFyN',
                        'Origen':           href,
                        'fecha de carga':   datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

                except Exception as e:
                    nombre_err = card.select_one('h4').text.strip() if card.select_one('h4') else "Desconocido"
                    registrar_rechazo(nombre_err, "", "", f"Error extracción: {str(e)}", "loop", "")
                    continue

            pagina += 1

        print(f"📊 FCEFyN: {len(eventos_procesados)} eventos recolectados antes de filtros")

        # --- ARMADO ---
        if eventos_procesados:
            df_final = pd.DataFrame(eventos_procesados)
            df_final = df_final.astype(str).replace('None', '').replace('nan', '')
            df_final = df_final.drop_duplicates(subset=['Origen'])

            # --- FILTRO 0: Por fecha ---
            ANIO_MINIMO = 2025
            df_final['anio_temp'] = df_final['Comienza'].str[:4].apply(lambda x: int(x) if x.isdigit() else 0)
            mask_antiguos = df_final['anio_temp'] < ANIO_MINIMO
            rechazados_antiguos = df_final[mask_antiguos].copy()
            df_final = df_final[~mask_antiguos].drop(columns=['anio_temp']).copy()
            for _, row in rechazados_antiguos.iterrows():
                registrar_rechazo(row['Eventos'], row['Lugar'], row['Comienza'], f'Evento anterior a {ANIO_MINIMO}', 'filtro_fecha', row['Origen'])
            log(f"🗑️ FCEFyN: {len(rechazados_antiguos)} eventos filtrados por fecha (anteriores a {ANIO_MINIMO})")

            # --- FILTRO 1: Por nombre de evento ---
            mask_evento = df_final['Eventos'].str.contains(PATRON_EVENTO, case=False, na=False, regex=True)
            rechazados_evento = df_final[mask_evento].copy()
            rechazados_evento['motivo_rechazo'] = 'Filtro nombre evento'
            df_final = df_final[~mask_evento].copy()
            print(f"🗑️ FCEFyN: {len(rechazados_evento)} eventos filtrados por nombre (colación, HCD, posgrado, tesis)")

            # Registrar rechazos de filtro 1
            for _, row in rechazados_evento.iterrows():
                registrar_rechazo(row['Eventos'], row['Lugar'], row['Comienza'], 'Filtro nombre evento', 'filtro1', row['Origen'])

            # --- FILTRO 2: Por lugar virtual ---
            mask_virtual = df_final['Lugar'].str.contains(PATRON_VIRTUAL, case=False, na=False, regex=True)
            rechazados_virtual = df_final[mask_virtual].copy()
            df_final = df_final[~mask_virtual].copy()
            print(f"🗑️ FCEFyN: {len(rechazados_virtual)} eventos filtrados por lugar virtual")

            # Registrar rechazos de filtro 2
            for _, row in rechazados_virtual.iterrows():
                registrar_rechazo(row['Eventos'], row['Lugar'], row['Comienza'], 'Filtro lugar virtual', 'filtro2', row['Origen'])

            print(f"✅ FCEFyN: {len(df_final)} eventos limpios tras filtros")

            # Preview local
            print(f"\n📋 df_final — {len(df_final)} filas x {len(df_final.columns)} columnas")
            print(df_final.to_string())

            # --- CLASIFICADOR ---
            df_final, metricas = aplicar_clasificador(df_final, 'Eventos', 'Lugar', 'Tipo de evento', 'confianza_clasificacion')
            log(f"🤖 FCEFyN — Predicciones: {metricas['predicciones']} | Confianza: {metricas['confianza_promedio']}")

            subir_a_google_sheets(df_final, 'FCEFyN historico (Auto)', 'Hoja 1')

            if primer_href_nueva_ejecucion:
                guardar_memoria(primer_href_nueva_ejecucion, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            reporte["filas_procesadas"] = len(df_final)
            reporte["estado"] = "Exitoso"

        else:
            log("FCEFyN: Sin eventos nuevos desde la última ejecución.")
            reporte["estado"] = "Sin novedades"

        # --- RECHAZADOS ---
        if not df_rechazados.empty:
            subir_a_google_sheets(df_rechazados.astype(str), 'Rechazados', 'Eventos')

    except Exception as e:
        log(f"❌ ERROR CRÍTICO EN FCEFYN: {e}")
        reporte["estado"] = "Fallido"
        reporte["error"] = str(e)

    finally:
        if driver: driver.quit()
        return reporte


# --- LLAMADO (comentar para desactivar) ---
log('')
log('FCEFYN')
#ejecutar_scraper_fcefyn()

def ejecutar_scraper_famaf():
    import pandas as pd
    import re
    import time
    import json
    import os
    from datetime import datetime
    from bs4 import BeautifulSoup

    driver = None
    df_rechazados = pd.DataFrame(columns=['Nombre', 'Locación', 'Fecha', 'Motivo', 'Linea', 'Fuente', 'Link', 'Fecha Scrp'])

    reporte = {
        "nombre": "FAMAF",
        "estado": "Pendiente",
        "filas_procesadas": 0,
        "error": None,
        "inicio": datetime.now().strftime('%H:%M:%S')
    }

    # --- CONFIGURACIÓN DE FILTROS ---
    PATRON_ORIGEN_ACADEMICA = r'academica'
    PATRON_EVENTO_AUTORIDADES = r'autoridades'
    ANIO_MINIMO = 2025

    MEMORIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memoria")
    os.makedirs(MEMORIA_DIR, exist_ok=True)
    MEMORIA_PATH = os.path.join(MEMORIA_DIR, "famaf_memoria.json")
    print(f"Memoria en: {MEMORIA_PATH}")

    def cargar_memoria():
        if not os.path.exists(MEMORIA_PATH):
            return {"ultimo_href": None, "ultima_fecha_scrp": None}
        try:
            with open(MEMORIA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"ultimo_href": None, "ultima_fecha_scrp": None}

    def guardar_memoria(href, fecha_scrp):
        try:
            datos = {"ultimo_href": href, "ultima_fecha_scrp": fecha_scrp}
            with open(MEMORIA_PATH, "w", encoding="utf-8") as f:
                json.dump(datos, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ No se pudo guardar memoria: {e}")

    def registrar_rechazo(nombre, loc, fecha, motivo, linea, href):
        nonlocal df_rechazados
        nuevo = pd.DataFrame([{
            'Nombre': nombre, 'Locación': loc, 'Fecha': fecha,
            'Motivo': motivo, 'Linea': str(linea), 'Fuente': 'FAMAF',
            'Link': href, 'Fecha Scrp': datetime.now().strftime('%Y-%m-%d')
        }])
        df_rechazados = pd.concat([df_rechazados, nuevo], ignore_index=True)

    def parsear_fechas_famaf(card):
        times = card.select('div.card-content div time')
        def iso_desde_attr(time_tag):
            try:
                attr = time_tag.get('datetime', '')
                dt = datetime.fromisoformat(attr)
                return dt.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S")
            except: return None
        if not times: return None, None
        comienza = iso_desde_attr(times[0])
        finaliza = iso_desde_attr(times[1]) if len(times) > 1 else comienza
        return comienza, finaliza

    try:
        BASE_URL  = "https://famaf.unc.edu.ar"
        LISTA_URL = f"{BASE_URL}/archivo/eventos"

        memoria = cargar_memoria()
        ultimo_href_conocido = memoria.get("ultimo_href", None)
        print(f"FAMAF: Último href en memoria → {ultimo_href_conocido or 'ninguno'}")

        driver = iniciar_driver()
        eventos_procesados = []
        pagina = 1
        detener = False
        hrefs_nueva_ejecucion = []

        while not detener:
            url_pagina = f"{LISTA_URL}?page={pagina}"
            driver.get(url_pagina)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            if soup.find('h1', string=lambda t: t and 'No encontrado' in t): break
            cards = soup.select('div.card.evento')
            if not cards: break

            for card in cards:
                try:
                    a_tag = card.select_one('div.card-image a')
                    href = BASE_URL + a_tag['href'] if a_tag else None
                    nombre_el = card.select_one('h3.title a')
                    nombre = nombre_el.text.strip() if nombre_el else None

                    lugar = ""
                    for div in card.select('div.card-content div'):
                        if div.select_one('svg.fa-location-dot'):
                            lugar = div.get_text(strip=True)
                            break

                    if not nombre or not href: continue
                    if len(hrefs_nueva_ejecucion) < 2: hrefs_nueva_ejecucion.append(href)
                    if ultimo_href_conocido and href == ultimo_href_conocido:
                        detener = True
                        break

                    comienza, finaliza = parsear_fechas_famaf(card)
                    if not comienza:
                        registrar_rechazo(nombre, lugar, "", "Sin fecha parseable", "loop", href)
                        continue

                    eventos_procesados.append({
                        'Eventos': nombre, 'Lugar': lugar, 'Comienza': comienza,
                        'Finaliza': finaliza, 'Tipo de evento': "", 'Detalle': "",
                        'Alcance': "", 'Costo de entrada': "", 'Fuente': 'FAMAF',
                        'Origen': href, 'fecha de carga': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                except Exception as e:
                    registrar_rechazo("Error", "", "", str(e), "loop", "")
            pagina += 1

        print(f"📊 FAMAF: {len(eventos_procesados)} eventos recolectados antes de filtros")

        if eventos_procesados:
            df_final = pd.DataFrame(eventos_procesados)
            df_final = df_final.astype(str).replace('None', '').replace('nan', '')
            df_final = df_final.drop_duplicates(subset=['Origen'])

            # --- FILTRO 1: Por Origen (contiene 'academica') ---
            mask_academica = df_final['Origen'].str.contains(PATRON_ORIGEN_ACADEMICA, case=False, na=False)
            rechazados_academica = df_final[mask_academica].copy()
            df_final = df_final[~mask_academica].copy()
            for _, row in rechazados_academica.iterrows():
                registrar_rechazo(row['Eventos'], row['Lugar'], row['Comienza'], 'Filtro Origen Académica', 'filtro_url', row['Origen'])

            # --- FILTRO 2: Por Eventos (contiene 'autoridades') ---
            mask_autoridades = df_final['Eventos'].str.contains(PATRON_EVENTO_AUTORIDADES, case=False, na=False)
            rechazados_autoridades = df_final[mask_autoridades].copy()
            df_final = df_final[~mask_autoridades].copy()
            for _, row in rechazados_autoridades.iterrows():
                registrar_rechazo(row['Eventos'], row['Lugar'], row['Comienza'], 'Filtro Palabra Autoridades', 'filtro_nombre', row['Origen'])

            # --- FILTRO 3: Por Año (mínimo 2025) ---
            # Comienza tiene formato 'YYYY-MM-DD HH:MM:SS'
            df_final['anio_temp'] = df_final['Comienza'].str[:4].apply(lambda x: int(x) if x.isdigit() else 0)
            mask_antiguos = df_final['anio_temp'] < ANIO_MINIMO
            rechazados_antiguos = df_final[mask_antiguos].copy()
            df_final = df_final[~mask_antiguos].drop(columns=['anio_temp']).copy()
            
            for _, row in rechazados_antiguos.iterrows():
                registrar_rechazo(row['Eventos'], row['Lugar'], row['Comienza'], f'Evento anterior a {ANIO_MINIMO}', 'filtro_fecha', row['Origen'])

            print(f"✅ FAMAF: {len(df_final)} eventos limpios tras filtros")

            # Clasificador y subida
            df_final, metricas = aplicar_clasificador(df_final, 'Eventos', 'Lugar', 'Tipo de evento', 'confianza_clasificacion')
            subir_a_google_sheets(df_final, 'FAMAF historico (Auto)', 'Hoja 1')

            href_a_guardar = hrefs_nueva_ejecucion[1] if len(hrefs_nueva_ejecucion) >= 2 else (hrefs_nueva_ejecucion[0] if hrefs_nueva_ejecucion else None)
            if href_a_guardar:
                guardar_memoria(href_a_guardar, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            reporte["filas_procesadas"] = len(df_final)
            reporte["estado"] = "Exitoso"
        else:
            reporte["estado"] = "Sin novedades"

        if not df_rechazados.empty:
            subir_a_google_sheets(df_rechazados.astype(str), 'Rechazados', 'Eventos')

    except Exception as e:
        log(f"❌ ERROR CRÍTICO EN FAMAF: {e}")
        reporte["estado"] = "Fallido"
        reporte["error"] = str(e)
    finally:
        if driver: driver.quit()
        return reporte


# --- LLAMADO (comentar para desactivar) ---
log('')
log('FAMAF')
#ejecutar_scraper_famaf()



#Importante el orden. Marca jerarquía. El primero se mantiene siempre a la hora de comparar duplicados y así...
dict_fuentes = {
    'FAMAF': 'FAMAF historico (Auto)',
    'FCEyN': 'FCEyN historico (Auto)',
    'FCEFyN': 'FCEFyN historico (Auto)',
    'Ferias y Congresos': 'Ferias y Congresos (Auto)',
    'Ticketek': 'Ticketek historico (Auto)',
    'Eden Entradas': 'Eden historico (Auto)',
    'Autoentrada': 'Autoentrada historico (Auto)',
    'Agencia Turismo Cba': 'Turismo CBA (Auto)',
    'Ente Metropolitano': 'Metropolitano historico (Auto)',
    'eventbrite': 'base_h_scrp_eventbrite',
    'AFA': 'Fixture Fútbol Córdoba'
}

def procesar_duplicados_y_normalizar():
    print("🚀 Iniciando proceso de limpieza con Jerarquía de Fuentes...")

    # --- CIUDADES BLACKLIST (no permitidas) ---
    # --- CIUDADES BLACKLIST (no permitidas) ---
    ciudades_blacklist = [
        r'neuqu[eé]n',
        r'curitiba',
        'huberman 1750',
        r'agua de oro',
        r'alta gracia',
        r'bialet mass[eé]',
        r'bouwer',
        r'calch[ií]n',
        r'capilla de los remedios',
        r'carlos paz',
        r'casa grande',
        r'colonia caroya',
        r'colonia tirolesa',
        r'colonia vicente ag[uü]ero',
        r'cosqu[ií]n',
        r'despeñaderos',
        r'el manzano',
        r'estaci[oó]n general paz',
        r'estaci[oó]n ju[aá]rez celman',
        r'estancia vieja',
        r'falda del carmen',
        r'icho cruz',
        r'jes[uú]s mar[ií]a',
        r'la calera',
        r'la falda',
        r'la granja',
        r'los cedros',
        r'lozada',
        r'malagueño',
        r'malvinas argentinas',
        r'mayu sumaj',
        r'mendiolaza',
        r'mi granja',
        r'montecristo',
        r'pilar',
        r'rafael garc[ií]a',
        r'r[ií]o ceballos',
        r'r[ií]o primero',
        r'r[ií]o segundo',
        r'sald[aá]n',
        r'salsipuedes',
        r'san antonio de arredondo',
        r'san roque',
        r'santa mar[ií]a de punilla',
        r'tala huasi',
        r'tanti',
        r'tinoco',
        r'toledo',
        r'unquillo',
        r'valle hermoso',
        r'villa allende',
        r'villa carlos paz',
        r'villa cerro azul',
        r'villa del prado',
        r'villa dolores',
        r'villa parque santa ana',
        r'villa parque s[ií]quiman',
        r'villa santa cruz del lago'
    ]

    # --- FUNCIONES AUXILIARES INTERNAS ---
    def obtener_df_de_sheets(nombre_tabla, nombre_hoja):
        import os, json, gspread
        from google.oauth2 import service_account
        secreto_json = os.environ.get('GCP_SERVICE_ACCOUNT_JSON')
        if not secreto_json:
            print(f"  ❌ No se encontró la variable de entorno GCP_SERVICE_ACCOUNT_JSON")
            return pd.DataFrame()
        try:
            info_claves = json.loads(secreto_json)
            creds = service_account.Credentials.from_service_account_info(
                info_claves, scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"
                ]
            )
            client = gspread.authorize(creds)
            sheet = client.open(nombre_tabla).worksheet(nombre_hoja)
            data = sheet.get_all_values()
            if len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
                print(f"  📗 Sheets OK: '{nombre_tabla}' / '{nombre_hoja}' → {len(df)} filas")
                return df
            else:
                print(f"  ⚠️ Hoja vacía o sin datos: '{nombre_tabla}' / '{nombre_hoja}'")
                return pd.DataFrame()
        except Exception as e:
            print(f"  ❌ Error leyendo '{nombre_tabla}' / '{nombre_hoja}': {e}")
            return pd.DataFrame()

    def borrar_fila_por_origen(nombre_tabla, nombre_hoja, origen_link):
        import os, json, gspread
        from google.oauth2 import service_account

        url_exceptuada = "https://www.feriasycongresos.com/calendario-de-eventos?busqueda=C%C3%B3rdoba"
        if str(origen_link).strip() == url_exceptuada:
            print(f"    🛡️ Excepción: origen protegido, no se borra.")
            return

        secreto_json = os.environ.get('GCP_SERVICE_ACCOUNT_JSON')
        if not secreto_json:
            print(f"    ❌ No se encontró GCP_SERVICE_ACCOUNT_JSON")
            return

        try:
            info_claves = json.loads(secreto_json)
            creds = service_account.Credentials.from_service_account_info(
                info_claves, scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"
                ]
            )
            client = gspread.authorize(creds)
            sheet = client.open(nombre_tabla).worksheet(nombre_hoja)

            data = sheet.get_all_values()
            if len(data) <= 1:
                print(f"    ⚠️ Hoja '{nombre_tabla}' vacía, nada que borrar.")
                return

            df_temp = pd.DataFrame(data[1:], columns=data[0])

            columnas_posibles = ['Origen', 'href', 'Link', 'URL']
            col_id = next((c for c in columnas_posibles if c in df_temp.columns), None)

            if col_id:
                match_idx = df_temp.index[df_temp[col_id].astype(str) == str(origen_link)].tolist()
                if match_idx:
                    fila_a_borrar = match_idx[0] + 2
                    sheet.delete_rows(fila_a_borrar)
                    print(f"    🗑️ Eliminado de '{nombre_tabla}' (col {col_id}): {origen_link}")
                else:
                    print(f"    ⚠️ Link no encontrado en '{nombre_tabla}': {origen_link}")
            else:
                print(f"    ❌ No se encontró columna de ID en '{nombre_tabla}'")

        except Exception as e:
            print(f"    ❌ Error borrando en '{nombre_tabla}': {e}")

    def normalizar_lugar_en_sheet(nombre_tabla, nombre_hoja, origen_link, lugar_normalizado,
                                   nombre_evento=None, fecha_evento=None,
                                   max_reintentos=4, cooldown_base=65):
        import os, json, time, gspread
        from google.oauth2 import service_account
    
        secreto_json = os.environ.get('GCP_SERVICE_ACCOUNT_JSON')
        if not secreto_json:
            print(f"    ❌ No se encontró GCP_SERVICE_ACCOUNT_JSON")
            return
    
        for intento in range(1, max_reintentos + 1):
            try:
                info_claves = json.loads(secreto_json)
                creds = service_account.Credentials.from_service_account_info(
                    info_claves, scopes=[
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive"
                    ]
                )
                client = gspread.authorize(creds)
                sheet = client.open(nombre_tabla).worksheet(nombre_hoja)
    
                data = sheet.get_all_values()
                if len(data) <= 1:
                    print(f"    ⚠️ Hoja '{nombre_tabla}' vacía.")
                    return
    
                headers = data[0]
                df_temp = pd.DataFrame(data[1:], columns=headers)
    
                columnas_lugar = ['Lugar', 'lugar', 'Location', 'Locacion', 'Locación']
                col_lugar = next((c for c in columnas_lugar if c in headers), None)
                if not col_lugar:
                    print(f"    ❌ No se encontró columna 'Lugar' en '{nombre_tabla}'")
                    return
                col_lugar_idx = headers.index(col_lugar) + 1
    
                columnas_posibles = ['Origen', 'href', 'Link', 'URL']
                col_id = next((c for c in columnas_posibles if c in headers), None)
                if not col_id:
                    print(f"    ❌ No se encontró columna de ID en '{nombre_tabla}'")
                    return
    
                # --- Buscar fila: primero intentar por origen único,
                #     si hay más de un match usar nombre+fecha como desempate ---
                match_por_origen = df_temp.index[
                    df_temp[col_id].astype(str) == str(origen_link)
                ].tolist()
    
                if len(match_por_origen) == 0:
                    print(f"    ⚠️ Link no encontrado en '{nombre_tabla}': {origen_link}")
                    return
    
                elif len(match_por_origen) == 1:
                    # Origen único → comportamiento original
                    match_idx = match_por_origen
    
                else:
                    # Origen compartido → desempatar por nombre + fecha
                    if nombre_evento is None or fecha_evento is None:
                        print(f"    ⚠️ Origen compartido en '{nombre_tabla}' pero no se pasó nombre/fecha. Se omite.")
                        return
    
                    col_nombre = next((c for c in ['Eventos', 'Nombre', 'Titulo', 'Title'] if c in headers), None)
                    col_fecha  = next((c for c in ['Comienza', 'Fecha', 'Date', 'Start']   if c in headers), None)
    
                    if not col_nombre or not col_fecha:
                        print(f"    ⚠️ No se encontraron columnas nombre/fecha en '{nombre_tabla}'. Se omite.")
                        return
    
                    mask = (
                        (df_temp[col_id].astype(str)     == str(origen_link))   &
                        (df_temp[col_nombre].astype(str)  == str(nombre_evento)) &
                        (df_temp[col_fecha].astype(str)   == str(fecha_evento))
                    )
                    match_idx = df_temp.index[mask].tolist()
    
                    if not match_idx:
                        print(f"    ⚠️ Sin match por nombre+fecha en '{nombre_tabla}': '{nombre_evento}' / {fecha_evento}")
                        return
    
                fila_sheet = match_idx[0] + 2  # +1 header, +1 base 1
                sheet.update_cell(fila_sheet, col_lugar_idx, lugar_normalizado)
                print(f"    ✏️ Lugar actualizado en '{nombre_tabla}' fila {fila_sheet}: '{lugar_normalizado}'")
                return
    
            except Exception as e:
                es_429 = '429' in str(e) or 'Quota exceeded' in str(e)
                if es_429 and intento < max_reintentos:
                    espera = cooldown_base * intento
                    print(f"    ⏳ Rate limit (429) en '{nombre_tabla}'. Esperando {espera}s antes del intento {intento+1}/{max_reintentos}...")
                    time.sleep(espera)
                else:
                    print(f"    ❌ Error normalizando lugar en '{nombre_tabla}': {e}")
                    return
                
    def actualizar_id_en_sheet(nombre_tabla, nombre_hoja, origen_link, nuevo_id,
                               nombre_evento=None, fecha_evento=None,
                               max_reintentos=4, cooldown_base=65):
        import os, json, time, gspread
        from google.oauth2 import service_account
    
        secreto_json = os.environ.get('GCP_SERVICE_ACCOUNT_JSON')
        if not secreto_json:
            print(f"    ❌ No se encontró GCP_SERVICE_ACCOUNT_JSON")
            return
    
        for intento in range(1, max_reintentos + 1):
            try:
                info_claves = json.loads(secreto_json)
                creds = service_account.Credentials.from_service_account_info(
                    info_claves, scopes=[
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive"
                    ]
                )
                client = gspread.authorize(creds)
                sheet = client.open(nombre_tabla).worksheet(nombre_hoja)
    
                data = sheet.get_all_values()
                if len(data) <= 1:
                    print(f"    ⚠️ Hoja '{nombre_tabla}' vacía.")
                    return
    
                headers = data[0]
                df_temp = pd.DataFrame(data[1:], columns=headers)
    
                if 'ID' not in headers:
                    print(f"    ❌ No se encontró columna 'ID' en '{nombre_tabla}'")
                    return
                col_id_idx = headers.index('ID') + 1
    
                columnas_posibles = ['Origen', 'href', 'Link', 'URL']
                col_origen = next((c for c in columnas_posibles if c in headers), None)
                if not col_origen:
                    print(f"    ❌ No se encontró columna de Origen en '{nombre_tabla}'")
                    return
    
                # --- Buscar fila: primero intentar por origen único,
                #     si hay más de un match usar nombre+fecha como desempate ---
                match_por_origen = df_temp.index[
                    df_temp[col_origen].astype(str) == str(origen_link)
                ].tolist()
    
                if len(match_por_origen) == 0:
                    print(f"    ⚠️ Link no encontrado en '{nombre_tabla}': {origen_link}")
                    return
    
                elif len(match_por_origen) == 1:
                    # Origen único → comportamiento original
                    match_idx = match_por_origen
    
                else:
                    # Origen compartido → desempatar por nombre + fecha
                    if nombre_evento is None or fecha_evento is None:
                        print(f"    ⚠️ Origen compartido en '{nombre_tabla}' pero no se pasó nombre/fecha. Se omite.")
                        return
    
                    col_nombre = next((c for c in ['Eventos', 'Nombre', 'Titulo', 'Title'] if c in headers), None)
                    col_fecha  = next((c for c in ['Comienza', 'Fecha', 'Date', 'Start']   if c in headers), None)
    
                    if not col_nombre or not col_fecha:
                        print(f"    ⚠️ No se encontraron columnas nombre/fecha en '{nombre_tabla}'. Se omite.")
                        return
    
                    mask = (
                        (df_temp[col_origen].astype(str)  == str(origen_link))   &
                        (df_temp[col_nombre].astype(str)  == str(nombre_evento)) &
                        (df_temp[col_fecha].astype(str)   == str(fecha_evento))
                    )
                    match_idx = df_temp.index[mask].tolist()
    
                    if not match_idx:
                        print(f"    ⚠️ Sin match por nombre+fecha en '{nombre_tabla}': '{nombre_evento}' / {fecha_evento}")
                        return
    
                fila_sheet = match_idx[0] + 2  # +1 header, +1 base 1
                sheet.update_cell(fila_sheet, col_id_idx, nuevo_id)
                print(f"    ✏️ ID actualizado en '{nombre_tabla}' fila {fila_sheet}: '{nuevo_id}'")
                return
    
            except Exception as e:
                es_429 = '429' in str(e) or 'Quota exceeded' in str(e)
                if es_429 and intento < max_reintentos:
                    espera = cooldown_base * intento
                    print(f"    ⏳ Rate limit (429) en '{nombre_tabla}'. Esperando {espera}s antes del intento {intento+1}/{max_reintentos}...")
                    time.sleep(espera)
                else:
                    print(f"    ❌ Error actualizando ID en '{nombre_tabla}': {e}")
                    return
                
    # --- CUERPO PRINCIPAL ---
    try:
        print("\n📥 Cargando datos principales de Sheets...")
        df_principal = obtener_df_de_sheets("Entradas auto", "Eventos")
        if df_principal.empty:
            print("⚠️ DataFrame principal vacío. Abortando.")
            return

        print(f"📋 Total de eventos cargados: {len(df_principal)}")
        print(f"📌 Columnas disponibles: {list(df_principal.columns)}")

        # --- GENERACIÓN DE IDs FALTANTES ---
        print("\n🆔 Generando IDs faltantes...")
        mapeo_prefijos = {
            'Ticketek': 'TKT',
            'Eden Entradas': 'EDE',
            'eventbrite': 'EVB',
            'FAMAF': 'FAM',
            'FCEyN': 'FCE',
            'FCEFyN': 'FCF',
            'Ferias y Congresos': 'FYC',
            'Autoentrada': 'AUT',
            'Agencia Turismo Cba': 'TCB',
            'Ente Metropolitano': 'TCM',
            'AFA': 'AFA',
        }
        ids_generados = 0
        fuentes_con_sufijos = ['Ferias y Congresos', 'AFA']
        for fuente in mapeo_prefijos.keys():
            if fuente in fuentes_con_sufijos:
                # Para fuentes con mismo URL, usar sufijos
                df_fuente = df_principal[df_principal['Fuente'] == fuente].copy()
                if not df_fuente.empty:
                    prefijo = mapeo_prefijos[fuente]
                    ids_serie = generar_ids_con_sufijo(df_fuente, 'Origen', prefijo)
                    for idx, nuevo_id in zip(df_fuente.index, ids_serie):
                        if pd.isna(df_principal.at[idx, 'ID']) or str(df_principal.at[idx, 'ID']).strip() == '':
                            df_principal.at[idx, 'ID'] = nuevo_id
                            ids_generados += 1
                            tabla_origen = dict_fuentes.get(fuente)
                            if tabla_origen:
                                actualizar_id_en_sheet(tabla_origen, "Hoja 1", df_principal.at[idx, 'Origen'], nuevo_id, nombre_evento=df_principal.at[idx, 'Eventos'], fecha_evento=df_principal.at[idx, 'Comienza'])
            else:
                # Para otras fuentes, IDs únicos por URL
                df_fuente = df_principal[df_principal['Fuente'] == fuente]
                for idx, row in df_fuente.iterrows():
                    if pd.isna(row.get('ID')) or str(row.get('ID', '')).strip() == '':
                        prefijo = mapeo_prefijos.get(fuente, 'EVT')
                        origen = str(row.get('Origen', ''))
                        nuevo_id = generar_id(origen, prefijo)
                        df_principal.at[idx, 'ID'] = nuevo_id
                        ids_generados += 1
                        tabla_origen = dict_fuentes.get(fuente)
                        if tabla_origen:
                            actualizar_id_en_sheet(tabla_origen, "Hoja 1", row['Origen'], nuevo_id, nombre_evento=row.get('Eventos'), fecha_evento=row.get('Comienza'))
        print(f"  ✅ IDs generados: {ids_generados}")

        # --- DataFrame para rechazados ---
        df_rechazados = pd.DataFrame(columns=['Nombre', 'Locación', 'Fecha', 'Motivo', 'Linea', 'Fuente', 'Link'])

        def registrar_rechazo(nombre, loc, fecha, motivo, linea, fuente, href):
            nonlocal df_rechazados
            nuevo = pd.DataFrame([{
                'Nombre': nombre, 'Locación': loc, 'Fecha': fecha,
                'Motivo': motivo, 'Linea': str(linea), 'Fuente': fuente, 'Link': href
            }])
            df_rechazados = pd.concat([df_rechazados, nuevo], ignore_index=True)

        # --- VERIFICACIÓN DE CIUDADES BLACKLIST ---
        print("\n🔍 Iniciando filtro de ciudades blacklist...")
        indices_a_eliminar = []

        for idx, row in df_principal.iterrows():
            nombre_evento = str(row.get('Eventos', ''))
            locacion = str(row.get('Lugar', ''))
            texto_combinado = f"{nombre_evento} {locacion}".lower()

            ciudad_detectada = None
            for patron in ciudades_blacklist:
                match = re.search(patron, texto_combinado, re.IGNORECASE)
                if match:
                    ciudad_detectada = match.group(0)
                    break

            if ciudad_detectada:
                registrar_rechazo(
                    nombre=nombre_evento,
                    loc=locacion,
                    fecha=row.get('Comienza', 'N/A'),
                    motivo=f"Ciudad en blacklist detectada: {ciudad_detectada}",
                    linea="filtro_blacklist",
                    fuente=row.get('Fuente', 'Desconocida'),
                    href=row.get('Origen', '')
                )
                indices_a_eliminar.append(idx)

                tabla_origen = dict_fuentes.get(row.get('Fuente'))
                if tabla_origen:
                    print(f"  🏙️ Blacklist ({ciudad_detectada}): Eliminando '{nombre_evento}' de {tabla_origen}")
                    borrar_fila_por_origen(tabla_origen, "Hoja 1", row.get('Origen'))

        print(f"  ✅ Filtro blacklist completado: {len(indices_a_eliminar)} eventos eliminados.")
        df_principal = df_principal.drop(indices_a_eliminar).reset_index(drop=True)
        print(f"  📋 Eventos restantes tras filtro: {len(df_principal)}")
            
        # --- 1. NORMALIZACIÓN DE LUGARES ---
        print("\n📍 Iniciando normalización de lugares...")
        df_equiv = obtener_df_de_sheets("Equiv Lugares", "Hoja 1")
        mapeo_lugares = {}
        valores_ya_normalizados = set()  # ← NUEVO: set con valores que YA son el resultado final
        
        if not df_equiv.empty:
            mapeo_lugares = {
                str(k).lower().strip(): str(v).strip()
                for k, v in zip(df_equiv.iloc[:, 0], df_equiv.iloc[:, 1])
            }
            # Construir set de valores normalizados (columna destino, col índice 1)
            # Si "Normalizado" es una columna específica, usar: df_equiv['Normalizado']
            # Si es simplemente la segunda columna:
            valores_ya_normalizados = {str(v).lower().strip() for v in df_equiv.iloc[:, 1]}
            print(f"  📖 Tabla de equivalencias cargada: {len(mapeo_lugares)} entradas.")
            print(f"  📖 Valores ya normalizados conocidos: {len(valores_ya_normalizados)}")
        else:
            print("  ⚠️ Tabla de equivalencias vacía o no encontrada.")
        
        lugares_no_encontrados = []
        lugares_normalizados = 0
        lugares_ya_ok = 0  # ← NUEVO: contador de los que se saltean
        
        for idx, row in df_principal.iterrows():
            lugar_raw = str(row.get('Lugar', ''))
            lugar_key = lugar_raw.lower().strip()
        
            # ← NUEVO: Si el valor actual ya ES un valor normalizado, saltearlo
            if lugar_key in valores_ya_normalizados:
                lugares_ya_ok += 1
                # Igual actualizamos Lugar_Norm en el df para la detección de duplicados
                # (el valor ya está bien, no hay que escribir nada en Sheets)
                continue
        
            if lugar_key in mapeo_lugares:
                lugar_norm = mapeo_lugares[lugar_key]
        
                if lugar_norm != lugar_raw:
                    df_principal.at[idx, 'Lugar'] = lugar_norm
        
                    tabla_origen = dict_fuentes.get(row.get('Fuente'))
                    if tabla_origen:
                        print(f"  ✏️ Normalizando '{lugar_raw}' → '{lugar_norm}' en {tabla_origen}")
                        normalizar_lugar_en_sheet(
                            tabla_origen, "Hoja 1", row.get('Origen'), lugar_norm,
                            nombre_evento=row.get('Eventos'),   # ← nuevo
                            fecha_evento=row.get('Comienza')    # ← nuevo
                        )
                        lugares_normalizados += 1
            else:
                if lugar_key not in lugares_no_encontrados:
                    lugares_no_encontrados.append(lugar_key)
        
        df_principal['Lugar_Norm'] = df_principal['Lugar']
        
        log(f"  ✅ Normalización completada: {lugares_normalizados} lugares actualizados en sheets.")
        log(f"  ⏭️ Lugares ya normalizados (salteados): {lugares_ya_ok}")
        log(f"  ⚠️ Lugares NO encontrados en tabla de equivalencias: {len(lugares_no_encontrados)}")
        if lugares_no_encontrados:
            print(f"  📝 Detalle lugares no encontrados:")
            for lugar in lugares_no_encontrados:
                print(f"      - '{lugar}'")

        # --- 2. PROCESAMIENTO DE FECHAS ---
        print("\n📅 Procesando fechas...")
        df_principal['Comienza_DT'] = pd.to_datetime(df_principal['Comienza'], errors='coerce').dt.date
        df_principal['Comienza_DTM'] = pd.to_datetime(df_principal['Comienza'], errors='coerce')  # ← datetime completo
        fechas_invalidas = df_principal['Comienza_DT'].isna().sum()
        print(f"  ✅ Fechas procesadas. Fechas inválidas/nulas: {fechas_invalidas}")

        duplicados_para_registro = []
        indices_ya_agrupados = set()

        # --- Calcular próximo ID ---
        print("\n🔢 Calculando próximo ID de duplicados...")
        df_hist_dups = obtener_df_de_sheets("Duplicados", "Hoja 1")
        prox_id_num = 1
        if not df_hist_dups.empty and 'id_dup' in df_hist_dups.columns:
            try:
                nums = df_hist_dups['id_dup'].astype(str).str.extract(r'(\d+)').dropna().astype(int)
                if not nums.empty:
                    prox_id_num = int(nums.max()) + 1
            except:
                prox_id_num = 1
        print(f"  📌 Próximo ID de duplicado: {prox_id_num}")

        # --- 3. BUCLE DE DETECCIÓN DE DUPLICADOS ---
        print("\n🔍 Iniciando detección de duplicados...")
        prioridad_fuentes = {fuente: i for i, fuente in enumerate(dict_fuentes.keys())}
        grupos_encontrados = 0

        for i in range(len(df_principal)):
            if i in indices_ya_agrupados:
                continue

            fila_a = df_principal.iloc[i]
            if pd.isna(fila_a['Comienza_DT']):
                continue

            grupo_actual_indices = [i]
            for j in range(i + 1, len(df_principal)):
                if j in indices_ya_agrupados:
                    continue
                fila_b = df_principal.iloc[j]

                mismo_lugar = (str(fila_a['Lugar_Norm']) == str(fila_b['Lugar_Norm'])) and fila_a['Lugar_Norm'] != ""
                misma_fecha = (fila_a['Comienza_DT'] == fila_b['Comienza_DT'])

                FUENTE_CON_HORA = "Agencia Turismo Cba"
                ambos_agencia = (
                    str(fila_a.get('Fuente', '')) == FUENTE_CON_HORA and
                    str(fila_b.get('Fuente', '')) == FUENTE_CON_HORA
                )
                if ambos_agencia:
                    misma_hora = (fila_a['Comienza_DTM'] == fila_b['Comienza_DTM'])
                    es_duplicado = mismo_lugar and misma_fecha and misma_hora
                else:
                    es_duplicado = mismo_lugar and misma_fecha
                
                if es_duplicado:
                    grupo_actual_indices.append(j)

            if len(grupo_actual_indices) > 1:
                grupos_encontrados += 1
                filas_grupo = df_principal.iloc[grupo_actual_indices].copy()
                filas_grupo['prioridad'] = filas_grupo['Fuente'].map(lambda x: prioridad_fuentes.get(x, 99))
                filas_grupo = filas_grupo.sort_values(by='prioridad', ascending=True)

                fuente_ganadora = filas_grupo.iloc[0]['Fuente']
                print(f"  🔁 Grupo #{grupos_encontrados} (ID {prox_id_num}): {len(filas_grupo)} eventos "
                      f"en '{filas_grupo.iloc[0]['Lugar_Norm']}' "
                      f"el {filas_grupo.iloc[0]['Comienza_DT']} → Ganador: {fuente_ganadora}")

                letras = "ABCDEFGHIJKL"
                for idx, (original_idx, row) in enumerate(filas_grupo.iterrows()):
                    indices_ya_agrupados.add(original_idx)

                    ev = row.copy()
                    ev['id_dup'] = f"{prox_id_num}{letras[idx]}"
                    duplicados_para_registro.append(ev.drop(['Lugar_Norm', 'Comienza_DT','Comienza_DTM', 'prioridad'], errors='ignore'))

                    if idx > 0:
                        tabla_dest = dict_fuentes.get(ev['Fuente'])
                        if tabla_dest:
                            print(f"    🗑️ Eliminando duplicado ({ev['Fuente']}): '{ev.get('Eventos', '')}'")
                            borrar_fila_por_origen(tabla_dest, "Hoja 1", ev['Origen'])

                prox_id_num += 1

        print(f"\n  ✅ Detección completada: {grupos_encontrados} grupos de duplicados encontrados.")

        # --- 4. GUARDAR DUPLICADOS ---
        print("\n💾 Guardando resultados...")
        if duplicados_para_registro:
            df_final = pd.DataFrame(duplicados_para_registro)
            subir_a_google_sheets(df_final, "Duplicados", "Hoja 1")
            print(f"  ✅ {len(duplicados_para_registro)} registros subidos a hoja 'Duplicados'.")
        else:
            print("  ✨ No se hallaron duplicados para procesar.")

        # --- 5. GUARDAR RECHAZADOS ---
        if not df_rechazados.empty:
            subir_a_google_sheets(df_rechazados, 'Rechazados', 'Eventos')
            print(f"  ✅ {len(df_rechazados)} eventos rechazados subidos a hoja 'Rechazados'.")
        else:
            print("  ✨ No hubo eventos rechazados por blacklist.")

        print("\n🏁 Proceso finalizado correctamente.")

    except Exception as e:
        import traceback
        print(f"💥 ERROR en procesar_duplicados: {e}")
        print(traceback.format_exc())

indices_processed = set()
log('')
log('Detección y procesamiento de duplicados')
procesar_duplicados_y_normalizar()

    # --- FUNCIONES AUXILIARES INTERNAS ---
def obtener_df_de_sheets(nombre_tabla, nombre_hoja):
    import os, json, gspread
    from google.oauth2 import service_account
    secreto_json = os.environ.get('GCP_SERVICE_ACCOUNT_JSON')
    if not secreto_json:
        print(f"  ❌ No se encontró la variable de entorno GCP_SERVICE_ACCOUNT_JSON")
        return pd.DataFrame()
    try:
        info_claves = json.loads(secreto_json)
        creds = service_account.Credentials.from_service_account_info(
            info_claves, scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        client = gspread.authorize(creds)
        sheet = client.open(nombre_tabla).worksheet(nombre_hoja)
        data = sheet.get_all_values()
        if len(data) > 1:
            df = pd.DataFrame(data[1:], columns=data[0])
            print(f"  📗 Sheets OK: '{nombre_tabla}' / '{nombre_hoja}' → {len(df)} filas")
            return df
        else:
            print(f"  ⚠️ Hoja vacía o sin datos: '{nombre_tabla}' / '{nombre_hoja}'")
            return pd.DataFrame()
    except Exception as e:
        print(f"  ❌ Error leyendo '{nombre_tabla}' / '{nombre_hoja}': {e}")
        return pd.DataFrame()

# --- 6. SNAPSHOT JSON EN DRIVE ---
print("\n🗂️ Generando snapshot JSON en Drive...")

import os, json as json_lib, io, time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

MAX_INTENTOS = 10
ESPERA_BASE = 15  # segundos

def obtener_df_con_reintentos(sheet_name, tab_name, max_intentos=MAX_INTENTOS):
    """Lee el DataFrame con reintentos ante errores 429."""
    for intento in range(1, max_intentos + 1):
        try:
            df = obtener_df_de_sheets(sheet_name, tab_name)
            if not df.empty:
                print(f"  ✅ DataFrame leído correctamente ({len(df)} filas) en intento {intento}")
                return df
            else:
                print(f"  ⚠️ DataFrame vacío en intento {intento}")
        except Exception as e:
            espera = ESPERA_BASE * intento  # espera lineal creciente: 15s, 30s, 45s...
            if "429" in str(e) or "Quota" in str(e):
                print(f"  ⏳ [Intento {intento}/{max_intentos}] Cuota excedida. Esperando {espera}s... ({e})")
            else:
                print(f"  ❌ [Intento {intento}/{max_intentos}] Error inesperado: {e}. Esperando {espera}s...")
            if intento < max_intentos:
                time.sleep(espera)
            else:
                print("  ❌ Se agotaron los intentos para leer el DataFrame.")
                raise
    return None

def subir_json_con_reintentos(drive_service, contenido_json, nombre_archivo, carpeta_id, max_intentos=MAX_INTENTOS):
    """Sube o actualiza el JSON en Drive con reintentos."""
    for intento in range(1, max_intentos + 1):
        try:
            buffer = io.BytesIO(contenido_json.encode('utf-8'))
            media = MediaIoBaseUpload(buffer, mimetype='application/json', resumable=False)

            resultado = drive_service.files().list(
                q=f"name='{nombre_archivo}' and '{carpeta_id}' in parents and trashed=false",
                fields="files(id, name)"
            ).execute()
            archivos = resultado.get('files', [])

            if archivos:
                file_id = archivos[0]['id']
                drive_service.files().update(fileId=file_id, media_body=media).execute()
                print(f"  ✅ [Intento {intento}] Snapshot ACTUALIZADO en Drive: {nombre_archivo}")
            else:
                drive_service.files().create(
                    body={
                        'name': nombre_archivo,
                        'mimeType': 'application/json',
                        'parents': [carpeta_id]
                    },
                    media_body=media
                ).execute()
                print(f"  ✅ [Intento {intento}] Snapshot CREADO en Drive: {nombre_archivo}")
            return True  # éxito

        except Exception as e:
            espera = ESPERA_BASE * intento
            if "429" in str(e) or "Quota" in str(e):
                print(f"  ⏳ [Intento {intento}/{max_intentos}] Cuota Drive excedida. Esperando {espera}s... ({e})")
            else:
                print(f"  ❌ [Intento {intento}/{max_intentos}] Error al subir: {e}. Esperando {espera}s...")
            if intento < max_intentos:
                time.sleep(espera)
            else:
                print("  ❌ Se agotaron los intentos para subir el JSON.")
                raise
    return False

try:
    secreto_json = os.environ.get('GCP_SERVICE_ACCOUNT_JSON')
    info_claves = json_lib.loads(secreto_json)
    creds = service_account.Credentials.from_service_account_info(
        info_claves,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )

    # 1. Leer el DataFrame con reintentos
    df_final_limpio = obtener_df_con_reintentos("Entradas auto", "Eventos")

    if df_final_limpio is None or df_final_limpio.empty:
        print("  ❌ No se pudo obtener el DataFrame. Abortando snapshot.")
    else:
        df_final_limpio = df_final_limpio.dropna(subset='ID')
        df_final_limpio = df_final_limpio[df_final_limpio['ID'].astype(str).str.strip() != ""]
        registros = df_final_limpio.to_dict(orient='records')
        contenido_json = json_lib.dumps(registros, ensure_ascii=False, indent=2)
        print(f"  📦 JSON generado: {len(registros)} eventos, {len(contenido_json)} bytes")

        # 2. Subir a Drive con reintentos
        drive_service = build('drive', 'v3', credentials=creds)
        nombre_archivo = "snapshot_eventos.json"
        CARPETA_ID = "1m2NZgQXznGYQU7ZAom05_2Ibu2PogyyC"

        # subir_json_con_reintentos(drive_service, contenido_json, nombre_archivo, CARPETA_ID)
        #log(f"  ✅ Snapshot finalizado: {nombre_archivo} → {len(registros)} eventos")

except Exception as e:
    import traceback
    print(f"  ❌ Error fatal en snapshot JSON: {e}")
    print(traceback.format_exc())




destinatarios=['rmansilla@cordobaacelera.com.ar']
destinatarios=['rmansilla@cordobaacelera.com.ar','meabeldano@cordobaacelera.com.ar','pgonzalez@cordobaacelera.com.ar']
contenido_final_log = log_buffer.getvalue()
enviar_log_smtp(contenido_final_log, destinatarios)






































































































































