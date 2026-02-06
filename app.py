#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import json
import requests
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from pyproj import Proj
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from functools import lru_cache

# ============================================================
# CONFIG
# ============================================================
PORT = int(os.environ.get("PORT", 8050))
MM = 1000.0
WINDOW_SIZE = 30  # dias para a média móvel

# Configurações para remoção de outliers
OUTLIER_CONFIG = {
    'method': 'iqr',  # 'iqr' ou 'zscore'
    'threshold': 3.0,  # Para zscore: número de desvios padrão
    'iqr_multiplier': 1.5,  # Para IQR: multiplicador do IQR
    'use_detrended': True,  # Usar série detrended para detecção
    'apply_to_all': True  # Aplicar a mesma máscara a todas as componentes
}

# Configurações EOTS Loading Service
EOTS_CONFIG = {
    'base_urls': {
        'CF': 'http://loading.u-strasbg.fr/ITRF/CF',
        'CM': 'http://loading.u-strasbg.fr/ITRF/CM'
    },
    'model_folders': {
        'ERA5': 'ERA5_hydro_daily',
        'GLDAS2': 'GLDAS2_daily',
        'GRACE': 'GRACE',
        'MERRA2': 'MERRA2_hyd_daily'
    },
    'cache_timeout': 3600  # 1 hora em segundos
}

# Google Drive folder IDs extracted from share links
GDRIVE_FOLDERS = {
    "linear": "1UQdyJahGg1gxb2WSUFs1ge9wUYUPXq93",
    "cm": "1Xdcp8j4m4VjOwDe3ryOdNm3ge2aVqmLf",
    "cf": "1Sc8g2GdEodO7NAeDZWkbM1MlbFNA_T5y",
}

GDRIVE_STATIONS_FILE = "17mi5FA44LvnWr-50-bLgrdbrBsuMU-bK"

# ============================================================
# GOOGLE DRIVE UTILITIES - IMPROVED
# ============================================================
def get_gdrive_download_url(file_id):
    """Converte ID do arquivo Google Drive para URL de download direto."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def get_gdrive_file_list_via_api(folder_id):
    """
    Tenta listar arquivos usando a API pública do Google Drive (sem auth).
    Isso funciona para pastas públicas.
    """
    try:
        url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            return {}
        
        pattern = r'/file/d/([a-zA-Z0-9_-]+)/view[^>]*>([^<]+\.pfiles)'
        matches = re.findall(pattern, response.text)
        
        files = {}
        for file_id, filename in matches:
            station_id = filename.replace('.pfiles', '').strip()
            files[station_id] = file_id
            
        if len(files) > 0:
            print(f"Found {len(files)} files via embedded view")
            return files
        
        pattern2 = r'"([a-zA-Z0-9_-]{25,})"[^}]*?"([^"]+\.pfiles)"'
        matches2 = re.findall(pattern2, response.text)
        
        for file_id, filename in matches2:
            station_id = filename.replace('.pfiles', '').strip()
            if station_id not in files:
                files[station_id] = file_id
        
        return files
        
    except Exception as e:
        print(f"Error listing folder {folder_id}: {e}")
        return {}

def parse_pfile_content(content):
    """Parseia o conteúdo de um arquivo .pfiles."""
    rows = []
    for line in content.split('\n'):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) >= 6:
            try:
                rows.append({
                    "time": float(p[0]),
                    "lon": float(p[3]),
                    "lat": float(p[4]),
                    "hgt": float(p[5])
                })
            except (ValueError, IndexError):
                continue
    
    return pd.DataFrame(rows)

# ============================================================
# LOAD STATIONS FROM GOOGLE DRIVE
# ============================================================
def load_stations():
    """Carrega a lista de estações do Google Drive."""
    try:
        url = get_gdrive_download_url(GDRIVE_STATIONS_FILE)
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        stations = pd.read_csv(
            io.StringIO(response.text),
            sep=r"\s+",
            header=None,
            names=["id", "lon", "lat", "hgt"],
            dtype={"id": str}
        )
        stations["id"] = stations["id"].str.upper()
        
        print(f"✅ Loaded {len(stations)} stations from Google Drive")
        return stations
        
    except Exception as e:
        print(f"❌ Erro ao carregar estações: {e}")
        
        return pd.DataFrame({
            "id": ["ERROR"],
            "lon": [-84.0],
            "lat": [44.0],
            "hgt": [200.0]
        })

stations = load_stations()

# Cache global para file IDs
file_index = {"linear": {}, "cm": {}, "cf": {}}
index_loaded = {"linear": False, "cm": False, "cf": False}

# Cache para dados EOTS
eots_cache = {}

# Tenta carregar índice pre-gerado (se existir)
try:
    if os.path.exists("file_index.json"):
        with open("file_index.json") as f:
            file_index = json.load(f)
            for source in file_index:
                index_loaded[source] = True
            print("✅ Loaded file index from file_index.json")
except:
    pass

# ============================================================
# EOTS LOADING SERVICE UTILITIES
# ============================================================
def get_eots_loading_data(station_id, model, frame):
    """
    Baixa dados de loading do EOTS para uma estação específica.
    
    Args:
        station_id: ID da estação (4 caracteres, ex: "DUBO")
        model: 'ERA5', 'GLDAS2', 'GRACE', 'MERRA2'
        frame: 'CF' ou 'CM'
    
    Returns:
        DataFrame com colunas: time (anos), east_mm, north_mm, up_mm
    """
    cache_key = f"{station_id}_{model}_{frame}"
    
    # Verifica cache
    if cache_key in eots_cache:
        print(f"📦 Cache hit for {cache_key}")
        return eots_cache[cache_key]
    
    try:
        # Primeiro, precisamos listar os arquivos na pasta para encontrar
        # o nome correto do arquivo para esta estação
        
        base_url = EOTS_CONFIG['base_urls'][frame]
        model_folder = EOTS_CONFIG['model_folders'][model]
        list_url = f"{base_url}/{model_folder}/"
        
        print(f"🔍 Searching for {station_id} in {model}/{frame}...")
        
        # Tenta listar os arquivos na pasta
        response = requests.get(list_url, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Cannot list directory: {list_url}")
            return pd.DataFrame(columns=['time', 'east_mm', 'north_mm', 'up_mm'])
        
        # Procura por arquivos que começam com o ID da estação
        # Padrão: DUBO_XXXXXXXXX_NEU_daily.era5 ou DUBO_XXXXXXXXX_NEU.grace
        html_content = response.text
        pattern = rf'({station_id}_[A-Za-z0-9]+_NEU[^"]*)\.(era5|gldas2|grace|merra2)'
        matches = re.findall(pattern, html_content)
        
        if not matches:
            print(f"❌ No file found for {station_id} in {model}/{frame}")
            return pd.DataFrame(columns=['time', 'east_mm', 'north_mm', 'up_mm'])
        
        # Pega o primeiro arquivo encontrado
        filename, ext = matches[0]
        full_filename = f"{filename}.{ext}"
        url = f"{base_url}/{model_folder}/{full_filename}"
        
        print(f"🌍 Downloading EOTS data: {url}")
        
        # Baixa o arquivo
        file_response = requests.get(url, timeout=30)
        file_response.raise_for_status()
        
        # Parse do conteúdo
        lines = file_response.text.strip().split('\n')
        data = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        mjd = float(parts[0])  # MJD
                        east = float(parts[1])  # mm
                        north = float(parts[2])  # mm
                        up = float(parts[3])    # mm
                        
                        # Converte MJD para anos (MJD 51544 = 2000.0)
                        year = 2000.0 + (mjd - 51544.0) / 365.25
                        
                        data.append([year, east, north, up])
                    except (ValueError, IndexError):
                        continue
        
        if data:
            df = pd.DataFrame(data, columns=['time', 'east_mm', 'north_mm', 'up_mm'])
            eots_cache[cache_key] = df
            print(f"✅ Loaded {len(df)} points for {cache_key}")
            return df
        else:
            print(f"⚠️ No data found in file for {cache_key}")
            return pd.DataFrame(columns=['time', 'east_mm', 'north_mm', 'up_mm'])
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading EOTS data for {cache_key}: {e}")
        return pd.DataFrame(columns=['time', 'east_mm', 'north_mm', 'up_mm'])
    except Exception as e:
        print(f"❌ Unexpected error for {cache_key}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['time', 'east_mm', 'north_mm', 'up_mm'])

def preload_eots_data(station_id):
    """
    Pré-carrega dados EOTS para uma estação em background.
    """
    def preload_task():
        models = ['ERA5', 'GLDAS2', 'GRACE', 'MERRA2']
        frames = ['CF', 'CM']
        
        print(f"🚀 Preloading EOTS data for {station_id}...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for model in models:
                for frame in frames:
                    futures.append(
                        executor.submit(get_eots_loading_data, station_id, model, frame)
                    )
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠️ Error in preload: {e}")
        
        print(f"✅ EOTS preload complete for {station_id}")
    
    # Inicia em thread separada
    thread = threading.Thread(target=preload_task, daemon=True)
    thread.start()

# ============================================================
# GNSS UTILITIES
# ============================================================
@lru_cache(maxsize=256)
def read_pfile_from_gdrive(source, station):
    """Lê arquivo .pfiles do Google Drive."""
    folder_id = GDRIVE_FOLDERS[source]
    
    if not index_loaded[source]:
        print(f"Loading index for {source}...")
        file_index[source] = get_gdrive_file_list_via_api(folder_id)
        index_loaded[source] = True
        print(f"Found {len(file_index[source])} files in {source}")
    
    file_ref = file_index[source].get(station)
    
    if file_ref is None:
        print(f"File not found for {station} in {source}")
        return pd.DataFrame(columns=["time", "lon", "lat", "hgt"])
    
    try:
        if file_ref.startswith("http"):
            url = file_ref
        else:
            url = get_gdrive_download_url(file_ref)
        
        print(f"Downloading {station} from {source}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        return parse_pfile_content(response.text)
        
    except Exception as e:
        print(f"Error downloading {station}: {e}")
        return pd.DataFrame(columns=["time", "lon", "lat", "hgt"])

def lonlat2local_mm(lon, lat, hgt):
    E, N = [], []
    for lo, la in zip(lon, lat):
        cm = int(lo / 3) * 3
        proj = Proj(
            proj="tmerc", lon_0=cm, lat_0=0,
            k=0.9999, x_0=250000,
            ellps="WGS84", units="m"
        )
        e, n = proj(lo, la)
        E.append(e)
        N.append(n)

    E = (np.array(E) - np.nanmean(E)) * MM
    N = (np.array(N) - np.nanmean(N)) * MM
    U = (np.array(hgt) - np.nanmean(hgt)) * MM
    return E, N, U

def trend_and_velocity(x, t):
    G = np.column_stack([np.ones(len(t)), t - t.mean()])
    m, _, _, _ = lstsq(G, x)
    return G @ m, m[1]

def apply_smoothing(x, t, window_days=WINDOW_SIZE, sigma=1.0):
    window_years = window_days / 365.25
    
    idx_sorted = np.argsort(t)
    t_sorted = t[idx_sorted]
    x_sorted = x[idx_sorted]
    
    x_smoothed = np.zeros_like(x_sorted)
    
    for i in range(len(t_sorted)):
        mask = np.abs(t_sorted - t_sorted[i]) <= window_years / 2
        if np.sum(mask) > 0:
            x_smoothed[i] = np.mean(x_sorted[mask])
        else:
            x_smoothed[i] = x_sorted[i]
    
    x_smoothed = gaussian_filter1d(x_smoothed, sigma=sigma)
    
    x_final = np.zeros_like(x)
    x_final[idx_sorted] = x_smoothed
    
    return x_final

def detect_outliers(data, method='iqr', threshold=3.0, iqr_multiplier=1.5):
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        mask = z_scores < threshold
        print(f"Z-score method: {np.sum(~mask)} outliers detected (threshold: {threshold}σ)")
    
    elif method == 'iqr':
        Q1 = np.nanpercentile(data, 25)
        Q3 = np.nanpercentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        mask = (data >= lower_bound) & (data <= upper_bound)
        print(f"IQR method: {np.sum(~mask)} outliers detected (IQR × {iqr_multiplier})")
    
    else:
        raise ValueError(f"Método desconhecido: {method}")
    
    return mask

def remove_outliers(t, E, N, U, E_d, N_d, U_d, config=OUTLIER_CONFIG):
    if config['use_detrended']:
        detection_series = [E_d, N_d, U_d]
        print("Using detrended series for outlier detection")
    else:
        detection_series = [E, N, U]
        print("Using original series for outlier detection")
    
    masks = []
    for i, series in enumerate(detection_series):
        mask = detect_outliers(
            series, 
            method=config['method'],
            threshold=config['threshold'],
            iqr_multiplier=config['iqr_multiplier']
        )
        masks.append(mask)
    
    if config['apply_to_all']:
        combined_mask = masks[0] & masks[1] & masks[2]
        print(f"Combined mask: keeping {np.sum(combined_mask)} of {len(t)} points")
    else:
        combined_mask = np.ones_like(t, dtype=bool)
        print("Using individual masks for each component")
    
    t_filtered = t[combined_mask]
    E_filtered = E[combined_mask]
    N_filtered = N[combined_mask]
    U_filtered = U[combined_mask]
    E_d_filtered = E_d[combined_mask]
    N_d_filtered = N_d[combined_mask]
    U_d_filtered = U_d[combined_mask]
    
    removed = len(t) - len(t_filtered)
    print(f"Removed {removed} outliers ({removed/len(t)*100:.1f}% of data)")
    
    return {
        't': t_filtered,
        'E': E_filtered,
        'N': N_filtered,
        'U': U_filtered,
        'E_d': E_d_filtered,
        'N_d': N_d_filtered,
        'U_d': U_d_filtered,
        'mask': combined_mask,
        'removed_count': removed,
        'removed_percentage': removed/len(t)*100
    }

# ============================================================
# DASH APP
# ============================================================
app = dash.Dash(__name__)
app.title = "GNSS Great Lakes Viewer"
server = app.server

# ============================================================
# MAP
# ============================================================
def make_map(show_labels=False, zoomed_station=None):
    fig = go.Figure()
    
    fig.add_scattermapbox(
        lon=stations.lon,
        lat=stations.lat,
        customdata=stations.id,
        text=stations.id if show_labels else None,
        mode="markers+text" if show_labels else "markers",
        marker=dict(
            size=8, 
            color="crimson",
            opacity=0.7
        ),
        hovertemplate="Station: %{customdata}<extra></extra>",
        name="All Stations"
    )
    
    if zoomed_station is not None:
        station_info = stations[stations["id"] == zoomed_station]
        if not station_info.empty:
            station_lon = station_info["lon"].iloc[0]
            station_lat = station_info["lat"].iloc[0]
            
            fig.add_scattermapbox(
                lon=[station_lon],
                lat=[station_lat],
                customdata=[zoomed_station],
                text=[zoomed_station] if show_labels else None,
                mode="markers+text" if show_labels else "markers",
                marker=dict(
                    size=15,
                    color="blue",
                    opacity=1.0
                ),
                hovertemplate="<b>Selected:</b> %{customdata}<extra></extra>",
                name="Selected Station"
            )
            
            map_center = dict(lon=station_lon, lat=station_lat)
            zoom_level = 8
        else:
            map_center = dict(lon=-84, lat=44)
            zoom_level = 4.2
    else:
        map_center = dict(lon=-84, lat=44)
        zoom_level = 4.2
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=map_center,
            zoom=zoom_level
        ),
        height=650,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ============================================================
# LAYOUT
# ============================================================
app.layout = html.Div(
    style={"width": "1200px", "margin": "auto"},
    children=[

        dcc.Store(id="ui-state", data={
            "station": None,
            "source": "linear",
            "view": "original",
            "smooth": True,
            "remove_outliers": False,
            "zoomed_station": None,
            "original_data": None,
            "filtered_data": None,
            "eots_model": None,  # Modelo EOTS selecionado
            "eots_frame": None   # Frame EOTS (CF/CM) - herdado do source quando possível
        }),

        html.H2("MSU Geodesy Lab - The Great Lakes GNSS Stations"),
        
        html.Div([
            html.P("⚠️ Note: Due to Google Drive limitations, file loading may be slow or fail. If data doesn't load, please try another station or source.", 
                   style={"fontSize": "12px", "color": "#666", "fontStyle": "italic"}),
            html.P(f"Outlier detection: {OUTLIER_CONFIG['method'].upper()} method, threshold: {OUTLIER_CONFIG['threshold']}, using {'detrended' if OUTLIER_CONFIG['use_detrended'] else 'original'} series",
                   style={"fontSize": "11px", "color": "#888", "fontStyle": "italic"})
        ]),

        html.Div([
            html.Button("Show IDs", id="btn-labels"),
            html.Span(" | ", style={"margin": "0 10px"}),
            html.Label("Search ID: ", style={"marginRight": "5px"}),
            dcc.Input(
                id="station-search-input",
                type="text",
                placeholder="Enter 4-character station ID",
                style={
                    "width": "150px",
                    "marginRight": "5px",
                    "padding": "5px"
                },
                maxLength=4
            ),
            html.Button("Search", id="btn-search", n_clicks=0),
            html.Div(id="search-status", style={"marginLeft": "10px", "display": "inline"})
        ], style={"marginBottom": "15px"}),

        dcc.Graph(id="station-map", figure=make_map(), config={"scrollZoom": True}),

        html.Hr(),

        html.Div([
            html.Div([
                html.Label("Data Source:", style={"fontWeight": "bold", "marginRight": "10px"}),
                html.Button("Linear", id="src-linear"),
                html.Button("CM", id="src-cm"),
                html.Button("CF", id="src-cf"),
            ], style={"marginBottom": "15px"}),
            
            html.Div([
                html.Label("Hydro Loading (EOTS):", style={"fontWeight": "bold", "marginRight": "10px"}),
                html.Button("ERA5", id="eots-era5"),
                html.Button("GRACE", id="eots-grace"),
                html.Button("GLDAS2", id="eots-gldas2"),
                html.Button("MERRA2", id="eots-merra2"),
                html.Button("None", id="eots-none"),
                html.Span(id="eots-status", style={"marginLeft": "10px", "fontSize": "12px", "color": "#666"})
            ], style={"marginBottom": "15px"}),

            html.Div([
                html.Label("View Options:", style={"fontWeight": "bold", "marginRight": "10px"}),
                html.Button("Original", id="btn-original"),
                html.Button("Detrended", id="btn-detrended"),
            ], style={"marginBottom": "15px"}),

            html.Div([
                html.Label("Processing:", style={"fontWeight": "bold", "marginRight": "10px"}),
                html.Button("Smooth ON", id="btn-smooth-on"),
                html.Button("Smooth OFF", id="btn-smooth-off"),
                html.Button("Remove Outliers", id="btn-outliers-on"),
                html.Button("Keep Outliers", id="btn-outliers-off"),
            ]),
        ], style={"marginBottom": "20px", "padding": "10px", "backgroundColor": "#f5f5f5", "borderRadius": "5px"}),

        html.Div(id="timeseries-container")
    ]
)

# ============================================================
# CALLBACKS
# ============================================================
@app.callback(
    [Output("station-map", "figure"),
     Output("ui-state", "data", allow_duplicate=True),
     Output("search-status", "children")],
    [Input("btn-labels", "n_clicks"),
     Input("btn-search", "n_clicks")],
    [State("station-search-input", "value"),
     State("ui-state", "data")],
    prevent_initial_call=True
)
def handle_search_and_labels(labels_clicks, search_clicks, search_input, state):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger == "btn-labels":
        show = (labels_clicks or 0) % 2 == 1
        return make_map(show_labels=show, zoomed_station=state.get("zoomed_station")), state, ""
    
    elif trigger == "btn-search":
        if not search_input:
            return make_map(zoomed_station=state.get("zoomed_station")), state, html.Span("Please enter a station ID", style={"color": "red"})
        
        search_id = search_input.strip().upper()
        
        if search_id not in stations["id"].values:
            return make_map(zoomed_station=state.get("zoomed_station")), state, html.Span(f"Station '{search_id}' not found", style={"color": "red"})
        
        new_state = state.copy()
        new_state["station"] = search_id
        new_state["zoomed_station"] = search_id
        new_state["original_data"] = None
        new_state["filtered_data"] = None
        new_state["eots_model"] = None  # Reseta modelo EOTS ao mudar estação
        
        # Pré-carrega dados EOTS em background
        preload_eots_data(search_id)
        
        return make_map(zoomed_station=search_id), new_state, html.Span(f"Showing station: {search_id}", style={"color": "green"})

@app.callback(
    [Output("ui-state", "data"),
     Output("eots-status", "children")],
    [
        Input("station-map", "clickData"),
        Input("src-linear", "n_clicks"),
        Input("src-cm", "n_clicks"),
        Input("src-cf", "n_clicks"),
        Input("btn-original", "n_clicks"),
        Input("btn-detrended", "n_clicks"),
        Input("btn-smooth-on", "n_clicks"),
        Input("btn-smooth-off", "n_clicks"),
        Input("btn-outliers-on", "n_clicks"),
        Input("btn-outliers-off", "n_clicks"),
        Input("eots-era5", "n_clicks"),
        Input("eots-grace", "n_clicks"),
        Input("eots-gldas2", "n_clicks"),
        Input("eots-merra2", "n_clicks"),
        Input("eots-none", "n_clicks"),
    ],
    State("ui-state", "data"),
    prevent_initial_call=True
)
def update_state(*args):
    ctx = dash.callback_context
    state = args[-1]
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "station-map":
        clicked_station = ctx.triggered[0]["value"]["points"][0]["customdata"]
        state["station"] = clicked_station
        state["zoomed_station"] = clicked_station
        state["view"] = "original"
        state["smooth"] = True
        state["remove_outliers"] = False
        state["original_data"] = None
        state["filtered_data"] = None
        state["eots_model"] = None
        # Pré-carrega dados EOTS
        preload_eots_data(clicked_station)
        return state, ""
    
    elif trigger.startswith("src-"):
        source = trigger.split("-")[1]
        state["source"] = source
        state["original_data"] = None
        state["filtered_data"] = None
        
        # Se EOTS ativo e source é CF ou CM, ajusta frame
        if state["eots_model"] is not None and source in ["cm", "cf"]:
            state["eots_frame"] = source.upper()
            return state, f"Source: {source.upper()}, EOTS: {state['eots_model']} ({state['eots_frame']})"
        return state, f"Source: {source.upper()}"
    
    elif trigger.startswith("eots-"):
        model = trigger.split("-")[1].upper()
        
        if model == "NONE":
            state["eots_model"] = None
            state["eots_frame"] = None
            return state, "EOTS: Disabled"
        else:
            state["eots_model"] = model
            # Se source atual é CF ou CM, usa como frame, senão usa CF como default
            if state["source"] in ["cm", "cf"]:
                state["eots_frame"] = state["source"].upper()
                return state, f"EOTS: {model} ({state['eots_frame']})"
            else:
                state["eots_frame"] = "CF"  # Default
                return state, f"EOTS: {model} (CF - default)"
    
    elif trigger == "btn-original":
        state["view"] = "original"
        return state, state.get("eots_status", "")
    elif trigger == "btn-detrended":
        state["view"] = "detrended"
        return state, state.get("eots_status", "")
    elif trigger == "btn-smooth-on":
        state["smooth"] = True
        return state, state.get("eots_status", "")
    elif trigger == "btn-smooth-off":
        state["smooth"] = False
        return state, state.get("eots_status", "")
    elif trigger == "btn-outliers-on":
        state["remove_outliers"] = True
        return state, state.get("eots_status", "")
    elif trigger == "btn-outliers-off":
        state["remove_outliers"] = False
        return state, state.get("eots_status", "")
    
    return state, ""

# ============================================================
# RENDER TIMESERIES
# ============================================================
@app.callback(
    Output("timeseries-container", "children"),
    Input("ui-state", "data")
)
def render_ts(state):
    if state["station"] is None:
        return html.Div("Click on a station or search for a station ID to view its time series.")

    try:
        df = read_pfile_from_gdrive(state["source"], state["station"])
        
        if df.empty:
            return html.Div([
                html.H3(f"Could not load data for station {state['station']}"),
                html.P("Possible reasons:"),
                html.Ul([
                    html.Li("File not found in the selected source (Linear/CM/CF)"),
                    html.Li("Google Drive temporarily blocking automated access"),
                    html.Li("Station ID mismatch between map and data files")
                ]),
                html.P("Try:"),
                html.Ul([
                    html.Li("Selecting a different source (Linear/CM/CF buttons above)"),
                    html.Li("Clicking on a different station"),
                    html.Li("Waiting a moment and trying again")
                ])
            ])
        
        # Processa dados básicos
        t = df.time.values
        E, N, U = lonlat2local_mm(df.lon, df.lat, df.hgt)
        
        # Calcula tendência
        E_tr, vE = trend_and_velocity(E, t)
        N_tr, vN = trend_and_velocity(N, t)
        U_tr, vU = trend_and_velocity(U, t)
        
        # Calcula série detrended
        E_d, N_d, U_d = E - E_tr, N - N_tr, U - U_tr
        
        # Armazena dados originais se ainda não armazenados
        if state.get("original_data") is None:
            state["original_data"] = {
                't': t, 'E': E, 'N': N, 'U': U,
                'E_d': E_d, 'N_d': N_d, 'U_d': U_d,
                'vE': vE, 'vN': vN, 'vU': vU
            }
        
        # Remove outliers se necessário
        if state.get("remove_outliers", False):
            # Se ainda não filtrou os dados, faz a filtragem
            if state.get("filtered_data") is None:
                filtered = remove_outliers(t, E, N, U, E_d, N_d, U_d, OUTLIER_CONFIG)
                # Recalcula velocidade com dados filtrados
                E_tr_f, vE_f = trend_and_velocity(filtered['E'], filtered['t'])
                N_tr_f, vN_f = trend_and_velocity(filtered['N'], filtered['t'])
                U_tr_f, vU_f = trend_and_velocity(filtered['U'], filtered['t'])
                
                state["filtered_data"] = {
                    **filtered,
                    'vE': vE_f, 'vN': vN_f, 'vU': vU_f,
                    'removed_info': f"Removed {filtered['removed_count']} points ({filtered['removed_percentage']:.1f}%)"
                }
            
            # Usa dados filtrados
            data = state["filtered_data"]
            t_plot = data['t']
            E_plot = data['E'] if state["view"] == "original" else data['E_d']
            N_plot = data['N'] if state["view"] == "original" else data['N_d']
            U_plot = data['U'] if state["view"] == "original" else data['U_d']
            vE_plot, vN_plot, vU_plot = data['vE'], data['vN'], data['vU']
            outlier_info = data.get('removed_info', '')
        else:
            # Usa dados originais
            data = state["original_data"]
            t_plot = data['t']
            E_plot = data['E'] if state["view"] == "original" else data['E_d']
            N_plot = data['N'] if state["view"] == "original" else data['N_d']
            U_plot = data['U'] if state["view"] == "original" else data['U_d']
            vE_plot, vN_plot, vU_plot = data['vE'], data['vN'], data['vU']
            outlier_info = ""
        
        # Aplica suavização se solicitado
        if state["smooth"]:
            E_smooth = apply_smoothing(E_plot, t_plot)
            N_smooth = apply_smoothing(N_plot, t_plot)
            U_smooth = apply_smoothing(U_plot, t_plot)

        # Carrega dados EOTS se solicitado
        eots_data = None
        if state.get("eots_model") is not None and state.get("eots_frame") is not None:
            # Se source atual não for CF/CM, tenta usar CF como default para EOTS
            frame_to_use = state["eots_frame"]
            if state["source"] not in ["cm", "cf"] and state["eots_frame"] not in ["CF", "CM"]:
                frame_to_use = "CF"  # Default para EOTS
            
            eots_data = get_eots_loading_data(
                state["station"],
                state["eots_model"],
                frame_to_use
            )
            
            if eots_data.empty:
                print(f"⚠️ No EOTS data available for {state['station']} with {state['eots_model']}/{frame_to_use}")

        hover = (
            "Year: %{x:.4f}<br>"
            "Disp: %{y:.2f} mm"
            "<extra></extra>"
        )

        # Título do gráfico
        title = f"Station {state['station']} — {state['source'].upper()} (Window: {WINDOW_SIZE} days)"
        if state.get("remove_outliers", False):
            title += f" — Outliers removed: {outlier_info}"
        
        if state.get("eots_model") is not None:
            title += f" — EOTS: {state['eots_model']} ({state.get('eots_frame', 'CF')})"
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
            subplot_titles=[
                f"East ({vE_plot:.2f} mm/yr)",
                f"North ({vN_plot:.2f} mm/yr)",
                f"Up ({vU_plot:.2f} mm/yr)"
            ]
        )
        
        # Dados principais (GNSS)
        fig.add_scatter(x=t_plot, y=E_plot, mode="markers", 
                       hovertemplate=hover, row=1, col=1,
                       marker=dict(size=5, opacity=0.6),
                       name="GNSS Data")
        fig.add_scatter(x=t_plot, y=N_plot, mode="markers", 
                       hovertemplate=hover, row=2, col=1,
                       marker=dict(size=5, opacity=0.6),
                       name="GNSS Data")
        fig.add_scatter(x=t_plot, y=U_plot, mode="markers", 
                       hovertemplate=hover, row=3, col=1,
                       marker=dict(size=5, opacity=0.6),
                       name="GNSS Data")

        # Linha suavizada se Smooth ON
        if state["smooth"]:
            fig.add_scatter(x=t_plot, y=E_smooth, mode="lines",
                            line=dict(color="magenta", width=2), 
                            hoverinfo="skip", row=1, col=1,
                            name="Smoothed GNSS")
            fig.add_scatter(x=t_plot, y=N_smooth, mode="lines",
                            line=dict(color="magenta", width=2), 
                            hoverinfo="skip", row=2, col=1,
                            name="Smoothed GNSS")
            fig.add_scatter(x=t_plot, y=U_smooth, mode="lines",
                            line=dict(color="magenta", width=2), 
                            hoverinfo="skip", row=3, col=1,
                            name="Smoothed GNSS")

        # Dados EOTS se disponíveis
        if eots_data is not None and not eots_data.empty:
            # Para visualização EOTS, sempre usamos detrended view
            # porque os dados EOTS já são detrended
            fig.add_scatter(x=eots_data['time'], y=eots_data['east_mm'], mode="lines",
                            line=dict(color="green", width=2, dash="dash"), 
                            hoverinfo="skip", row=1, col=1,
                            name=f"EOTS {state['eots_model']}")
            fig.add_scatter(x=eots_data['time'], y=eots_data['north_mm'], mode="lines",
                            line=dict(color="green", width=2, dash="dash"), 
                            hoverinfo="skip", row=2, col=1,
                            name=f"EOTS {state['eots_model']}")
            fig.add_scatter(x=eots_data['time'], y=eots_data['up_mm'], mode="lines",
                            line=dict(color="green", width=2, dash="dash"), 
                            hoverinfo="skip", row=3, col=1,
                            name=f"EOTS {state['eots_model']}")

        fig.update_layout(
            title=title,
            height=900,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return dcc.Graph(figure=fig, config={"scrollZoom": True})
        
    except Exception as e:
        import traceback
        return html.Div([
            html.H3(f"Error loading data for station {state['station']}"),
            html.P(f"Error: {str(e)}"),
            html.Pre(traceback.format_exc(), style={"fontSize": "10px", "overflow": "auto"})
        ])

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import io
# import json
# import requests
# import numpy as np
# import pandas as pd
# from scipy.linalg import lstsq
# from scipy.ndimage import gaussian_filter1d
# from scipy import stats
# from pyproj import Proj
# import re

# import dash
# from dash import dcc, html, Input, Output, State
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# from functools import lru_cache

# # ============================================================
# # CONFIG
# # ============================================================
# PORT = int(os.environ.get("PORT", 8050))
# MM = 1000.0
# WINDOW_SIZE = 30  # dias para a média móvel

# # Configurações para remoção de outliers
# OUTLIER_CONFIG = {
#     'method': 'iqr',  # 'iqr' ou 'zscore'
#     'threshold': 3.0,  # Para zscore: número de desvios padrão
#     'iqr_multiplier': 1.5,  # Para IQR: multiplicador do IQR
#     'use_detrended': True,  # Usar série detrended para detecção
#     'apply_to_all': True  # Aplicar a mesma máscara a todas as componentes
# }

# # Google Drive folder IDs extracted from share links
# GDRIVE_FOLDERS = {
#     "linear": "1UQdyJahGg1gxb2WSUFs1ge9wUYUPXq93",
#     "cm": "1Xdcp8j4m4VjOwDe3ryOdNm3ge2aVqmLf",
#     "cf": "1Sc8g2GdEodO7NAeDZWkbM1MlbFNA_T5y",
# }

# GDRIVE_STATIONS_FILE = "17mi5FA44LvnWr-50-bLgrdbrBsuMU-bK"

# # ============================================================
# # GOOGLE DRIVE UTILITIES - IMPROVED
# # ============================================================
# def get_gdrive_download_url(file_id):
#     """Converte ID do arquivo Google Drive para URL de download direto."""
#     return f"https://drive.google.com/uc?export=download&id={file_id}"

# def get_gdrive_file_list_via_api(folder_id):
#     """
#     Tenta listar arquivos usando a API pública do Google Drive (sem auth).
#     Isso funciona para pastas públicas.
#     """
#     try:
#         # Usa a API v3 do Google Drive sem autenticação (apenas para pastas públicas)
#         # Formato: https://www.googleapis.com/drive/v3/files?q='FOLDER_ID'+in+parents&fields=files(id,name)&key=API_KEY
#         # Como não temos API key, vamos tentar uma abordagem diferente
        
#         # Tenta acessar via web scraping melhorado
#         url = f"https://drive.google.com/embeddedfolderview?id={folder_id}"
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         }
        
#         response = requests.get(url, headers=headers, timeout=30)
        
#         if response.status_code != 200:
#             return {}
        
#         # Procura por IDs de arquivo no HTML da view embutida
#         # Formato: <a href="/file/d/FILE_ID/view
#         pattern = r'/file/d/([a-zA-Z0-9_-]+)/view[^>]*>([^<]+\.pfiles)'
#         matches = re.findall(pattern, response.text)
        
#         files = {}
#         for file_id, filename in matches:
#             station_id = filename.replace('.pfiles', '').strip()
#             files[station_id] = file_id
            
#         if len(files) > 0:
#             print(f"Found {len(files)} files via embedded view")
#             return files
        
#         # Tenta outro padrão
#         pattern2 = r'"([a-zA-Z0-9_-]{25,})"[^}]*?"([^"]+\.pfiles)"'
#         matches2 = re.findall(pattern2, response.text)
        
#         for file_id, filename in matches2:
#             station_id = filename.replace('.pfiles', '').strip()
#             if station_id not in files:
#                 files[station_id] = file_id
        
#         return files
        
#     except Exception as e:
#         print(f"Error listing folder {folder_id}: {e}")
#         return {}

# def parse_pfile_content(content):
#     """Parseia o conteúdo de um arquivo .pfiles."""
#     rows = []
#     for line in content.split('\n'):
#         if line.startswith("#") or not line.strip():
#             continue
#         p = line.split()
#         if len(p) >= 6:
#             try:
#                 rows.append({
#                     "time": float(p[0]),
#                     "lon": float(p[3]),
#                     "lat": float(p[4]),
#                     "hgt": float(p[5])
#                 })
#             except (ValueError, IndexError):
#                 continue
    
#     return pd.DataFrame(rows)

# # ============================================================
# # LOAD STATIONS FROM GOOGLE DRIVE
# # ============================================================
# def load_stations():
#     """Carrega a lista de estações do Google Drive."""
#     try:
#         url = get_gdrive_download_url(GDRIVE_STATIONS_FILE)
        
#         response = requests.get(url, timeout=15)
#         response.raise_for_status()
        
#         # Lê como CSV
#         stations = pd.read_csv(
#             io.StringIO(response.text),
#             sep=r"\s+",
#             header=None,
#             names=["id", "lon", "lat", "hgt"],
#             dtype={"id": str}
#         )
#         stations["id"] = stations["id"].str.upper()
        
#         print(f"✅ Loaded {len(stations)} stations from Google Drive")
#         return stations
        
#     except Exception as e:
#         print(f"❌ Erro ao carregar estações: {e}")
        
#         # Fallback
#         return pd.DataFrame({
#             "id": ["ERROR"],
#             "lon": [-84.0],
#             "lat": [44.0],
#             "hgt": [200.0]
#         })

# stations = load_stations()

# # Cache global para file IDs
# file_index = {"linear": {}, "cm": {}, "cf": {}}
# index_loaded = {"linear": False, "cm": False, "cf": False}

# # Tenta carregar índice pre-gerado (se existir)
# try:
#     if os.path.exists("file_index.json"):
#         with open("file_index.json") as f:
#             file_index = json.load(f)
#             for source in file_index:
#                 index_loaded[source] = True
#             print("✅ Loaded file index from file_index.json")
# except:
#     pass

# # ============================================================
# # GNSS UTILITIES
# # ============================================================
# @lru_cache(maxsize=256)
# def read_pfile_from_gdrive(source, station):
#     """Lê arquivo .pfiles do Google Drive."""
#     folder_id = GDRIVE_FOLDERS[source]
    
#     # Se ainda não carregou o índice desta pasta, tenta carregar
#     if not index_loaded[source]:
#         print(f"Loading index for {source}...")
#         file_index[source] = get_gdrive_file_list_via_api(folder_id)
#         index_loaded[source] = True
#         print(f"Found {len(file_index[source])} files in {source}")
    
#     # Procura o file ID ou URL
#     file_ref = file_index[source].get(station)
    
#     if file_ref is None:
#         print(f"File not found for {station} in {source}")
#         return pd.DataFrame(columns=["time", "lon", "lat", "hgt"])
    
#     try:
#         # Se file_ref já é uma URL completa, usa direto
#         # Senão, converte file ID para URL
#         if file_ref.startswith("http"):
#             url = file_ref
#         else:
#             url = get_gdrive_download_url(file_ref)
        
#         print(f"Downloading {station} from {source}...")
#         response = requests.get(url, timeout=30)
#         response.raise_for_status()
        
#         return parse_pfile_content(response.text)
        
#     except Exception as e:
#         print(f"Error downloading {station}: {e}")
#         return pd.DataFrame(columns=["time", "lon", "lat", "hgt"])

# def lonlat2local_mm(lon, lat, hgt):
#     E, N = [], []
#     for lo, la in zip(lon, lat):
#         cm = int(lo / 3) * 3
#         proj = Proj(
#             proj="tmerc", lon_0=cm, lat_0=0,
#             k=0.9999, x_0=250000,
#             ellps="WGS84", units="m"
#         )
#         e, n = proj(lo, la)
#         E.append(e)
#         N.append(n)

#     E = (np.array(E) - np.nanmean(E)) * MM
#     N = (np.array(N) - np.nanmean(N)) * MM
#     U = (np.array(hgt) - np.nanmean(hgt)) * MM
#     return E, N, U

# def trend_and_velocity(x, t):
#     G = np.column_stack([np.ones(len(t)), t - t.mean()])
#     m, _, _, _ = lstsq(G, x)
#     return G @ m, m[1]

# def apply_smoothing(x, t, window_days=WINDOW_SIZE, sigma=1.0):
#     """
#     Aplica suavização com média móvel de janela fixa (em dias) seguida de filtro gaussiano.
    
#     Args:
#         x: array de valores
#         t: array de tempos (em anos)
#         window_days: tamanho da janela em dias
#         sigma: parâmetro sigma para o filtro gaussiano
#     """
#     # Converte window_days para anos
#     window_years = window_days / 365.25
    
#     # Ordena os dados por tempo
#     idx_sorted = np.argsort(t)
#     t_sorted = t[idx_sorted]
#     x_sorted = x[idx_sorted]
    
#     # Aplica média móvel
#     x_smoothed = np.zeros_like(x_sorted)
    
#     for i in range(len(t_sorted)):
#         # Encontra pontos dentro da janela
#         mask = np.abs(t_sorted - t_sorted[i]) <= window_years / 2
#         if np.sum(mask) > 0:
#             x_smoothed[i] = np.mean(x_sorted[mask])
#         else:
#             x_smoothed[i] = x_sorted[i]
    
#     # Aplica filtro gaussiano
#     x_smoothed = gaussian_filter1d(x_smoothed, sigma=sigma)
    
#     # Retorna aos índices originais
#     x_final = np.zeros_like(x)
#     x_final[idx_sorted] = x_smoothed
    
#     return x_final

# def detect_outliers(data, method='iqr', threshold=3.0, iqr_multiplier=1.5):
#     """
#     Detecta outliers em um conjunto de dados usando diferentes métodos.
    
#     Args:
#         data: array numpy com os dados
#         method: 'iqr' (Interquartile Range) ou 'zscore' (Z-score)
#         threshold: para zscore, número de desvios padrão
#         iqr_multiplier: para IQR, multiplicador do IQR
    
#     Returns:
#         mask: array booleano onde True indica dados NÃO outliers (para manter)
#     """
#     if method == 'zscore':
#         # Método Z-score
#         z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
#         mask = z_scores < threshold
#         print(f"Z-score method: {np.sum(~mask)} outliers detected (threshold: {threshold}σ)")
    
#     elif method == 'iqr':
#         # Método IQR (Interquartile Range)
#         Q1 = np.nanpercentile(data, 25)
#         Q3 = np.nanpercentile(data, 75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - iqr_multiplier * IQR
#         upper_bound = Q3 + iqr_multiplier * IQR
#         mask = (data >= lower_bound) & (data <= upper_bound)
#         print(f"IQR method: {np.sum(~mask)} outliers detected (IQR × {iqr_multiplier})")
    
#     else:
#         raise ValueError(f"Método desconhecido: {method}")
    
#     return mask

# def remove_outliers(t, E, N, U, E_d, N_d, U_d, config=OUTLIER_CONFIG):
#     """
#     Remove outliers das séries GNSS.
    
#     Args:
#         t: array de tempos
#         E, N, U: arrays das componentes originais
#         E_d, N_d, U_d: arrays das componentes detrended
#         config: dicionário com configurações de detecção
    
#     Returns:
#         Dicionário com dados filtrados e máscara aplicada
#     """
#     # Escolhe qual série usar para detecção de outliers
#     if config['use_detrended']:
#         detection_series = [E_d, N_d, U_d]
#         print("Using detrended series for outlier detection")
#     else:
#         detection_series = [E, N, U]
#         print("Using original series for outlier detection")
    
#     # Detecta outliers
#     masks = []
#     for i, series in enumerate(detection_series):
#         mask = detect_outliers(
#             series, 
#             method=config['method'],
#             threshold=config['threshold'],
#             iqr_multiplier=config['iqr_multiplier']
#         )
#         masks.append(mask)
    
#     # Combina máscaras
#     if config['apply_to_all']:
#         # Usa a mesma máscara para todas as componentes (AND lógico)
#         combined_mask = masks[0] & masks[1] & masks[2]
#         print(f"Combined mask: keeping {np.sum(combined_mask)} of {len(t)} points")
#     else:
#         # Mantém máscaras individuais
#         combined_mask = np.ones_like(t, dtype=bool)
#         print("Using individual masks for each component")
    
#     # Aplica máscara aos dados
#     t_filtered = t[combined_mask]
#     E_filtered = E[combined_mask]
#     N_filtered = N[combined_mask]
#     U_filtered = U[combined_mask]
#     E_d_filtered = E_d[combined_mask]
#     N_d_filtered = N_d[combined_mask]
#     U_d_filtered = U_d[combined_mask]
    
#     # Relatório
#     removed = len(t) - len(t_filtered)
#     print(f"Removed {removed} outliers ({removed/len(t)*100:.1f}% of data)")
    
#     return {
#         't': t_filtered,
#         'E': E_filtered,
#         'N': N_filtered,
#         'U': U_filtered,
#         'E_d': E_d_filtered,
#         'N_d': N_d_filtered,
#         'U_d': U_d_filtered,
#         'mask': combined_mask,
#         'removed_count': removed,
#         'removed_percentage': removed/len(t)*100
#     }

# # ============================================================
# # DASH APP
# # ============================================================
# app = dash.Dash(__name__)
# app.title = "GNSS Great Lakes Viewer"
# server = app.server

# # ============================================================
# # MAP
# # ============================================================
# def make_map(show_labels=False, zoomed_station=None):
#     """
#     Cria o mapa com opções de zoom.
    
#     Args:
#         show_labels: mostra os IDs das estações
#         zoomed_station: ID da estação para dar zoom (None para visão geral)
#     """
#     fig = go.Figure()
    
#     # Adiciona todas as estações
#     fig.add_scattermapbox(
#         lon=stations.lon,
#         lat=stations.lat,
#         customdata=stations.id,
#         text=stations.id if show_labels else None,
#         mode="markers+text" if show_labels else "markers",
#         marker=dict(
#             size=8, 
#             color="crimson",
#             opacity=0.7
#         ),
#         hovertemplate="Station: %{customdata}<extra></extra>",
#         name="All Stations"
#     )
    
#     # Configuração inicial do mapa
#     if zoomed_station is not None:
#         # Encontra a estação específica
#         station_info = stations[stations["id"] == zoomed_station]
#         if not station_info.empty:
#             station_lon = station_info["lon"].iloc[0]
#             station_lat = station_info["lat"].iloc[0]
            
#             # Destaca a estação procurada com cor diferente
#             fig.add_scattermapbox(
#                 lon=[station_lon],
#                 lat=[station_lat],
#                 customdata=[zoomed_station],
#                 text=[zoomed_station] if show_labels else None,
#                 mode="markers+text" if show_labels else "markers",
#                 marker=dict(
#                     size=15,
#                     color="blue",
#                     opacity=1.0
#                 ),
#                 hovertemplate="<b>Selected:</b> %{customdata}<extra></extra>",
#                 name="Selected Station"
#             )
            
#             # Configura o zoom para a estação
#             map_center = dict(lon=station_lon, lat=station_lat)
#             zoom_level = 8
#         else:
#             # Se estação não encontrada, mantém visão geral
#             map_center = dict(lon=-84, lat=44)
#             zoom_level = 4.2
#     else:
#         # Visão geral de todas as estações
#         map_center = dict(lon=-84, lat=44)
#         zoom_level = 4.2
    
#     fig.update_layout(
#         mapbox=dict(
#             style="open-street-map",
#             center=map_center,
#             zoom=zoom_level
#         ),
#         height=650,
#         margin=dict(l=20, r=20, t=40, b=20)
#     )
    
#     return fig

# # ============================================================
# # LAYOUT
# # ============================================================
# app.layout = html.Div(
#     style={"width": "1200px", "margin": "auto"},
#     children=[

#         dcc.Store(id="ui-state", data={
#             "station": None,
#             "source": "linear",
#             "view": "original",
#             "smooth": True,
#             "remove_outliers": False,
#             "zoomed_station": None,
#             "original_data": None,  # Para armazenar dados originais quando outliers são removidos
#             "filtered_data": None   # Para armazenar dados filtrados
#         }),

#         html.H2("MSU Geodesy Lab - The Great Lakes GNSS Stations"),
        
#         html.Div([
#             html.P("⚠️ Note: Due to Google Drive limitations, file loading may be slow or fail. If data doesn't load, please try another station or source.", 
#                    style={"fontSize": "12px", "color": "#666", "fontStyle": "italic"}),
#             html.P(f"Outlier detection: {OUTLIER_CONFIG['method'].upper()} method, threshold: {OUTLIER_CONFIG['threshold']}, using {'detrended' if OUTLIER_CONFIG['use_detrended'] else 'original'} series",
#                    style={"fontSize": "11px", "color": "#888", "fontStyle": "italic"})
#         ]),

#         html.Div([
#             html.Button("Show IDs", id="btn-labels"),
#             html.Span(" | ", style={"margin": "0 10px"}),
#             html.Label("Search ID: ", style={"marginRight": "5px"}),
#             dcc.Input(
#                 id="station-search-input",
#                 type="text",
#                 placeholder="Enter 4-character station ID",
#                 style={
#                     "width": "150px",
#                     "marginRight": "5px",
#                     "padding": "5px"
#                 },
#                 maxLength=4
#             ),
#             html.Button("Search", id="btn-search", n_clicks=0),
#             html.Div(id="search-status", style={"marginLeft": "10px", "display": "inline"})
#         ], style={"marginBottom": "15px"}),

#         dcc.Graph(id="station-map", figure=make_map(), config={"scrollZoom": True}),

#         html.Hr(),

#         html.Div([
#             html.Div([
#                 html.Button("Linear", id="src-linear"),
#                 html.Button("CM", id="src-cm"),
#                 html.Button("CF", id="src-cf"),
#             ], style={"marginBottom": "10px"}),

#             html.Div([
#                 html.Button("Original", id="btn-original"),
#                 html.Button("Detrended", id="btn-detrended"),
#             ], style={"marginBottom": "10px"}),

#             html.Div([
#                 html.Button("Smooth ON", id="btn-smooth-on"),
#                 html.Button("Smooth OFF", id="btn-smooth-off"),
#                 html.Button("Remove Outliers", id="btn-outliers-on"),
#                 html.Button("Keep Outliers", id="btn-outliers-off"),
#             ]),
#         ], style={"marginBottom": "20px"}),

#         html.Div(id="timeseries-container")
#     ]
# )

# # ============================================================
# # CALLBACKS
# # ============================================================
# @app.callback(
#     [Output("station-map", "figure"),
#      Output("ui-state", "data", allow_duplicate=True),
#      Output("search-status", "children")],
#     [Input("btn-labels", "n_clicks"),
#      Input("btn-search", "n_clicks")],
#     [State("station-search-input", "value"),
#      State("ui-state", "data")],
#     prevent_initial_call=True
# )
# def handle_search_and_labels(labels_clicks, search_clicks, search_input, state):
#     ctx = dash.callback_context
#     trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    
#     if trigger == "btn-labels":
#         # Alterna labels sem afetar o zoom
#         show = (labels_clicks or 0) % 2 == 1
#         return make_map(show_labels=show, zoomed_station=state.get("zoomed_station")), state, ""
    
#     elif trigger == "btn-search":
#         if not search_input:
#             return make_map(zoomed_station=state.get("zoomed_station")), state, html.Span("Please enter a station ID", style={"color": "red"})
        
#         search_id = search_input.strip().upper()
        
#         # Verifica se a estação existe
#         if search_id not in stations["id"].values:
#             return make_map(zoomed_station=state.get("zoomed_station")), state, html.Span(f"Station '{search_id}' not found", style={"color": "red"})
        
#         # Atualiza o estado para dar zoom na estação e mostrar os dados
#         new_state = state.copy()
#         new_state["station"] = search_id
#         new_state["zoomed_station"] = search_id
#         new_state["original_data"] = None  # Reseta dados ao mudar de estação
#         new_state["filtered_data"] = None
        
#         return make_map(zoomed_station=search_id), new_state, html.Span(f"Showing station: {search_id}", style={"color": "green"})

# @app.callback(
#     Output("ui-state", "data"),
#     [
#         Input("station-map", "clickData"),
#         Input("src-linear", "n_clicks"),
#         Input("src-cm", "n_clicks"),
#         Input("src-cf", "n_clicks"),
#         Input("btn-original", "n_clicks"),
#         Input("btn-detrended", "n_clicks"),
#         Input("btn-smooth-on", "n_clicks"),
#         Input("btn-smooth-off", "n_clicks"),
#         Input("btn-outliers-on", "n_clicks"),
#         Input("btn-outliers-off", "n_clicks"),
#     ],
#     State("ui-state", "data"),
#     prevent_initial_call=True
# )
# def update_state(*args):
#     ctx = dash.callback_context
#     state = args[-1]
#     trigger = ctx.triggered[0]["prop_id"].split(".")[0]

#     if trigger == "station-map":
#         clicked_station = ctx.triggered[0]["value"]["points"][0]["customdata"]
#         state["station"] = clicked_station
#         state["zoomed_station"] = clicked_station
#         state["view"] = "original"
#         state["smooth"] = True
#         state["remove_outliers"] = False
#         state["original_data"] = None  # Reseta dados ao mudar de estação
#         state["filtered_data"] = None
#     elif trigger.startswith("src-"):
#         state["source"] = trigger.split("-")[1]
#         state["original_data"] = None  # Reseta dados ao mudar de fonte
#         state["filtered_data"] = None
#     elif trigger == "btn-original":
#         state["view"] = "original"
#     elif trigger == "btn-detrended":
#         state["view"] = "detrended"
#     elif trigger == "btn-smooth-on":
#         state["smooth"] = True
#     elif trigger == "btn-smooth-off":
#         state["smooth"] = False
#     elif trigger == "btn-outliers-on":
#         state["remove_outliers"] = True
#     elif trigger == "btn-outliers-off":
#         state["remove_outliers"] = False
    
#     return state

# # ============================================================
# # RENDER TIMESERIES
# # ============================================================
# @app.callback(
#     Output("timeseries-container", "children"),
#     Input("ui-state", "data")
# )
# def render_ts(state):
#     if state["station"] is None:
#         return html.Div("Click on a station or search for a station ID to view its time series.")

#     try:
#         df = read_pfile_from_gdrive(state["source"], state["station"])
        
#         if df.empty:
#             return html.Div([
#                 html.H3(f"Could not load data for station {state['station']}"),
#                 html.P("Possible reasons:"),
#                 html.Ul([
#                     html.Li("File not found in the selected source (Linear/CM/CF)"),
#                     html.Li("Google Drive temporarily blocking automated access"),
#                     html.Li("Station ID mismatch between map and data files")
#                 ]),
#                 html.P("Try:"),
#                 html.Ul([
#                     html.Li("Selecting a different source (Linear/CM/CF buttons above)"),
#                     html.Li("Clicking on a different station"),
#                     html.Li("Waiting a moment and trying again")
#                 ])
#             ])
        
#         # Processa dados básicos
#         t = df.time.values
#         E, N, U = lonlat2local_mm(df.lon, df.lat, df.hgt)
        
#         # Calcula tendência
#         E_tr, vE = trend_and_velocity(E, t)
#         N_tr, vN = trend_and_velocity(N, t)
#         U_tr, vU = trend_and_velocity(U, t)
        
#         # Calcula série detrended
#         E_d, N_d, U_d = E - E_tr, N - N_tr, U - U_tr
        
#         # Armazena dados originais se ainda não armazenados
#         if state.get("original_data") is None:
#             state["original_data"] = {
#                 't': t, 'E': E, 'N': N, 'U': U,
#                 'E_d': E_d, 'N_d': N_d, 'U_d': U_d,
#                 'vE': vE, 'vN': vN, 'vU': vU
#             }
        
#         # Remove outliers se necessário
#         if state.get("remove_outliers", False):
#             # Se ainda não filtrou os dados, faz a filtragem
#             if state.get("filtered_data") is None:
#                 filtered = remove_outliers(t, E, N, U, E_d, N_d, U_d, OUTLIER_CONFIG)
#                 # Recalcula velocidade com dados filtrados
#                 E_tr_f, vE_f = trend_and_velocity(filtered['E'], filtered['t'])
#                 N_tr_f, vN_f = trend_and_velocity(filtered['N'], filtered['t'])
#                 U_tr_f, vU_f = trend_and_velocity(filtered['U'], filtered['t'])
                
#                 state["filtered_data"] = {
#                     **filtered,
#                     'vE': vE_f, 'vN': vN_f, 'vU': vU_f,
#                     'removed_info': f"Removed {filtered['removed_count']} points ({filtered['removed_percentage']:.1f}%)"
#                 }
            
#             # Usa dados filtrados
#             data = state["filtered_data"]
#             t_plot = data['t']
#             E_plot = data['E'] if state["view"] == "original" else data['E_d']
#             N_plot = data['N'] if state["view"] == "original" else data['N_d']
#             U_plot = data['U'] if state["view"] == "original" else data['U_d']
#             vE_plot, vN_plot, vU_plot = data['vE'], data['vN'], data['vU']
#             outlier_info = data.get('removed_info', '')
#         else:
#             # Usa dados originais
#             data = state["original_data"]
#             t_plot = data['t']
#             E_plot = data['E'] if state["view"] == "original" else data['E_d']
#             N_plot = data['N'] if state["view"] == "original" else data['N_d']
#             U_plot = data['U'] if state["view"] == "original" else data['U_d']
#             vE_plot, vN_plot, vU_plot = data['vE'], data['vN'], data['vU']
#             outlier_info = ""
        
#         # Aplica suavização se solicitado
#         if state["smooth"]:
#             E_smooth = apply_smoothing(E_plot, t_plot)
#             N_smooth = apply_smoothing(N_plot, t_plot)
#             U_smooth = apply_smoothing(U_plot, t_plot)

#         hover = (
#             "Year: %{x:.4f}<br>"
#             "Disp: %{y:.2f} mm"
#             "<extra></extra>"
#         )

#         # Título do gráfico
#         title = f"Station {state['station']} — {state['source'].upper()} (Window: {WINDOW_SIZE} days)"
#         if state.get("remove_outliers", False):
#             title += f" — Outliers removed: {outlier_info}"
        
#         fig = make_subplots(
#             rows=3, cols=1,
#             shared_xaxes=True,
#             vertical_spacing=0.035,
#             subplot_titles=[
#                 f"East ({vE_plot:.2f} mm/yr)",
#                 f"North ({vN_plot:.2f} mm/yr)",
#                 f"Up ({vU_plot:.2f} mm/yr)"
#             ]
#         )
        
#         # Dados principais
#         fig.add_scatter(x=t_plot, y=E_plot, mode="markers", 
#                        hovertemplate=hover, row=1, col=1,
#                        marker=dict(size=5, opacity=0.6))
#         fig.add_scatter(x=t_plot, y=N_plot, mode="markers", 
#                        hovertemplate=hover, row=2, col=1,
#                        marker=dict(size=5, opacity=0.6))
#         fig.add_scatter(x=t_plot, y=U_plot, mode="markers", 
#                        hovertemplate=hover, row=3, col=1,
#                        marker=dict(size=5, opacity=0.6))

#         # Linha suavizada se Smooth ON
#         if state["smooth"]:
#             fig.add_scatter(x=t_plot, y=E_smooth, mode="lines",
#                             line=dict(color="magenta", width=2), 
#                             hoverinfo="skip", row=1, col=1)
#             fig.add_scatter(x=t_plot, y=N_smooth, mode="lines",
#                             line=dict(color="magenta", width=2), 
#                             hoverinfo="skip", row=2, col=1)
#             fig.add_scatter(x=t_plot, y=U_smooth, mode="lines",
#                             line=dict(color="magenta", width=2), 
#                             hoverinfo="skip", row=3, col=1)

#         fig.update_layout(
#             title=title,
#             height=900,
#             showlegend=False
#         )

#         return dcc.Graph(figure=fig, config={"scrollZoom": True})
        
#     except Exception as e:
#         import traceback
#         return html.Div([
#             html.H3(f"Error loading data for station {state['station']}"),
#             html.P(f"Error: {str(e)}"),
#             html.Pre(traceback.format_exc(), style={"fontSize": "10px", "overflow": "auto"})
#         ])

# # ============================================================
# # RUN
# # ============================================================
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=PORT, debug=False)
