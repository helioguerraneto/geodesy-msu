#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import json
import requests
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from pyproj import Proj
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

GDRIVE_STATIONS_FILE = "17mi5FA44LvnWr-50-bLgrdbrBsuMU-bK"

# ============================================================
# GOOGLE DRIVE UTILITIES - IMPROVED
# ============================================================
def get_gdrive_download_url(file_id):
    """Converte ID do arquivo Google Drive para URL de download direto."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"

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
        
        # Lê como CSV
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
        
        # Fallback
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
# GNSS UTILITIES
# ============================================================
@lru_cache(maxsize=256)
def read_pfile_from_gdrive(source, station):
    """Lê arquivo .pfiles do Google Drive."""
    folder_id = GDRIVE_FOLDERS[source]
    
    # Se ainda não carregou o índice desta pasta, tenta carregar
    if not index_loaded[source]:
        print(f"Loading index for {source}...")
        file_index[source] = get_gdrive_file_list_via_api(folder_id)
        index_loaded[source] = True
        print(f"Found {len(file_index[source])} files in {source}")
    
    # Procura o file ID ou URL
    file_ref = file_index[source].get(station)
    
    if file_ref is None:
        print(f"File not found for {station} in {source}")
        return pd.DataFrame(columns=["time", "lon", "lat", "hgt"])
    
    try:
        # Se file_ref já é uma URL completa, usa direto
        # Senão, converte file ID para URL
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

def seasonal_model(x, t, remove_trend=False):
    G = np.column_stack([
        np.ones(len(t)),
        t - t.mean(),
        np.sin(2*np.pi*t),
        np.cos(2*np.pi*t),
        np.sin(4*np.pi*t),
        np.cos(4*np.pi*t)
    ])
    m, _, _, _ = lstsq(G, x)
    if remove_trend:
        m[1] = 0.0
    return G @ m

# ============================================================
# DASH APP
# ============================================================
app = dash.Dash(__name__)
app.title = "GNSS Great Lakes Viewer"
server = app.server

# ============================================================
# MAP
# ============================================================
def make_map(show_labels=False):
    fig = go.Figure()
    fig.add_scattermapbox(
        lon=stations.lon,
        lat=stations.lat,
        customdata=stations.id,
        text=stations.id if show_labels else None,
        mode="markers+text" if show_labels else "markers",
        marker=dict(size=8, color="crimson"),
        hovertemplate="Station: %{customdata}<extra></extra>"
    )
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lon=-84, lat=44),
            zoom=4.2
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
            "model": True
        }),

        html.H2("MSU Geodesy Lab - The Great Lakes GNSS Stations"),
        
        html.Div([
            html.P("⚠️ Note: Due to Google Drive limitations, file loading may be slow or fail. If data doesn't load, please try another station or source.", 
                   style={"fontSize": "12px", "color": "#666", "fontStyle": "italic"})
        ]),

        dcc.Graph(id="station-map", figure=make_map(), config={"scrollZoom": True}),
        html.Button("Show IDs", id="btn-labels"),

        html.Hr(),

        html.Div([
            html.Div([
                html.Button("Linear", id="src-linear"),
                html.Button("CM", id="src-cm"),
                html.Button("CF", id="src-cf"),
            ], style={"marginBottom": "10px"}),

            html.Div([
                html.Button("Original", id="btn-original"),
                html.Button("Detrended", id="btn-detrended"),
            ], style={"marginBottom": "10px"}),

            html.Div([
                html.Button("Model ON", id="btn-model-on"),
                html.Button("Model OFF", id="btn-model-off"),
            ]),
        ], style={"marginBottom": "20px"}),

        html.Div(id="timeseries-container")
    ]
)

# ============================================================
# CALLBACKS
# ============================================================
@app.callback(
    Output("station-map", "figure"),
    Input("btn-labels", "n_clicks")
)
def toggle_station_labels(n_clicks):
    show = (n_clicks or 0) % 2 == 1
    return make_map(show_labels=show)

@app.callback(
    Output("ui-state", "data"),
    [
        Input("station-map", "clickData"),
        Input("src-linear", "n_clicks"),
        Input("src-cm", "n_clicks"),
        Input("src-cf", "n_clicks"),
        Input("btn-original", "n_clicks"),
        Input("btn-detrended", "n_clicks"),
        Input("btn-model-on", "n_clicks"),
        Input("btn-model-off", "n_clicks"),
    ],
    State("ui-state", "data"),
    prevent_initial_call=True
)
def update_state(*args):
    ctx = dash.callback_context
    state = args[-1]
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "station-map":
        state["station"] = ctx.triggered[0]["value"]["points"][0]["customdata"]
        state["view"] = "original"
        state["model"] = True
    elif trigger.startswith("src-"):
        state["source"] = trigger.split("-")[1]
    elif trigger == "btn-original":
        state["view"] = "original"
    elif trigger == "btn-detrended":
        state["view"] = "detrended"
    elif trigger == "btn-model-on":
        state["model"] = True
    elif trigger == "btn-model-off":
        state["model"] = False

    return state

# ============================================================
# RENDER TIMESERIES
# ============================================================
@app.callback(
    Output("timeseries-container", "children"),
    Input("ui-state", "data")
)
def render_ts(state):

    if state["station"] is None:
        return html.Div("Click on a station to view its time series.")

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
        
        t = df.time.values
        E, N, U = lonlat2local_mm(df.lon, df.lat, df.hgt)

        E_tr, vE = trend_and_velocity(E, t)
        N_tr, vN = trend_and_velocity(N, t)
        U_tr, vU = trend_and_velocity(U, t)

        E_d, N_d, U_d = E - E_tr, N - N_tr, U - U_tr

        E_m = seasonal_model(E, t)
        N_m = seasonal_model(N, t)
        U_m = seasonal_model(U, t)

        E_md = seasonal_model(E_d, t, True)
        N_md = seasonal_model(N_d, t, True)
        U_md = seasonal_model(U_d, t, True)

        hover = (
            "Year: %{x:.4f}<br>"
            "Disp: %{y:.2f} mm"
            "<extra></extra>"
        )

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.035,
            subplot_titles=[
                f"East ({vE:.2f} mm/yr)",
                f"North ({vN:.2f} mm/yr)",
                f"Up ({vU:.2f} mm/yr)"
            ]
        )

        use_detr = state["view"] == "detrended"

        fig.add_scatter(x=t, y=E_d if use_detr else E, mode="markers", hovertemplate=hover, row=1, col=1)
        fig.add_scatter(x=t, y=N_d if use_detr else N, mode="markers", hovertemplate=hover, row=2, col=1)
        fig.add_scatter(x=t, y=U_d if use_detr else U, mode="markers", hovertemplate=hover, row=3, col=1)

        if state["model"]:
            fig.add_scatter(x=t, y=E_md if use_detr else E_m, mode="lines",
                            line=dict(color="magenta"), hoverinfo="skip", row=1, col=1)
            fig.add_scatter(x=t, y=N_md if use_detr else N_m, mode="lines",
                            line=dict(color="magenta"), hoverinfo="skip", row=2, col=1)
            fig.add_scatter(x=t, y=U_md if use_detr else U_m, mode="lines",
                            line=dict(color="magenta"), hoverinfo="skip", row=3, col=1)

        fig.update_layout(
            title=f"Station {state['station']} — {state['source'].upper()}",
            height=900,
            showlegend=False
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
