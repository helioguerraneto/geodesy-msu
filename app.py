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

# ============================================================
# CONFIG
# ============================================================
PORT = int(os.environ.get("PORT", 8050))
MM = 1000.0

# ID do arquivo glstations.txt no Google Drive
GDRIVE_STATIONS_FILE = "17mi5FA44LvnWr-50-bLgrdbrBsuMU-bK"

FILE_INDEX_PATH = "file_index.json"

# ============================================================
# GOOGLE DRIVE UTIL
# ============================================================
def get_gdrive_download_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# ============================================================
# PARSE PFILES
# ============================================================
def parse_pfile_content(content):
    rows = []
    for line in content.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 6:
            try:
                rows.append({
                    "time": float(p[0]),
                    "lon":  float(p[3]),
                    "lat":  float(p[4]),
                    "hgt":  float(p[5]),
                })
            except ValueError:
                pass
    return pd.DataFrame(rows)

# ============================================================
# LOAD STATIONS
# ============================================================
def load_stations():
    url = get_gdrive_download_url(GDRIVE_STATIONS_FILE)
    r = requests.get(url, timeout=20)
    r.raise_for_status()

    stations = pd.read_csv(
        io.StringIO(r.text),
        sep=r"\s+",
        header=None,
        names=["id", "lon", "lat", "hgt"],
        dtype={"id": str},
    )
    stations["id"] = stations["id"].str.upper()
    print(f"✅ Loaded {len(stations)} stations")
    return stations

stations = load_stations()

# ============================================================
# LOAD FILE INDEX (OBRIGATÓRIO)
# ============================================================
if not os.path.exists(FILE_INDEX_PATH):
    raise RuntimeError("❌ file_index.json NOT FOUND – app cannot start")

with open(FILE_INDEX_PATH) as f:
    file_index = json.load(f)

print("✅ file_index.json loaded")

# ============================================================
# GNSS UTILITIES
# ============================================================
def read_pfile_from_gdrive(source, station):
    source = source.lower()
    station = station.upper()

    if source not in file_index:
        return pd.DataFrame()

    if station not in file_index[source]:
        return pd.DataFrame()

    url = file_index[source][station]

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return parse_pfile_content(r.text)
    except Exception as e:
        print(f"❌ Error downloading {station} ({source}): {e}")
        return pd.DataFrame()

def lonlat2local_mm(lon, lat, hgt):
    E, N = [], []
    for lo, la in zip(lon, lat):
        cm = int(lo / 3) * 3
        proj = Proj(
            proj="tmerc",
            lon_0=cm,
            lat_0=0,
            k=0.9999,
            x_0=250000,
            ellps="WGS84",
            units="m",
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
        np.cos(4*np.pi*t),
    ])
    m, _, _, _ = lstsq(G, x)
    if remove_trend:
        m[1] = 0.0
    return G @ m

# ============================================================
# DASH APP
# ============================================================
app = dash.Dash(__name__)
server = app.server
app.title = "GNSS Great Lakes Viewer"

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
        hovertemplate="Station: %{customdata}<extra></extra>",
    )
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lon=-84, lat=44),
            zoom=4.2,
        ),
        height=650,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

# ============================================================
# LAYOUT
# ============================================================
app.layout = html.Div(
    style={"width": "1200px", "margin": "auto"},
    children=[
        dcc.Store(
            id="ui-state",
            data=dict(station=None, source="linear", view="original", model=True),
        ),
        html.H2("MSU Geodesy Lab – Great Lakes GNSS Stations"),
        dcc.Graph(id="station-map", figure=make_map(), config={"scrollZoom": True}),
        html.Button("Show IDs", id="btn-labels"),
        html.Hr(),
        html.Div([
            html.Button("Linear", id="src-linear"),
            html.Button("CM", id="src-cm"),
            html.Button("CF", id="src-cf"),
        ]),
        html.Div([
            html.Button("Original", id="btn-original"),
            html.Button("Detrended", id="btn-detrended"),
        ]),
        html.Div([
            html.Button("Model ON", id="btn-model-on"),
            html.Button("Model OFF", id="btn-model-off"),
        ]),
        html.Div(id="timeseries-container"),
    ],
)

# ============================================================
# CALLBACKS
# ============================================================
@app.callback(
    Output("station-map", "figure"),
    Input("btn-labels", "n_clicks"),
)
def toggle_station_labels(n):
    return make_map(show_labels=(n or 0) % 2 == 1)

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
    prevent_initial_call=True,
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
# RENDER TS
# ============================================================
@app.callback(
    Output("timeseries-container", "children"),
    Input("ui-state", "data"),
)
def render_ts(state):
    if state["station"] is None:
        return html.Div("Click on a station")

    df = read_pfile_from_gdrive(state["source"], state["station"])
    if df.empty:
        return html.Div("No data available")

    t = df.time.values
    E, N, U = lonlat2local_mm(df.lon, df.lat, df.hgt)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_scatter(x=t, y=E, mode="markers", row=1, col=1)
    fig.add_scatter(x=t, y=N, mode="markers", row=2, col=1)
    fig.add_scatter(x=t, y=U, mode="markers", row=3, col=1)

    fig.update_layout(height=900, showlegend=False)
    return dcc.Graph(figure=fig)

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
