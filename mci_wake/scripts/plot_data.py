#!/usr/bin/env python3
"""
Dash Application for Inspecting MCI Wake Word Training Data.

Usage:
    python scripts/dash_app.py [--port 8050] [--host 127.0.0.1] [--debug]
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure workspace src directory is in Python path
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
SRC_DIR = WORKSPACE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Gesture label mappings
GESTURE_MAP: Dict[int, str] = {
    0: "noGesture",
    1: "fist",
    2: "waveIn",
    3: "waveOut",
    4: "open",
    5: "pinch"
}
REV_GESTURE_MAP: Dict[str, int] = {v: k for k, v in GESTURE_MAP.items()}

# Colors for channels and gestures
CHANNEL_COLORS = [
    "#38bdf8", "#34d399", "#fbbf24", "#f87171",
    "#a855f7", "#ec4899", "#818cf8", "#2dd4bf"
]
GESTURE_COLORS = {
    "noGesture": "#94a3b8",
    "fist": "#ef4444",
    "waveIn": "#3b82f6",
    "waveOut": "#10b981",
    "open": "#f59e0b",
    "pinch": "#8b5cf6",
    "ADL": "#64748b"
}


def load_dataset() -> Dict[str, Any]:
    """
    Loads dataset from cached dataset.pkl if available, or via load_raw_data.
    Returns structured data dictionary for fast lookup.
    """
    pkl_candidates = [
        SCRIPT_DIR / "dataset.pkl",
        WORKSPACE_DIR / "dataset.pkl",
        Path("dataset.pkl"),
        Path("scripts/dataset.pkl")
    ]
    
    loaded_tuple = None
    for pkl_path in pkl_candidates:
        if pkl_path.exists():
            print(f"[Info] Loading dataset from pickle: {pkl_path}")
            try:
                with open(pkl_path, "rb") as f:
                    loaded_tuple = pickle.load(f)
                break
            except Exception as e:
                print(f"[Warning] Failed loading {pkl_path}: {e}")

    dataset = {
        "samples": [],
        "num_subjects": 0,
        "total_samples": 0,
        "gestures_count": {},
        "subjects_list": []
    }

    if loaded_tuple is not None and len(loaded_tuple) >= 5:
        emg_data, imu_data, labels, myo_labels, subject_ids = loaded_tuple[:5]
        
        # Process training & testing splits
        for split in ["training", "testing"]:
            emg_list = emg_data.get(split, [])
            imu_list = imu_data.get(split, [])
            lbl_list = labels.get(split, [])
            sub_list = subject_ids.get(split, [])

            for i in range(len(emg_list)):
                emg_arr = np.array(emg_list[i], dtype=np.float32)
                imu_arr = np.array(imu_list[i], dtype=np.float32) if i < len(imu_list) and imu_list[i] is not None else None
                lbl = int(lbl_list[i]) if i < len(lbl_list) else 0
                sub = int(sub_list[i]) if i < len(sub_list) else -1
                
                dataset["samples"].append({
                    "id": len(dataset["samples"]),
                    "split": split,
                    "subject_id": sub,
                    "label_id": lbl,
                    "gesture": GESTURE_MAP.get(lbl, f"Unknown ({lbl})"),
                    "emg": emg_arr,
                    "imu": imu_arr,
                    "length": emg_arr.shape[0] if emg_arr.ndim >= 1 else 0,
                    "channels": emg_arr.shape[1] if emg_arr.ndim >= 2 else 0,
                    "is_adl": False
                })

    # Try loading ADL data if train_utils is available
    try:
        from mci_wake.data.train_utils import load_disco_adls, ADL_DATA
        adl_data = load_disco_adls(ADL_DATA)
        print(f"[Info] Loaded {len(adl_data)} ADL noise segments.")
        for i, adl_sample in enumerate(adl_data):
            adl_arr = np.array(adl_sample, dtype=np.float32)
            if adl_arr.ndim == 2:
                dataset["samples"].append({
                    "id": len(dataset["samples"]),
                    "split": "ADL",
                    "subject_id": -1,
                    "label_id": 0,
                    "gesture": "ADL",
                    "emg": adl_arr,
                    "imu": None,
                    "length": adl_arr.shape[0],
                    "channels": adl_arr.shape[1],
                    "is_adl": True
                })
    except Exception as e:
        print(f"[Note] Could not load ADL dataset: {e}")

    # Fallback to load_raw_data if no pickle was found
    if len(dataset["samples"]) == 0:
        print("[Info] No cached pickle found, attempting load_raw_data()...")
        try:
            from mci_wake.data.train_utils import load_raw_data
            emg_data_all, labels_all, subject_ids_all, adl_data = load_raw_data()
            for i in range(len(emg_data_all)):
                emg_arr = np.array(emg_data_all[i], dtype=np.float32)
                lbl = int(labels_all[i])
                sub = int(subject_ids_all[i])
                dataset["samples"].append({
                    "id": len(dataset["samples"]),
                    "split": "all",
                    "subject_id": sub,
                    "label_id": lbl,
                    "gesture": GESTURE_MAP.get(lbl, f"Unknown ({lbl})"),
                    "emg": emg_arr,
                    "imu": None,
                    "length": emg_arr.shape[0] if emg_arr.ndim >= 1 else 0,
                    "channels": emg_arr.shape[1] if emg_arr.ndim >= 2 else 0,
                    "is_adl": False
                })
            for i, adl_sample in enumerate(adl_data):
                adl_arr = np.array(adl_sample, dtype=np.float32)
                if adl_arr.ndim == 2:
                    dataset["samples"].append({
                        "id": len(dataset["samples"]),
                        "split": "ADL",
                        "subject_id": -1,
                        "label_id": 0,
                        "gesture": "ADL",
                        "emg": adl_arr,
                        "imu": None,
                        "length": adl_arr.shape[0],
                        "channels": adl_arr.shape[1],
                        "is_adl": True
                    })
        except Exception as e:
            print(f"[Error] Failed to load dataset: {e}")

    # Pre-calculate summary statistics
    all_subs = sorted(list({s["subject_id"] for s in dataset["samples"] if s["subject_id"] != -1}))
    dataset["subjects_list"] = all_subs
    dataset["num_subjects"] = len(all_subs)
    dataset["total_samples"] = len(dataset["samples"])
    
    gest_counts = {}
    for s in dataset["samples"]:
        g = s["gesture"]
        gest_counts[g] = gest_counts.get(g, 0) + 1
    dataset["gestures_count"] = gest_counts

    print(f"[Success] Total loaded samples: {dataset['total_samples']} across {dataset['num_subjects']} subjects.")
    return dataset


# Initialize dataset globally
DATASET = load_dataset()

# Initialize Dash App
app = dash.Dash(
    __name__,
    title="MCI Wake - Dataset Inspector",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server

# Options for Filters
SPLIT_OPTIONS = [{"label": "All Splits", "value": "ALL"}] + [
    {"label": s.capitalize(), "value": s} for s in sorted(list({s["split"] for s in DATASET["samples"]}))
]
SUBJECT_OPTIONS = [{"label": "All Subjects", "value": "ALL"}] + [
    {"label": f"User {s}", "value": s} for s in DATASET["subjects_list"]
]
GESTURE_OPTIONS = [{"label": "All Gestures", "value": "ALL"}] + [
    {"label": g, "value": g} for g in list(GESTURE_MAP.values()) + (["ADL"] if "ADL" in DATASET["gestures_count"] else [])
]

# Custom CSS / Dark Theme Styles
CARD_STYLE = {
    "backgroundColor": "#1e293b",
    "borderRadius": "12px",
    "padding": "20px",
    "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)",
    "marginBottom": "20px",
    "border": "1px solid #334155"
}

STAT_BOX_STYLE = {
    "backgroundColor": "#0f172a",
    "borderRadius": "8px",
    "padding": "15px",
    "textAlign": "center",
    "border": "1px solid #334155",
    "flex": "1",
    "margin": "0 8px"
}

# Layout
app.layout = html.Div(
    style={
        "backgroundColor": "#0f172a",
        "color": "#f8fafc",
        "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "minHeight": "100vh",
        "padding": "24px"
    },
    children=[
        # Header
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "24px"},
            children=[
                html.Div([
                    html.H1("MCI Wake Dataset Inspector", style={"margin": "0", "fontSize": "28px", "fontWeight": "700", "color": "#f8fafc"}),
                    html.P("Explore & visualize EMG & IMU wake-word training dataset", style={"margin": "4px 0 0 0", "color": "#94a3b8", "fontSize": "14px"})
                ]),
                html.Div([
                    html.Span("Calibration-Free Wakeword", style={"backgroundColor": "#3b82f6", "color": "#ffffff", "padding": "6px 14px", "borderRadius": "20px", "fontSize": "13px", "fontWeight": "600"})
                ])
            ]
        ),

        # Dataset Summary Stat Cards
        html.Div(
            style={"display": "flex", "justifyContent": "space-between", "marginBottom": "24px"},
            children=[
                html.Div([
                    html.Div("Total Samples", style={"color": "#94a3b8", "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                    html.Div(f"{DATASET['total_samples']:,}", style={"fontSize": "24px", "fontWeight": "700", "color": "#38bdf8", "marginTop": "4px"})
                ], style=STAT_BOX_STYLE),

                html.Div([
                    html.Div("Unique Subjects", style={"color": "#94a3b8", "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                    html.Div(f"{DATASET['num_subjects']}", style={"fontSize": "24px", "fontWeight": "700", "color": "#34d399", "marginTop": "4px"})
                ], style=STAT_BOX_STYLE),

                html.Div([
                    html.Div("EMG Channels", style={"color": "#94a3b8", "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                    html.Div("8 Channels", style={"fontSize": "24px", "fontWeight": "700", "color": "#fbbf24", "marginTop": "4px"})
                ], style=STAT_BOX_STYLE),

                html.Div([
                    html.Div("Gestures Count", style={"color": "#94a3b8", "fontSize": "12px", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
                    html.Div(f"{len(GESTURE_MAP)} Classes", style={"fontSize": "24px", "fontWeight": "700", "color": "#a855f7", "marginTop": "4px"})
                ], style=STAT_BOX_STYLE),
            ]
        ),

        # Filter Control Panel
        html.Div(
            style=CARD_STYLE,
            children=[
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "16px", "alignItems": "end"},
                    children=[
                        # Split Dropdown
                        html.Div([
                            html.Label("Split / Dataset", style={"fontSize": "13px", "fontWeight": "600", "color": "#cbd5e1", "marginBottom": "6px", "display": "block"}),
                            dcc.Dropdown(id="split-dropdown", options=SPLIT_OPTIONS, value="ALL", clearable=False, style={"color": "#000000"})
                        ]),

                        # Subject Dropdown
                        html.Div([
                            html.Label("Subject ID", style={"fontSize": "13px", "fontWeight": "600", "color": "#cbd5e1", "marginBottom": "6px", "display": "block"}),
                            dcc.Dropdown(id="subject-dropdown", options=SUBJECT_OPTIONS, value="ALL", clearable=False, style={"color": "#000000"})
                        ]),

                        # Gesture Dropdown
                        html.Div([
                            html.Label("Gesture Label", style={"fontSize": "13px", "fontWeight": "600", "color": "#cbd5e1", "marginBottom": "6px", "display": "block"}),
                            dcc.Dropdown(id="gesture-dropdown", options=GESTURE_OPTIONS, value="ALL", clearable=False, style={"color": "#000000"})
                        ]),

                        # Sample Index Controls
                        html.Div([
                            html.Label("Sample Navigation", style={"fontSize": "13px", "fontWeight": "600", "color": "#cbd5e1", "marginBottom": "6px", "display": "block"}),
                            html.Div(
                                style={"display": "flex", "gap": "8px", "alignItems": "center"},
                                children=[
                                    html.Button("◀ Prev", id="btn-prev", n_clicks=0, style={"backgroundColor": "#334155", "color": "#ffffff", "border": "none", "borderRadius": "6px", "padding": "8px 12px", "cursor": "pointer"}),
                                    dcc.Input(id="sample-index-input", type="number", value=0, min=0, step=1, style={"width": "80px", "padding": "6px", "borderRadius": "6px", "border": "1px solid #475569", "textAlign": "center"}),
                                    html.Button("Next ▶", id="btn-next", n_clicks=0, style={"backgroundColor": "#334155", "color": "#ffffff", "border": "none", "borderRadius": "6px", "padding": "8px 12px", "cursor": "pointer"}),
                                    html.Button("🎲 Random", id="btn-random", n_clicks=0, style={"backgroundColor": "#3b82f6", "color": "#ffffff", "border": "none", "borderRadius": "6px", "padding": "8px 12px", "cursor": "pointer"})
                                ]
                            )
                        ])
                    ]
                ),

                # Counter Badge
                html.Div(
                    id="matching-count-badge",
                    style={"marginTop": "14px", "fontSize": "13px", "color": "#94a3b8", "textAlign": "right"}
                )
            ]
        ),

        # Tabs for Dashboard sections
        dcc.Tabs(
            id="main-tabs",
            value="tab-inspector",
            colors={"border": "#334155", "primary": "#38bdf8", "background": "#0f172a"},
            style={"marginBottom": "20px"},
            children=[
                dcc.Tab(label="📈 Sample Signal Inspector", value="tab-inspector", style={"backgroundColor": "#1e293b", "color": "#cbd5e1", "padding": "12px"}, selected_style={"backgroundColor": "#3b82f6", "color": "#ffffff", "padding": "12px"}),
                dcc.Tab(label="📊 Dataset Statistics", value="tab-stats", style={"backgroundColor": "#1e293b", "color": "#cbd5e1", "padding": "12px"}, selected_style={"backgroundColor": "#3b82f6", "color": "#ffffff", "padding": "12px"}),
                dcc.Tab(label="⚡ Frequency & Feature Spectrum", value="tab-spectrum", style={"backgroundColor": "#1e293b", "color": "#cbd5e1", "padding": "12px"}, selected_style={"backgroundColor": "#3b82f6", "color": "#ffffff", "padding": "12px"})
            ]
        ),

        # Tab Content Container
        html.Div(id="tab-content")
    ]
)


# Store filtered indices in a dcc.Store
app.layout.children.append(dcc.Store(id="filtered-indices-store", data=[]))


@app.callback(
    Output("filtered-indices-store", "data"),
    Output("matching-count-badge", "children"),
    Input("split-dropdown", "value"),
    Input("subject-dropdown", "value"),
    Input("gesture-dropdown", "value")
)
def update_filtered_indices(split_val, subject_val, gesture_val):
    filtered = []
    for idx, s in enumerate(DATASET["samples"]):
        if split_val != "ALL" and s["split"] != split_val:
            continue
        if subject_val != "ALL" and s["subject_id"] != subject_val:
            continue
        if gesture_val != "ALL" and s["gesture"] != gesture_val:
            continue
        filtered.append(idx)

    cnt = len(filtered)
    badge = f"Matching Samples: {cnt:,} / {DATASET['total_samples']:,}"
    return filtered, badge


@app.callback(
    Output("sample-index-input", "value"),
    Output("sample-index-input", "max"),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    Input("btn-random", "n_clicks"),
    Input("filtered-indices-store", "data"),
    State("sample-index-input", "value")
)
def handle_sample_navigation(btn_prev, btn_next, btn_rand, filtered_indices, curr_val):
    triggered_id = ctx.triggered_id
    max_idx = max(0, len(filtered_indices) - 1)
    
    if not filtered_indices:
        return 0, 0

    curr_val = curr_val or 0
    if curr_val > max_idx:
        curr_val = max_idx

    if triggered_id == "btn-prev":
        curr_val = max(0, curr_val - 1)
    elif triggered_id == "btn-next":
        curr_val = min(max_idx, curr_val + 1)
    elif triggered_id == "btn-random":
        curr_val = int(np.random.randint(0, len(filtered_indices)))

    return curr_val, max_idx


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("filtered-indices-store", "data"),
    Input("sample-index-input", "value")
)
def render_tab_content(active_tab, filtered_indices, filtered_pos):
    if not filtered_indices:
        return html.Div(
            style=CARD_STYLE,
            children=html.Div("No samples match the selected filters. Please adjust filter options.", style={"color": "#f87171", "textAlign": "center", "fontSize": "16px"})
        )

    pos = min(max(0, filtered_pos or 0), len(filtered_indices) - 1)
    sample_global_idx = filtered_indices[pos]
    sample = DATASET["samples"][sample_global_idx]

    if active_tab == "tab-inspector":
        return build_inspector_tab(sample, pos, len(filtered_indices))
    elif active_tab == "tab-stats":
        return build_stats_tab()
    elif active_tab == "tab-spectrum":
        return build_spectrum_tab(sample)
    return html.Div("Select a tab.")


def build_inspector_tab(sample: Dict[str, Any], pos: int, total_filtered: int) -> html.Div:
    emg = sample["emg"]
    imu = sample["imu"]
    timesteps = np.arange(sample["length"])

    # Build 8-Channel EMG Figure
    fig_emg = make_subplots(
        rows=8, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"Ch {ch+1}" for ch in range(8)]
    )

    for ch in range(min(8, emg.shape[1] if emg.ndim > 1 else 1)):
        ch_data = emg[:, ch] if emg.ndim > 1 else emg
        fig_emg.add_trace(
            go.Scatter(
                x=timesteps,
                y=ch_data,
                mode="lines",
                name=f"Channel {ch+1}",
                line=dict(color=CHANNEL_COLORS[ch % len(CHANNEL_COLORS)], width=1.5),
                hovertemplate=f"Ch {ch+1}: %{{y:.2f}}<extra></extra>"
            ),
            row=ch+1, col=1
        )

    fig_emg.update_layout(
        height=750,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5e1"),
        margin=dict(l=40, r=40, t=30, b=40),
        showlegend=False
    )
    fig_emg.update_xaxes(showgrid=True, gridcolor="#334155", title_text="Timesteps (samples)")
    fig_emg.update_yaxes(showgrid=True, gridcolor="#334155")

    # Build Combined Overlay Figure
    fig_overlay = go.Figure()
    for ch in range(min(8, emg.shape[1] if emg.ndim > 1 else 1)):
        ch_data = emg[:, ch] if emg.ndim > 1 else emg
        fig_overlay.add_trace(
            go.Scatter(
                x=timesteps,
                y=ch_data,
                mode="lines",
                name=f"Ch {ch+1}",
                line=dict(color=CHANNEL_COLORS[ch % len(CHANNEL_COLORS)], width=1.2)
            )
        )
    fig_overlay.update_layout(
        title="8-Channel EMG Overlay",
        height=320,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5e1"),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", y=1.15)
    )
    fig_overlay.update_xaxes(showgrid=True, gridcolor="#334155", title_text="Timesteps")
    fig_overlay.update_yaxes(showgrid=True, gridcolor="#334155", title_text="Raw Amplitude")

    # Build IMU Figure if available
    fig_imu = None
    if imu is not None and imu.ndim == 2:
        fig_imu = go.Figure()
        imu_labels = ["w", "x", "y", "z"]
        imu_colors = ["#38bdf8", "#34d399", "#f59e0b", "#f43f5e"]
        imu_time = np.arange(imu.shape[0])
        for i in range(min(4, imu.shape[1])):
            fig_imu.add_trace(
                go.Scatter(
                    x=imu_time,
                    y=imu[:, i],
                    mode="lines",
                    name=f"Quat {imu_labels[i]}",
                    line=dict(color=imu_colors[i], width=2)
                )
            )
        fig_imu.update_layout(
            title="IMU Quaternion Signals (w, x, y, z)",
            height=280,
            paper_bgcolor="#1e293b",
            plot_bgcolor="#0f172a",
            font=dict(color="#cbd5e1"),
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", y=1.15)
        )
        fig_imu.update_xaxes(showgrid=True, gridcolor="#334155", title_text="Timesteps")
        fig_imu.update_yaxes(showgrid=True, gridcolor="#334155")

    # Sample details card
    details_card = html.Div(
        style=CARD_STYLE,
        children=[
            html.H3(f"Sample #{sample['id']} Metadata", style={"margin": "0 0 12px 0", "fontSize": "18px", "color": "#38bdf8"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))", "gap": "12px"},
                children=[
                    html.Div([html.Span("Gesture: ", style={"color": "#94a3b8"}), html.Strong(sample["gesture"], style={"color": GESTURE_COLORS.get(sample["gesture"], "#ffffff")})]),
                    html.Div([html.Span("Subject ID: ", style={"color": "#94a3b8"}), html.Strong(f"User {sample['subject_id']}" if sample["subject_id"] != -1 else "ADL Noise")]),
                    html.Div([html.Span("Split: ", style={"color": "#94a3b8"}), html.Strong(sample["split"].capitalize())]),
                    html.Div([html.Span("Duration: ", style={"color": "#94a3b8"}), html.Strong(f"{sample['length']} samples")]),
                    html.Div([html.Span("Channels: ", style={"color": "#94a3b8"}), html.Strong(f"{sample['channels']} EMG")]),
                    html.Div([html.Span("Filtered Pos: ", style={"color": "#94a3b8"}), html.Strong(f"{pos + 1} of {total_filtered}")])
                ]
            )
        ]
    )

    return html.Div([
        details_card,
        html.Div(
            style=CARD_STYLE,
            children=[
                html.H3("Multi-Channel EMG Waveforms", style={"margin": "0 0 12px 0", "fontSize": "18px", "color": "#f8fafc"}),
                dcc.Graph(figure=fig_overlay, config={"responsive": True}),
                html.Hr(style={"borderColor": "#334155", "margin": "20px 0"}),
                dcc.Graph(figure=fig_emg, config={"responsive": True})
            ]
        ),
        html.Div(
            style=CARD_STYLE,
            children=[
                html.H3("IMU Quaternion Signals", style={"margin": "0 0 12px 0", "fontSize": "18px", "color": "#f8fafc"}),
                dcc.Graph(figure=fig_imu, config={"responsive": True}) if fig_imu else html.Div("No IMU data available for this sample.", style={"color": "#94a3b8", "padding": "10px"})
            ]
        )
    ])


def build_stats_tab() -> html.Div:
    # Gesture Distribution Bar Chart
    df_samples = pd.DataFrame([
        {"gesture": s["gesture"], "split": s["split"], "subject_id": s["subject_id"], "length": s["length"]}
        for s in DATASET["samples"]
    ])

    fig_gest = go.Figure()
    gest_counts = df_samples["gesture"].value_counts()
    fig_gest.add_trace(
        go.Bar(
            x=gest_counts.index,
            y=gest_counts.values,
            marker_color=[GESTURE_COLORS.get(g, "#3b82f6") for g in gest_counts.index],
            text=gest_counts.values,
            textposition="auto"
        )
    )
    fig_gest.update_layout(
        title="Sample Distribution Across Gestures",
        height=350,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5e1"),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_gest.update_xaxes(showgrid=True, gridcolor="#334155")
    fig_gest.update_yaxes(showgrid=True, gridcolor="#334155")

    # Sample Length Histogram
    fig_len = go.Figure()
    fig_len.add_trace(
        go.Histogram(
            x=df_samples["length"],
            nbinsx=40,
            marker_color="#38bdf8",
            opacity=0.8
        )
    )
    fig_len.update_layout(
        title="Sample Duration / Timesteps Distribution",
        height=350,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5e1"),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_len.update_xaxes(showgrid=True, gridcolor="#334155", title_text="Sample Length (timesteps)")
    fig_len.update_yaxes(showgrid=True, gridcolor="#334155", title_text="Count")

    # Subject Samples Distribution
    sub_df = df_samples[df_samples["subject_id"] != -1]
    sub_counts = sub_df["subject_id"].value_counts().sort_index()

    fig_sub = go.Figure()
    fig_sub.add_trace(
        go.Bar(
            x=[f"U{s}" for s in sub_counts.index],
            y=sub_counts.values,
            marker_color="#a855f7"
        )
    )
    fig_sub.update_layout(
        title="Samples Per Subject",
        height=350,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5e1"),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_sub.update_xaxes(showgrid=False, gridcolor="#334155")
    fig_sub.update_yaxes(showgrid=True, gridcolor="#334155")

    return html.Div([
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
            children=[
                html.Div(style=CARD_STYLE, children=[dcc.Graph(figure=fig_gest, config={"responsive": True})]),
                html.Div(style=CARD_STYLE, children=[dcc.Graph(figure=fig_len, config={"responsive": True})])
            ]
        ),
        html.Div(style=CARD_STYLE, children=[dcc.Graph(figure=fig_sub, config={"responsive": True})])
    ])


def build_spectrum_tab(sample: Dict[str, Any]) -> html.Div:
    emg = sample["emg"]
    
    # Feature Metrics Bar Chart (RMS & Mean Absolute Value per channel)
    rms_per_ch = []
    mav_per_ch = []
    ch_names = [f"Ch {i+1}" for i in range(emg.shape[1] if emg.ndim > 1 else 1)]

    for ch in range(emg.shape[1] if emg.ndim > 1 else 1):
        c_data = emg[:, ch] if emg.ndim > 1 else emg
        rms_per_ch.append(np.sqrt(np.mean(c_data ** 2)))
        mav_per_ch.append(np.mean(np.abs(c_data)))

    fig_feats = go.Figure()
    fig_feats.add_trace(go.Bar(x=ch_names, y=rms_per_ch, name="RMS", marker_color="#38bdf8"))
    fig_feats.add_trace(go.Bar(x=ch_names, y=mav_per_ch, name="MAV (Mean Abs)", marker_color="#34d399"))
    fig_feats.update_layout(
        title="Channel Power & Activation Features (RMS vs MAV)",
        height=350,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5e1"),
        barmode="group",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig_feats.update_xaxes(showgrid=True, gridcolor="#334155")
    fig_feats.update_yaxes(showgrid=True, gridcolor="#334155")

    # Power Spectral Density (FFT)
    fig_fft = go.Figure()
    for ch in range(min(8, emg.shape[1] if emg.ndim > 1 else 1)):
        c_data = emg[:, ch] if emg.ndim > 1 else emg
        fft_vals = np.abs(np.fft.rfft(c_data))
        freqs = np.fft.rfftfreq(len(c_data), d=1.0/200.0)  # Assume ~200Hz sampling rate
        fig_fft.add_trace(
            go.Scatter(
                x=freqs,
                y=fft_vals,
                mode="lines",
                name=f"Ch {ch+1}",
                line=dict(color=CHANNEL_COLORS[ch % len(CHANNEL_COLORS)], width=1.5)
            )
        )
    fig_fft.update_layout(
        title="Power Spectral Density / FFT Spectrum (0 - 100 Hz)",
        height=380,
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#cbd5e1"),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", y=1.15)
    )
    fig_fft.update_xaxes(showgrid=True, gridcolor="#334155", title_text="Frequency (Hz)")
    fig_fft.update_yaxes(showgrid=True, gridcolor="#334155", title_text="Magnitude")

    return html.Div([
        html.Div(style=CARD_STYLE, children=[dcc.Graph(figure=fig_feats, config={"responsive": True})]),
        html.Div(style=CARD_STYLE, children=[dcc.Graph(figure=fig_fft, config={"responsive": True})])
    ])


def main():
    parser = argparse.ArgumentParser(description="MCI Wake Dataset Inspector Dash App")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="Port to run Dash server (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"🚀 Starting Dash Dataset Inspector at http://{args.host}:{args.port}")
    print(f"=======================================================\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
