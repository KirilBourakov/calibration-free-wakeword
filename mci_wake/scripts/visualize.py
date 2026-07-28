"""
Dash Web Application for visualizing EPN and DISCO EMG datasets,
as well as testing and visualizing Hanning cross-fade stitching on arbitrary/random data.

Run with:
    python scripts/visualize_dash.py
"""

import sys
import os
from pathlib import Path
import random
import pickle
import time
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure workspace src is on Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mci_wake.stitching.hanning import stitch
from mci_wake.utils.normalize import safe_znormalize_global
from mci_wake.data.train_utils import load_epn_data, load_disco_adls, split_disco_adls, EPN_DATA, ADL_DATA, gesture_mapping

# Inverse gesture map
GESTURE_NAMES = {v: k for k, v in gesture_mapping.items()}
GESTURE_NAMES[0] = 'noGesture'

# 8 EMG Channel Color Palette (Vibrant Neon)
CHANNEL_COLORS = [
    '#38bdf8',  # Ch 1: Sky Blue
    '#34d399',  # Ch 2: Emerald Green
    '#fbbf24',  # Ch 3: Amber
    '#f87171',  # Ch 4: Coral Red
    '#a78bfa',  # Ch 5: Violet
    '#f472b6',  # Ch 6: Pink
    '#22d3ee',  # Ch 7: Cyan
    '#fb923c',  # Ch 8: Orange
]


# ==========================================
# Data Loading & Fallback Generation
# ==========================================

_EPN_CACHE = None
_DISCO_CACHE = None


def get_epn_data_cached():
    """Loads EPN dataset from pkl/json or generates synthetic fallback if missing."""
    global _EPN_CACHE
    if _EPN_CACHE is not None:
        return _EPN_CACHE

    pkl_path = ROOT_DIR / "scripts" / "dataset.pkl"
    if pkl_path.exists():
        try:
            print(f"Loading EPN dataset from {pkl_path}...")
            with open(pkl_path, 'rb') as f:
                emg_data, imu_data, labels, myo_labels, epn_subjects = pickle.load(f)
            _EPN_CACHE = (emg_data, labels, epn_subjects)
            print(f"Loaded EPN: {len(emg_data.get('training', []))} train, {len(emg_data.get('testing', []))} test samples.")
            return _EPN_CACHE
        except Exception as e:
            print(f"Warning: Failed to load dataset.pkl: {e}")

    # Fallback to load_epn_data loader
    try:
        print("Attempting to load raw EPN dataset...")
        emg_data, imu_data, labels, myo_labels, epn_subjects = load_epn_data(EPN_DATA)
        _EPN_CACHE = (emg_data, labels, epn_subjects)
        return _EPN_CACHE
    except Exception as e:
        print(f"Notice: EPN dataset not available on disk ({e}). Using synthetic EPN data.")

    # Generate Synthetic EPN Data
    rng = np.random.default_rng(42)
    emg_data = {'training': [], 'testing': []}
    labels = {'training': [], 'testing': []}
    epn_subjects = {'training': [], 'testing': []}

    for split in ['training', 'testing']:
        n_subs = 10 if split == 'training' else 3
        for sub in range(1, n_subs + 1):
            for g_id in range(6):
                for rep in range(3):
                    T = rng.integers(150, 350)
                    t = np.linspace(0, 2 * np.pi, T)
                    # Base noise + gesture burst in center
                    sig = rng.normal(0, 0.1, size=(T, 8))
                    if g_id > 0:
                        burst = np.sin(t * (g_id + 1))[:, None] * np.exp(-((t - np.pi) ** 2) / 2)
                        sig += burst * (0.5 + 0.3 * rng.random(8))
                    emg_data[split].append(sig)
                    labels[split].append(g_id)
                    epn_subjects[split].append(sub)

    _EPN_CACHE = (emg_data, labels, epn_subjects)
    return _EPN_CACHE


def get_disco_data_cached():
    """Loads DISCO dataset from disk or generates synthetic fallback if missing."""
    global _DISCO_CACHE
    if _DISCO_CACHE is not None:
        return _DISCO_CACHE

    try:
        print(f"Loading DISCO dataset from {ADL_DATA}...")
        recs, subs = load_disco_adls(ADL_DATA)
        if len(recs) > 0:
            _DISCO_CACHE = (recs, subs)
            print(f"Loaded DISCO: {len(recs)} recordings across {len(np.unique(subs))} subjects.")
            return _DISCO_CACHE
    except Exception as e:
        print(f"Notice: DISCO dataset load notice ({e}).")

    # Synthetic fallback
    print("Generating synthetic DISCO dataset...")
    rng = np.random.default_rng(123)
    recs = []
    subs = []
    for s in range(1, 16):
        for r in range(4):
            T = rng.integers(500, 1000)
            t = np.linspace(0, 10, T)
            noise = rng.normal(0, 0.15, size=(T, 8))
            drift = 0.2 * np.sin(0.5 * t)[:, None]
            recs.append(noise + drift)
            subs.append(s)

    _DISCO_CACHE = (recs, np.array(subs, dtype=int))
    return _DISCO_CACHE


# Pre-load data structures
epn_emg, epn_labels, epn_subs = get_epn_data_cached()
disco_recs, disco_subs = get_disco_data_cached()


# ==========================================
# Dash Application Initialization
# ==========================================

app = dash.Dash(
    __name__,
    title="MCI Wake - Visualizer & Stitching Studio",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Dark Theme Styling Dictionary
DARK_STYLE = {
    'bg': '#0f172a',
    'card_bg': '#1e293b',
    'border': '#334155',
    'text': '#f8fafc',
    'text_muted': '#94a3b8',
    'accent': '#6366f1',
    'accent_hover': '#4f46e5',
}

card_style = {
    'backgroundColor': DARK_STYLE['card_bg'],
    'border': f"1px solid {DARK_STYLE['border']}",
    'borderRadius': '12px',
    'padding': '20px',
    'marginBottom': '20px',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.3)',
}

control_group_style = {
    'marginBottom': '16px',
}

label_style = {
    'color': DARK_STYLE['text'],
    'fontWeight': '600',
    'fontSize': '0.9rem',
    'marginBottom': '6px',
    'display': 'block',
}


# ==========================================
# Layout Components
# ==========================================

header_component = html.Div(
    style={
        'backgroundColor': DARK_STYLE['card_bg'],
        'borderBottom': f"1px solid {DARK_STYLE['border']}",
        'padding': '24px 32px',
        'marginBottom': '24px',
        'borderRadius': '0 0 16px 16px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'boxShadow': '0 10px 15px -3px rgba(0, 0, 0, 0.3)',
    },
    children=[
        html.Div([
            html.H1(
                "⚡ MCI Wake Visualizer & Stitching Studio",
                style={'color': '#f8fafc', 'margin': '0', 'fontSize': '1.75rem', 'fontWeight': '700'},
            ),
            html.P(
                "Interactive exploration of EPN gesture data, DISCO ADL noise data, and constant-power Hanning cross-fade stitching",
                style={'color': DARK_STYLE['text_muted'], 'margin': '4px 0 0 0', 'fontSize': '0.95rem'},
            ),
        ]),
        html.Div([
            html.Span("Status: ", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.85rem'}),
            html.Span(
                "Active",
                style={
                    'backgroundColor': '#065f46',
                    'color': '#34d399',
                    'padding': '4px 12px',
                    'borderRadius': '9999px',
                    'fontSize': '0.85rem',
                    'fontWeight': '600',
                },
            ),
        ]),
    ],
)

tab_style = {
    'backgroundColor': DARK_STYLE['card_bg'],
    'color': DARK_STYLE['text_muted'],
    'border': f"1px solid {DARK_STYLE['border']}",
    'padding': '12px 24px',
    'fontWeight': '600',
    'borderRadius': '8px 8px 0 0',
    'marginRight': '4px',
}

selected_tab_style = {
    'backgroundColor': DARK_STYLE['accent'],
    'color': '#ffffff',
    'border': f"1px solid {DARK_STYLE['accent']}",
    'padding': '12px 24px',
    'fontWeight': '700',
    'borderRadius': '8px 8px 0 0',
    'marginRight': '4px',
}

# --- TAB 1: EPN LAYOUT ---
tab_epn_layout = html.Div([
    html.Div(
        style={'display': 'grid', 'gridTemplateColumns': '320px 1fr', 'gap': '24px'},
        children=[
            # Sidebar Controls
            html.Div(
                style=card_style,
                children=[
                    html.H3("EPN Data Controls", style={'color': DARK_STYLE['text'], 'marginTop': '0', 'marginBottom': '16px'}),
                    html.Div([
                        html.Label("Dataset Split", style=label_style),
                        dcc.RadioItems(
                            id="epn-split-radio",
                            options=[
                                {'label': ' Training', 'value': 'training'},
                                {'label': ' Testing', 'value': 'testing'},
                            ],
                            value='training',
                            labelStyle={'color': DARK_STYLE['text'], 'marginRight': '16px', 'cursor': 'pointer'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Subject ID", style=label_style),
                        dcc.Dropdown(
                            id="epn-subject-dd",
                            clearable=False,
                            style={'color': '#000'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Gesture Filter", style=label_style),
                        dcc.Dropdown(
                            id="epn-gesture-dd",
                            options=[{'label': 'All Gestures', 'value': -1}] + [
                                {'label': f"{v} (id: {k})", 'value': k} for k, v in GESTURE_NAMES.items()
                            ],
                            value=-1,
                            clearable=False,
                            style={'color': '#000'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Sample Selector", style=label_style),
                        dcc.Dropdown(
                            id="epn-sample-dd",
                            clearable=False,
                            style={'color': '#000'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Signal Processing", style=label_style),
                        dcc.Checklist(
                            id="epn-normalize-chk",
                            options=[{'label': ' Apply safe z-normalization', 'value': 'norm'}],
                            value=['norm'],
                            labelStyle={'color': DARK_STYLE['text'], 'cursor': 'pointer'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Visible EMG Channels", style=label_style),
                        dcc.Checklist(
                            id="epn-channels-chk",
                            options=[{'label': f' Ch {ch+1}', 'value': ch} for ch in range(8)],
                            value=list(range(8)),
                            labelStyle={'color': DARK_STYLE['text'], 'display': 'inline-block', 'width': '50%', 'cursor': 'pointer'},
                        ),
                    ], style=control_group_style),
                ],
            ),
            # Main View Display
            html.Div([
                html.Div(id="epn-info-card", style=card_style),
                html.Div(
                    style=card_style,
                    children=[
                        dcc.Graph(id="epn-waveform-graph", config={'responsive': True}),
                    ],
                ),
                html.Div(
                    style=card_style,
                    children=[
                        dcc.Graph(id="epn-rms-graph", config={'responsive': True}),
                    ],
                ),
            ]),
        ],
    ),
])

# --- TAB 2: DISCO LAYOUT ---
tab_disco_layout = html.Div([
    html.Div(
        style={'display': 'grid', 'gridTemplateColumns': '320px 1fr', 'gap': '24px'},
        children=[
            # Sidebar Controls
            html.Div(
                style=card_style,
                children=[
                    html.H3("DISCO Data Controls", style={'color': DARK_STYLE['text'], 'marginTop': '0', 'marginBottom': '16px'}),
                    html.Div([
                        html.Label("Subject ID", style=label_style),
                        dcc.Dropdown(
                            id="disco-subject-dd",
                            options=[{'label': f"Subject S{s}", 'value': s} for s in sorted(list(set(disco_subs)))],
                            value=sorted(list(set(disco_subs)))[0] if len(disco_subs) > 0 else 1,
                            clearable=False,
                            style={'color': '#000'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Recording File", style=label_style),
                        dcc.Dropdown(
                            id="disco-recording-dd",
                            clearable=False,
                            style={'color': '#000'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Display Mode", style=label_style),
                        dcc.RadioItems(
                            id="disco-mode-radio",
                            options=[
                                {'label': ' Full Recording', 'value': 'full'},
                                {'label': ' Windowed Slices', 'value': 'windowed'},
                            ],
                            value='full',
                            labelStyle={'color': DARK_STYLE['text'], 'marginRight': '12px', 'cursor': 'pointer'},
                        ),
                    ], style=control_group_style),
                    html.Div(
                        id="disco-window-controls",
                        style=control_group_style,
                        children=[
                            html.Label("Window Slice Index", style=label_style),
                            dcc.Slider(id="disco-window-idx-slider", min=0, max=10, step=1, value=0),
                        ],
                    ),
                    html.Div([
                        html.Label("Signal Processing", style=label_style),
                        dcc.Checklist(
                            id="disco-normalize-chk",
                            options=[{'label': ' Apply safe z-normalization', 'value': 'norm'}],
                            value=['norm'],
                            labelStyle={'color': DARK_STYLE['text'], 'cursor': 'pointer'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Visible Channels", style=label_style),
                        dcc.Checklist(
                            id="disco-channels-chk",
                            options=[{'label': f' Ch {ch+1}', 'value': ch} for ch in range(8)],
                            value=list(range(8)),
                            labelStyle={'color': DARK_STYLE['text'], 'display': 'inline-block', 'width': '50%', 'cursor': 'pointer'},
                        ),
                    ], style=control_group_style),
                ],
            ),
            # Main View Display
            html.Div([
                html.Div(id="disco-info-card", style=card_style),
                html.Div(
                    style=card_style,
                    children=[
                        dcc.Graph(id="disco-waveform-graph", config={'responsive': True}),
                    ],
                ),
            ]),
        ],
    ),
])

# --- TAB 3: STITCHING LAYOUT ---
tab_stitch_layout = html.Div([
    html.Div(
        style={'display': 'grid', 'gridTemplateColumns': '340px 1fr', 'gap': '24px'},
        children=[
            # Sidebar Controls
            html.Div(
                style=card_style,
                children=[
                    html.H3("Stitching Controls", style={'color': DARK_STYLE['text'], 'marginTop': '0', 'marginBottom': '16px'}),
                    html.Div([
                        html.Label("Data Source", style=label_style),
                        dcc.Dropdown(
                            id="stitch-source-dd",
                            options=[
                                {'label': '🎲 Random Gaussian Noise', 'value': 'noise'},
                                {'label': '🌊 Synthetic Multi-Sine Waves', 'value': 'sine'},
                                {'label': '📈 Random Walk / Smooth Drifts', 'value': 'walk'},
                                {'label': '💪 Real EPN Gesture Slices', 'value': 'epn'},
                                {'label': '🔊 Real DISCO Noise Slices', 'value': 'disco'},
                                {'label': '🔀 Mixed Sequence (EPN + DISCO + Synth)', 'value': 'mixed'},
                            ],
                            value='sine',
                            clearable=False,
                            style={'color': '#000'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Number of Segments", style=label_style),
                        dcc.Slider(
                            id="stitch-num-segments-slider",
                            min=2, max=8, step=1, value=3,
                            marks={i: str(i) for i in range(2, 9)},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Segment Length (samples)", style=label_style),
                        dcc.Slider(
                            id="stitch-seg-len-slider",
                            min=60, max=400, step=20, value=150,
                            marks={60: '60', 150: '150', 300: '300', 400: '400'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Overlap Samples (cross-fade)", style=label_style),
                        dcc.Slider(
                            id="stitch-overlap-slider",
                            min=2, max=50, step=1, value=15,
                            marks={2: '2', 15: '15 (def)', 30: '30', 50: '50'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Channel View", style=label_style),
                        dcc.Dropdown(
                            id="stitch-channel-dd",
                            options=[{'label': 'All 8 Channels', 'value': -1}] + [
                                {'label': f'Channel {ch+1}', 'value': ch} for ch in range(8)
                            ],
                            value=-1,
                            clearable=False,
                            style={'color': '#000'},
                        ),
                    ], style=control_group_style),
                    html.Div([
                        html.Label("Inspect Seam Close-up", style=label_style),
                        dcc.Dropdown(
                            id="stitch-seam-dd",
                            clearable=False,
                            style={'color': '#000'},
                        ),
                    ], style=control_group_style),
                    html.Button(
                        "🎲 Re-generate / Re-stitch",
                        id="stitch-regen-btn",
                        n_clicks=0,
                        style={
                            'width': '100%',
                            'padding': '12px',
                            'backgroundColor': DARK_STYLE['accent'],
                            'color': '#fff',
                            'border': 'none',
                            'borderRadius': '8px',
                            'fontWeight': '700',
                            'cursor': 'pointer',
                            'marginTop': '12px',
                        },
                    ),
                ],
            ),
            # Main View Display
            html.Div([
                html.Div(id="stitch-summary-card", style=card_style),
                html.Div(
                    style=card_style,
                    children=[
                        dcc.Graph(id="stitch-main-graph", config={'responsive': True}),
                    ],
                ),
                html.Div(
                    style=card_style,
                    children=[
                        dcc.Graph(id="stitch-seam-graph", config={'responsive': True}),
                    ],
                ),
            ]),
        ],
    ),
])

# App Root Layout
app.layout = html.Div(
    style={
        'backgroundColor': DARK_STYLE['bg'],
        'color': DARK_STYLE['text'],
        'minHeight': '100vh',
        'fontFamily': 'Inter, system-ui, -apple-system, sans-serif',
        'paddingBottom': '40px',
    },
    children=[
        header_component,
        html.Div(
            style={'padding': '0 32px'},
            children=[
                dcc.Tabs(
                    id="main-tabs",
                    value="tab-stitch",
                    children=[
                        dcc.Tab(
                            label="🧵 Stitching Visualizer (Arbitrary/Random Data)",
                            value="tab-stitch",
                            children=tab_stitch_layout,
                            style=tab_style,
                            selected_style=selected_tab_style,
                        ),
                        dcc.Tab(
                            label="📊 EPN Dataset",
                            value="tab-epn",
                            children=tab_epn_layout,
                            style=tab_style,
                            selected_style=selected_tab_style,
                        ),
                        dcc.Tab(
                            label="🌊 DISCO ADL Dataset",
                            value="tab-disco",
                            children=tab_disco_layout,
                            style=tab_style,
                            selected_style=selected_tab_style,
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ==========================================
# Callbacks
# ==========================================


# --- TAB 1: EPN CALLBACKS ---

@app.callback(
    Output("epn-subject-dd", "options"),
    Output("epn-subject-dd", "value"),
    Input("epn-split-radio", "value"),
)
def update_epn_subjects(split):
    emg_dict, labels_dict, subs_dict = get_epn_data_cached()
    subs = subs_dict.get(split, [])
    unique_subs = sorted(list(set(subs)))
    options = [{'label': f"Subject {s}", 'value': s} for s in unique_subs]
    default_val = unique_subs[0] if len(unique_subs) > 0 else 1
    return options, default_val


@app.callback(
    Output("epn-sample-dd", "options"),
    Output("epn-sample-dd", "value"),
    Input("epn-split-radio", "value"),
    Input("epn-subject-dd", "value"),
    Input("epn-gesture-dd", "value"),
)
def update_epn_samples(split, subject_id, gesture_id):
    emg_dict, labels_dict, subs_dict = get_epn_data_cached()
    emg_list = emg_dict.get(split, [])
    lab_list = labels_dict.get(split, [])
    sub_list = subs_dict.get(split, [])

    matching_indices = []
    for idx, (sub, lab) in enumerate(zip(sub_list, lab_list)):
        if sub == subject_id:
            if gesture_id == -1 or lab == gesture_id:
                matching_indices.append(idx)

    if not matching_indices:
        return [{'label': 'No matching samples', 'value': -1}], -1

    options = [
        {'label': f"Sample #{i} ({GESTURE_NAMES.get(lab_list[i], 'Unknown')}, {len(emg_list[i])} pts)", 'value': i}
        for i in matching_indices[:100]  # Limit dropdown length for speed
    ]
    return options, options[0]['value']


@app.callback(
    Output("epn-info-card", "children"),
    Output("epn-waveform-graph", "figure"),
    Output("epn-rms-graph", "figure"),
    Input("epn-split-radio", "value"),
    Input("epn-sample-dd", "value"),
    Input("epn-normalize-chk", "value"),
    Input("epn-channels-chk", "value"),
)
def update_epn_view(split, sample_idx, norm_chk, selected_channels):
    if sample_idx is None or sample_idx < 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(paper_bgcolor=DARK_STYLE['card_bg'], plot_bgcolor=DARK_STYLE['card_bg'])
        return html.Div("No sample selected"), empty_fig, empty_fig

    emg_dict, labels_dict, subs_dict = get_epn_data_cached()
    data = emg_dict[split][sample_idx]
    label_id = labels_dict[split][sample_idx]
    sub_id = subs_dict[split][sample_idx]

    if 'norm' in norm_chk:
        data = safe_znormalize_global(data)

    T, n_ch = data.shape
    duration_ms = (T / 200.0) * 1000.0  # Assume 200 Hz Myo armband sampling rate

    rms_per_ch = np.sqrt(np.mean(data ** 2, axis=0))

    # Info card
    info = html.Div(
        style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(180px, 1fr))', 'gap': '16px'},
        children=[
            html.Div([html.Div("Subject ID", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"User {sub_id}", style={'fontSize': '1.2rem', 'fontWeight': '700'})]),
            html.Div([html.Div("Gesture Name", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(GESTURE_NAMES.get(label_id, 'Unknown'), style={'fontSize': '1.2rem', 'fontWeight': '700', 'color': DARK_STYLE['accent']})]),
            html.Div([html.Div("Sample Count", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"{T} timesteps", style={'fontSize': '1.2rem', 'fontWeight': '700'})]),
            html.Div([html.Div("Duration (@ 200 Hz)", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"{duration_ms:.0f} ms", style={'fontSize': '1.2rem', 'fontWeight': '700'})]),
            html.Div([html.Div("Overall Mean RMS", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"{np.mean(rms_per_ch):.3f}", style={'fontSize': '1.2rem', 'fontWeight': '700'})]),
        ],
    )

    # Waveform Figure
    fig_wave = go.Figure()
    time_axis = np.arange(T)
    for ch in range(n_ch):
        if ch in selected_channels:
            fig_wave.add_trace(go.Scatter(
                x=time_axis,
                y=data[:, ch],
                mode='lines',
                name=f'Channel {ch+1}',
                line=dict(color=CHANNEL_COLORS[ch % 8], width=1.5),
            ))

    fig_wave.update_layout(
        title=f"EPN Sample Waveform (Subject {sub_id} - {GESTURE_NAMES.get(label_id, 'Unknown')})",
        xaxis_title="Timestep (samples)",
        yaxis_title="Amplitude (normalized)" if 'norm' in norm_chk else "Raw Amplitude",
        template="plotly_dark",
        paper_bgcolor=DARK_STYLE['card_bg'],
        plot_bgcolor=DARK_STYLE['card_bg'],
        font=dict(color=DARK_STYLE['text']),
        margin=dict(l=40, r=20, t=50, b=40),
        height=350,
    )

    # RMS Bar Chart
    fig_rms = go.Figure()
    fig_rms.add_trace(go.Bar(
        x=[f"Ch {ch+1}" for ch in range(8)],
        y=rms_per_ch,
        marker_color=CHANNEL_COLORS,
    ))
    fig_rms.update_layout(
        title="Channel Power Distribution (RMS Amplitude)",
        xaxis_title="EMG Channel",
        yaxis_title="RMS Amplitude",
        template="plotly_dark",
        paper_bgcolor=DARK_STYLE['card_bg'],
        plot_bgcolor=DARK_STYLE['card_bg'],
        font=dict(color=DARK_STYLE['text']),
        margin=dict(l=40, r=20, t=50, b=40),
        height=250,
    )

    return info, fig_wave, fig_rms


# --- TAB 2: DISCO CALLBACKS ---

@app.callback(
    Output("disco-recording-dd", "options"),
    Output("disco-recording-dd", "value"),
    Input("disco-subject-dd", "value"),
)
def update_disco_recordings(subject_id):
    recs, subs = get_disco_data_cached()
    indices = [i for i, s in enumerate(subs) if s == subject_id]
    options = [{'label': f"Recording #{i+1} ({len(recs[i])} samples)", 'value': i} for i in indices]
    default_val = indices[0] if len(indices) > 0 else 0
    return options, default_val


@app.callback(
    Output("disco-window-controls", "style"),
    Output("disco-window-idx-slider", "max"),
    Input("disco-mode-radio", "value"),
    Input("disco-recording-dd", "value"),
)
def toggle_disco_window_controls(mode, rec_idx):
    if mode != 'windowed' or rec_idx is None:
        return {'display': 'none'}, 10

    recs, subs = get_disco_data_cached()
    data = recs[rec_idx]
    # Estimate window count with min length 150, step 50
    n_windows = max(1, (len(data) - 150) // 50)
    return control_group_style, n_windows - 1


@app.callback(
    Output("disco-info-card", "children"),
    Output("disco-waveform-graph", "figure"),
    Input("disco-subject-dd", "value"),
    Input("disco-recording-dd", "value"),
    Input("disco-mode-radio", "value"),
    Input("disco-window-idx-slider", "value"),
    Input("disco-normalize-chk", "value"),
    Input("disco-channels-chk", "value"),
)
def update_disco_view(subject_id, rec_idx, view_mode, win_idx, norm_chk, selected_channels):
    if rec_idx is None:
        empty_fig = go.Figure()
        return html.Div("No recording selected"), empty_fig

    recs, subs = get_disco_data_cached()
    data = recs[rec_idx]

    if view_mode == 'windowed':
        # Slice out window
        start_i = min(win_idx * 50, max(0, len(data) - 150))
        end_i = min(start_i + 200, len(data))
        data = data[start_i:end_i]

    if 'norm' in norm_chk:
        data = safe_znormalize_global(data)

    T, n_ch = data.shape
    rms_per_ch = np.sqrt(np.mean(data ** 2, axis=0))

    info = html.Div(
        style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(180px, 1fr))', 'gap': '16px'},
        children=[
            html.Div([html.Div("Subject ID", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"Subject S{subject_id}", style={'fontSize': '1.2rem', 'fontWeight': '700'})]),
            html.Div([html.Div("Recording ID", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"Rec #{rec_idx+1}", style={'fontSize': '1.2rem', 'fontWeight': '700', 'color': DARK_STYLE['accent']})]),
            html.Div([html.Div("View Mode", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(view_mode.capitalize(), style={'fontSize': '1.2rem', 'fontWeight': '700'})]),
            html.Div([html.Div("Sample Timesteps", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"{T} pts", style={'fontSize': '1.2rem', 'fontWeight': '700'})]),
            html.Div([html.Div("Mean RMS Noise Level", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"{np.mean(rms_per_ch):.3f}", style={'fontSize': '1.2rem', 'fontWeight': '700'})]),
        ],
    )

    fig = go.Figure()
    time_axis = np.arange(T)
    for ch in range(n_ch):
        if ch in selected_channels:
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=data[:, ch],
                mode='lines',
                name=f'Channel {ch+1}',
                line=dict(color=CHANNEL_COLORS[ch % 8], width=1.5),
            ))

    fig.update_layout(
        title=f"DISCO ADL Recording (Subject S{subject_id} - Rec #{rec_idx+1})",
        xaxis_title="Timestep (samples)",
        yaxis_title="Amplitude (normalized)" if 'norm' in norm_chk else "Raw Amplitude",
        template="plotly_dark",
        paper_bgcolor=DARK_STYLE['card_bg'],
        plot_bgcolor=DARK_STYLE['card_bg'],
        font=dict(color=DARK_STYLE['text']),
        margin=dict(l=40, r=20, t=50, b=40),
        height=400,
    )

    return info, fig


# --- TAB 3: STITCHING CALLBACKS ---

@app.callback(
    Output("stitch-seam-dd", "options"),
    Output("stitch-seam-dd", "value"),
    Input("stitch-num-segments-slider", "value"),
)
def update_seam_dropdown(num_segments):
    n_seams = max(1, num_segments - 1)
    options = [{'label': f"Seam Boundary {i+1} (Seg {i+1} ➔ Seg {i+2})", 'value': i} for i in range(n_seams)]
    return options, 0


@app.callback(
    Output("stitch-summary-card", "children"),
    Output("stitch-main-graph", "figure"),
    Output("stitch-seam-graph", "figure"),
    Input("stitch-source-dd", "value"),
    Input("stitch-num-segments-slider", "value"),
    Input("stitch-seg-len-slider", "value"),
    Input("stitch-overlap-slider", "value"),
    Input("stitch-channel-dd", "value"),
    Input("stitch-seam-dd", "value"),
    Input("stitch-regen-btn", "n_clicks"),
)
def update_stitching_view(source_type, n_segs, seg_len, overlap_samples, selected_ch, seam_idx, n_clicks):
    # Set seed based on n_clicks + parameters so re-generate button works
    seed = (n_clicks * 1337 + n_segs * 42 + seg_len + overlap_samples) % 1000000
    rng = np.random.default_rng(seed)

    raw_segments = []
    segment_labels = []

    # Generate or fetch segment slices
    for i in range(n_segs):
        length = seg_len + rng.integers(-10, 15)  # slight variation in segment length

        if source_type == 'noise':
            arr = rng.normal(0, 0.5, size=(length, 8))
            label = f"Noise Seg {i+1}"
        elif source_type == 'sine':
            t = np.linspace(0, 4 * np.pi, length)
            freq = (i + 1) * 1.5
            arr = np.zeros((length, 8))
            for ch in range(8):
                arr[:, ch] = np.sin(t * freq + ch * 0.5) * (0.8 + 0.4 * np.cos(t))
            label = f"Sine Seg {i+1} ({freq:.1f}Hz)"
        elif source_type == 'walk':
            arr = np.cumsum(rng.normal(0, 0.1, size=(length, 8)), axis=0)
            label = f"Walk Seg {i+1}"
        elif source_type == 'epn':
            emg_dict, labels_dict, _ = get_epn_data_cached()
            train_emg = emg_dict['training']
            train_labs = labels_dict['training']
            idx = rng.integers(0, len(train_emg))
            sample = train_emg[idx]
            g_name = GESTURE_NAMES.get(train_labs[idx], 'Gesture')
            if len(sample) > length:
                st = rng.integers(0, len(sample) - length)
                arr = sample[st:st + length]
            else:
                arr = sample
            label = f"EPN ({g_name}) Seg {i+1}"
        elif source_type == 'disco':
            recs, _ = get_disco_data_cached()
            idx = rng.integers(0, len(recs))
            sample = recs[idx]
            if len(sample) > length:
                st = rng.integers(0, len(sample) - length)
                arr = sample[st:st + length]
            else:
                arr = sample
            label = f"DISCO Noise Seg {i+1}"
        elif source_type == 'mixed':
            if i % 2 == 0:
                emg_dict, labels_dict, _ = get_epn_data_cached()
                idx = rng.integers(0, len(emg_dict['training']))
                arr = emg_dict['training'][idx][:length]
                label = f"EPN Gesture Seg {i+1}"
            else:
                recs, _ = get_disco_data_cached()
                idx = rng.integers(0, len(recs))
                arr = recs[idx][:length]
                label = f"DISCO Noise Seg {i+1}"

        # Ensure correct 2D shape (T, 8)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] < 8:
            arr = np.pad(arr, ((0, 0), (0, 8 - arr.shape[1])))
        elif arr.shape[1] > 8:
            arr = arr[:, :8]

        raw_segments.append(arr)
        segment_labels.append(label)

    # Perform Hanning cross-fade stitch
    start_time = time.perf_counter()
    stitched_arr = stitch(raw_segments, overlap_samples=overlap_samples)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    total_input_len = sum(len(s) for s in raw_segments)
    actual_stitched_len = len(stitched_arr)
    expected_overlap_loss = (n_segs - 1) * overlap_samples

    # Compute seam boundary positions in the stitched array
    seam_boundaries = []
    curr_pos = 0
    for i in range(len(raw_segments) - 1):
        seg_len_curr = len(raw_segments[i])
        seam_start = curr_pos + seg_len_curr - overlap_samples
        seam_end = seam_start + overlap_samples
        seam_boundaries.append((seam_start, seam_end))
        curr_pos = seam_start

    # Summary KPI Card
    summary_card = html.Div(
        style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))', 'gap': '16px'},
        children=[
            html.Div([html.Div("Input Segments", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"{n_segs} segments ({total_input_len} samples)", style={'fontSize': '1.1rem', 'fontWeight': '700'})]),
            html.Div([html.Div("Overlap Parameter", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"{overlap_samples} samples cross-fade", style={'fontSize': '1.1rem', 'fontWeight': '700', 'color': DARK_STYLE['accent']})]),
            html.Div([html.Div("Stitched Array Length", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"{actual_stitched_len} samples", style={'fontSize': '1.1rem', 'fontWeight': '700', 'color': '#34d399'})]),
            html.Div([html.Div("Stitch Compute Time", style={'color': DARK_STYLE['text_muted'], 'fontSize': '0.8rem'}), html.Div(f"{elapsed_ms:.3f} ms", style={'fontSize': '1.1rem', 'fontWeight': '700'})]),
        ],
    )

    # --- Plot 1: Main Stitched Output & Seam Overlay ---
    fig_main = go.Figure()

    channels_to_plot = list(range(8)) if selected_ch == -1 else [selected_ch]

    time_axis = np.arange(actual_stitched_len)
    for ch in channels_to_plot:
        fig_main.add_trace(go.Scatter(
            x=time_axis,
            y=stitched_arr[:, ch],
            mode='lines',
            name=f'Stitched Ch {ch+1}',
            line=dict(color=CHANNEL_COLORS[ch % 8], width=2),
        ))

    # Add shaded vertical region shapes for seams
    shapes = []
    annotations = []
    for idx, (s_start, s_end) in enumerate(seam_boundaries):
        shapes.append(dict(
            type="rect",
            xref="x", yref="paper",
            x0=s_start, x1=s_end,
            y0=0, y1=1,
            fillcolor="rgba(129, 140, 248, 0.25)",
            line=dict(width=1, color="rgba(129, 140, 248, 0.6)"),
        ))
        annotations.append(dict(
            x=(s_start + s_end) / 2,
            y=1.05,
            xref="x", yref="paper",
            text=f"Seam {idx+1}",
            showarrow=False,
            font=dict(color="#a5b4fc", size=11, weight="bold"),
        ))

    fig_main.update_layout(
        title="Continuous Stitched Signal (Shaded regions indicate Hanning cross-fade seams)",
        xaxis_title="Timestep (samples)",
        yaxis_title="Signal Amplitude",
        template="plotly_dark",
        paper_bgcolor=DARK_STYLE['card_bg'],
        plot_bgcolor=DARK_STYLE['card_bg'],
        font=dict(color=DARK_STYLE['text']),
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )

    # --- Plot 2: Seam Close-up Cross-fade Inspector ---
    fig_seam = go.Figure()

    if seam_idx is not None and seam_idx < len(seam_boundaries):
        s_start, s_end = seam_boundaries[seam_idx]
        inspect_ch = 0 if selected_ch == -1 else selected_ch

        # Extract boundary math
        seg_out = raw_segments[seam_idx]
        seg_in = raw_segments[seam_idx + 1]

        overlap = min(len(seg_out), len(seg_in), overlap_samples)
        theta = np.linspace(0, np.pi / 2, overlap)
        w_out = (np.cos(theta) ** 2)
        w_in = (np.sin(theta) ** 2)

        seam_out_sig = seg_out[-overlap:, inspect_ch]
        seam_in_sig = seg_in[:overlap, inspect_ch]

        blended = (seam_out_sig * w_out) + (seam_in_sig * w_in)

        x_seam = np.arange(overlap)

        # Plot components
        fig_seam.add_trace(go.Scatter(
            x=x_seam, y=seam_out_sig * w_out,
            mode='lines+markers', name=f'Seg {seam_idx+1} Tail (x cos²θ)',
            line=dict(color='#f87171', dash='dash', width=2),
        ))
        fig_seam.add_trace(go.Scatter(
            x=x_seam, y=seam_in_sig * w_in,
            mode='lines+markers', name=f'Seg {seam_idx+2} Head (x sin²θ)',
            line=dict(color='#38bdf8', dash='dash', width=2),
        ))
        fig_seam.add_trace(go.Scatter(
            x=x_seam, y=blended,
            mode='lines+markers', name='Blended Seam Output',
            line=dict(color='#34d399', width=3),
        ))

        # Weighting curves on secondary Y-axis
        fig_seam.add_trace(go.Scatter(
            x=x_seam, y=w_out,
            mode='lines', name='Fade-Out Weight w_out (cos²θ)',
            line=dict(color='#fbbf24', width=1.5, dash='dot'),
            yaxis='y2',
        ))
        fig_seam.add_trace(go.Scatter(
            x=x_seam, y=w_in,
            mode='lines', name='Fade-In Weight w_in (sin²θ)',
            line=dict(color='#a78bfa', width=1.5, dash='dot'),
            yaxis='y2',
        ))

        fig_seam.update_layout(
            title=f"Seam Boundary {seam_idx+1} Close-Up Inspector (Channel {inspect_ch+1})",
            xaxis_title="Overlap Timestep (samples)",
            yaxis_title="Signal Amplitude",
            yaxis2=dict(
                title="Hanning Weight (0.0 to 1.0)",
                overlaying="y",
                side="right",
                range=[-0.05, 1.05],
            ),
            template="plotly_dark",
            paper_bgcolor=DARK_STYLE['card_bg'],
            plot_bgcolor=DARK_STYLE['card_bg'],
            font=dict(color=DARK_STYLE['text']),
            margin=dict(l=40, r=40, t=50, b=40),
            height=350,
        )

    return summary_card, fig_main, fig_seam


# ==========================================
# Script Entry Point
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"\n=======================================================")
    print(f"🚀 Starting MCI Wake Dash App at http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(debug=True, port=port)
