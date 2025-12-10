import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, callback
from Isolation_Forest import IsolationForestAnomalyDetector

# -----------------------------------------------------------------------------
# Load ORCL data ONCE at module import
# -----------------------------------------------------------------------------
# Assumes ORCL.csv is available in the working directory where the Dash app runs.
df_orcl = pd.read_csv("ORCL.csv").sort_values(by="Date", ascending=True)

# 1-day returns in percent
df_orcl["Return_1d"] = df_orcl["Close"].pct_change() * 100.0

# Drop the first NaN return, shape (M, 1)
orcl_returns = df_orcl["Return_1d"].dropna().values.reshape(-1, 1)

# Matching dates for the returns (same length as orcl_returns)
orcl_dates = df_orcl.loc[df_orcl["Return_1d"].notna(), "Date"].values

# Matching prices for the returns
orcl_prices = df_orcl.loc[df_orcl["Return_1d"].notna(), "Close"].values

M = orcl_returns.shape[0]

# -----------------------------------------------------------------------------
# Page layout
# -----------------------------------------------------------------------------
binary_partition_layout = [
    # Controls
    html.Div([
        html.Div([
            html.Button(
                "Run Binary Partition on ORCL Returns",
                id='bp-run-orcl-button',
                n_clicks=0,
                style={
                    'backgroundColor': '#2b2d42',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '8px',
                    'padding': '12px 24px',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                    'boxShadow': '2px 2px 5px rgba(0,0,0,0.15)',
                    'transition': '0.2s',
                    'width': '100%'
                }
            )
        ], style={'flex': '1'})
    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'center',
        'padding': '15px',
        'backgroundColor': 'white',
        'borderRadius': '10px',
        'boxShadow': '0px 3px 8px rgba(0,0,0,0.07)',
        'border': '1px solid #edf2f4',
        'marginBottom': '20px'
    }),
        
    # Main figure
    html.Div(
        [
            dcc.Graph(
                id='bp-scatter-plot',
                figure={},
                style={
                    'height': '80vh',
                    'width': '100%',
                }
            )
        ],
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'width': '100%'
        }
    ),
]


# -----------------------------------------------------------------------------
# Callback: run a single random binary partition on fixed ORCL 1d returns
# -----------------------------------------------------------------------------
@callback(
    Output('bp-scatter-plot', 'figure'),
    Input('bp-run-orcl-button', 'n_clicks')
)
def generate_single_step_orcl_binary_partition(n_clicks):
    """
    Each time the button is clicked, we run a single binary partition step
    on the fixed ORCL 1-day return series (in %).

    - Data is deterministic (loaded once from ORCL.csv at import).
    - Split is random because IsolationForestAnomalyDetector.binary_partition
      samples a cut point uniformly between min and max of the returns.
    """

    # -------------------------
    # 1) Build 1 x 2 figure (big, like iTree)
    # -------------------------
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "1D Binary Partition on ORCL 1-Day Returns (%)",
            "Tree Node After Split"
        ),
        horizontal_spacing=0.12,
        column_widths=[0.6, 0.4]
    )

    # -------------------------
    # 2) Run binary partition on returns
    # -------------------------
    one_dimensional_iForest_model = IsolationForestAnomalyDetector(orcl_returns)

    (split_1d_from_left,
     split_1d_from_right,
     random_1d_axis_to_cut,
     random_1d_point_on_axis_to_cut) = one_dimensional_iForest_model.binary_partition()

    # -------------------------
    # Left plot: 1D scatter + split line + shading
    # -------------------------
    all_returns = orcl_returns.flatten()
    left_vals = split_1d_from_left.flatten()
    right_vals = split_1d_from_right.flatten()

    # Left subset
    fig.add_trace(
        go.Scatter(
            x=left_vals,
            y=np.zeros(len(left_vals)),
            mode='markers',
            name='Left Split',
            marker=dict(
                size=9,
                opacity=0.75,
                color='rgb(65, 105, 225)',
                line=dict(color='rgb(30, 60, 180)', width=1)
            )
        ),
        row=1, col=1
    )

    # Right subset
    fig.add_trace(
        go.Scatter(
            x=right_vals,
            y=np.zeros(len(right_vals)),
            mode='markers',
            name='Right Split',
            marker=dict(
                size=9,
                opacity=0.75,
                color='rgb(220, 20, 60)',
                line=dict(color='rgb(180, 10, 40)', width=1)
            )
        ),
        row=1, col=1
    )

    # Split line
    fig.add_vline(
        x=random_1d_point_on_axis_to_cut,
        line_dash="dash",
        line_color="black",
        line_width=2,
        row=1, col=1
    )

    # Bounds for shading
    x_min = all_returns.min()
    x_max = all_returns.max()

    # Small x padding so it doesn’t look cramped
    pad_x = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    x_min -= pad_x
    x_max += pad_x

    # Shaded regions
    fig.add_shape(
        type="rect",
        x0=x_min,
        x1=random_1d_point_on_axis_to_cut,
        y0=-0.3,
        y1=0.3,
        fillcolor="rgba(65, 105, 225, 0.15)",
        line=dict(width=0),
        layer='below',
        row=1, col=1
    )
    fig.add_shape(
        type="rect",
        x0=random_1d_point_on_axis_to_cut,
        x1=x_max,
        y0=-0.3,
        y1=0.3,
        fillcolor="rgba(220, 20, 60, 0.15)",
        line=dict(width=0),
        layer='below',
        row=1, col=1
    )

    # Counts annotations (kept, but you can remove if you want even cleaner)
    fig.add_annotation(
        x=(x_min + random_1d_point_on_axis_to_cut) / 2,
        y=0.25,
        text=f"Left: n = {len(split_1d_from_left)}",
        showarrow=False,
        font=dict(size=11, color='rgb(65, 105, 225)'),
        row=1, col=1
    )
    fig.add_annotation(
        x=(random_1d_point_on_axis_to_cut + x_max) / 2,
        y=0.25,
        text=f"Right: n = {len(split_1d_from_right)}",
        showarrow=False,
        font=dict(size=11, color='rgb(220, 20, 60)'),
        row=1, col=1
    )

    fig.update_xaxes(
        title_text="1-Day Return (%)",
        range=[x_min, x_max],
        row=1, col=1
    )
    fig.update_yaxes(
        visible=False,
        row=1, col=1
    )

    # -------------------------
    # Right plot: simple tree node diagram (fixed-size markers)
    # -------------------------
    n_total_1d = len(all_returns)
    n_left_1d = len(split_1d_from_left)
    n_right_1d = len(split_1d_from_right)

    NODE_SIZE = 12

    # Root node
    fig.add_trace(
        go.Scatter(
            x=[0.5],
            y=[1.0],
            mode="markers+text",
            text=[f"<b>Root</b><br>n={n_total_1d}"],
            textposition="top center",
            textfont=dict(size=11, color='#2b2d42'),
            marker=dict(
                size=NODE_SIZE,
                color='#f8f9fa',
                line=dict(color='#2b2d42', width=1.5),
                symbol='circle'
            ),
            showlegend=False,
            hovertext=[(
                f"Node role: root<br>"
                f"Node type: internal<br>"
                f"Depth: 0<br>"
                f"Size (n): {n_total_1d}<br>"
                f"Split: r = {random_1d_point_on_axis_to_cut:.2f}%"
            )],
            hoverinfo='text'
        ),
        row=1, col=2
    )

    # Left child node
    fig.add_trace(
        go.Scatter(
            x=[0.2],
            y=[0.0],
            mode="markers+text",
            text=[f"<b>Left</b><br>n={n_left_1d}"],
            textposition="bottom center",
            textfont=dict(size=10, color='rgb(65, 105, 225)'),
            marker=dict(
                size=NODE_SIZE,
                color='rgba(65, 105, 225, 0.15)',
                line=dict(color='rgb(65, 105, 225)', width=1.5),
                symbol='circle'
            ),
            showlegend=False,
            hovertext=[(
                f"Node role: left<br>"
                f"Node type: external<br>"
                f"Depth: 1<br>"
                f"Size (n): {n_left_1d}<br>"
                f"Condition: r < {random_1d_point_on_axis_to_cut:.2f}%"
            )],
            hoverinfo='text'
        ),
        row=1, col=2
    )

    # Right child node
    fig.add_trace(
        go.Scatter(
            x=[0.8],
            y=[0.0],
            mode="markers+text",
            text=[f"<b>Right</b><br>n={n_right_1d}"],
            textposition="bottom center",
            textfont=dict(size=10, color='rgb(220, 20, 60)'),
            marker=dict(
                size=NODE_SIZE,
                color='rgba(220, 20, 60, 0.15)',
                line=dict(color='rgb(220, 20, 60)', width=1.5),
                symbol='circle'
            ),
            showlegend=False,
            hovertext=[(
                f"Node role: right<br>"
                f"Node type: external<br>"
                f"Depth: 1<br>"
                f"Size (n): {n_right_1d}<br>"
                f"Condition: r ≥ {random_1d_point_on_axis_to_cut:.2f}%"
            )],
            hoverinfo='text'
        ),
        row=1, col=2
    )

    # Connecting lines
    fig.add_trace(
        go.Scatter(
            x=[0.5, 0.2, None, 0.5, 0.8],
            y=[1.0, 0.0, None, 1.0, 0.0],
            mode="lines",
            line=dict(color='#8d99ae', width=2.0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=2
    )

    # Remove inline annotations (kept tree clean; hover has details)
    fig.update_xaxes(
        visible=False,
        range=[0, 1],
        row=1, col=2
    )
    fig.update_yaxes(
        visible=False,
        range=[-0.3, 1.35],
        row=1, col=2
    )

    # Layout: make it big and consistent with iTree
    fig.update_layout(
        height=800,
        width=1600,
        margin=dict(l=50, r=50, t=80, b=50),
        template="simple_white",
        showlegend=False,
        title={
            "text": "Single-Step Random Binary Partition on ORCL 1-Day Returns (%)",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.98,
            "font": dict(size=16)
        }
    )

    return fig
