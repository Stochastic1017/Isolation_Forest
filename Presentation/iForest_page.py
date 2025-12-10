import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, callback
from Isolation_Forest import IsolationForestAnomalyDetector

# =============================================================================
# Load ORCL data ONCE at module import (Oracle daily closes, 1-day returns in %)
# =============================================================================
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

# =============================================================================
# Layout – ORCL-only iForest page
# =============================================================================
iForest_layout = [
    html.Div([
        # Run button
        html.Button(
            "Run iForest on ORCL Returns",
            id='iforest-run-orcl-button',
            n_clicks=0,
            style={
                'backgroundColor': '#2b2d42',
                'color': 'white',
                'border': 'none',
                'borderRadius': '8px',
                'cursor': 'pointer',
                'fontWeight': 'bold',
                'padding': '10px 22px',
                'boxShadow': '2px 2px 5px rgba(0,0,0,0.15)',
                'transition': '0.2s',
                'marginRight': '18px'
            }
        ),

        # Info tiles
        html.Div([
            html.Div(
                f"ORCL Observations: {M}",
                style={
                    'padding': '6px 10px',
                    'borderRadius': '6px',
                    'border': '1px solid #ddd',
                    'backgroundColor': '#fafafa',
                    'marginRight': '10px'
                }
            ),
        ], style={'display': 'flex', 'alignItems': 'center'}),

    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'center',
        'justifyContent': 'flex-start',
        'gap': '12px',
        'padding': '10px 6px'
    }),

    dcc.Loading(
        id="iforest-loading",
        type="circle",
        children=dcc.Graph(
            id='iforest-scatter-plot',
            figure={},
            style={
                'height': '80vh',
                'width': '100%',
            }
        ),
        style={'width': '100%'}
    ),
]

@callback(
    Output("iforest-scatter-plot", "figure"),
    Input("iforest-run-orcl-button", "n_clicks"),
    prevent_initial_call=False,
)
def generate_iforest_orcl(n_clicks):
    """
    Run Isolation Forest on ORCL 1-day returns and show:

    - Row 1: price vs date (line) + anomaly-coloured markers
    - Row 2: 1-day returns vs date (line) + anomaly-coloured markers
    """

    # Use the global returns; on initial load (n_clicks == 0), still generate the figure once.
    X = orcl_returns
    if X is None or len(X) == 0:
        return go.Figure()

    # ---------------------------------------------------------------------
    # 1. Fit Isolation Forest and get anomaly scores
    # ---------------------------------------------------------------------
    detector = IsolationForestAnomalyDetector(X)
    scores = np.array(detector.anomaly_scores())  # shape (M,)

    # Ensure consistent length
    M_local = X.shape[0]
    if scores.shape[0] != M_local:
        scores = scores[:M_local]

    dates = orcl_dates[:M_local]
    prices = orcl_prices[:M_local]
    returns_1d = X[:M_local, 0]

    # ---------------------------------------------------------------------
    # 2. Build a sorted DataFrame to avoid weird line connections
    # ---------------------------------------------------------------------
    df_plot = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "price": prices,
            "ret": returns_1d,
            "score": scores,
        }
    ).sort_values("date")

    # Extract sorted arrays
    dates_dt = df_plot["date"]
    prices_sorted = df_plot["price"]
    returns_sorted = df_plot["ret"]
    scores_sorted = df_plot["score"]

    # ---------------------------------------------------------------------
    # 3. Make 2×1 subplots (price on top, returns bottom)
    # ---------------------------------------------------------------------
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[
            "ORCL Closing Price (colored by anomaly score)",
            "ORCL 1-Day Returns (%) (colored by anomaly score)",
        ],
    )

    # =========================
    # Row 1: PRICE
    # =========================
    # Line (plain)
    fig.add_trace(
        go.Scatter(
            x=dates_dt,
            y=prices_sorted,
            mode="lines",
            name="ORCL Close",
            line=dict(width=1.5),
            connectgaps=False,
        ),
        row=1,
        col=1,
    )

    # Anomaly-coloured markers
    fig.add_trace(
        go.Scatter(
            x=dates_dt,
            y=prices_sorted,
            mode="markers",
            name="Price (anomaly-coloured)",
            marker=dict(
                size=7,
                color=scores_sorted,
                colorscale="Turbo",
                showscale=False,  # keep colorbar only on returns
                opacity=0.9,
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Price: %{y:.2f}<br>"
                "Anomaly score: %{marker.color:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)

    # =========================
    # Row 2: RETURNS
    # =========================
    # Line (plain)
    fig.add_trace(
        go.Scatter(
            x=dates_dt,
            y=returns_sorted,
            mode="lines",
            name="1-Day Return",
            line=dict(width=1.2),
            connectgaps=False,
        ),
        row=2,
        col=1,
    )

    # Anomaly-coloured markers + colorbar
    fig.add_trace(
        go.Scatter(
            x=dates_dt,
            y=returns_sorted,
            mode="markers",
            name="Returns (anomaly-coloured)",
            marker=dict(
                size=7,
                color=scores_sorted,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="Anomaly score"),
                opacity=0.9,
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Return: %{y:.2f}%<br>"
                "Anomaly score: %{marker.color:.4f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="1-Day Return (%)", row=2, col=1)

    # ---------------------------------------------------------------------
    # 4. De-clutter date axis: monthly ticks
    # ---------------------------------------------------------------------
    monthly_ticks = pd.date_range(dates_dt.min(), dates_dt.max(), freq="MS")

    fig.update_xaxes(
        tickvals=monthly_ticks,
        tickformat="%b %Y",
        tickangle=-45,
        row=2,  # only need to set on bottom when shared_xaxes=True
        col=1,
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # ---------------------------------------------------------------------
    # 5. Layout
    # ---------------------------------------------------------------------
    fig.update_layout(
        template="plotly_white",
        height=750,
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
        title="Isolation Forest Anomaly Scores on ORCL 1-Day Returns",
    )

    return fig
