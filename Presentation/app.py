import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dash import Dash, html, dcc

from binary_partition_page import binary_partition_layout
from iTree_page import iTree_layout
from path_length_page import path_length_layout
from iForest_page import iForest_layout


app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_scripts=["https://cdn.plot.ly/plotly-latest.min.js"],
)
server = app.server

# If you have a central colors dict, feel free to replace this:
COLORS = {
    "background": "#ffffff",
    "text": "#2b2d42",
    "primary": "#2b2d42",
}

app.layout = html.Div(
    style={
        "background": COLORS["background"],
        "minHeight": "100vh",
        "padding": "20px",
        "boxSizing": "border-box",
        "color": COLORS["text"],
        "fontFamily": '"Inter", system-ui, -apple-system, sans-serif',
    },
    children=[
        dcc.Tabs(
            id="main-tabs",
            value="tab-binary",
            children=[
                dcc.Tab(
                    label="Binary Partition",
                    value="tab-binary",
                    # IMPORTANT: children is your layout (list of components), NOT a dict/object
                    children=binary_partition_layout,
                    style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["text"],
                        "padding": "10px 20px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": "2px solid transparent",
                        "transition": "color 0.3s ease",
                    },
                    selected_style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["primary"],
                        "padding": "6px 18px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": f"2px solid {COLORS['primary']}",
                    },
                ),
                dcc.Tab(
                    label="iTrees",
                    value="tab-itrees",
                    children=iTree_layout,
                    style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["text"],
                        "padding": "10px 20px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": "2px solid transparent",
                        "transition": "color 0.3s ease",
                    },
                    selected_style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["primary"],
                        "padding": "6px 18px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": f"2px solid {COLORS['primary']}",
                    },
                ),
                dcc.Tab(
                    label="Path Length",
                    value="tab-path-length",
                    children=path_length_layout,
                    style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["text"],
                        "padding": "10px 20px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": "2px solid transparent",
                        "transition": "color 0.3s ease",
                    },
                    selected_style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["primary"],
                        "padding": "6px 18px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": f"2px solid {COLORS['primary']}",
                    },
                ),
                dcc.Tab(
                    label="iForest",
                    value="tab-iforest",
                    children=iForest_layout,
                    style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["text"],
                        "padding": "10px 20px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": "2px solid transparent",
                        "transition": "color 0.3s ease",
                    },
                    selected_style={
                        "backgroundColor": COLORS["background"],
                        "color": COLORS["primary"],
                        "padding": "6px 18px",
                        "fontWeight": "bold",
                        "fontSize": "14px",
                        "border": "none",
                        "borderBottom": f"2px solid {COLORS['primary']}",
                    },
                ),
            ],
        )
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
