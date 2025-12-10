import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, State, callback, callback_context
from Isolation_Forest import IsolationForestAnomalyDetector

# =============================================================================
# Load ORCL data ONCE at module import
# =============================================================================
# Assumes ORCL.csv is in the working directory of the Dash app.
df_orcl = pd.read_csv("ORCL.csv").sort_values(by="Date", ascending=True)

# 1-day returns in percent
df_orcl["Return_1d"] = df_orcl["Close"].pct_change() * 100.0
orcl_returns = df_orcl["Return_1d"].dropna().values.reshape(-1, 1)

M = orcl_returns.shape[0]
HEIGHT_LIMIT = int(np.ceil(np.log2(M))) if M > 0 else 0


# =============================================================================
# Layout
# =============================================================================
iTree_layout = [

    # Controls row
    html.Div([
        # Reset / New Tree button
        html.Button(
            "Reset / New Random Tree",
            id='itree-reset-tree-button',
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

        # Expand button
        html.Button(
            "One-Step Partition",
            id='itree-expand-tree-button',
            n_clicks=0,
            style={
                'backgroundColor': '#2b2d42',
                'color': 'white',
                'border': 'none',
                'borderRadius': '8px',
                'padding': '10px 22px',
                'cursor': 'pointer',
                'fontWeight': 'bold',
                'boxShadow': '2px 2px 5px rgba(0,0,0,0.15)',
                'transition': '0.2s',
                'marginRight': '18px'
            }
        ),

        # Info: sample size and height limit
        html.Div([
            html.Div(
                f"Observations: {M}",
                style={
                    'padding': '4px 8px',
                    'borderRadius': '6px',
                    'border': '1px solid #ddd',
                    'backgroundColor': '#fafafa',
                    'marginRight': '10px'
                }
            ),
            html.Label("Height Limit:", style={'fontWeight': '600', 'marginRight': '6px'}),
            html.Div(
                id='itree-height-limit',
                children=str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—",
                style={
                    'padding': '6px 10px',
                    'borderRadius': '6px',
                    'border': '1px solid #ddd',
                    'backgroundColor': '#fafafa',
                    'minWidth': '40px',
                    'textAlign': 'center'
                }
            )
        ], style={'display': 'flex', 'alignItems': 'center'}),

    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'center',
        'justifyContent': 'flex-start',
        'gap': '12px',
        'padding': '10px 6px'
    }),

    # Stores for tree and depth
    html.Div([
        dcc.Store(id='itree-tree-store'),
        dcc.Store(id='itree-depth-store'),
    ]),

    # Main figure
    html.Div(
        [
            dcc.Graph(
                id='itree-scatter-plot',
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

    html.Hr(),

    html.Div([
        dcc.Link(
            'Go to Binary Partition',
            href='/binary_partition',
            style={
                'color': '#2b2d42',
                'fontSize': '18px',
                'textDecoration': 'none',
                'fontWeight': 'bold',
                'padding': '8px 14px',
                'border': '2px solid #2b2d42',
                'borderRadius': '8px',
                'backgroundColor': '#f8f9fa',
                'textAlign': 'center',
                'display': 'inline-block',
                'boxShadow': '2px 2px 4px rgba(0, 0, 0, 0.15)'
            }
        ),

        dcc.Link(
            'Go to Path Length',
            href='/path_length',
            style={
                'color': '#2b2d42',
                'fontSize': '18px',
                'textDecoration': 'none',
                'fontWeight': 'bold',
                'padding': '8px 14px',
                'border': '2px solid #2b2d42',
                'borderRadius': '8px',
                'backgroundColor': '#f8f9fa',
                'textAlign': 'center',
                'display': 'inline-block',
                'boxShadow': '2px 2px 4px rgba(0, 0, 0, 0.15)'
            }
        ),

    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'padding': '20px 0'
    })
]

@callback(
    Output('itree-scatter-plot', 'figure'),
    Output('itree-tree-store', 'data'),
    Output('itree-depth-store', 'data'),
    Output('itree-height-limit', 'children'),
    Input('itree-reset-tree-button', 'n_clicks'),
    Input('itree-expand-tree-button', 'n_clicks'),
    State('itree-tree-store', 'data'),
    State('itree-depth-store', 'data'),
)
def update_itree_fig(
    n_clicks_reset,
    n_clicks_expand,
    stored_tree,
    stored_depth
):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    X = orcl_returns  # fixed 1D ORCL returns (percent)
    tree = stored_tree
    depth = stored_depth if stored_depth is not None else 0

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def get_max_depth(node):
        if not isinstance(node, dict):
            return 0
        if node.get('type') == 'external':
            return 0
        return 1 + max(get_max_depth(node.get('left')), get_max_depth(node.get('right')))

    def build_new_tree():
        """Build a new random isolation tree on fixed ORCL returns."""
        if X is None or len(X) == 0:
            return None
        model = IsolationForestAnomalyDetector(X)
        full_tree = model.iTree(S=None, c=0, l=HEIGHT_LIMIT)
        return full_tree

    def collect_splits(itree, depth_limit, bounds, current_depth=1, splits=None):
        """
        Collect all splits (1D) up to a given depth for drawing partition lines.
        """
        if splits is None:
            splits = []

        if (itree is None or
            not isinstance(itree, dict) or
            itree.get('type') == 'external' or
            current_depth > depth_limit):
            return splits

        q = itree['split_axis']
        p = itree['split_point']
        xmin, xmax, ymin, ymax = bounds

        splits.append({'axis': q, 'point': p, 'bounds': bounds, 'depth': current_depth})

        if current_depth < depth_limit:
            if q == 0:
                left_bounds = (xmin, p, ymin, ymax)
                right_bounds = (p, xmax, ymin, ymax)
            else:
                # in 1D we never hit this, but keep for completeness
                left_bounds = (xmin, xmax, ymin, p)
                right_bounds = (xmin, xmax, p, ymax)

            collect_splits(itree['left'], depth_limit, left_bounds, current_depth + 1, splits)
            collect_splits(itree['right'], depth_limit, right_bounds, current_depth + 1, splits)

        return splits

    def collect_depth_splits(itree, X_data, depth_target):
        """
        Collect all (left_idx, right_idx) pairs for nodes exactly at depth_target.
        Used to color points at the current depth.
        """
        if (itree is None or
            not isinstance(itree, dict) or
            depth_target <= 0):
            return []

        N = X_data.shape[0]
        indices_all = np.arange(N)
        result = []

        def _rec(node, idx, depth_cur):
            if (node is None or
                not isinstance(node, dict) or
                node.get('type') == 'external' or
                len(idx) == 0):
                return

            q = node['split_axis']
            p = node['split_point']

            vals = X_data[idx, q]
            left_mask = vals < p
            left_idx = idx[left_mask]
            right_idx = idx[~left_mask]

            if depth_cur == depth_target:
                result.append({
                    'left_idx': left_idx,
                    'right_idx': right_idx,
                    'axis': q,
                    'point': p
                })
                return

            if depth_cur < depth_target:
                _rec(node['left'], left_idx, depth_cur + 1)
                _rec(node['right'], right_idx, depth_cur + 1)

        _rec(itree, indices_all, 1)
        return result

    def build_tree_traces(itree, X_data, depth_limit):
        """
        Build full binary tree visualization (right subplot), simplified:

        - Nodes are fixed-size points (no dynamic marker scaling).
        - No fig.add_annotation() calls.
        - Node hover shows all relevant information.
        """

        if (
            itree is None
            or X_data is None
            or len(X_data) == 0
            or depth_limit <= 0
        ):
            return []

        N = X_data.shape[0]
        mask_all = np.ones(N, dtype=bool)

        # Fixed marker size
        NODE_SIZE = 12

        # Visual lists
        root_x, root_y, root_label, root_hover = [], [], [], []
        left_x, left_y, left_label, left_hover = [], [], [], []
        right_x, right_y, right_label, right_hover = [], [], [], []
        edge_x, edge_y = [], []

        # Layout constants
        effective_depth = max(depth_limit, 1)
        dy = 0.8 / effective_depth
        dx0 = 0.32

        def traverse(node, mask, depth_cur, x, y, dx, role, condition_str):
            if (
                node is None or
                not isinstance(node, dict) or
                depth_cur > depth_limit
            ):
                return

            size_here = int(mask.sum())
            if size_here == 0:
                return

            node_type = node.get("type", "internal")
            is_external = (node_type == "external")

            # Label on plot
            if role == "root":
                label = f"Root\nn={size_here}"
            elif role == "left":
                label = f"L\nn={size_here}"
            else:
                label = f"R\nn={size_here}"

            # Hover info
            if not is_external:
                q = node["split_axis"]
                p = node["split_point"]
                split_info = f"Split: r = {p:.2f}%"
            else:
                split_info = "Split: leaf"

            hover_text = (
                f"Node role: {role}<br>"
                f"Node type: {node_type}<br>"
                f"Depth: {depth_cur}<br>"
                f"n={size_here}<br>"
                f"{condition_str}<br>"
                f"{split_info}"
            )

            # Store depending on role
            if role == "root":
                root_x.append(x)
                root_y.append(y)
                root_label.append(label)
                root_hover.append(hover_text)
            elif role == "left":
                left_x.append(x)
                left_y.append(y)
                left_label.append(label)
                left_hover.append(hover_text)
            else:
                right_x.append(x)
                right_y.append(y)
                right_label.append(label)
                right_hover.append(hover_text)

            # Stop recursion if leaf or we've reached visible depth
            if is_external or depth_cur >= depth_limit:
                return

            # Compute child partitions
            q = node["split_axis"]
            p = node["split_point"]
            vals = X_data[:, q]

            left_mask = mask & (vals < p)
            right_mask = mask & (~(vals < p))

            if not (left_mask.any() or right_mask.any()):
                return

            # Child positioning
            child_y = y - dy
            child_dx = dx / 2.0

            cond_left = f"Condition: r < {p:.2f}%"
            cond_right = f"Condition: r ≥ {p:.2f}%"

            # Left child
            if left_mask.any():
                x_left = x - child_dx
                edge_x.extend([x, x_left, None])
                edge_y.extend([y, child_y, None])
                traverse(
                    node["left"],
                    left_mask,
                    depth_cur + 1,
                    x_left,
                    child_y,
                    child_dx,
                    "left",
                    cond_left,
                )

            # Right child
            if right_mask.any():
                x_right = x + child_dx
                edge_x.extend([x, x_right, None])
                edge_y.extend([y, child_y, None])
                traverse(
                    node["right"],
                    right_mask,
                    depth_cur + 1,
                    x_right,
                    child_y,
                    child_dx,
                    "right",
                    cond_right,
                )

        # Start traversal
        traverse(
            itree,
            mask_all,
            depth_cur=1,
            x=0.5,
            y=1.05,
            dx=dx0,
            role="root",
            condition_str="Root",
        )

        # Build Plotly traces
        traces = []

        # Edges
        if edge_x:
            traces.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(color="#adb5bd", width=1.8),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Root
        if root_x:
            traces.append(
                go.Scatter(
                    x=root_x,
                    y=root_y,
                    mode="markers+text",
                    text=root_label,
                    textposition="top center",
                    textfont=dict(size=10, color="#2b2d42"),
                    marker=dict(
                        size=NODE_SIZE,
                        color="#f8f9fa",
                        line=dict(color="#2b2d42", width=1.2),
                        symbol="circle",
                    ),
                    hovertext=root_hover,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

        # Left nodes
        if left_x:
            traces.append(
                go.Scatter(
                    x=left_x,
                    y=left_y,
                    mode="markers+text",
                    text=left_label,
                    textposition="bottom center",
                    textfont=dict(size=9, color="rgb(65, 105, 225)"),
                    marker=dict(
                        size=NODE_SIZE,
                        color="rgba(65, 105, 225, 0.15)",
                        line=dict(color="rgb(65, 105, 225)", width=1.2),
                        symbol="circle",
                    ),
                    hovertext=left_hover,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

        # Right nodes
        if right_x:
            traces.append(
                go.Scatter(
                    x=right_x,
                    y=right_y,
                    mode="markers+text",
                    text=right_label,
                    textposition="bottom center",
                    textfont=dict(size=9, color="rgb(220, 20, 60)"),
                    marker=dict(
                        size=NODE_SIZE,
                        color="rgba(220, 20, 60, 0.15)",
                        line=dict(color="rgb(220, 20, 60)", width=1.2),
                        symbol="circle",
                    ),
                    hovertext=right_hover,
                    hoverinfo="text",
                    showlegend=False,
                )
            )

        return traces

    # -------------------------------------------------------------------------
    # Ensure we have a tree at least once (first load or after reset)
    # -------------------------------------------------------------------------
    if tree is None:
        tree = build_new_tree()
        depth = 0

    if triggered_id == 'itree-reset-tree-button':
        tree = build_new_tree()
        depth = 0

    elif triggered_id == 'itree-expand-tree-button' and tree is not None:
        max_depth = get_max_depth(tree)
        if depth < max_depth:
            depth += 1

    # -------------------------------------------------------------------------
    # Build 1×2 subplot: (1) data space, (2) tree diagram
    # -------------------------------------------------------------------------
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=("", ""),
        horizontal_spacing=0.08
    )

    # ===== Left: ORCL 1-day returns with partition lines =====
    x_vals = X[:, 0]
    x_min = float(x_vals.min())
    x_max = float(x_vals.max())
    y_min, y_max = -0.3, 0.3

    pad_x = 0.08 * (x_max - x_min if x_max > x_min else 1.0)
    x_min -= pad_x
    x_max += pad_x

    if depth <= 0 or tree is None or not isinstance(tree, dict):
        # all points neutral
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=np.zeros_like(x_vals),
                mode='markers',
                name='ORCL 1-day returns',
                marker=dict(
                    size=8,
                    opacity=0.85,
                    color='rgb(120, 120, 120)',
                    line=dict(color='rgb(60, 60, 60)', width=0.8)
                )
            ),
            row=1, col=1
        )
    else:
        depth_splits = collect_depth_splits(tree, X, depth_target=depth)

        N = len(x_vals)
        mask_left_union = np.zeros(N, dtype=bool)
        mask_right_union = np.zeros(N, dtype=bool)

        for s in depth_splits:
            mask_left_union[s['left_idx']] = True
            mask_right_union[s['right_idx']] = True

        mask_neutral = ~(mask_left_union | mask_right_union)

        if mask_neutral.any():
            fig.add_trace(
                go.Scatter(
                    x=x_vals[mask_neutral],
                    y=np.zeros(mask_neutral.sum()),
                    mode='markers',
                    name='Previous levels',
                    marker=dict(
                        size=10,
                        opacity=1.0,
                        color='black',
                        line=dict(color='rgb(140, 140, 140)', width=0.5)
                    )
                ),
                row=1, col=1
            )

        if mask_left_union.any():
            fig.add_trace(
                go.Scatter(
                    x=x_vals[mask_left_union],
                    y=np.zeros(mask_left_union.sum()),
                    mode='markers',
                    name='Left split (current depth)',
                    marker=dict(
                        size=10,
                        opacity=1.0,
                        color='rgb(65, 105, 225)',
                        line=dict(color='rgb(30, 60, 180)', width=1)
                    )
                ),
                row=1, col=1
            )

        if mask_right_union.any():
            fig.add_trace(
                go.Scatter(
                    x=x_vals[mask_right_union],
                    y=np.zeros(mask_right_union.sum()),
                    mode='markers',
                    name='Right split (current depth)',
                    marker=dict(
                        size=10,
                        opacity=1.0,
                        color='rgb(220, 20, 60)',
                        line=dict(color='rgb(180, 10, 40)', width=1)
                    )
                ),
                row=1, col=1
            )

    # partition lines up to current depth
    if tree is not None and depth > 0:
        splits_all = collect_splits(
            tree,
            depth_limit=depth,
            bounds=(x_min, x_max, y_min, y_max)
        )
        for s in splits_all:
            p = s['point']
            w = 1.5 if s['depth'] == 1 else 1.0
            dash = "dash" if s['depth'] == depth else "dot"
            fig.add_vline(
                x=p,
                line_dash=dash,
                line_color="#888888",
                line_width=w,
                row=1, col=1
            )

    fig.update_yaxes(visible=False, range=[y_min, y_max], row=1, col=1)
    fig.update_xaxes(title_text="1-Day Return (%)", range=[x_min, x_max], row=1, col=1)

    # ===== Right: Tree diagram =====
    tree_depth_limit = depth + 1 if depth > 0 else 1
    tree_traces = build_tree_traces(tree, X, depth_limit=tree_depth_limit)
    for tr in tree_traces:
        fig.add_trace(tr, row=1, col=2)

    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    fig.update_layout(
        height=800,
        width=1600,
        margin=dict(l=50, r=50, t=80, b=50),
        template="simple_white",
        showlegend=False,
        title={
            "text": f"Isolation Tree on ORCL 1-Day Returns – splits revealed up to depth {depth}",
            "x": 0.5,
            "xanchor": "center"
        }
    )

    height_limit_display = str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—"
    return fig, tree, depth, height_limit_display
