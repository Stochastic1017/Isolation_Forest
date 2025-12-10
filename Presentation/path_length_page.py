import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, State, callback, callback_context, no_update
from Isolation_Forest import IsolationForestAnomalyDetector

# =============================================================================
# Load ORCL data ONCE at module import (Oracle only, 1-day returns in %)
# =============================================================================
df_orcl = pd.read_csv("ORCL.csv").sort_values(by="Date", ascending=True)

# 1-day returns in percent, using Adjusted Close to match other pages
df_orcl["Return_1d"] = df_orcl["Close"].pct_change() * 100.0
orcl_returns = df_orcl["Return_1d"].dropna().values.reshape(-1, 1)

N_ORCL = orcl_returns.shape[0]
HEIGHT_LIMIT = int(np.ceil(np.log2(max(1, N_ORCL)))) if N_ORCL > 0 else 0

# =============================================================================
# Layout — Oracle-only Path Length page
# =============================================================================
path_length_layout = [

       html.Div([
           # Reset / New Tree
           html.Div([
               html.Button(
                   "Reset / New Random Tree",
                   id='path-length-reset-tree-button',
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
                       'marginBottom': '10px',
                       'width': '100%'
                   }
               )
           ], style={'flex': '0.7', 'marginRight': '20px'}),

           # ORCL observations + height limit
           html.Div([
               html.Label(
                   "ORCL Observations:",
                   style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}
               ),
               html.Div(
                   f"{N_ORCL}",
                   style={
                       'padding': '8px 12px',
                       'borderRadius': '8px',
                       'border': '1px solid #e9ecef',
                       'backgroundColor': '#f8f9fa',
                       'minWidth': '70px',
                       'textAlign': 'center',
                       'fontWeight': '600',
                       'marginBottom': '8px'
                   }
               ),
               html.Label(
                   "Height Limit:",
                   style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}
               ),
               html.Div(
                   id='path-length-height-limit',
                   children=str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—",
                   style={
                       'padding': '8px 12px',
                       'borderRadius': '8px',
                       'border': '1px solid #e9ecef',
                       'backgroundColor': '#f8f9fa',
                       'minWidth': '70px',
                       'textAlign': 'center',
                       'fontWeight': '600'
                   }
               )
           ], style={'flex': '0.7', 'marginRight': '20px'}),

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
       
    # Stores
    html.Div([
        dcc.Store(id='path-length-data-store'),
        dcc.Store(id='path-length-tree-store'),
        dcc.Store(id='path-length-depth-store'),
        dcc.Store(id='path-length-selected-index-store')
    ]),

    # Main figure (big, like other pages)
    html.Div(
        [
            dcc.Graph(
                id='path-length-scatter-plot',
                figure={},
                style={
                    'height': '70vh',
                    'width': '100%',
                    'margin': '0 auto'
                }
            )
        ],
        style={
            'textAlign': 'center',
            'width': '100%'
        }
    ),

    html.Div(
        id='path-length-summary-output',
        style={
            'textAlign': 'center',
            'fontSize': '18px',
            'marginTop': '15px',
            'color': '#2b2d42',
            'fontWeight': 'bold'
        }
    ),
]

# =============================================================================
# Helpers
# =============================================================================

def assign_leaf_indices(tree, X):
    """
    For each external node in the tree, attach a list of data-point indices
    that end up in that leaf: node['indices'] = [i0, i1, ...].
    """
    if tree is None:
        return

    def clear_indices(node):
        if node['type'] == 'external':
            node['indices'] = []
        else:
            clear_indices(node['left'])
            clear_indices(node['right'])

    clear_indices(tree)

    X_arr = np.array(X)

    for i, point in enumerate(X_arr):
        node = tree
        while node['type'] == 'internal':
            q = node['split_axis']
            p = node['split_point']
            if point[q] < p:
                node = node['left']
            else:
                node = node['right']
        node.setdefault('indices', []).append(i)

def build_orcl_tree():
    """
    Build a random isolation tree on fixed ORCL returns using HEIGHT_LIMIT.
    """
    X = orcl_returns
    if X is None or len(X) == 0:
        return None

    detector = IsolationForestAnomalyDetector(X)
    tree = detector.iTree(S=None, c=0, l=HEIGHT_LIMIT)
    assign_leaf_indices(tree, X)
    return tree

def assign_tree_layout(tree):
    """
    Assign x/y positions to nodes for a clear tree plot:

    - Leaf x positions: consecutive integers via inorder traversal.
    - Internal x: midpoint of children.
    - y: -depth.
    """
    if tree is None:
        return [], [], [], [], [], []

    leaf_counter = [0]

    def assign_x(node):
        if node['type'] == 'external':
            node['x'] = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            assign_x(node['left'])
            assign_x(node['right'])
            node['x'] = 0.5 * (node['left']['x'] + node['right']['x'])

    assign_x(tree)

    node_x = []
    node_y = []
    node_text = []
    node_refs = []
    edge_x = []
    edge_y = []

    def collect(node, depth):
        node['y'] = -depth
        node_x.append(node['x'])
        node_y.append(node['y'])
        node_refs.append(node)

        if node['type'] == 'internal':
            text = f"split: r < {node['split_point']:.2f}%"
            node_text.append(text)

            for child_key in ['left', 'right']:
                child = node[child_key]
                collect(child, depth + 1)
                edge_x.extend([node['x'], child['x'], None])
                edge_y.extend([node['y'], child['y'], None])
        else:
            node_text.append(f"external, s={node['size']}")

    collect(tree, depth=0)

    return node_x, node_y, node_text, edge_x, edge_y, node_refs


def get_path_nodes_and_length(point, X, tree):
    """
    For a selected point, follow it down the iTree and collect the nodes
    along the path + compute path length using IsolationForestAnomalyDetector.
    """
    X_arr = np.array(X)
    detector = IsolationForestAnomalyDetector(X_arr)

    length = detector.path_length_by_point(point, tree, current_length=0)

    path_nodes = []
    node = tree
    while node['type'] == 'internal':
        path_nodes.append(node)
        q = node['split_axis']
        p = node['split_point']
        if point[q] < p:
            node = node['left']
        else:
            node = node['right']
    path_nodes.append(node)

    return path_nodes, length


def build_figure(X, tree, selected_index=None):
    """
    Two subplots:
    - Left: ORCL returns (1D), with a light histogram behind the scatter.
    - Right: Isolation tree with node layout and optional highlighted path.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.45, 0.55],
        subplot_titles=("ORCL 1-Day Returns (%)", "Isolation Tree"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )

    if X is None or len(X) == 0 or tree is None:
        return fig

    X_arr = np.array(X)

    # ------- Left panel: 1D returns + light histogram -------
    xs = X_arr[:, 0]
    ys = np.zeros_like(xs)

    # 1) Light histogram in the background
    fig.add_trace(
        go.Histogram(
            x=xs,
            nbinsx=40,  # adjust if you want coarser/finer bins
            marker=dict(
                color="rgba(100, 100, 100, 0.18)"  # light gray, semi-transparent
            ),
            hoverinfo="skip",
            showlegend=False,
            name="distribution"
        ),
        row=1, col=1
    )

    # 2) Scatter on top (for interaction and clarity)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            marker=dict(size=9, color="black"),
            name='ORCL returns'
        ),
        row=1, col=1
    )

    # Highlight selected point, if any
    if selected_index is not None and 0 <= selected_index < len(xs):
        fig.add_trace(
            go.Scatter(
                x=[xs[selected_index]],
                y=[ys[selected_index]],
                mode='markers',
                marker=dict(size=13, symbol='circle-open', line=dict(width=2)),
                name='selected point'
            ),
            row=1, col=1
        )

    # Hide y-axis (counts / 0-line) and label x-axis as returns in %
    fig.update_xaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # Overlay histogram and scatter nicely
    fig.update_layout(barmode="overlay")

    # ------- Right panel: tree plot -------
    node_x, node_y, node_text, edge_x, edge_y, node_refs = assign_tree_layout(tree)

    # Tree edges
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=1),
            hoverinfo='none',
            showlegend=False
        ),
        row=1, col=2
    )

    # Tree nodes
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(size=8),
            text=node_text,
            hoverinfo='text',
            name='nodes'
        ),
        row=1, col=2
    )

    # Highlight path for selected point
    if selected_index is not None and 0 <= selected_index < len(xs):
        point = X_arr[selected_index]
        path_nodes, _ = get_path_nodes_and_length(point, X, tree)

        path_node_x = [n['x'] for n in path_nodes]
        path_node_y = [n['y'] for n in path_nodes]

        path_edge_x = []
        path_edge_y = []
        for i in range(len(path_nodes) - 1):
            p0 = path_nodes[i]
            p1 = path_nodes[i + 1]
            path_edge_x.extend([p0['x'], p1['x'], None])
            path_edge_y.extend([p0['y'], p1['y'], None])

        fig.add_trace(
            go.Scatter(
                x=path_edge_x,
                y=path_edge_y,
                mode='lines',
                line=dict(width=3),
                name='path',
                hoverinfo='none'
            ),
            row=1, col=2
        )

        # Separate internal and external nodes
        internal_x = []
        internal_y = []
        internal_text = []

        external_x = []
        external_y = []
        external_text = []

        for x, y, txt, node in zip(node_x, node_y, node_text, node_refs):
            if node['type'] == 'internal':
                internal_x.append(x)
                internal_y.append(y)
                internal_text.append(txt)
            else:
                external_x.append(x)
                external_y.append(y)
                external_text.append(txt)

        # Internal nodes: one color
        fig.add_trace(
            go.Scatter(
                x=internal_x,
                y=internal_y,
                mode="markers",
                marker=dict(size=8, color="rgba(0, 0, 180, 0.9)"),  # deep blue
                text=internal_text,
                hoverinfo="text",
                name="internal nodes"
            ),
            row=1, col=2
        )

        # External nodes: another color
        fig.add_trace(
            go.Scatter(
                x=external_x,
                y=external_y,
                mode="markers",
                marker=dict(size=8, color="rgba(200, 0, 0, 0.9)"),  # red
                text=external_text,
                hoverinfo="text",
                name="external nodes"
            ),
            row=1, col=2
        )

    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        template='simple_white',
        showlegend=False
    )

    return fig


# =============================================================================
# Callback — ORCL-only path length
# =============================================================================
@callback(
    [
        Output('path-length-data-store', 'data'),
        Output('path-length-tree-store', 'data'),
        Output('path-length-depth-store', 'data'),
        Output('path-length-selected-index-store', 'data'),
        Output('path-length-scatter-plot', 'figure'),
        Output('path-length-summary-output', 'children'),
        Output('path-length-height-limit', 'children'),
    ],
    [
        Input('path-length-reset-tree-button', 'n_clicks'),
        Input('path-length-scatter-plot', 'clickData'),
    ],
    [
        State('path-length-data-store', 'data'),
        State('path-length-tree-store', 'data'),
        State('path-length-selected-index-store', 'data'),
    ]
)
def update_itree(
        reset_clicks,
        click_data,
        stored_X,
        stored_tree,
        stored_selected_index
):

    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    X = orcl_returns
    # ---------------------------
    # Case 0: initial load / reset
    # ---------------------------
    if trigger is None or trigger == 'path-length-reset-tree-button':
        tree = build_orcl_tree()

        if tree is None or X is None or len(X) == 0:
            fig = build_figure(X, tree, selected_index=None)
            return (
                [],
                tree,
                0,
                None,
                fig,
                "",
                str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—"
            )

        # Randomly choose a starting point for demonstration
        selected_index = int(np.random.choice(len(X), 1)[0])

        # Compute tree depth
        def depth(node, d=0):
            if node is None:
                return d
            if node['type'] == 'external':
                return d
            return max(depth(node['left'], d + 1), depth(node['right'], d + 1))

        depth_info = depth(tree)

        fig = build_figure(X, tree, selected_index=selected_index)

        # Path length for the selected point
        point = X[selected_index]
        path_nodes, length = get_path_nodes_and_length(point, X, tree)
        h = len(path_nodes) - 1
        s = path_nodes[-1]['size']

        if s <= 1:
            c = 0.0
        elif s == 2:
            c = 1.0
        else:
            H = np.log(s - 1) + np.euler_gamma
            c = 2 * (H - (1 - 1 / s))

        message = f"Path length for selected ORCL return: {h:.0f} + {c:.3f} = {length:.3f}"

        return (
            X.tolist(),
            tree,
            depth_info,
            selected_index,
            fig,
            message,
            str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—"
        )

    # ---------------------------
    # Case 1: user clicked
    # ---------------------------
    elif trigger == 'path-length-scatter-plot':
        if stored_X is None or stored_tree is None or click_data is None:
            return (
                stored_X if stored_X is not None else [],
                stored_tree,
                no_update,
                stored_selected_index,
                no_update,
                no_update,
                str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—"
            )

        X_arr = np.array(orcl_returns)
        tree = stored_tree
        click_point = click_data['points'][0]
        curve = click_point['curveNumber']

        # traces:
        # 0 = base scatter, 1 = selected point,
        # 2 = tree edges, 3 = tree nodes, 4/5 = path overlays
        if curve in (0, 1):
            # clicked on data scatter: choose nearest x
            x_clicked = click_point['x']
            selected_index = int(np.argmin(np.abs(X_arr[:, 0] - x_clicked)))

            if selected_index < 0 or selected_index >= len(X_arr):
                return (
                    stored_X,
                    stored_tree,
                    no_update,
                    stored_selected_index,
                    no_update,
                    no_update,
                    str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—"
                )

        elif curve == 3:
            # clicked on tree nodes
            _, _, _, _, _, node_refs = assign_tree_layout(tree)
            node_idx = int(click_point['pointIndex'])

            if node_idx < 0 or node_idx >= len(node_refs):
                return (
                    stored_X,
                    stored_tree,
                    no_update,
                    stored_selected_index,
                    no_update,
                    no_update,
                    str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—"
                )

            node = node_refs[node_idx]

            if node['type'] != 'external' or 'indices' not in node or len(node['indices']) == 0:
                return (
                    stored_X,
                    stored_tree,
                    no_update,
                    stored_selected_index,
                    no_update,
                    no_update,
                    str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—"
                )

            selected_index = int(node['indices'][0])

        else:
            # Ignore clicks on edges/path overlays
            return (
                stored_X,
                stored_tree,
                no_update,
                stored_selected_index,
                no_update,
                no_update,
                str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—"
            )

        # Shared logic for path length once we have selected_index
        point = X_arr[selected_index]
        path_nodes, length = get_path_nodes_and_length(point, orcl_returns, tree)

        h = len(path_nodes) - 1
        s = path_nodes[-1]['size']

        if s <= 1:
            c = 0.0
        elif s == 2:
            c = 1.0
            # keep same formula as earlier
        else:
            H = np.log(s - 1) + np.euler_gamma
            c = 2 * (H - (1 - 1 / s))

        fig = build_figure(orcl_returns, tree, selected_index=selected_index)
        message = f"Path length for selected ORCL return: {h:.0f} + {c:.3f} = {length:.3f}"

        return (
            orcl_returns.tolist(),
            tree,
            no_update,
            selected_index,
            fig,
            message,
            str(HEIGHT_LIMIT) if HEIGHT_LIMIT > 0 else "—"
        )
