import os
import sys
import numpy as np
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, State, callback, callback_context, no_update
from Isolation_Forest import IsolationForestAnomalyDetector

path_length_layout = [
    html.H1(
        "Path Length in Isolation Trees",
        style={'textAlign': 'center', 'color': '#2b2d42'}
    ),

    html.P(
        "The underlying assumption of Isolation Forest is that anomalies are easier to isolate than normal points. "
        "The 'ease' of isolating points is quantified using path lengths of a point as it travels down a randomly "
        "a randomly generated isolation tree.",
        style={'textAlign': 'left', 'color': '#333'}
    ),

    html.Div([
        # Left Column (Code)
        html.Div([
            html.H3("Python Code"),
            dcc.Markdown(
                """
                ```python
                def expected_length_of_unsuccessful_search_in_RBST(self, s):

                    # trivial case: point is fully isolated.             
                    if s <= 1:
                        return 0.0

                    # trivial case: external node has two points
                    # one additional binary partition will yield isolated points
                    if s == 2:
                        return 1.0

                    # Approximation of harmonic series 1 + 1/2 + 1/3 + ... + 1/(s-1)
                    H = np.log(s-1) + np.euler_gamma

                    # RBST expected path in unsuccessful search formula
                    # Refer: The Art of Computer Programming (Volume 3) - Donald Knuth
                    return 2 * (H - (1 - 1/s))

                def path_length_by_point(self, point, iChild, current_length=0):
                
                    # base case: if node is external, compute final path length
                    if iChild['type'] == 'external':
                        return current_length + self.expected_length_of_unsuccessful_search_in_RBST(iChild['size'])

                    else:
                        # Fetch the axis and point of split for this internal node
                        split_axis = iChild['split_axis']
                        split_point = iChild['split_point']

                        # Go left if point is on the left-side of the split
                        if point[split_axis] < split_point:
                            return self.path_length_by_point(point, iChild['left'], current_length+1)
                
                        # Go right if point is on the right-side of the split
                        else:
                            return self.path_length_by_point(point, iChild['right'], current_length+1)
                ```
                """
            )
        ], style={
            'flex': '1',
            'padding': '20px',
            'borderRight': '2px solid #ccc'
        }),

        # Right Column (Math)
        html.Div([
            html.H3("Mathematical Description"),
            html.H4("External Node and Randomized Binary Search Tree Approximation:"),
            dcc.Markdown(
                r'''
                Suppose we are at an external node with size $s$.

                $$
                \operatorname{ExpectedPathLengthExternalNode}(s) = \begin{cases}
                0 & \text{if } s \le 1\\
                1 & \text{if } s=2\\
                2 \cdot \bigg(H_{s-1} - \bigg[1 - \displaystyle\frac{1}{s}\bigg]\bigg) & \text{Otherwise} 
                \end{cases}
                $$

                Here, $H_{s-1} = 1+\frac{1}{2}+\dots+\frac{1}{s-1} \approx log(s-1) + \gamma$ where
                $\gamma \approx 0.57721 \text{ is the Euler–Mascheroni constant.}$
                ''', mathjax=True
            ),
            html.H4("Path Length Calculation:"),
            dcc.Markdown(
                r'''
                Let $\operatorname{PathLength}(\mathbf{x}, \text{iChild}, h)$ denote the path length of $\mathbf{x} \in \mathbf{X}$
                in a subset tree $\text{iChild}$ at current depth $h$.

                **Stopping rule (External node):**  

                If the external node has $s$ points, then
                
                $$
                \operatorname{PathLength}(\mathbf{x}, \text{iChild}, h) =
                h + \operatorname{ExpectedPathLengthExternalNode}(s).
                $$

                **Recursive step (Internal node):**  

                The node stores split axis $q$ and split point $p$.
                
                $$
                \text{iChild} = \begin{cases}
                \text{left child}, & x_q < p, \\[4pt]
                \text{right child}, & x_q \ge p \end{cases}
                $$

                Then recurse with

                $$
                \operatorname{PathLength}(\mathbf{x}, \text{iChild}, h+1).
                $$
                ''', mathjax=True
            ),
        ], style={
            'flex': '1',
            'padding': '20px'
        }),

    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'justifyContent': 'space-between'
    }),

    html.Div([
       html.H3(
           "Path Length Controls",
           style={
               'marginBottom': '10px',
               'fontWeight': 'bold',
               'color': '#2b2d42',
               'borderBottom': '2px solid #edf2f4',
               'paddingBottom': '8px'
           }
       ),

       html.Div([
           # --- Generate random data ---
           html.Div([
               html.Button(
                   "Generate Random Data",
                   id='path-length-generate-data-button',
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

           # --- Number of points slider ---
           html.Div([
               html.Label(
                   "Number of Points:",
                   style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}
               ),
               dcc.Slider(
                   0, 20, 1,
                   value=5,
                   id='path-length-number-of-points-slider',
                   marks={0: '0', 10: '10', 20: '20'},
                   tooltip={"always_visible": False},
               )
           ], style={'flex': '1.2', 'marginRight': '20px'}),

           # --- 1D / 2D toggle ---
           html.Div([
               html.Label(
                   "Dimension:",
                   style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}
               ),
               dcc.RadioItems(
                   id='path-length-dimension-toggle',
                   options=[
                       {'label': '1D', 'value': '1d'},
                       {'label': '2D', 'value': '2d'},
                   ],
                   value='2d',
                   labelStyle={'display': 'inline-block', 'marginRight': '15px'}
               )
           ], style={'flex': '0.8', 'marginRight': '10px'}),

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
       })
    ]),

    html.Div([
            dcc.Store(id='path-length-data-store'),
            dcc.Store(id='path-length-tree-store'),
            dcc.Store(id='path-length-depth-store'),
            dcc.Store(id='path-length-selected-index-store')
    ]),

    html.Div(
        [
            dcc.Graph(
                id='path-length-scatter-plot',
                figure={},
                style={
                    'height': '80vh',
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

    html.Hr(),

    html.Div([
        dcc.Link(
            'Go to iTrees',
            href='/itrees',
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
            'Go to iTrees',
            href='/error404',
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

# ---------- Helpers ----------

def build_random_data(num_points, dim):
    """
    Generate random 1D or 2D data and a single iTree over that data.
    We intentionally inject duplicates so some external nodes have size s > 2.
    """
    if num_points <= 0:
        X = np.empty((0, 2 if dim == '2d' else 1))
        tree = {'type': 'external', 'size': 0}
        return X, tree

    # base random data
    if dim == '1d':
        X = np.random.uniform(-5, 5, size=(num_points, 1))
    else:
        X = np.random.uniform(-5, 5, size=(num_points, 2))

    # --- inject duplicates to create leaves with s > 2 ---
    if num_points >= 3:
        # choose how many points should be identical (a small cluster)
        cluster_size = max(3, num_points // 4)   # e.g. 25% of points, at least 3
        cluster_size = min(cluster_size, num_points)

        # pick random indices that will form the duplicate cluster
        idx = np.random.choice(num_points, size=cluster_size, replace=False)

        # choose a "center" and copy it to all chosen indices
        center = X[idx[0]].copy()
        X[idx] = center
        # now those `cluster_size` points are exactly equal, so when the iTree
        # recurses down to a node containing only them, np.all(isclose(...)) is
        # True and you get an external node with size = cluster_size (>2)

    detector = IsolationForestAnomalyDetector(X)
    tree = detector.iTree()   # single isolation tree
    return X, tree

def assign_tree_layout(tree):

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
    node_refs = []  # to later check membership in a path
    edge_x = []
    edge_y = []

    def collect(node, depth):
        node['y'] = -depth
        node_x.append(node['x'])
        node_y.append(node['y'])
        node_refs.append(node)

        if node['type'] == 'internal':
            text = f"split: x[{node['split_axis']}] < {node['split_point']:.2f}"
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
    For a clicked point, follow it down the iTree and collect the nodes
    along the path + compute path length using IsolationForestAnomalyDetector.
    """
    X_arr = np.array(X)
    detector = IsolationForestAnomalyDetector(X_arr)

    # path length using your existing method
    length = detector.path_length_by_point(point, tree, current_length=0)

    # now collect the concrete nodes visited
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
    path_nodes.append(node)  # final external node

    return path_nodes, length


def build_figure(X, tree, selected_index=None):
    
    # Create two subplots: scatter (col=1) and tree (col=2)
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.45, 0.55],
        subplot_titles=("Data", "Isolation Tree"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )

    if X is None or len(X) == 0 or tree is None:
        return fig

    X_arr = np.array(X)
    dim = X_arr.shape[1]

    # ------- Scatter Plot -------
    if dim == 1:
        xs = X_arr[:, 0]
        ys = np.zeros_like(xs)
    else:
        xs = X_arr[:, 0]
        ys = X_arr[:, 1]

    # Base scatter
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            marker=dict(size=10),
            name='points'
        ),
        row=1, col=1
    )

    # Highlight selected point
    if selected_index is not None and 0 <= selected_index < len(xs):
        fig.add_trace(
            go.Scatter(
                x=[xs[selected_index]],
                y=[ys[selected_index]],
                mode='markers',
                marker=dict(size=14, symbol='circle-open'),
                name='selected point'
            ),
            row=1, col=1
        )

    fig.update_xaxes(title_text="x₁", row=1, col=1)
    fig.update_yaxes(title_text="x₂" if dim == 2 else "", row=1, col=1)

    # ------- Tree Plot -------
    node_x, node_y, node_text, edge_x, edge_y, node_refs = assign_tree_layout(tree)

    # Full tree edges
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

    # Full tree nodes
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

    # If a point is selected, draw its path on top
    if selected_index is not None and 0 <= selected_index < len(xs):
        point = X_arr[selected_index]
        path_nodes, _ = get_path_nodes_and_length(point, X, tree)

        # path nodes
        path_node_x = [n['x'] for n in path_nodes]
        path_node_y = [n['y'] for n in path_nodes]

        # path edges
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

        fig.add_trace(
            go.Scatter(
                x=path_node_x,
                y=path_node_y,
                mode='markers',
                marker=dict(size=12, symbol='diamond'),
                name='path nodes',
                hoverinfo='skip'
            ),
            row=1, col=2
        )

    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), template='simple_white')

    return fig

@callback(
    [
        Output('path-length-data-store', 'data'),
        Output('path-length-tree-store', 'data'),
        Output('path-length-depth-store', 'data'),
        Output('path-length-selected-index-store', 'data'),
        Output('path-length-scatter-plot', 'figure'),
        Output('path-length-summary-output', 'children'),
    ],
    [
        Input('path-length-generate-data-button', 'n_clicks'),
        Input('path-length-scatter-plot', 'clickData'),
    ],
    [
        State('path-length-number-of-points-slider', 'value'),
        State('path-length-dimension-toggle', 'value'),
        State('path-length-data-store', 'data'),
        State('path-length-tree-store', 'data'),
        State('path-length-selected-index-store', 'data'),
    ]
)
def update_itree(
        generate_clicks,
        click_data,
        num_points,
        dim,
        stored_X,
        stored_tree,
        stored_selected_index
):

    ctx = callback_context
    # figure out what triggered this callback
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # ------------------------------------------------------------------
    # Case 0: initial page load OR "Generate Random Data" button clicked
    # ------------------------------------------------------------------
    if trigger is None or trigger == 'path-length-generate-data-button':
        # behave like user clicked "generate random data"
        X, tree = build_random_data(num_points, dim)
        selected_index = None

        def depth(node, d=0):
            if node['type'] == 'external':
                return d
            return max(depth(node['left'], d + 1), depth(node['right'], d + 1))

        depth_info = depth(tree) if tree is not None else 0
        fig = build_figure(X, tree, selected_index=None)
        message = "Click a point in the scatter plot to see its path length."

        return (
            X.tolist(),
            tree,
            depth_info,
            selected_index,
            fig,
            message
        )

    # ----------------------------------------------
    # Case 1: user clicked on a point in the scatter
    # ----------------------------------------------
    elif trigger == 'path-length-scatter-plot':
        if stored_X is None or stored_tree is None or click_data is None:
            return (
                stored_X,
                stored_tree,
                no_update,
                stored_selected_index,
                no_update,
                no_update,
            )

        X = np.array(stored_X)
        tree = stored_tree

        # --- Robust selection: 1D uses nearest x, 2D keeps pointIndex ---
        dim = X.shape[1]
        click_point = click_data['points'][0]

        if dim == 1:
            # 1D: use the x-coordinate from the click and find nearest point in X
            x_clicked = click_point['x']
            selected_index = int(np.argmin(np.abs(X[:, 0] - x_clicked)))
        else:
            # 2D: pointIndex is reliable here
            selected_index = int(click_point['pointIndex'])

        # extra safety, in case anything weird happens
        if selected_index < 0 or selected_index >= len(X):
            return (
                stored_X,
                stored_tree,
                no_update,
                stored_selected_index,
                no_update,
                no_update,
            )

        point = X[selected_index]
        path_nodes, length = get_path_nodes_and_length(point, stored_X, tree)

        fig = build_figure(X, tree, selected_index=selected_index)
        message = f"Path length for selected point: {length:.3f}"

        return (
            stored_X,        # data store unchanged
            tree,            # tree store unchanged
            no_update,       # depth store unchanged
            selected_index,
            fig,
            message,
        )
