import os
import sys
import numpy as np
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, callback
from Isolation_Forest import IsolationForestAnomalyDetector

binary_partition_layout = [
    html.H1(
        "Step 1: Binary Partition in Isolation Forest",
        style={'textAlign': 'center', 'color': '#2b2d42'}
    ),

    html.P(
        "In an Isolation Tree, each split is created by performing a binary partition "
        "on the data. We randomly pick one feature (axis), then randomly choose a "
        "cut point between that feature's minimum and maximum values. All points "
        "with values less than the cut go to the left subset; the rest go to the right.",
        style={'textAlign': 'left', 'color': '#333'}
    ),

    html.Div([
        # Left Column (Code)
        html.Div([
            html.H3("Python Code"),
            dcc.Markdown(
                """
                ```python
                def binary_partition(self, X=None):

                    # fallback to use all data if X is None
                    if X is None:
                        X = self.X

                    # base case: X is a single isolated point
                    if len(X) <= 1:
                        return (X, np.array([]).reshape(0, self.d), 0, 0)

                    # Choose a random axis from {0, ..., d-1} on which to cut
                    # Discrete uniform distribution, i.e., P(X=i) = 1/d for all i
                    random_axis_to_cut = np.random.randint(low=0, high=self.d)

                    # Access the chosen axis (column) chosen above
                    X_d = X[:, random_axis_to_cut]

                    # Choose a random point on chosen axis (column) from the respective (min, max)
                    # Continuous uniform distribution, i.e., P(X <= x) = 1/(max-min) for all x : min <= x <= max
                    random_point_on_axis_to_cut = np.random.uniform(low=X_d.min(), high=X_d.max())

                    # Apply condition to split points (rows of X) from left and right of random cut
                    split_condition = X_d < random_point_on_axis_to_cut
                    split_X_from_left = X[np.where(split_condition)[0]]   # Split X where condition is TRUE (i.e., left)
                    split_X_from_right = X[np.where(~split_condition)[0]] # Split X where condition is FALSE (i.e., right)

                    return (split_X_from_left, split_X_from_right, random_axis_to_cut, random_point_on_axis_to_cut)
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
            dcc.Markdown(
                r'''
                Let $X \in \mathbb{R}^{M \times d}$, i.e., $X = \begin{pmatrix} \vec{x_{1}}\\ \vec{x_{2}}\\ \vdots\\ \vec{x_{M}} \end{pmatrix}
                                                              = \begin{pmatrix} x_{11} & x_{12} & \dots & x_{1d}\\
                                                                                x_{21} & x_{21} & \dots & x_{2d}\\
                                                                                \vdots & \vdots & \vdots & \vdots\\
                                                                                x_{M1} & x_{M2} & \dots & x_{Md} \end{pmatrix}$
                be the Data Matrix.

                A binary partition chooses:

                1. A random feature (axis) along which to make the cut, where $\mathbb{P}(X=q) = \frac{1}{d}$ and $q \in \{1,\dots,d\}$
                   $$
                   q \sim \text{DiscreteUniform}\{ 1, 2, \dots, d \}
                   $$

                2. A random point along the axis where we make the cut, such that $P(X \ge p) = \frac{1}{x_{q,max} - x_{q,min}}$
                   where $x_{q,\text{max}} = \max_{1 \le j \le M}\;x_{jq}$ and $x_{q,\text{min}} = \min_{1 \le k \le M}\;x_{kq}$
                   $$
                   p \sim \text{ContinuousUniform}(x_{q,\text{min}}, x_{q,\text{max}})
                   $$

                3. Create two subsets such that:
                   $$
                   X_{\text{left}} = \{ \vec{x_{l}} \in X : x_{lq} < p \}, \quad
                   X_{\text{right}} = \{ \vec{x_{r}} \in X : x_{rq} \ge p \}
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
            "Binary Partition Controls",
            style={
                'marginBottom': '10px',
                'fontWeight': 'bold',
                'color': '#2b2d42',
                'borderBottom': '2px solid #edf2f4',
                'paddingBottom': '8px'
            }
        ),

        html.Div([
            html.Div([
                html.Label(
                    "Number of Points:",
                    style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}
                ),
                dcc.Slider(
                    0, 520, 10,
                    value=20,
                    id='bp-number-of-points-slider',
                    marks={0: '0', 252: '252', 520: '520'},
                    tooltip={"always_visible": False},
                )
            ], style={'flex': '1', 'marginRight': '20px'}),

            html.Div([
                html.Button(
                    "Generate Random Split",
                    id='bp-generate-split-button',
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
                        'transition': '0.2s'
                    }
                )
            ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'})
        ],
        style={
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

    html.Div(
        [
            dcc.Graph(
                id='bp-scatter-plot',
                figure={},
                style={
                    'display': 'inline-block'
                }
            )
        ],
        style={
            'textAlign': 'center',
            'width': '100%'
        }
    ),

    html.Hr(),

    html.Div([
        dcc.Link(
            '← Go to Overview / Intro',
            href='/intro',
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
            'Go to iTrees →',
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

    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'padding': '20px 0'
    })
]

@callback(
    Output('bp-scatter-plot', 'figure'),
    [
        Input('bp-number-of-points-slider', 'value'),
        Input('bp-generate-split-button', 'n_clicks')
    ]
)
def generate_single_step_random_binary_partition(M, n_clicks):

    ##################
    # Build 2x2 figure
    ##################
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "1D Binary Partition",
            "2D Binary Partition",
            "",
            ""
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.15
    )

    ##################
    # One-Dimensional
    ##################
    # Make it 2D (n, 1) because the class expects X with shape (M, d)
    random_one_dimensional_array = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=M
    ).reshape(-1, 1)

    one_dimensional_iForest_model = IsolationForestAnomalyDetector(
        random_one_dimensional_array
    )

    (split_1d_from_left,
     split_1d_from_right,
     random_1d_axis_to_cut,
     random_1d_point_on_axis_to_cut) = one_dimensional_iForest_model.binary_partition()

    # === Top-left: 1D points + split + shaded intervals ===
    fig.add_trace(
        go.Scatter(
            x=split_1d_from_left.flatten(),
            y=np.zeros(len(split_1d_from_left.flatten())),
            mode='markers',
            name='1D Left Split',
            marker=dict(
                size=12,
                opacity=0.75,
                color='rgb(65, 105, 225)',
                line=dict(color='rgb(30, 60, 180)', width=1)
            )
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=split_1d_from_right.flatten(),
            y=np.zeros(len(split_1d_from_right.flatten())),
            mode='markers',
            name='1D Right Split',
            marker=dict(
                size=12,
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

    # Get bounds for shading
    x1_min = random_one_dimensional_array.min()
    x1_max = random_one_dimensional_array.max()

    # Shaded intervals (layer='below' puts them behind the points)
    fig.add_shape(
        type="rect",
        x0=x1_min,
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
        x1=x1_max,
        y0=-0.3,
        y1=0.3,
        fillcolor="rgba(220, 20, 60, 0.15)",
        line=dict(width=0),
        layer='below',
        row=1, col=1
    )

    fig.add_annotation(
        x=(x1_min + random_1d_point_on_axis_to_cut) / 2,
        y=0.25,
        text=f"Left: n = {len(split_1d_from_left)}",
        showarrow=False,
        font=dict(size=12, color='rgb(65, 105, 225)'),
        row=1, col=1
    )
    fig.add_annotation(
        x=(random_1d_point_on_axis_to_cut + x1_max) / 2,
        y=0.25,
        text=f"Right: n = {len(split_1d_from_right)}",
        showarrow=False,
        font=dict(size=12, color='rgb(220, 20, 60)'),
        row=1, col=1
    )

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # === Bottom-left: 1D tree node diagram ===
    
    # === Bottom-left: 1D tree node diagram ===
    
    # Calculate proportional sizes for nodes
    n_total_1d = len(random_one_dimensional_array)
    n_left_1d = len(split_1d_from_left)
    n_right_1d = len(split_1d_from_right)
    
    # Scale sizes proportionally (with min/max bounds for visibility)
    base_size = 35
    left_size_1d = max(20, min(45, base_size * (n_left_1d / n_total_1d)))
    right_size_1d = max(20, min(45, base_size * (n_right_1d / n_total_1d)))
    
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
                size=base_size,
                color='#f8f9fa',
                line=dict(color='#2b2d42', width=2.5),
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
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
                size=left_size_1d,
                color='rgba(65, 105, 225, 0.15)',
                line=dict(color='rgb(65, 105, 225)', width=2.5),
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
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
                size=right_size_1d,
                color='rgba(220, 20, 60, 0.15)',
                line=dict(color='rgb(220, 20, 60)', width=2.5),
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # Connecting lines
    fig.add_trace(
        go.Scatter(
            x=[0.5, 0.2, None, 0.5, 0.8],
            y=[1.0, 0.0, None, 1.0, 0.0],
            mode="lines",
            line=dict(color='#8d99ae', width=2.5),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # Add split condition annotations on the edges
    fig.add_annotation(
        x=0.35,
        y=0.5,
        text=f"x < {random_1d_point_on_axis_to_cut:.2f}",
        showarrow=False,
        font=dict(size=9, color='#495057', family='monospace'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#dee2e6',
        borderwidth=1,
        borderpad=3,
        row=2, col=1
    )
    
    fig.add_annotation(
        x=0.65,
        y=0.5,
        text=f"x ≥ {random_1d_point_on_axis_to_cut:.2f}",
        showarrow=False,
        font=dict(size=9, color='#495057', family='monospace'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#dee2e6',
        borderwidth=1,
        borderpad=3,
        row=2, col=1
    )
    
    # Add split point info box (to mirror 2D style)
    fig.add_annotation(
        x=0.5,
        y=1.35,
        text=f"<b>Split:</b> at x = {random_1d_point_on_axis_to_cut:.3f}",
        showarrow=False,
        font=dict(size=10, color='#2b2d42', family='Arial'),
        bgcolor='rgba(237, 242, 244, 0.95)',
        bordercolor='#2b2d42',
        borderwidth=2,
        borderpad=6,
        row=2, col=1
    )
    
    fig.update_xaxes(
        visible=False, 
        range=[0, 1],
        row=2, col=1
    )
    fig.update_yaxes(
        visible=False, 
        range=[-0.3, 1.35],
        row=2, col=1
    )
    
    ##################
    # Two-Dimensional
    ##################
    random_two_dimensional_array = np.random.multivariate_normal(
        mean=[0.0, 0.0],
        cov=[[1.0, 0.0], [0.0, 1.0]],
        size=M
    )

    two_dimensional_iForest_model = IsolationForestAnomalyDetector(
        random_two_dimensional_array
    )

    (split_2d_from_left,
     split_2d_from_right,
     random_2d_axis_to_cut,
     random_2d_point_on_axis_to_cut) = two_dimensional_iForest_model.binary_partition()

    # Determine split orientation
    if random_2d_axis_to_cut == 0:
        split_axis = "vertical"
    else:
        split_axis = "horizontal"

    # Calculate bounds
    x2_min = random_two_dimensional_array[:, 0].min()
    x2_max = random_two_dimensional_array[:, 0].max()
    y2_min = random_two_dimensional_array[:, 1].min()
    y2_max = random_two_dimensional_array[:, 1].max()
    
    n_left_2d = len(split_2d_from_left)
    n_right_2d = len(split_2d_from_right)

    # === Top-right: 2D scatter + split + shaded rectangles ===
    fig.add_trace(
        go.Scatter(
            x=split_2d_from_left[:, 0],
            y=split_2d_from_left[:, 1],
            mode="markers",
            name="2D Left",
            marker=dict(
                size=12,
                opacity=0.75,
                color='rgb(65, 105, 225)',
                line=dict(color='rgb(30, 60, 180)', width=1)
            ),
            showlegend=False
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=split_2d_from_right[:, 0],
            y=split_2d_from_right[:, 1],
            mode="markers",
            name="2D Right",
            marker=dict(
                size=12,
                opacity=0.75,
                color='rgb(220, 20, 60)',
                line=dict(color='rgb(180, 10, 40)', width=1)
            ),
            showlegend=False
        ),
        row=1, col=2
    )

    if split_axis == "vertical":
        # Split line
        fig.add_vline(
            x=random_2d_point_on_axis_to_cut,
            line_dash="dash",
            line_color="black",
            line_width=2,
            row=1, col=2
        )

        # Left region (layer='below' puts it behind the points)
        fig.add_shape(
            type="rect",
            x0=x2_min,
            x1=random_2d_point_on_axis_to_cut,
            y0=y2_min,
            y1=y2_max,
            fillcolor="rgba(65, 105, 225, 0.15)",
            line=dict(width=0),
            layer='below',
            row=1, col=2
        )
        # Right region
        fig.add_shape(
            type="rect",
            x0=random_2d_point_on_axis_to_cut,
            x1=x2_max,
            y0=y2_min,
            y1=y2_max,
            fillcolor="rgba(220, 20, 60, 0.15)",
            line=dict(width=0),
            layer='below',
            row=1, col=2
        )

        fig.add_annotation(
            x=(x2_min + random_2d_point_on_axis_to_cut) / 2,
            y=y2_max - 0.2,
            text=f"Left region<br>n = {n_left_2d}",
            showarrow=False,
            font=dict(size=11, color='rgb(65, 105, 225)'),
            row=1, col=2
        )
        fig.add_annotation(
            x=(random_2d_point_on_axis_to_cut + x2_max) / 2,
            y=y2_max - 0.2,
            text=f"Right region<br>n = {n_right_2d}",
            showarrow=False,
            font=dict(size=11, color='rgb(220, 20, 60)'),
            row=1, col=2
        )

    else:  # horizontal split
        fig.add_hline(
            y=random_2d_point_on_axis_to_cut,
            line_dash="dash",
            line_color="black",
            line_width=2,
            row=1, col=2
        )

        # Bottom region (layer='below' puts it behind the points)
        fig.add_shape(
            type="rect",
            x0=x2_min,
            x1=x2_max,
            y0=y2_min,
            y1=random_2d_point_on_axis_to_cut,
            fillcolor="rgba(65, 105, 225, 0.15)",
            line=dict(width=0),
            layer='below',
            row=1, col=2
        )
        # Top region
        fig.add_shape(
            type="rect",
            x0=x2_min,
            x1=x2_max,
            y0=random_2d_point_on_axis_to_cut,
            y1=y2_max,
            fillcolor="rgba(220, 20, 60, 0.15)",
            line=dict(width=0),
            layer='below',
            row=1, col=2
        )

        fig.add_annotation(
            x=(x2_min + x2_max) / 2,
            y=(y2_min + random_2d_point_on_axis_to_cut) / 2,
            text=f"Lower region<br>n = {n_left_2d}",
            showarrow=False,
            font=dict(size=11, color='rgb(65, 105, 225)'),
            row=1, col=2
        )
        fig.add_annotation(
            x=(x2_min + x2_max) / 2,
            y=(random_2d_point_on_axis_to_cut + y2_max) / 2,
            text=f"Upper region<br>n = {n_right_2d}",
            showarrow=False,
            font=dict(size=11, color='rgb(220, 20, 60)'),
            row=1, col=2
        )

    fig.update_xaxes(title_text="x₁", row=1, col=2)
    fig.update_yaxes(title_text="x₂", row=1, col=2)

    # === Bottom-right: 2D tree node diagram ===
    
    # Calculate proportional sizes for nodes
    n_total_2d = len(random_two_dimensional_array)
    n_left_2d = len(split_2d_from_left)
    n_right_2d = len(split_2d_from_right)
    
    # Scale sizes proportionally (with min/max bounds for visibility)
    base_size = 35
    left_size_2d = max(20, min(45, base_size * (n_left_2d / n_total_2d)))
    right_size_2d = max(20, min(45, base_size * (n_right_2d / n_total_2d)))
    
    # Root node
    fig.add_trace(
        go.Scatter(
            x=[0.5],
            y=[1.0],
            mode="markers+text",
            text=[f"<b>Root</b><br>n={n_total_2d}"],
            textposition="top center",
            textfont=dict(size=11, color='#2b2d42'),
            marker=dict(
                size=base_size,
                color='#f8f9fa',
                line=dict(color='#2b2d42', width=2.5),
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=2
    )
    
    # Left child node
    fig.add_trace(
        go.Scatter(
            x=[0.2],
            y=[0.0],
            mode="markers+text",
            text=[f"<b>Left</b><br>n={n_left_2d}"],
            textposition="bottom center",
            textfont=dict(size=10, color='rgb(65, 105, 225)'),
            marker=dict(
                size=left_size_2d,
                color='rgba(65, 105, 225, 0.15)',
                line=dict(color='rgb(65, 105, 225)', width=2.5),
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=2
    )
    
    # Right child node
    fig.add_trace(
        go.Scatter(
            x=[0.8],
            y=[0.0],
            mode="markers+text",
            text=[f"<b>Right</b><br>n={n_right_2d}"],
            textposition="bottom center",
            textfont=dict(size=10, color='rgb(220, 20, 60)'),
            marker=dict(
                size=right_size_2d,
                color='rgba(220, 20, 60, 0.15)',
                line=dict(color='rgb(220, 20, 60)', width=2.5),
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=2
    )
    
    # Connecting lines
    fig.add_trace(
        go.Scatter(
            x=[0.5, 0.2, None, 0.5, 0.8],
            y=[1.0, 0.0, None, 1.0, 0.0],
            mode="lines",
            line=dict(color='#8d99ae', width=2.5),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=2
    )
    
    # Determine which axis was split and format accordingly
    if random_2d_axis_to_cut == 0:
        axis_name = "x₁"
    else:
        axis_name = "x₂"
    
    # Add split condition annotations on the edges
    fig.add_annotation(
        x=0.35,
        y=0.5,
        text=f"{axis_name} < {random_2d_point_on_axis_to_cut:.2f}",
        showarrow=False,
        font=dict(size=9, color='#495057', family='monospace'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#dee2e6',
        borderwidth=1,
        borderpad=3,
        row=2, col=2
    )
    
    fig.add_annotation(
        x=0.65,
        y=0.5,
        text=f"{axis_name} ≥ {random_2d_point_on_axis_to_cut:.2f}",
        showarrow=False,
        font=dict(size=9, color='#495057', family='monospace'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#dee2e6',
        borderwidth=1,
        borderpad=3,
        row=2, col=2
    )
    
    # Add split point info box
    split_direction = "Vertical" if random_2d_axis_to_cut == 0 else "Horizontal"
    fig.add_annotation(
        x=0.5,
        y=1.35,
        text=f"<b>Split:</b> {split_direction} at {axis_name} = {random_2d_point_on_axis_to_cut:.3f}",
        showarrow=False,
        font=dict(size=10, color='#2b2d42', family='Arial'),
        bgcolor='rgba(237, 242, 244, 0.95)',
        bordercolor='#2b2d42',
        borderwidth=2,
        borderpad=6,
        row=2, col=2
    )
    
    fig.update_xaxes(
        visible=False, 
        range=[0, 1],
        row=2, col=2
    )
    fig.update_yaxes(
        visible=False, 
        range=[-0.3, 1.35],
        row=2, col=2
    )
    
    fig.update_layout(
        height=900,
        width=1200,
        margin=dict(l=40, r=40, t=80, b=50),
        template="simple_white",
        title={
            "text": "Single-Step Binary Partition (Using IsolationForestAnomalyDetector.binary_partition)",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.0,
            "font": dict(size=15)
        },
        showlegend=False
    )

    return fig
