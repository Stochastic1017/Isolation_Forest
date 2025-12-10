import os
import sys
import numpy as np
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, State, callback, callback_context
from Isolation_Forest import IsolationForestAnomalyDetector

iForest_layout = [
    html.H1(
        "Computing Anomaly Scores and Isolation Forest Implementation",
        style={'textAlign': 'center', 'color': '#2b2d42'}
    ),

    html.P(
        "Using path length, expected length of failed search, and a bunch of iTrees, we compute anomaly scores for each point in the dataset. "
        "For this page, daily closing prices of Oracle is used, and iForest is implemented on the 1-day returns. ",
        style={'textAlign': 'left', 'color': '#333'}
    ),

    html.Div([
        # Left Column (Code)
        html.Div([
            html.H3("Python Code"),
            dcc.Markdown(
                """
                ```python
                def iForest(self):
                    # Initialize array to store all iTrees
                    self.forest = []
                    for idx in range(self.M):
                        # bootstrap sample full dataset
                        # max_samples=1.0 => 1:1 ratio of data and samples (with replacement)
                        bootstrapped_X = self.X[
                            np.random.choice(self.M, size=self.M, replace=True)
                        ]
                
                        limit = int(np.ceil(np.log2(len(bootstrapped_X))))
                        iTree_idx = self.iTree(bootstrapped_X, counter=0, limit=limit)
                        self.forest.append(iTree_idx)

                    return self.forest

                def anomaly_scores(self):

                    if not hasattr(self, 'forest'):
                        # build an iForest if not already built
                        self.iForest()

                    # Average path length of unsuccessful search in RBST
                    C_M = self.expected_length_of_unsuccessful_search_in_RBST(self.M)
                    scores = []
                    for point in self.X:
                        # Average path length for each point across all trees
                        E_hx = np.mean([self.path_length_by_point(point, itree, 0) for itree in self.forest])

                        # Anomaly scores in [0,1], higher -> more anomalous             
                        scores.append( 2**(-E_hx / C_M) )

                    return scores
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
                Let $\mathbf{X} \in \mathbb{R}^{M \times d}$ be the data matrix as defined earlier.


                Then, we create an iForest with $M$ iTrees as follows:

                1. Take $M$ bootstrap samples, where each sample is of the form:
                $$
                X^{\text{bootstrap}}_{i} = \{\mathbf{x}_{i1}, \dots, \mathbf{x}_{iM}\} \quad \text{where } i_j \sim \text{DiscreteUniform}(1,M)
                $$

                2. Fit $M$ iTree's with bootstrapped samples and height limit $l=\lceil\text{log}_2(M)\rceil$:
                $$
                \mathcal{T}_{i} = \text{iTree}(X^{\text{bootstrap}}_{i}, c=0, l=l)
                $$


                Finally, we compute anomaly scores for each point $\mathbf{x} \in \mathbf{X}$ across iForest $\{\mathcal{T}_1, \dots, \mathcal{T}_M\}$ as follows:
                
                1. Compute Expected Path Length of Unsuccessful Search for all points $M$:
                $$
                C_M = \text{ExpectedPathLengthNode}(M)
                $$

                2. Compute Average Path Length for $\mathbf{x}$ across all $M$ trees:
                $$
                E_{h\mathbf{x}} = \frac{1}{M}\displaystyle\sum_{i=1}^{M} \text{PathLength}(\mathbf{x}, \mathcal{T}_i, 0)
                $$

                3. Compute Anomaly Score for $\mathbf{x}$:
                $$
                \text{score}_{\mathbf{x}} = 2^{-E_{hx} / C_M}
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
        # Generate button
        html.Button(
            "Generate Random Data",
            id='iforest-generate-data-button',
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
                'transition': '0.2s'
            }
        ),

        # --- Number of points slider ---
        html.Div([
            html.Label(
                "Number of Points:",
                style={'fontWeight': '600', 'marginBottom': '6px', 'display': 'block'}
            ),
            dcc.Slider(
                2, 520, 1,
                value=252,
                id='iforest-number-of-points-slider',
                marks={2: '2', 252: '252', 520: '520'},
                tooltip={"always_visible": False},
            )
        ], style={'flex': '1.0', 'marginRight': '10px'}),

        # Dimension toggle
        html.Div([
            html.Label("Dimension", style={'fontWeight': '600', 'marginRight': '6px'}),
            dcc.RadioItems(
                id='iforest-dimension-toggle',
                options=[
                    {'label': '1D', 'value': '1d'},
                    {'label': '2D', 'value': '2d'},
                ],
                value='2d',
                labelStyle={'display': 'inline-block', 'marginRight': '10px'},
                inputStyle={'marginRight': '4px'}
            )
        ], style={'display': 'flex', 'alignItems': 'center'}),

    ], style={
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'center',
        'justifyContent': 'flex-start',
        'gap': '18px',
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

    html.Hr(),

    html.Div([
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
    Output('iforest-scatter-plot', 'figure'),
    Input('iforest-generate-data-button', 'n_clicks'),
    State('iforest-number-of-points-slider', 'value'),
    State('iforest-dimension-toggle', 'value'),
    prevent_initial_call=False
)
def generate_iforest_demo(n_clicks, M, dim):
    # Initial empty state
    if not n_clicks:
        return go.Figure()

    # Default M if slider somehow None
    if M is None:
        M = 252

    # -------------------------
    # 1. Generate random Gaussian samples
    # -------------------------
    if dim == '1d':
        # 1D standard normal
        X = np.random.normal(loc=0.0, scale=1.0, size=(M, 1))
    else:
        # 2D correlated Gaussian
        mean = np.array([0.0, 0.0])
        cov = np.array([[1.0, 0.5],
                        [0.5, 1.0]])
        X = np.random.multivariate_normal(mean, cov, size=M)

    # -------------------------
    # 2. Fit Isolation Forest (1:1 trees) and get anomaly scores
    # -------------------------
    detector = IsolationForestAnomalyDetector(X)
    scores = np.array(detector.anomaly_scores())

    # -------------------------
    # 3. Build figure with two subplots:
    #    left: scatter colored by anomaly score
    #    right: anomaly score vs index
    # -------------------------
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=("", "")
    )

    # Left: scatter plot
    if dim == '1d':
        # map the 1D value to x, and keep y = 0 for visualization
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=np.zeros(M),
                mode='markers',
                marker=dict(
                    size=8,
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='score')
                ),
                name='samples (1D)'
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text="", showticklabels=False, row=1, col=1)
    else:
        # 2D scatter
        fig.add_trace(
            go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='score')
                ),
                name='samples (2D)'
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text="x₁", row=1, col=1)
        fig.update_yaxes(title_text="x₂", row=1, col=1)

    # Right: anomaly score vs index
    idx = np.arange(M)
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=scores,
            mode='lines+markers',
            name='anomaly score',
            line=dict(width=2)
        ),
        row=1, col=2
    )
    fig.update_xaxes(title_text="point index", row=1, col=2)
    fig.update_yaxes(title_text="score", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=600,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        title="Isolation Forest anomaly scores on random Gaussian data"
    )

    return fig
