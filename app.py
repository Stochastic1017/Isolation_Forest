
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from binary_partition_page import binary_partition_layout
from iTree_page import iTree_layout
from path_length_page import path_length_layout
from iForest_page import iForest_layout

app = Dash(__name__, 
           suppress_callback_exceptions=True, 
           external_scripts=['https://cdn.plot.ly/plotly-latest.min.js'])
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    # Binary Partition Page
    if pathname in ['/', '/binary_partition']:
        return binary_partition_layout

    # iTree Page
    elif pathname in ['/', '/itrees']:
        return iTree_layout

    # Path Length Page
    elif pathname in ['/', '/path_length']:
        return path_length_layout

    elif pathname in ['/', '/iforest']:
        return iForest_layout 
    
    else:
        # Fallback (404)
        return html.H1("Page not found")

if __name__ == '__main__':
    app.run(debug=True)
