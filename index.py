import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Importar la app base
from app import app

# Importar tus módulos (asegúrate de que k-means sea kmeans)
import arboles_decision
import histograma_boxplots_dispersion
import kmeans 
import naive_bayes
import redes_neuronales
import regresion

# --- Estilos CSS ---
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "overflow": "auto",
    "font-family": "Arial, sans-serif"
}

CONTENT_STYLE = {
    "margin-left": "19rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "font-family": "Arial, sans-serif"
}

LINK_STYLE = {
    "display": "block",
    "padding": "10px 15px",
    "text-decoration": "none",
    "color": "#333",
    "border-radius": "5px",
    "margin-bottom": "5px",
    "transition": "background-color 0.3s"
}

# --- Layout del Menú ---
sidebar = html.Div(
    [
        html.H2("Data Science", style={'fontSize': '24px', 'marginBottom': '20px'}),
        html.Hr(),
        html.P("Selecciona un modelo:", style={'color': '#666'}),
        
        # Aquí reemplazamos el dcc.Nav erróneo por un html.Div contenedor
        html.Div([
            dcc.Link(' Histograma y Boxplots', href='/apps/histograma', style=LINK_STYLE),
            dcc.Link(' Árboles de Decisión', href='/apps/arboles', style=LINK_STYLE),
            dcc.Link(' K-Means Clustering', href='/apps/kmeans', style=LINK_STYLE),
            dcc.Link(' Naive Bayes', href='/apps/naive', style=LINK_STYLE),
            dcc.Link(' Redes Neuronales', href='/apps/redes', style=LINK_STYLE),
            dcc.Link(' Regresión Logística', href='/apps/regresion', style=LINK_STYLE),
        ])
    ],
    style=SIDEBAR_STYLE,
)

# --- Layout Principal ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    sidebar,
    html.Div(id='page-content', style=CONTENT_STYLE)
])

# --- Callback para cambiar de página ---
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/histograma':
        return histograma_boxplots_dispersion.layout
    elif pathname == '/apps/arboles':
        return arboles_decision.layout
    elif pathname == '/apps/kmeans':
        return kmeans.layout
    elif pathname == '/apps/naive':
        return naive_bayes.layout
    elif pathname == '/apps/redes':
        return redes_neuronales.layout
    elif pathname == '/apps/regresion':
        return regresion.layout
    else:
        # Página de bienvenida
        return html.Div([
            html.H1("Bienvenido a tu Portafolio"),
            html.P("Selecciona una opción del menú de la izquierda para ver tus códigos en acción."),
        ])

if __name__ == '__main__':
    app.run(debug=True)