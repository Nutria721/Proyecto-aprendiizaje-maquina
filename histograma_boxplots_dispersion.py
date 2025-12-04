import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import os

# --- IMPORTANTE: Importamos la app del archivo principal ---
from app import app

# === 1. Configuración Inicial y Funciones Auxiliares ===

def get_csv_files():
    """Busca archivos CSV en el directorio actual."""
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    return files

def load_and_clean_data(filename):
    """Carga el CSV, limpia NaNs y detecta columnas."""
    try:
        # Intenta leer con coma, si falla intenta punto y coma
        try:
            df = pd.read_csv(filename)
            if df.shape[1] < 2: 
                df = pd.read_csv(filename, sep=';')
        except:
            df = pd.read_csv(filename, sep=';')
            
        # 1. Eliminar nulos
        df.dropna(inplace=True)
        
        # 2. Normalizar nombres de columnas
        df.columns = [str(c).replace('.', '_').strip() for c in df.columns]
        
        # 3. Detectar columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 4. Detectar columna categórica (para "Variety")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        target_col = "variety" # Default name
        
        # Lógica para encontrar la mejor columna de categoría
        if 'variety' in df.columns:
            target_col = 'variety'
        elif 'species' in df.columns:
            df.rename(columns={'species': 'variety'}, inplace=True)
        elif 'class' in df.columns:
            df.rename(columns={'class': 'variety'}, inplace=True)
        elif 'target' in df.columns:
            df.rename(columns={'target': 'variety'}, inplace=True)
        elif len(cat_cols) > 0:
            # Si no hay nombres conocidos, usamos la última columna de texto encontrada
            target_col = cat_cols[-1]
            df.rename(columns={target_col: 'variety'}, inplace=True)
        else:
            # Si todo es numérico, creamos una categoría ficticia
            df['variety'] = 'Todos'
            
        return df, numeric_cols
    except Exception as e:
        print(f"Error cargando {filename}: {e}")
        return None, []

# NOTA: app = dash.Dash(__name__) ELIMINADO

# Obtener archivos al inicio
csv_files = get_csv_files()
default_file = csv_files[0] if csv_files else None

# === 3. Layout Principal ===
# Asignamos a 'layout' en vez de 'app.layout'
layout = html.Div([
    html.H1("Dashboard de Análisis Exploratorio Dinámico", style={"textAlign": "center", "marginBottom": "20px", "color": "#2c3e50"}),

    # --- SELECTOR DE ARCHIVO ---
    html.Div([
        html.Label("Selecciona un Dataset:", style={'fontWeight': 'bold', 'fontSize': '18px'}),
        dcc.Dropdown(
            # ID RENOMBRADO con prefijo 'hbd-'
            id='hbd-file-selector',
            options=[{'label': f, 'value': f} for f in csv_files],
            value=default_file,
            clearable=False,
            style={'width': '60%', 'margin': '0 auto'}
        ),
    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1', 'marginBottom': '20px'}),

    # --- CONTENIDO (TABS) ---
    dcc.Tabs([
        # --- PESTAÑA 1: Gráfica de Dispersión ---
        dcc.Tab(label='Gráfica de Dispersión', children=[
            html.Div([
                html.H3("Exploración de Dispersión", style={"textAlign": "center"}),
                
                # Controles de Ejes
                html.Div([
                    html.Div([
                        html.Label("Eje X:"),
                        # ID RENOMBRADO
                        dcc.Dropdown(id="hbd-x-axis", clearable=False),
                    ], style={"width": "45%", "display": "inline-block"}),

                    html.Div([
                        html.Label("Eje Y:"),
                        # ID RENOMBRADO
                        dcc.Dropdown(id="hbd-y-axis", clearable=False),
                    ], style={"width": "45%", "display": "inline-block", "marginLeft": "5%"}),
                ], style={'padding': '10px'}),

                html.Div([
                    html.Label("Filtrar por Categoría (Color):"),
                    # ID RENOMBRADO
                    dcc.Checklist(id="hbd-variety-filter", inline=True),
                ], style={"marginTop": "20px", "textAlign": "center", "padding": "10px"}),

                # ID RENOMBRADO
                dcc.Loading(dcc.Graph(id="hbd-scatter-plot"), type="default")
            ], style={"padding": "20px"})
        ]),

        # --- PESTAÑA 2: Análisis Estadístico ---
        dcc.Tab(label='Análisis Estadístico', children=[
            html.Div([
                html.H3("1. Tabla de Estadísticas Descriptivas"),
                dash_table.DataTable(
                    # ID RENOMBRADO
                    id="hbd-table-summary",
                    style_table={"margin": "20px", "overflowX": "auto"},
                    style_cell={"textAlign": "center", "padding": "5px"},
                    style_header={"backgroundColor": "#f4f4f4", "fontWeight": "bold"}
                ),

                html.H3("2. Histogramas"),
                html.Div([
                    html.Label("Selecciona variable para Histograma:"),
                    # ID RENOMBRADO
                    dcc.Dropdown(id="hbd-hist-col", clearable=False),
                    # ID RENOMBRADO
                    dcc.Loading(dcc.Graph(id="hbd-histogram"), type="default")
                ], style={"marginBottom": "40px"}),

                html.H3("3. Diagramas de Caja (Boxplots)"),
                html.Div([
                    html.Label("Selecciona variable para Boxplot:"),
                    # ID RENOMBRADO
                    dcc.Dropdown(id="hbd-box-col", clearable=False),
                    # ID RENOMBRADO
                    dcc.Loading(dcc.Graph(id="hbd-boxplot"), type="default")
                ])
            ], style={"padding": "20px"})
        ])
    ])
], style={"fontFamily": "sans-serif", "maxWidth": "1200px", "margin": "0 auto"})


# === 4. Callbacks (Lógica Reactiva) ===

# --- CALLBACK MAESTRO: Actualiza todas las opciones cuando cambia el archivo ---
@app.callback(
    # Outputs actualizados con prefijo 'hbd-'
    [Output("hbd-x-axis", "options"), Output("hbd-x-axis", "value"),
     Output("hbd-y-axis", "options"), Output("hbd-y-axis", "value"),
     Output("hbd-variety-filter", "options"), Output("hbd-variety-filter", "value"),
     Output("hbd-hist-col", "options"), Output("hbd-hist-col", "value"),
     Output("hbd-box-col", "options"), Output("hbd-box-col", "value"),
     Output("hbd-table-summary", "data"), Output("hbd-table-summary", "columns")],
    [Input("hbd-file-selector", "value")]
)
def update_controls_and_data(filename):
    if not filename:
        return [], None, [], None, [], [], [], None, [], None, [], []
    
    df, numeric_cols = load_and_clean_data(filename)
    
    if df is None or not numeric_cols:
        return [], None, [], None, [], [], [], None, [], None, [], []

    # Opciones para dropdowns numéricos
    opts = [{"label": c, "value": c} for c in numeric_cols]
    
    # Valores por defecto (primero y segundo de la lista si existen)
    val_x = numeric_cols[0]
    val_y = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
    
    # Opciones para el filtro de variedad
    varieties = sorted(df["variety"].unique().astype(str))
    var_opts = [{"label": v, "value": v} for v in varieties]
    var_vals = varieties # Todos seleccionados por defecto
    
    # Calcular tabla de estadísticas
    summary = []
    for col in numeric_cols:
        summary.append({
            "Variable": col,
            "Media": round(df[col].mean(), 2),
            "Mediana": round(df[col].median(), 2),
            "Varianza": round(df[col].var(), 2)
        })
    table_cols = [{"name": i, "id": i} for i in ["Variable", "Media", "Mediana", "Varianza"]]

    return (opts, val_x, 
            opts, val_y, 
            var_opts, var_vals, 
            opts, val_x, 
            opts, val_x, 
            summary, table_cols)


# --- CALLBACK GRÁFICO 1: Dispersión ---
@app.callback(
    Output("hbd-scatter-plot", "figure"),
    [Input("hbd-file-selector", "value"),
     Input("hbd-x-axis", "value"),
     Input("hbd-y-axis", "value"),
     Input("hbd-variety-filter", "value")]
)
def update_scatter(filename, x_col, y_col, selected_varieties):
    if not filename or not x_col or not y_col: return {}
    
    df, _ = load_and_clean_data(filename)
    
    # Filtrar por variedad seleccionada
    df["variety"] = df["variety"].astype(str)
    filtered_df = df[df["variety"].isin(selected_varieties)]
    
    fig = px.scatter(
        filtered_df, x=x_col, y=y_col, color="variety",
        title=f"Dispersión: {x_col} vs {y_col}",
        template="plotly_white",
        size_max=15
    )
    return fig


# --- CALLBACK GRÁFICO 2: Histograma ---
@app.callback(
    Output("hbd-histogram", "figure"),
    [Input("hbd-file-selector", "value"), Input("hbd-hist-col", "value")]
)
def update_histogram(filename, selected_col):
    if not filename or not selected_col: return {}
    
    df, _ = load_and_clean_data(filename)
    
    fig = px.histogram(
        df, x=selected_col, color="variety", 
        nbins=20, marginal="box", barmode="overlay",
        title=f"Distribución de {selected_col}",
        template="plotly_white"
    )
    fig.update_layout(bargap=0.1)
    return fig


# --- CALLBACK GRÁFICO 3: Boxplot ---
@app.callback(
    Output("hbd-boxplot", "figure"),
    [Input("hbd-file-selector", "value"), Input("hbd-box-col", "value")]
)
def update_boxplot(filename, selected_col):
    if not filename or not selected_col: return {}
    
    df, _ = load_and_clean_data(filename)
    
    fig = px.box(
        df, y=selected_col, color="variety", points="all",
        title=f"Boxplot: {selected_col} por Categoría",
        template="plotly_white"
    )
    return fig