import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
import os

# --- IMPORTANTE: Importamos la app del archivo principal ---
from app import app

# ==========================================
# 0. Funciones de Ayuda para Archivos
# ==========================================
def get_csv_files():
    return [f for f in os.listdir('.') if f.endswith('.csv')]

# Generar iris.csv si no existe para tener un ejemplo
if "iris.csv" not in get_csv_files():
    try:
        from sklearn.datasets import load_iris
        iris_raw = load_iris()
        temp_df = pd.DataFrame(data=iris_raw.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        temp_df['variety'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)
        temp_df.to_csv("iris.csv", index=False)
    except:
        pass

# ==========================================
# 1. Clase Naive Bayes Manual (Original)
# ==========================================
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _gaussian_prob(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        # Agregamos epsilon para evitar división por cero
        var = var + 1e-9 
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                conditional = np.sum(np.log(self._gaussian_prob(c, x)))
                posterior = prior + conditional
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

# ==========================================
# 2. Layout de la App
# ==========================================
# NOTA: Eliminamos app = dash.Dash(__name__)

# --- Panel de Configuración ---
config_panel = html.Div([
    html.H4("Configuración del Dataset", style={"color": "#2c3e50"}),
    html.Div([
        html.Div([
            html.Label("1. Seleccionar Archivo:"),
            dcc.Dropdown(
                # RENOMBRADO ID para evitar conflicto con otros archivos
                id="nb-file-selector",
                options=[{'label': f, 'value': f} for f in get_csv_files()],
                value=get_csv_files()[0] if get_csv_files() else None,
                clearable=False
            )
        ], style={"width": "30%", "display": "inline-block", "marginRight": "10px"}),

        html.Div([
            html.Label("2. Columna Objetivo (Clase):"),
            # RENOMBRADO ID
            dcc.Dropdown(id="nb-target-selector", clearable=False)
        ], style={"width": "30%", "display": "inline-block", "marginRight": "10px"}),

        html.Div([
            html.Label("3. Features (Numéricas):"),
            # RENOMBRADO ID
            dcc.Dropdown(id="nb-feature-selector", multi=True)
        ], style={"width": "35%", "display": "inline-block"}),
    ], style={"display": "flex", "alignItems": "flex-end"})
], style={"backgroundColor": "#f8f9fa", "padding": "20px", "borderRadius": "8px", "marginBottom": "20px", "border": "1px solid #dee2e6"})

# --- Layout Principal ---
# Usamos 'layout' en vez de 'app.layout'
layout = html.Div([
    html.H1("Clasificador Naive Bayes (Gaussiano) Adaptable", style={"textAlign": "center", "color": "#2c3e50"}),
    html.P("Implementación manual visualizando las distribuciones de probabilidad para cualquier dataset.", style={"textAlign": "center", "color": "#7f8c8d"}),
    
    html.Div([
        config_panel,
        
        # Contenedor de Resultados (Se oculta si hay error o falta selección)
        # RENOMBRADO ID
        html.Div(id="nb-results-container", children=[
            # Sección 1: Métricas
            html.H3("1. Rendimiento por Clase", style={"borderBottom": "2px solid #3498db", "marginTop": "20px"}),
            # RENOMBRADO ID
            html.Div(id="nb-metrics-table-container"),

            # Sección 2: Visualización (Gráficos)
            html.Div([
                # Izquierda: Heatmap
                html.Div([
                    html.H3("2. Matriz de Confusión", style={"borderBottom": "2px solid #3498db"}),
                    # RENOMBRADO ID
                    dcc.Graph(id="nb-heatmap-graph", style={"height": "400px"})
                ], style={"width": "40%", "minWidth": "300px", "marginRight": "20px"}),
                
                # Derecha: Distribuciones
                html.Div([
                    html.H3("3. Distribuciones Gaussianas (Priors)", style={"borderBottom": "2px solid #3498db"}),
                    html.P("Estas curvas muestran la probabilidad aprendida para cada característica según la clase.", style={"fontSize": "0.9em", "fontStyle": "italic"}),
                    # RENOMBRADO ID
                    dcc.Graph(id="nb-gaussian-graph", style={"height": "600px"})
                ], style={"width": "55%", "minWidth": "300px"})
            ], style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center", "marginTop": "20px"})
        ])
    ], style={"padding": "20px", "maxWidth": "1200px", "margin": "0 auto"})
], style={"fontFamily": "sans-serif", "paddingBottom": "50px"})


# ==========================================
# 3. Callbacks (Lógica Reactiva)
# ==========================================

# Callback A: Cargar columnas cuando cambia el archivo
@app.callback(
    Output("nb-target-selector", "options"),
    Output("nb-feature-selector", "options"),
    Output("nb-target-selector", "value"),
    Output("nb-feature-selector", "value"),
    Input("nb-file-selector", "value")
)
def load_file_columns(filename):
    if not filename: return [], [], None, []
    try:
        df = pd.read_csv(filename)
        # Limpieza básica de columnas
        df.columns = [c.replace('.', '_').strip() for c in df.columns]
        
        all_cols = [{'label': c, 'value': c} for c in df.columns]
        
        # Detectar numéricas
        num_df = df.select_dtypes(include=[np.number])
        num_cols = [{'label': c, 'value': c} for c in num_df.columns]
        
        # Valores por defecto inteligentes
        target_val = df.columns[-1] # Asumimos la última es target
        feat_vals = [c for c in num_df.columns if c != target_val] # Resto numérico son features
        
        return all_cols, num_cols, target_val, feat_vals
    except Exception as e:
        print(f"Error: {e}")
        return [], [], None, []

# Callback B: Ejecutar Modelo y Actualizar Gráficos
@app.callback(
    Output("nb-metrics-table-container", "children"),
    Output("nb-heatmap-graph", "figure"),
    Output("nb-gaussian-graph", "figure"),
    Input("nb-file-selector", "value"),
    Input("nb-target-selector", "value"),
    Input("nb-feature-selector", "value")
)
def update_dashboard(filename, target_col, feature_cols):
    if not filename or not target_col or not feature_cols:
        return html.Div("Seleccione datos completos."), go.Figure(), go.Figure()

    # 1. Carga y Preparación
    df = pd.read_csv(filename)
    df.columns = [c.replace('.', '_').strip() for c in df.columns]
    
    # Filtrar NaN en las columnas seleccionadas
    df_clean = df[[target_col] + feature_cols].dropna()
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    classes = np.unique(y)

    # 2. Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    # 3. Métricas
    metrics_list = []
    cm_matrix = pd.DataFrame(0, index=classes, columns=classes)
    
    # Calcular matriz confusión
    for t, p in zip(y_test, y_pred):
        if t in classes and p in classes:
            cm_matrix.loc[t, p] += 1
            
    for cls in classes:
        TP = np.sum((y_test == cls) & (y_pred == cls))
        FP = np.sum((y_test != cls) & (y_pred == cls))
        TN = np.sum((y_test != cls) & (y_pred != cls))
        FN = np.sum((y_test == cls) & (y_pred != cls))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_list.append({
            "Clase": cls,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "Accuracy": round(accuracy, 3),
            "F1": round(f1, 3)
        })

    # Crear Tabla Dash
    metrics_table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in pd.DataFrame(metrics_list).columns],
        data=metrics_list,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '10px', 'fontFamily': 'sans-serif'},
        style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]
    )

    # 4. Gráfico Heatmap
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=cm_matrix.values,
        x=cm_matrix.columns,
        y=cm_matrix.index,
        colorscale="Blues",
        text=cm_matrix.values,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    fig_heatmap.update_layout(title="Matriz de Confusión", xaxis_title="Predicción", yaxis_title="Real")

    # 5. Gráfico Distribuciones (Adaptado dinámicamente)
    n_features = len(feature_cols)
    rows = (n_features + 1) // 2
    cols = 2 if n_features > 1 else 1
    
    fig_gauss = make_subplots(rows=rows, cols=cols, subplot_titles=feature_cols)
    colors = ['#EF553B', '#00CC96', '#636EFA', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    
    for i, feature_name in enumerate(feature_cols):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Datos para graficar curva (min/max del dataset completo para consistencia)
        x_min = df_clean[feature_name].min()
        x_max = df_clean[feature_name].max()
        padding = (x_max - x_min) * 0.2
        x_axis = np.linspace(x_min - padding, x_max + padding, 200)
        
        for j, cls in enumerate(nb.classes):
            # Obtener medias/vars del modelo entrenado
            # Nota: el indice 'i' corresponde al orden en feature_cols
            mean = nb.means[cls][i]
            var = nb.vars[cls][i]
            std = np.sqrt(var)
            
            # PDF Gaussiana
            y_axis = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_axis - mean) / std) ** 2)
            
            fig_gauss.add_trace(
                go.Scatter(
                    x=x_axis, y=y_axis, mode='lines', name=str(cls),
                    line=dict(color=colors[j % len(colors)]),
                    showlegend=(i==0), # Solo mostrar leyenda en el primer subplot
                    fill='tozeroy', 
                    fillcolor=f"rgba{tuple(list(int(colors[j%len(colors)].lstrip('#')[k:k+2], 16) for k in (0, 2, 4)) + [0.1])}"
                ),
                row=row, col=col
            )
            
    fig_gauss.update_layout(height=300 * rows, title_text="Distribuciones Gaussianas Aprendidas", showlegend=True)

    return metrics_table, fig_heatmap, fig_gauss