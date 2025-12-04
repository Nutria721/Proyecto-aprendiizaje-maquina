import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import os

# === Importaciones Scikit-Learn ===
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# --- IMPORTANTE: Importamos la app del archivo principal ---
from app import app

# ==========================================
# 0. Funciones de Ayuda (Archivos)
# ==========================================
def get_csv_files():
    return [f for f in os.listdir('.') if f.endswith('.csv')]

# Generar iris.csv por defecto si no existe
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
# 1. Lógica de Modelado (Backend)
# ==========================================

def train_eval_classification(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    
    try:
        X_test_sc = scaler.transform(X_test)
    except:
        return 0, 0, 0, 0, []

    model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')
    try:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        return acc, prec, rec, f1, y_pred
    except Exception as e:
        return 0, 0, 0, 0, np.zeros_like(y_test)

def run_validation_suite(method_name, X, y):
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    y_true_all, y_pred_all = [], []

    if method_name == "Validación Simple (70/30)":
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
            a, p, r, f, preds = train_eval_classification(X_train, X_test, y_train, y_test)
            acc_list.append(a); prec_list.append(p); rec_list.append(r); f1_list.append(f)
            y_true_all.extend(y_test)
            y_pred_all.extend(preds)

    elif method_name == "K-Fold (k=10)":
        k = min(10, len(X))
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            a, p, r, f, preds = train_eval_classification(X_train, X_test, y_train, y_test)
            acc_list.append(a); prec_list.append(p); rec_list.append(r); f1_list.append(f)
            y_true_all.extend(y_test)
            y_pred_all.extend(preds)

    elif method_name == "K-Fold Estratificado (k=10)":
        try:
            k = min(10, len(X), min(np.bincount(pd.factorize(y)[0])))
            if k < 2: k = 2
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                a, p, r, f, preds = train_eval_classification(X_train, X_test, y_train, y_test)
                acc_list.append(a); prec_list.append(p); rec_list.append(r); f1_list.append(f)
                y_true_all.extend(y_test)
                y_pred_all.extend(preds)
        except ValueError:
            return None

    elif method_name == "Leave-One-Out (LOO)":
        if len(X) > 500: return None
        loo = LeaveOneOut()
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=200, solver='lbfgs')
        from sklearn.model_selection import cross_val_predict
        try:
            y_pred_all = cross_val_predict(model, X_sc, y, cv=loo)
            y_true_all = y
            acc_list = [accuracy_score(y_true_all, y_pred_all)]
            prec_list = [precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)]
            rec_list = [recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)]
            f1_list = [f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)]
        except:
            return None

    elif method_name == "Bootstrap (OOB)":
        n_iterations = 20
        for i in range(n_iterations):
            train_idx = resample(np.arange(len(X)), replace=True, random_state=i)
            test_idx = np.setdiff1d(np.arange(len(X)), train_idx)
            if len(test_idx) == 0: continue
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            a, p, r, f, preds = train_eval_classification(X_train, X_test, y_train, y_test)
            acc_list.append(a); prec_list.append(p); rec_list.append(r); f1_list.append(f)
            y_true_all.extend(y_test)
            y_pred_all.extend(preds)

    if not acc_list: return None

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

    return {
        "Método": method_name,
        "Accuracy": np.mean(acc_list),
        "Precision": np.mean(prec_list),
        "Recall": np.mean(rec_list),
        "F1 Score": np.mean(f1_list),
        "CM": cm,
        "Labels": labels
    }

# ==========================================
# 2. Layout de la App
# ==========================================

# Panel de Configuración
config_panel = html.Div([
    html.H4("1. Configuración del Dataset (Universal)", style={"color": "#2c3e50"}),
    html.Div([
        html.Div([
            html.Label("Seleccionar Archivo:"),
            dcc.Dropdown(
                id="file-selector",
                options=[{'label': f, 'value': f} for f in get_csv_files()],
                value=get_csv_files()[0] if get_csv_files() else None,
                clearable=False
            )
        ], style={"width": "30%", "display": "inline-block", "marginRight": "10px"}),

        html.Div([
            html.Label("Columna Objetivo (Clasificación):"),
            dcc.Dropdown(id="target-selector", clearable=False)
        ], style={"width": "30%", "display": "inline-block", "marginRight": "10px"}),

        html.Div([
            html.Label("Features (Variables Numéricas):"),
            dcc.Dropdown(id="feature-selector", multi=True)
        ], style={"width": "35%", "display": "inline-block"}),
    ], style={"display": "flex", "alignItems": "flex-end"})
], style={"backgroundColor": "#f8f9fa", "padding": "20px", "borderRadius": "8px", "marginBottom": "20px", "border": "1px solid #dee2e6"})

# --- Layout MODIFICADO: Se asigna a 'layout' en vez de 'app.layout' ---
layout = html.Div([
    html.H1("Dashboard Predictivo (Regresión & Clasificación)", style={"textAlign": "center", "color": "#2c3e50"}),
    
    html.Div([
        config_panel,
        
        dcc.Tabs([
            dcc.Tab(label='2. Regresión Lineal (Numérica)', children=[
                html.Div([
                    html.H3("Predicción de Variables Numéricas", style={"textAlign": "center"}),
                    html.P("Elige una de tus 'Features' para que sea la variable dependiente (Y) y las demás serán predictoras (X).", style={"textAlign": "center", "color": "gray"}),
                    
                    html.Div([
                        html.Label("Variable a Predecir (Y):", style={"fontWeight": "bold"}),
                        dcc.Dropdown(id="reg-target-sub", style={"width": "50%"})
                    ], style={"padding": "20px", "backgroundColor": "#eef2f3", "borderRadius": "10px", "marginBottom": "20px"}),

                    html.Div([
                        html.Div(id="reg-results-container", style={"width": "35%", "display": "inline-block", "verticalAlign": "top"}),
                        html.Div([dcc.Graph(id="reg-plot")], style={"width": "60%", "display": "inline-block", "verticalAlign": "top"})
                    ])
                ], style={"padding": "20px"})
            ]),

            dcc.Tab(label='3. Clasificación (Regresión Logística)', children=[
                html.Div([
                    html.H3("Validación de Modelos de Clasificación", style={"textAlign": "center"}),
                    html.P("Se usará la 'Columna Objetivo' del panel superior como clase a predecir.", style={"textAlign": "center"}),
                    
                    html.Button("Ejecutar Validación (5 Métodos)", id="btn-validate", n_clicks=0, style={"marginTop": "10px", "marginBottom": "20px", "width": "100%", "padding": "15px", "backgroundColor": "#3498db", "color": "white", "border": "none", "cursor": "pointer", "fontSize": "16px", "fontWeight": "bold"}),
                    
                    html.Div(id="classification-error-msg", style={"color": "red", "textAlign": "center", "fontWeight": "bold"}),

                    dcc.Loading(children=[
                        html.H4("Tabla Comparativa de Métricas", style={"textAlign": "center"}),
                        dash_table.DataTable(
                            id="validation-table",
                            style_table={"overflowX": "auto", "marginBottom": "40px"},
                            style_cell={"textAlign": "center", "fontFamily": "sans-serif", "padding": "10px"},
                            style_header={"backgroundColor": "#34495e", "color": "white", "fontWeight": "bold"},
                            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'}]
                        ),
                        
                        html.H4("Matrices de Confusión Acumuladas", style={"textAlign": "center"}),
                        html.Div(id="confusion-matrices-container", style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center", "gap": "20px"})
                    ], type="circle")

                ], style={"padding": "20px"})
            ])
        ])

    ], style={"padding": "20px", "maxWidth": "1200px", "margin": "0 auto"})
], style={"fontFamily": "sans-serif", "padding": "20px"})


# ==========================================
# 3. Callbacks
# ==========================================

# A. Cargar Columnas
@app.callback(
    Output("target-selector", "options"),
    Output("feature-selector", "options"),
    Output("target-selector", "value"),
    Output("feature-selector", "value"),
    Input("file-selector", "value")
)
def load_file_columns(filename):
    if not filename: return [], [], None, []
    try:
        df = pd.read_csv(filename)
        df.columns = [c.replace('.', '_').strip() for c in df.columns] 
        
        all_cols = [{'label': c, 'value': c} for c in df.columns]
        num_df = df.select_dtypes(include=[np.number])
        num_cols = [{'label': c, 'value': c} for c in num_df.columns]
        
        target_val = df.columns[-1]
        feat_vals = [c for c in num_df.columns if c != target_val]
        
        return all_cols, num_cols, target_val, feat_vals
    except Exception as e:
        return [], [], None, []

# B. Actualizar Dropdown de Regresión
@app.callback(
    Output("reg-target-sub", "options"),
    Output("reg-target-sub", "value"),
    Input("feature-selector", "value")
)
def update_reg_options(features):
    if not features: return [], None
    return [{'label': f, 'value': f} for f in features], features[0]

# C. Lógica Regresión Lineal
@app.callback(
    [Output("reg-results-container", "children"),
     Output("reg-plot", "figure")],
    [Input("file-selector", "value"),
     Input("reg-target-sub", "value"),
     Input("feature-selector", "value")]
)
def update_linear_regression(filename, reg_target, features):
    if not filename or not reg_target or not features:
        return html.Div("Seleccione variables..."), go.Figure()
    
    try:
        df = pd.read_csv(filename)
        df.columns = [c.replace('.', '_').strip() for c in df.columns]
        
        x_cols = [f for f in features if f != reg_target]
        if not x_cols:
            return html.Div("Necesitas al menos 2 variables numéricas (1 Predictora, 1 Objetivo)."), go.Figure()
            
        df_clean = df[x_cols + [reg_target]].dropna()
        X = df_clean[x_cols]
        y = df_clean[reg_target]
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        coeff_items = [html.Li(f"{col}: {coef:.4f}") for col, coef in zip(x_cols, model.coef_)]
        
        results_html = html.Div([
            html.H4(f"Objetivo: {reg_target}", style={"color": "#2980b9"}),
            html.Div([
                html.P([html.Strong("R² Score: "), f"{r2:.4f}"], style={"fontSize": "1.2em", "color": "green"}),
                html.P([html.Strong("MSE: "), f"{mse:.4f}"]),
                html.Hr(),
                html.P([html.Strong("Intercepto: "), f"{model.intercept_:.4f}"]),
                html.P([html.Strong("Coeficientes:")]),
                html.Ul(coeff_items)
            ], style={"backgroundColor": "#fff", "padding": "15px", "border": "1px solid #ddd", "borderRadius": "5px"})
        ])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=y_pred, mode='markers', name='Datos', marker=dict(color='#3498db', size=8, opacity=0.6)))
        min_val, max_val = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal', line=dict(color='#e74c3c', dash='dash')))
        fig.update_layout(title=f"Real vs Predicho ({reg_target})", xaxis_title="Valor Real", yaxis_title="Predicción", template="plotly_white")
        
        return results_html, fig
    except Exception as e:
        return html.Div(f"Error: {str(e)}"), go.Figure()

# D. Lógica Clasificación
@app.callback(
    [Output("confusion-matrices-container", "children"),
     Output("validation-table", "data"),
     Output("validation-table", "columns"),
     Output("classification-error-msg", "children")],
    [Input("btn-validate", "n_clicks")],
    [State("file-selector", "value"),
     State("target-selector", "value"),
     State("feature-selector", "value")]
)
def update_classification_tab(n_clicks, filename, target, features):
    if n_clicks == 0: return [], [], [], ""
    if not filename or not target or not features:
        return [], [], [], "Por favor configura el dataset arriba (Archivo, Objetivo, Features)."

    try:
        df = pd.read_csv(filename)
        df.columns = [c.replace('.', '_').strip() for c in df.columns]
        
        df_clean = df[features + [target]].dropna()
        X = df_clean[features].values
        y = df_clean[target].values
        
        if len(np.unique(y)) < 2:
             return [], [], [], "Error: La columna objetivo debe tener al menos 2 clases distintas."

        methods = ["Validación Simple (70/30)", "K-Fold (k=10)", "K-Fold Estratificado (k=10)", "Leave-One-Out (LOO)", "Bootstrap (OOB)"]
        results_data = []
        matrix_graphs = []

        for method in methods:
            res = run_validation_suite(method, X, y)
            if res is None: continue

            row = {k: res[k] for k in ["Método", "Accuracy", "Precision", "Recall", "F1 Score"]}
            for k, v in row.items():
                if isinstance(v, float): row[k] = round(v, 4)
            results_data.append(row)

            fig_cm = px.imshow(res["CM"], text_auto=True, 
                               labels=dict(x="Predicción", y="Real", color="N"),
                               x=res["Labels"], y=res["Labels"],
                               color_continuous_scale="Blues")
            fig_cm.update_layout(title=dict(text=method, font=dict(size=11)), margin=dict(l=10, r=10, t=30, b=10), height=250, width=250)
            
            matrix_graphs.append(html.Div([dcc.Graph(figure=fig_cm, config={'displayModeBar': False})], style={"border": "1px solid #ddd", "padding": "5px", "borderRadius": "5px"}))

        cols = [{"name": k, "id": k} for k in ["Método", "Accuracy", "Precision", "Recall", "F1 Score"]]
        return matrix_graphs, results_data, cols, ""
        
    except Exception as e:
        return [], [], [], f"Error procesando: {str(e)}"
