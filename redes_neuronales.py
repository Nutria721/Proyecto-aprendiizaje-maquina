import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
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
# 1. Funciones de Preprocesamiento Manual
# ==========================================
def manual_train_test_split(X, y, test_size=0.3, seed=42):
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def manual_one_hot_encoder(y_labels):
    # Convertir a string para uniformidad
    y_labels = np.array([str(x) for x in y_labels])
    class_names = sorted(list(set(y_labels)))
    class_map = {label: i for i, label in enumerate(class_names)}
    num_classes = len(class_names)
    num_samples = len(y_labels)
    
    y_one_hot = np.zeros((num_samples, num_classes))
    indices = [class_map[label] for label in y_labels]
    y_one_hot[np.arange(num_samples), indices] = 1
    
    return y_one_hot, class_names, class_map

def manual_standard_scaler(X_train, X_test):
    # Ajustar solo con train
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1 # Evitar división por cero
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std
    return X_train_scaled, X_test_scaled

# ==========================================
# 2. Clase Neural Network (NumPy Puro)
# ==========================================
class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output, seed=42):
        np.random.seed(seed)
        # Inicialización He/Xavier básica
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2./n_input)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2./n_hidden)
        self.b2 = np.zeros((1, n_output))
        self.loss_history = []

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def compute_loss(self, y_true_oh, y_pred_probs):
        m = y_true_oh.shape[0]
        # Cross Entropy Loss con clip para estabilidad numérica
        y_pred_probs = np.clip(y_pred_probs, 1e-12, 1. - 1e-12)
        logprobs = -np.log(y_pred_probs[range(m), np.argmax(y_true_oh, axis=1)])
        loss = np.sum(logprobs) / m
        return loss

    def backward(self, X, y_true_oh, lr):
        m = X.shape[0]
        
        # Output Layer Gradients
        dZ2 = self.A2 - y_true_oh
        dW2 = (1/m) * (self.A1.T @ dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden Layer Gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1/m) * (X.T @ dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, y_true_oh, epochs, lr):
        self.loss_history = []
        for i in range(epochs):
            y_probs = self.forward(X)
            loss = self.compute_loss(y_true_oh, y_probs)
            self.loss_history.append(loss)
            self.backward(X, y_true_oh, lr)
            
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# ==========================================
# 3. Layout de la App
# ==========================================
# NOTA: Ya no creamos app = dash.Dash(__name__) aquí

# --- Panel de Configuración ---
config_panel = html.Div([
    html.H4("Configuración del Dataset", style={"color": "#2c3e50"}),
    html.Div([
        html.Div([
            html.Label("1. Seleccionar Archivo:"),
            dcc.Dropdown(
                # RENOMBRADO ID para evitar conflictos con otros archivos
                id="nn-file-selector",
                options=[{'label': f, 'value': f} for f in get_csv_files()],
                value=get_csv_files()[0] if get_csv_files() else None,
                clearable=False
            )
        ], style={"width": "30%", "display": "inline-block", "marginRight": "10px"}),

        html.Div([
            html.Label("2. Columna Objetivo (Clase):"),
            dcc.Dropdown(id="nn-target-selector", clearable=False)
        ], style={"width": "30%", "display": "inline-block", "marginRight": "10px"}),

        html.Div([
            html.Label("3. Features (Numéricas):"),
            dcc.Dropdown(id="nn-feature-selector", multi=True)
        ], style={"width": "35%", "display": "inline-block"}),
    ], style={"display": "flex", "alignItems": "flex-end"})
], style={"backgroundColor": "#f8f9fa", "padding": "20px", "borderRadius": "8px", "marginBottom": "20px", "border": "1px solid #dee2e6"})

# --- Layout Principal ---
# Asignamos a 'layout' en lugar de 'app.layout'
layout = html.Div([
    html.H1("Red Neuronal 'Desde Cero' (NumPy) - Universal", style={"textAlign": "center", "color": "#2c3e50"}),
    
    html.Div([
        config_panel,
        
        # Contenedor de Resultados (Se oculta hasta cargar datos)
        dcc.Loading(id="nn-loading-results", children=[
            html.Div(id="nn-results-container")
        ], type="circle")

    ], style={"padding": "20px", "maxWidth": "1200px", "margin": "0 auto"})
], style={"fontFamily": "sans-serif", "padding": "20px"})


# ==========================================
# 4. Callbacks (Lógica Reactiva)
# ==========================================

# Callback A: Cargar columnas cuando cambia el archivo
@app.callback(
    Output("nn-target-selector", "options"),
    Output("nn-feature-selector", "options"),
    Output("nn-target-selector", "value"),
    Output("nn-feature-selector", "value"),
    Input("nn-file-selector", "value")
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
        print(f"Error: {e}")
        return [], [], None, []

# Callback B: Ejecutar Entrenamiento y Actualizar UI
@app.callback(
    Output("nn-results-container", "children"),
    Input("nn-file-selector", "value"),
    Input("nn-target-selector", "value"),
    Input("nn-feature-selector", "value")
)
def train_and_update(filename, target_col, feature_cols):
    if not filename or not target_col or not feature_cols:
        return html.Div("Seleccione datos completos.")

    try:
        # 1. Carga de Datos
        df = pd.read_csv(filename)
        df.columns = [c.replace('.', '_').strip() for c in df.columns]
        
        df_clean = df[[target_col] + feature_cols].dropna()
        X_raw = df_clean[feature_cols].values
        y_raw = df_clean[target_col].values
        
        # 2. Preprocesamiento Manual
        X_train_raw, X_test_raw, y_train_labels, y_test_labels = manual_train_test_split(X_raw, y_raw)
        
        # One-Hot Encoder (Train)
        y_train_oh, class_names, class_map = manual_one_hot_encoder(y_train_labels)
        
        # Standard Scaler
        X_train, X_test = manual_standard_scaler(X_train_raw, X_test_raw)
        
        # 3. Configuración y Entrenamiento NN
        n_input = X_train.shape[1]
        n_output = len(class_names)
        n_hidden = 8 # Mantenemos fijo por simplicidad, podría ser parametrizable
        
        nn = NeuralNetwork(n_input=n_input, n_hidden=n_hidden, n_output=n_output)
        EPOCHS = 1000 
        LEARNING_RATE = 0.1
        
        nn.train(X_train, y_train_oh, EPOCHS, LEARNING_RATE)
        
        # 4. Predicción y Evaluación
        y_test_pred_idx = nn.predict(X_test)
        
        # Mapeo inverso de índices a etiquetas originales
        y_test_pred_labels = [class_names[idx] for idx in y_test_pred_idx]
        
        # 5. Métricas y Gráficos
        
        # Matriz Confusión y Accuracy
        cm = pd.DataFrame(0, index=class_names, columns=class_names)
        correct_count = 0
        
        y_test_labels_str = [str(x) for x in y_test_labels]
        
        for t, p in zip(y_test_labels_str, y_test_pred_labels):
            if t in class_names and p in class_names:
                cm.loc[t, p] += 1
            if t == p:
                correct_count += 1
                
        acc_score = correct_count / len(y_test_labels) if len(y_test_labels) > 0 else 0
        
        # Tabla de Métricas
        metrics_list = []
        for cls in class_names:
            TP = cm.loc[cls, cls]
            FP = cm.loc[:, cls].sum() - TP
            FN = cm.loc[cls, :].sum() - TP
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics_list.append({"Clase": cls, "Precision": round(precision,3), "Recall": round(recall,3), "F1": round(f1,3)})
            
        metrics_df = pd.DataFrame(metrics_list)
        
        # Gráficos
        
        # A. Loss
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=list(range(len(nn.loss_history))), y=nn.loss_history, mode='lines', name='Loss', line=dict(color='#e74c3c', width=2)))
        fig_loss.update_layout(title="Curva de Aprendizaje (Loss vs Épocas)", xaxis_title="Época", yaxis_title="Loss", height=350, margin=dict(l=20, r=20, t=40, b=20))

        # B. Heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=cm.values, x=cm.columns, y=cm.index, colorscale="Blues", text=cm.values, texttemplate="%{text}", textfont={"size": 16}
        ))
        fig_heatmap.update_layout(title="Matriz de Confusión (Test Set)", xaxis_title="Predicción", yaxis_title="Real", height=350, margin=dict(l=20, r=20, t=40, b=20))
        
        # C. Tabla Resultados Detallados
        df_results = pd.DataFrame({
            "Índice": range(len(y_test_labels)),
            "Real": y_test_labels_str,
            "Predicción": y_test_pred_labels,
            "Correcto": ["o" if r == p else "x" for r, p in zip(y_test_labels_str, y_test_pred_labels)]
        })

        # Construcción del Layout de Resultados
        return html.Div([
            # Resumen Superior
            html.Div([
                html.Div([
                    html.H4("Resumen del Modelo"),
                    html.Ul([
                        html.Li(f"Entradas: {n_input} (Features)"),
                        html.Li(f"Oculta: {n_hidden} neuronas"),
                        html.Li(f"Salida: {n_output} neuronas (Clases)"),
                        html.Li(f"Accuracy en Test: {acc_score:.2%}", style={"fontWeight": "bold", "color": "green"})
                    ])
                ], style={"width": "30%", "padding": "20px", "backgroundColor": "#ecf0f1", "borderRadius": "10px"}),
                
                html.Div([
                    dcc.Graph(figure=fig_loss)
                ], style={"width": "65%", "paddingLeft": "20px"})
            ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "marginBottom": "30px"}),

            # Métricas y Matriz
            html.Div([
                html.Div([
                    html.H4("Métricas por Clase"),
                    dash_table.DataTable(
                        data=metrics_df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in metrics_df.columns],
                        style_header={'backgroundColor': '#2c3e50', 'color': 'white'},
                        style_cell={'textAlign': 'center', 'padding': '10px'},
                    )
                ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top"}),
                
                html.Div([
                    dcc.Graph(figure=fig_heatmap)
                ], style={"width": "50%", "display": "inline-block", "float": "right"})
            ], style={"marginBottom": "30px"}),

            # Tabla Detallada
            html.Div([
                html.H3("Detalle de Predicciones", style={"textAlign": "center"}),
                dash_table.DataTable(
                    data=df_results.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df_results.columns],
                    page_size=10,
                    sort_action="native",
                    style_table={'overflowX': 'auto'},
                    style_header={'backgroundColor': '#34495e', 'color': 'white'},
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{Correcto} eq "x"'}, 'backgroundColor': '#ffcccc', 'color': 'red'}
                    ]
                )
            ])
        ])

    except Exception as e:
        import traceback
        return html.Div([
            html.H3("Error procesando el dataset", style={"color": "red"}),
            html.Pre(str(e)),
            html.Pre(traceback.format_exc())
        ])