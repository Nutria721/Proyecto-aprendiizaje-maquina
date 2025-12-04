import pandas as pd
import numpy as np
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os

# --- IMPORTANTE: Importamos la app del archivo principal ---
from app import app

# ==========================================
# === 1. Funciones Lógicas del Árbol (INTACTAS) ===
# ==========================================

def gini_impurity(y):
    """Calcula la impureza Gini de un vector de etiquetas."""
    if len(y) == 0: return 0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def best_split_verbose(X, y):
    n_features = X.shape[1]
    best_gain = -1
    best_attr, best_thresh = None, None
    gini_parent = gini_impurity(y)
    
    best_split_details = {}
    
    # Optimización: Elegir solo hasta 10 umbrales por feature para velocidad en datasets grandes
    for feature in range(n_features):
        unique_values = np.unique(X[:, feature])
        if len(unique_values) > 50:
             thresholds = np.percentile(unique_values, np.linspace(0, 100, 10))
        else:
             thresholds = unique_values

        for t in thresholds:
            left_mask = X[:, feature] <= t
            right_mask = ~left_mask
            
            if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                continue
            
            gini_left = gini_impurity(y[left_mask])
            gini_right = gini_impurity(y[right_mask])
            
            w_left = len(y[left_mask]) / len(y)
            w_right = len(y[right_mask]) / len(y)
            
            gain = gini_parent - (w_left * gini_left + w_right * gini_right)
            
            if gain > best_gain:
                best_gain = gain
                best_attr = feature
                best_thresh = t
                best_split_details = {
                    "gini_parent": gini_parent,
                    "gini_left": gini_left,
                    "gini_right": gini_right
                }

    return best_attr, best_thresh, best_gain, best_split_details

def build_tree_verbose(X, y, depth=0, max_depth=3):
    if len(np.unique(y)) == 1 or depth == max_depth:
        vals, counts = np.unique(y, return_counts=True)
        return {"class": vals[np.argmax(counts)]}
    
    attr, thresh, gain, details = best_split_verbose(X, y)

    if attr is None:
        vals, counts = np.unique(y, return_counts=True)
        return {"class": vals[np.argmax(counts)]}
        
    left_mask = X[:, attr] <= thresh
    right_mask = ~left_mask
    
    node = {
        "feature": attr,
        "threshold": thresh,
        "gain": gain,
        "gini_parent": details.get("gini_parent", 0),
        "gini_left": details.get("gini_left", 0),
        "gini_right": details.get("gini_right", 0),
        "left": build_tree_verbose(X[left_mask], y[left_mask], depth + 1, max_depth),
        "right": build_tree_verbose(X[right_mask], y[right_mask], depth + 1, max_depth)
    }
    return node

def predict_one(tree, x):
    if "class" in tree:
        return tree["class"]
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_one(tree["left"], x)
    else:
        return predict_one(tree["right"], x)

def predict_tree(tree, X):
    return np.array([predict_one(tree, x) for x in X])

def sklearn_to_manual_structure(clf, class_names):
    t = clf.tree_
    def recurse(node):
        if t.children_left[node] == t.children_right[node]:
            val = t.value[node][0]
            class_idx = np.argmax(val)
            # Protección contra índices fuera de rango si hay clases no predichas
            safe_class_idx = class_idx if class_idx < len(class_names) else 0
            return {"class": class_names[safe_class_idx]}
        
        feature_idx = t.feature[node]
        threshold = t.threshold[node]
        impurity = t.impurity[node]
        
        left_id = t.children_left[node]
        right_id = t.children_right[node]
        
        n = t.n_node_samples[node]
        n_l = t.n_node_samples[left_id]
        n_r = t.n_node_samples[right_id]
        imp_l = t.impurity[left_id]
        imp_r = t.impurity[right_id]
        
        gain = impurity - ((n_l/n)*imp_l + (n_r/n)*imp_r)
        
        return {
            "feature": feature_idx,
            "threshold": threshold,
            "gain": gain,
            "gini_parent": impurity,
            "gini_left": imp_l,
            "gini_right": imp_r,
            "left": recurse(left_id),
            "right": recurse(right_id)
        }
    return recurse(0)

# ==========================================
# === 2. Funciones de Visualización ===
# ==========================================

def calculate_metrics_per_class(y_true, y_pred, classes):
    metrics_list = []
    # Asegurar que las clases sean strings para evitar errores de índice
    classes = [str(c) for c in classes]
    y_true = [str(c) for c in y_true]
    y_pred = [str(c) for c in y_pred]
    
    unique_labels = sorted(list(set(classes) | set(y_true) | set(y_pred)))
    
    cm_matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels)
    
    for t, p in zip(y_true, y_pred):
        cm_matrix.loc[t, p] += 1
        
    for cls in unique_labels:
        TP = np.sum((np.array(y_true) == cls) & (np.array(y_pred) == cls))
        FP = np.sum((np.array(y_true) != cls) & (np.array(y_pred) == cls))
        TN = np.sum((np.array(y_true) != cls) & (np.array(y_pred) != cls))
        FN = np.sum((np.array(y_true) == cls) & (np.array(y_pred) != cls))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_list.append({
            "Clase": cls,
            "TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "Accuracy": round(accuracy, 3),
            "F1": round(f1, 3)
        })
        
    return metrics_list, cm_matrix

def create_tree_figure(tree, feature_names, title_suffix=""):
    fig = go.Figure()
    
    def get_positions(node, depth=0, x_min=0, x_max=1, parent_pos=None):
        x = (x_min + x_max) / 2
        y = 1 - depth * 0.25
        
        if "class" in node:
            text = f"<b>{node['class']}</b>"
            color = "lightgreen"
        else:
            f_name = feature_names[node['feature']] if node['feature'] < len(feature_names) else f"Feat {node['feature']}"
            text = f"{f_name} <= {node['threshold']:.2f}<br>Gain: {node['gain']:.3f}"
            color = "lightblue"
            
        if parent_pos:
            fig.add_trace(go.Scatter(
                x=[parent_pos[0], x], y=[parent_pos[1], y],
                mode='lines', line=dict(color='gray', width=1), hoverinfo='none'
            ))
            
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            text=[text], textposition="top center",
            marker=dict(size=25, color=color, line=dict(width=1, color='black')),
            hoverinfo='text', hovertext=f"Gini Parent: {node.get('gini_parent',0):.3f}"
        ))
        
        if "left" in node:
            get_positions(node["left"], depth+1, x_min, x, (x, y))
        if "right" in node:
            get_positions(node["right"], depth+1, x, x_max, (x, y))

    get_positions(tree)
    
    fig.update_layout(
        title=f"Estructura del Árbol {title_suffix}",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=500
    )
    return fig

def extract_gini_info(tree, feature_names, parent="Root", nodes_list=None):
    if nodes_list is None: nodes_list = []
    if "class" in tree: return nodes_list
    
    f_name = feature_names[tree['feature']] if tree['feature'] < len(feature_names) else f"Feat {tree['feature']}"
    current_node = f"{f_name} <= {tree['threshold']:.2f}"
    
    nodes_list.append({
        "Nodo Padre": parent,
        "Regla": current_node,
        "Gini Padre": round(tree['gini_parent'], 3),
        "Gini Izq": round(tree['gini_left'], 3),
        "Gini Der": round(tree['gini_right'], 3),
        "Ganancia": round(tree['gain'], 3)
    })
    
    extract_gini_info(tree["left"], feature_names, current_node + " (Sí)", nodes_list)
    extract_gini_info(tree["right"], feature_names, current_node + " (No)", nodes_list)
    return nodes_list

# ==========================================
# === 3. Layout de la App ===
# ==========================================
# NOTA: app = dash.Dash(__name__) ELIMINADO

# Buscar archivos CSV
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
dropdown_options = [{'label': f, 'value': f} for f in csv_files]
default_value = csv_files[0] if csv_files else 'dummy'
if not csv_files:
    dropdown_options.append({'label': 'Generar Datos Dummy', 'value': 'dummy'})

# Asignamos a 'layout' en vez de 'app.layout'
layout = html.Div([
    html.H1("Comparativa: Árbol de Decisión Manual vs Sklearn", style={"textAlign": "center", "color": "#2c3e50"}),
    
    html.Div([
        html.Label("Selecciona un Dataset:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            # ID RENOMBRADO con prefijo 'ad-'
            id='ad-dataset-selector',
            options=dropdown_options,
            value=default_value,
            clearable=False,
            style={'width': '50%', 'margin': '0 auto'}
        ),
    ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#f8f9fa'}),

    dcc.Loading(
        # ID RENOMBRADO
        id="ad-loading-content",
        type="default",
        children=html.Div(id='ad-main-content') # ID RENOMBRADO
    )
], style={"fontFamily": "sans-serif", "maxWidth": "1200px", "margin": "0 auto"})

# ==========================================
# === 4. Callback Principal ===
# ==========================================

@app.callback(
    # Outputs e Inputs actualizados con los nuevos IDs
    Output('ad-main-content', 'children'),
    Input('ad-dataset-selector', 'value')
)
def update_dashboard(selected_dataset):
    # --- 1. CARGA SEGURA DE DATOS ---
    if selected_dataset == 'dummy' or not selected_dataset:
        df = pd.DataFrame({
            "A": np.random.rand(50), "B": np.random.rand(50), 
            "Target": np.random.choice(["X", "Y"], 50)
        })
    else:
        try:
            # Intenta leer con coma, si falla intenta con punto y coma
            try:
                df = pd.read_csv(selected_dataset)
                if df.shape[1] < 2: # Si solo detecta 1 columna, probablemente el separador está mal
                    df = pd.read_csv(selected_dataset, sep=';')
            except:
                 df = pd.read_csv(selected_dataset, sep=';')
        except Exception as e:
            return html.Div(f"Error crítico cargando CSV: {str(e)}", style={'color': 'red'})

    # --- 2. LIMPIEZA DE DATOS ---
    # a) Eliminar filas con valores nulos
    df.dropna(inplace=True)
    
    if df.empty:
        return html.Div("Error: El dataset quedó vacío después de eliminar nulos.", style={'color': 'red'})

    # b) Normalizar nombres de columnas
    df.columns = [str(c).replace('.', '_').strip() for c in df.columns]
    target_col = df.columns[-1]

    # --- 3. PREPARACIÓN DE FEATURES Y TARGET ---
    # Separar X e y
    raw_X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    # IMPORTANTE: Convertir Features categóricas a numéricas si existen
    X_processed = raw_X.copy()
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    
    X = X_processed.values
    
    # Asegurar que el Target sea tratable como string para las visualizaciones
    y = np.array([str(val) for val in y])
    classes = np.unique(y)
    numeric_cols = list(X_processed.columns)

    if len(classes) < 2:
        return html.Div("Error: El dataset necesita al menos 2 clases en la última columna para clasificar.", style={'color': 'red'})

    # --- 4. ENTRENAMIENTO ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Modelo Sklearn
        clf = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3)
        clf.fit(X_train, y_train)
        y_pred_lib = clf.predict(X_test)

        # Modelo Manual
        tree_manual = build_tree_verbose(X_train, y_train, max_depth=3)
        y_pred_manual = predict_tree(tree_manual, X_test)
        
        tree_sklearn_dict = sklearn_to_manual_structure(clf, classes)
    except Exception as e:
        return html.Div(f"Error durante el entrenamiento: {str(e)}", style={'color': 'red'})

    # --- 5. GENERACIÓN DE VISUALIZACIONES ---
    metrics_lib, matrix_lib = calculate_metrics_per_class(y_test, y_pred_lib, classes)
    metrics_manual, matrix_manual = calculate_metrics_per_class(y_test, y_pred_manual, classes)

    df_metrics_combined = pd.concat([
        pd.DataFrame(metrics_lib).assign(Método="Sklearn"),
        pd.DataFrame(metrics_manual).assign(Método="Manual")
    ]) if metrics_lib else pd.DataFrame()

    fig_tree_manual = create_tree_figure(tree_manual, numeric_cols, "(Manual)")
    fig_tree_sklearn = create_tree_figure(tree_sklearn_dict, numeric_cols, "(Sklearn)")

    df_gini_manual = pd.DataFrame(extract_gini_info(tree_manual, numeric_cols))
    df_gini_sklearn = pd.DataFrame(extract_gini_info(tree_sklearn_dict, numeric_cols))

    # --- 6. RETORNO DEL LAYOUT ---
    return html.Div([
        html.H4(f"Dataset: {selected_dataset} | Filas limpias: {len(df)}", style={'textAlign': 'center', 'color': '#7f8c8d'}),

        html.Div([
            html.H3("1. Métricas", style={"borderBottom": "2px solid #3498db"}),
            dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in df_metrics_combined.columns],
                data=df_metrics_combined.to_dict("records"),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'padding': '10px'},
                style_header={'backgroundColor': '#ecf0f1', 'fontWeight': 'bold'},
            )
        ], style={"marginBottom": "40px", "padding": "20px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),

        html.Div([
            html.H3("2. Matrices de Confusión", style={"borderBottom": "2px solid #3498db"}),
            html.Div([
                html.Div([html.H4("Sklearn"), dcc.Graph(figure=go.Figure(data=go.Heatmap(z=matrix_lib.values, x=matrix_lib.columns, y=matrix_lib.index, colorscale="Blues", text=matrix_lib.values, texttemplate="%{text}"), layout=go.Layout(margin=dict(t=20, b=20))))], style={"width": "48%", "display": "inline-block"}),
                html.Div([html.H4("Manual"), dcc.Graph(figure=go.Figure(data=go.Heatmap(z=matrix_manual.values, x=matrix_manual.columns, y=matrix_manual.index, colorscale="Oranges", text=matrix_manual.values, texttemplate="%{text}"), layout=go.Layout(margin=dict(t=20, b=20))))], style={"width": "48%", "display": "inline-block", "float": "right"})
            ])
        ], style={"marginBottom": "40px", "padding": "20px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),

        html.Div([
            html.H3("3. Visualización de Árboles", style={"borderBottom": "2px solid #3498db"}),
            html.Div([
                html.Div([dcc.Graph(figure=fig_tree_manual)], style={"width": "48%", "display": "inline-block"}),
                html.Div([dcc.Graph(figure=fig_tree_sklearn)], style={"width": "48%", "display": "inline-block", "float": "right"})
            ])
        ], style={"marginBottom": "40px", "padding": "20px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),
        
        html.Div([
            html.H3("4. Tablas Gini", style={"borderBottom": "2px solid #3498db"}),
            html.Div([html.H4("Manual"), dash_table.DataTable(data=df_gini_manual.to_dict("records"), style_table={'overflowX': 'auto'})], style={"marginBottom": "20px"}),
            html.Div([html.H4("Sklearn"), dash_table.DataTable(data=df_gini_sklearn.to_dict("records"), style_table={'overflowX': 'auto'})])
        ], style={"padding": "20px"})
    ])