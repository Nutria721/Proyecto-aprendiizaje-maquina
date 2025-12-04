import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.utils import resample

# --- IMPORTANTE: Importamos la app del archivo principal ---
from app import app

# ===================== GESTIÓN DE DATASETS =====================
def get_csv_files():
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    return files

if "iris.csv" not in get_csv_files():
    try:
        from sklearn.datasets import load_iris
        iris_raw = load_iris()
        temp_df = pd.DataFrame(data=iris_raw.data, columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
        temp_df['variety'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)
        temp_df.to_csv("iris.csv", index=False)
    except:
        pass

# Variables Globales Iniciales
df = pd.DataFrame()
numeric_cols = []
class_col = ""
classes = []
X = np.array([])
y = np.array([])

# Variables globales para la Simulación
K_global = 3
centroids = None
cluster_assignments = None
iteration = 0
centroids_history = []
assignments_history = []
sk_centroids_history = []
sk_assignments_history = []
sk_iteration = 0
initial_centroids = None
metrics_data = {}

# NOTA: Ya no creamos app = dash.Dash(__name__)

# ===================== FUNCIONES MATEMÁTICAS (COMUNES) =====================
def initialize_centroids(data, k, seed=124):
    if len(data) < k: return data
    np.random.seed(seed)
    indices = np.random.choice(len(data), k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    assignments = []
    for point in data:
        distances = np.linalg.norm(point - centroids, axis=1)
        assignments.append(np.argmin(distances))
    return np.array(assignments)

def update_centroids(data, assignments, k, prev_centroids):
    new_centroids = []
    for i in range(k):
        cluster_points = data[assignments == i]
        if len(cluster_points) > 0:
            new_centroids.append(cluster_points.mean(axis=0))
        else:
            new_centroids.append(prev_centroids[i])
    return np.array(new_centroids)

# ===================== LÓGICA DE EVALUACIÓN Y VALIDACIÓN =====================
def get_cluster_mapping(train_X, train_y, centroids, K):
    train_assignments = assign_clusters(train_X, centroids)
    mapping = {}
    for i in range(K):
        mask = (train_assignments == i)
        if mask.sum() == 0:
            if len(train_y) > 0: mapping[i] = train_y[0]
            else: mapping[i] = 0
            continue
        labels, counts = np.unique(train_y[mask], return_counts=True)
        mapping[i] = labels[np.argmax(counts)]
    return mapping

def predict_with_mapping(test_X, centroids, mapping):
    test_assignments = assign_clusters(test_X, centroids)
    return np.array([mapping[a] for a in test_assignments])

def confusion_matrix_full(y_true, y_pred, class_list):
    matrix = pd.DataFrame(0, index=class_list, columns=class_list)
    for t, p in zip(y_true, y_pred):
        if p in class_list and t in class_list:
            matrix.loc[t, p] += 1
    return matrix

def confusion_matrix_per_class(y_true, y_pred, classes):
    cm = {}
    for cls in classes:
        TP = np.sum((y_true == cls) & (y_pred == cls))
        FP = np.sum((y_true != cls) & (y_pred == cls))
        TN = np.sum((y_true != cls) & (y_pred != cls))
        FN = np.sum((y_true == cls) & (y_pred != cls))
        cm[cls] = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
    return cm

def classification_metrics(cm):
    metrics_res = {}
    for cls, vals in cm.items():
        TP, FP, FN, TN = vals["TP"], vals["FP"], vals["FN"], vals["TN"]
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics_res[cls] = {
            "Precision": precision, "Recall": recall, "Accuracy": accuracy, "F1": f1
        }
    return metrics_res

# ===================== MÉTODOS DE VALIDACIÓN =====================
def run_kmeans_fold(train_X, train_y, test_X, test_y, K, seed):
    centroids = initialize_centroids(train_X, K, seed=seed)
    for _ in range(10): 
        assignments = assign_clusters(train_X, centroids)
        centroids = update_centroids(train_X, assignments, K, centroids)
    
    mapping = get_cluster_mapping(train_X, train_y, centroids, K)
    y_pred = predict_with_mapping(test_X, centroids, mapping)
    return y_pred

def simple_validation(data, labels, K, classes_list, test_size=0.3):
    n_test = int(len(data) * test_size)
    if n_test == 0: n_test = 1
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    train_idx, test_idx = indices[:-n_test], indices[-n_test:]
    y_pred = run_kmeans_fold(data[train_idx], labels[train_idx], data[test_idx], labels[test_idx], K, seed=42)
    return confusion_matrix_per_class(labels[test_idx], y_pred, classes_list), confusion_matrix_full(labels[test_idx], y_pred, classes_list)

def kfold_validation(data, labels, K, classes_list, n_splits=5, stratified=False):
    all_y_true, all_y_pred = [], []
    if n_splits > len(data): n_splits = len(data)

    if stratified:
        class_indices = {cls: np.where(labels == cls)[0] for cls in classes_list}
        folds = [[] for _ in range(n_splits)]
        np.random.seed(42)
        for cls, idxs in class_indices.items():
            np.random.shuffle(idxs)
            split = np.array_split(idxs, n_splits)
            for i in range(len(split)):
                 if i < n_splits: folds[i].extend(split[i])
        splits_indices = []
        for i in range(n_splits):
            test_idx = np.array(folds[i])
            if len(test_idx) == 0: continue
            train_idx = np.setdiff1d(np.arange(len(data)), test_idx)
            splits_indices.append((train_idx, test_idx))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits_indices = list(kf.split(data))

    for i, (train_idx, test_idx) in enumerate(splits_indices):
        y_pred_fold = run_kmeans_fold(data[train_idx], labels[train_idx], data[test_idx], labels[test_idx], K, seed=i)
        all_y_true.extend(labels[test_idx])
        all_y_pred.extend(y_pred_fold)
    
    all_y_true, all_y_pred = np.array(all_y_true), np.array(all_y_pred)
    return confusion_matrix_per_class(all_y_true, all_y_pred, classes_list), confusion_matrix_full(all_y_true, all_y_pred, classes_list)

def leave_one_out_validation(data, labels, K, classes_list):
    loo = LeaveOneOut()
    all_y_true, all_y_pred = [], []
    limit = 200
    indices = np.arange(len(data))
    if len(data) > limit:
        np.random.shuffle(indices)
        indices = indices[:limit]
        data_subset, labels_subset = data[indices], labels[indices]
    else:
        data_subset, labels_subset = data, labels

    for i, (train_idx, test_idx) in enumerate(loo.split(data_subset)):
        y_pred_fold = run_kmeans_fold(data_subset[train_idx], labels_subset[train_idx], data_subset[test_idx], labels_subset[test_idx], K, seed=i)
        all_y_true.extend(labels_subset[test_idx])
        all_y_pred.extend(y_pred_fold)

    all_y_true, all_y_pred = np.array(all_y_true), np.array(all_y_pred)
    return confusion_matrix_per_class(all_y_true, all_y_pred, classes_list), confusion_matrix_full(all_y_true, all_y_pred, classes_list)

def bootstrap_validation(data, labels, K, classes_list, n_iterations=20):
    matrices, cms = [], []
    np.random.seed(99)
    for i in range(n_iterations):
        indices = np.arange(len(data))
        train_idx = resample(indices, replace=True, random_state=i)
        test_idx = np.setdiff1d(indices, train_idx)
        if len(test_idx) == 0: continue
        y_pred_oob = run_kmeans_fold(data[train_idx], labels[train_idx], data[test_idx], labels[test_idx], K, seed=i)
        cms.append(confusion_matrix_per_class(labels[test_idx], y_pred_oob, classes_list))
        matrices.append(confusion_matrix_full(labels[test_idx], y_pred_oob, classes_list))

    avg_cm = {cls: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for cls in classes_list}
    if cms:
        for cm_iter in cms:
            for cls in classes_list:
                for key in cm_iter[cls]: avg_cm[cls][key] += cm_iter[cls][key] / len(cms)
        avg_matrix = sum(matrices) / len(matrices)
    else:
        avg_matrix = pd.DataFrame(0, index=classes_list, columns=classes_list)
    return avg_cm, avg_matrix

# ===================== LAYOUT DE LA APP =====================

# --- Configuración de Dataset ---
dataset_selection_layout = html.Div([
    html.H4("1. Configuración del Dataset", style={"color": "darkblue"}),
    html.Div([
        html.Label("Seleccionar Archivo CSV (Carpeta Raíz):"),
        dcc.Dropdown(
            # ID RENOMBRADO
            id="km-file-selector",
            options=[{'label': f, 'value': f} for f in get_csv_files()],
            value=get_csv_files()[0] if get_csv_files() else None
        )
    ], style={"width": "30%", "display": "inline-block", "marginRight": "2%"}),

    html.Div([
        html.Label("Columna Objetivo (Clase/Target):"),
        # ID RENOMBRADO
        dcc.Dropdown(id="km-target-col-selector")
    ], style={"width": "30%", "display": "inline-block", "marginRight": "2%"}),

    html.Div([
        html.Label("Columnas Numéricas (Features):"),
        # ID RENOMBRADO
        dcc.Dropdown(id="km-feature-cols-selector", multi=True)
    ], style={"width": "35%", "display": "inline-block"}),
    html.Hr()
], style={"backgroundColor": "#f9f9f9", "padding": "20px", "borderRadius": "5px", "marginBottom": "20px"})

# --- Layout Pestaña 1: Simulación ---
tab1_layout = html.Div([
    html.H3("Comparación Interactiva: Algoritmo Propio vs Scikit-learn", style={"textAlign": "center"}),
    
    html.Div([
        html.Div([
            html.Label("Eje X:"),
            # ID RENOMBRADO
            dcc.Dropdown(id="km-x-axis"),
        ], style={"width": "45%", "display": "inline-block"}),

        html.Div([
            html.Label("Eje Y:"),
            # ID RENOMBRADO
            dcc.Dropdown(id="km-y-axis"),
        ], style={"width": "45%", "display": "inline-block", "marginLeft": "5%"}),
    ]),

    html.Div([
        html.Label("Número de Clusters (K):"),
        # IDs RENOMBRADOS
        dcc.Input(id="km-num-clusters", type="number", value=3, min=1, max=10, step=1),
        html.Button("Reiniciar", id="km-reset-centroids", n_clicks=0, style={"marginLeft": "10px"}),
        html.Button("Anterior Iteración", id="km-prev-iteration", n_clicks=0, style={"marginLeft": "10px"}),
        html.Button("Siguiente Iteración", id="km-next-iteration", n_clicks=0, style={"marginLeft": "10px"}),
    ], style={"marginTop": "20px", "marginBottom": "20px"}),

    html.Div([
        html.Div([
            html.H4("Algoritmo Propio"),
            # IDs RENOMBRADOS
            html.Div(id="km-info-propio", style={"marginBottom": "10px"}),
            dcc.Graph(id="km-kmeans-propio")
        ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),

        html.Div([
            html.H4("Scikit-learn KMeans"),
            # IDs RENOMBRADOS
            html.Div(id="km-info-sklearn", style={"marginBottom": "10px"}),
            dcc.Graph(id="km-kmeans-sklearn")
        ], style={"width": "48%", "display": "inline-block", "marginLeft": "4%", "verticalAlign": "top"}),
    ])
])

# --- Layout Pestaña 2: Métricas ---
# ID RENOMBRADO
tab2_layout = html.Div(id="km-metrics-content")

# Usamos 'layout' en lugar de 'app.layout'
layout = html.Div([
    html.H1("Dashboard K-Means Universal", style={"textAlign": "center"}),
    dataset_selection_layout,
    dcc.Tabs([
        dcc.Tab(label='Simulación Interactiva', children=tab1_layout),
        dcc.Tab(label='Métricas de Evaluación', children=tab2_layout),
    ])
])

# ===================== CALLBACKS DE CARGA DE DATOS =====================
@app.callback(
    Output("km-target-col-selector", "options"),
    Output("km-feature-cols-selector", "options"),
    Output("km-target-col-selector", "value"),
    Output("km-feature-cols-selector", "value"),
    Output("km-x-axis", "options"),
    Output("km-y-axis", "options"),
    Output("km-x-axis", "value"),
    Output("km-y-axis", "value"),
    Input("km-file-selector", "value")
)
def load_data(filename):
    global df
    if not filename: return [], [], None, [], [], [], None, None
    try:
        df = pd.read_csv(filename)
        df = df.dropna()
        all_cols = [{"label": c, "value": c} for c in df.columns]
        num_df = df.select_dtypes(include=[np.number])
        num_options = [{"label": c, "value": c} for c in num_df.columns]
        
        default_target = df.columns[-1]
        default_features = [c for c in num_df.columns if c != default_target]
        default_x = default_features[0] if len(default_features) > 0 else None
        default_y = default_features[1] if len(default_features) > 1 else default_x

        return (all_cols, num_options, default_target, default_features, num_options, num_options, default_x, default_y)
    except Exception as e:
        print(f"Error cargando archivo: {e}")
        return [], [], None, [], [], [], None, None

# ===================== CALLBACKS DE SIMULACIÓN =====================
@app.callback(
    Output("km-kmeans-propio", "figure"),
    Output("km-kmeans-sklearn", "figure"),
    Output("km-info-propio", "children"),
    Output("km-info-sklearn", "children"),
    Input("km-next-iteration", "n_clicks"),
    Input("km-prev-iteration", "n_clicks"),
    Input("km-reset-centroids", "n_clicks"),
    Input("km-x-axis", "value"),
    Input("km-y-axis", "value"),
    Input("km-num-clusters", "value"),
    Input("km-feature-cols-selector", "value"),
    State("km-kmeans-propio", "figure")
)
def update_kmeans_simulation(next_clicks, prev_clicks, reset_clicks, x_col, y_col, num_clusters, selected_features, current_fig):
    global centroids, cluster_assignments, iteration, K_global
    global centroids_history, assignments_history
    global sk_centroids_history, sk_assignments_history, sk_iteration
    global initial_centroids, df
    
    if df.empty or not x_col or not y_col or not selected_features:
        return {}, {}, "Seleccione datos", "Seleccione datos"

    K_global = num_clusters
    X_subset = df[selected_features].values
    
    ctx = dash.callback_context
    if not ctx.triggered: trigger_id = "km-reset-centroids"
    else: trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    is_reset = (centroids is None or trigger_id == "km-reset-centroids" or 
               trigger_id == "km-num-clusters" or trigger_id == "km-feature-cols-selector")

    if is_reset:
        initial_centroids = initialize_centroids(X_subset, K_global)
        centroids = initial_centroids.copy()
        cluster_assignments = assign_clusters(X_subset, centroids)
        iteration = 0
        centroids_history = [centroids.copy()]
        assignments_history = [cluster_assignments.copy()]

        try:
            kmeans = KMeans(n_clusters=K_global, init=initial_centroids, n_init=1, max_iter=1, random_state=0)
            kmeans.fit(X_subset)
            sk_centroids_history = [kmeans.cluster_centers_.copy()]
            sk_assignments_history = [kmeans.labels_.copy()]
        except:
            sk_centroids_history = [initial_centroids]
            sk_assignments_history = [np.zeros(len(X_subset))]
        sk_iteration = 0

    elif trigger_id == "km-next-iteration":
        cluster_assignments = assign_clusters(X_subset, centroids)
        centroids = update_centroids(X_subset, cluster_assignments, K_global, centroids)
        iteration += 1
        centroids_history.append(centroids.copy())
        assignments_history.append(cluster_assignments.copy())

        try:
            kmeans = KMeans(n_clusters=K_global, init=sk_centroids_history[-1], n_init=1, max_iter=1, random_state=0)
            kmeans.fit(X_subset)
            sk_centroids_history.append(kmeans.cluster_centers_.copy())
            sk_assignments_history.append(kmeans.labels_.copy())
        except: pass
        sk_iteration += 1

    elif trigger_id == "km-prev-iteration":
        if iteration > 0:
            iteration -= 1
            centroids = centroids_history[iteration].copy()
            cluster_assignments = assignments_history[iteration].copy()
        if sk_iteration > 0: sk_iteration -= 1

    idx_x = selected_features.index(x_col)
    idx_y = selected_features.index(y_col)
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'gray']

    def create_figure(hist_centroids, hist_assignments, current_iter, title_prefix):
        safe_iter = min(current_iter, len(hist_centroids) - 1)
        if safe_iter < 0: safe_iter = 0
        current_cents = hist_centroids[safe_iter]
        current_assigns = hist_assignments[safe_iter]
        
        fig_data, info_text = [], []
        for i in range(K_global):
            mask = (current_assigns == i)
            points_x = df.loc[mask, x_col]
            points_y = df.loc[mask, y_col]
            color = colors[i % len(colors)]
            
            fig_data.append({"x": points_x, "y": points_y, "mode": "markers", "name": f"Cluster {i+1}", "marker": {"color": color, "size": 8, "opacity": 0.6}})
            
            cx, cy = current_cents[i, idx_x], current_cents[i, idx_y]
            fig_data.append({"x": [cx], "y": [cy], "mode": "markers", "name": f"Centroide {i+1}", "marker": {"color": "black", "symbol": "x", "size": 15, "line": {"width": 2}}})
            
            for px, py in zip(points_x, points_y):
                fig_data.append({"x": [px, cx], "y": [py, cy], "mode": "lines", "line": {"color": color, "width": 0.5}, "showlegend": False, "hoverinfo": "skip"})
            
            info_text.append(f"Centr {i+1}: Puntos: {len(points_x)}")

        layout = {"title": f"{title_prefix} - Iteración {safe_iter}", "xaxis": {"title": x_col}, "yaxis": {"title": y_col}, "showlegend": True, "height": 500}
        return {"data": fig_data, "layout": layout}, info_text

    fig_propio, text_propio = create_figure(centroids_history, assignments_history, iteration, "Propio")
    fig_sklearn, text_sklearn = create_figure(sk_centroids_history, sk_assignments_history, sk_iteration, "Scikit-learn")

    return fig_propio, fig_sklearn, html.Ul([html.Li(info) for info in text_propio]), html.Ul([html.Li(info) for info in text_sklearn])

# ===================== CALLBACK DE MÉTRICAS (PESTAÑA 2) =====================
@app.callback(
    Output("km-metrics-content", "children"),
    Input("km-target-col-selector", "value"),
    Input("km-feature-cols-selector", "value"),
    Input("km-num-clusters", "value")
)
def update_metrics_tab(target_c, feature_c, k_val):
    global df, metrics_data
    if df.empty or not target_c or not feature_c: return html.Div("Configure dataset completo.")

    X_local = df[feature_c].values
    y_local = df[target_c].values
    classes_local = df[target_c].unique()
    
    cm_simple, mat_simple = simple_validation(X_local, y_local, k_val, classes_local)
    cm_strat, mat_strat = kfold_validation(X_local, y_local, k_val, classes_local, n_splits=5, stratified=True)
    cm_kfold, mat_kfold = kfold_validation(X_local, y_local, k_val, classes_local, n_splits=5, stratified=False)
    cm_loo, mat_loo = leave_one_out_validation(X_local, y_local, k_val, classes_local)
    cm_boot, mat_boot = bootstrap_validation(X_local, y_local, k_val, classes_local, n_iterations=10)

    metrics_data = {
        "Validación Simple (70/30)": {"Confusion": cm_simple, "Metrics": classification_metrics(cm_simple), "Matrix": mat_simple},
        "K-Fold Estratificado (k=5)": {"Confusion": cm_strat, "Metrics": classification_metrics(cm_strat), "Matrix": mat_strat},
        "K-Fold Normal (k=5)": {"Confusion": cm_kfold, "Metrics": classification_metrics(cm_kfold), "Matrix": mat_kfold},
        "Leave-One-Out (LOO)": {"Confusion": cm_loo, "Metrics": classification_metrics(cm_loo), "Matrix": mat_loo},
        "Bootstrap (10 iters)": {"Confusion": cm_boot, "Metrics": classification_metrics(cm_boot), "Matrix": mat_boot}
    }

    def generate_metrics_table(local_classes):
        rows = []
        for method in metrics_data:
            for cls in local_classes:
                m_conf = metrics_data[method]["Confusion"].get(cls, {"TP":0, "FP":0, "TN":0, "FN":0})
                m_meta = metrics_data[method]["Metrics"].get(cls, {"Precision":0, "Recall":0, "Accuracy":0, "F1":0})
                rows.append(html.Tr([
                    html.Td(method), html.Td(str(cls)), html.Td(f"{m_conf['TP']:.1f}"), html.Td(f"{m_conf['FP']:.1f}"),
                    html.Td(f"{m_conf['TN']:.1f}"), html.Td(f"{m_conf['FN']:.1f}"), html.Td(f"{m_meta['Precision']:.3f}"),
                    html.Td(f"{m_meta['Recall']:.3f}"), html.Td(f"{m_meta['Accuracy']:.3f}"), html.Td(f"{m_meta['F1']:.3f}")
                ]))
        return rows

    graphs = []
    for method in metrics_data:
        graphs.append(html.Div([
            html.H5(method, style={"textAlign": "center"}),
            dcc.Graph(style={'height': '300px'}, figure={
                "data": [{"z": metrics_data[method]["Matrix"].values, "x": [str(c) for c in metrics_data[method]["Matrix"].columns], "y": [str(i) for i in metrics_data[method]["Matrix"].index], "type": "heatmap", "colorscale": "Blues", "texttemplate": "%{z:.1f}"}],
                "layout": {"title": f"{method}", "margin": {"l": 40, "r": 40, "t": 40, "b": 40}, "height": 300}
            })
        ], style={"width": "30%", "minWidth": "300px", "display": "inline-block", "padding": "10px"}))

    return html.Div([
        html.H3("Evaluación Exhaustiva de Modelos"),
        html.H4("Tabla Detallada de Métricas"),
        html.Div([html.Table([
            html.Thead([html.Tr([html.Th("Método"), html.Th("Clase"), html.Th("TP"), html.Th("FP"), html.Th("TN"), html.Th("FN"), html.Th("Precision"), html.Th("Recall"), html.Th("Accuracy"), html.Th("F1")])]),
            html.Tbody(generate_metrics_table(classes_local))
        ], style={"border": "1px solid #ddd", "width": "100%", "textAlign": "center"})], style={"overflowX": "auto"}),
        html.H4("Matrices de Confusión", style={"marginTop": "30px"}),
        html.Div(graphs, style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"})
    ])