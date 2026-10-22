"""
Guía de Práctica de Laboratorio 06
Desarrollo, implementación y versionamiento de redes recurrentes (RNN y LSTM)

Curso : Inteligencia Artificial / Aprendizaje Profundo
Autor : Luidman Zanes

El script parte del código base de la guía (generación de datos, SimpleRNNModel y
bucle de entrenamiento) y lo organiza en funciones para poder repetir cada
experimento bajo condiciones idénticas. Se ejecuta en este orden:

    Actividad 1 · Inspección de dimensiones (tensores 3D)
    Sección 3   · Entrenamiento del modelo base (RNN simple, SEQ_LEN = 10)
    Actividad 2 · Modificación de hiperparámetros (hidden_size 16 -> 64)
    Actividad 3 · Análisis de ventanas temporales (SEQ_LEN 10 -> 50)
    Reto        · LSTM frente a RNN simple bajo condiciones idénticas (SEQ_LEN = 50)

Uso:
    python practica_rnn.py            # ejecuta todo y guarda las gráficas en resultados/
    python practica_rnn.py --mostrar  # además abre las ventanas de matplotlib
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. Configuración general
# ---------------------------------------------------------------------------
N_PUNTOS = 600              # puntos de la onda senoidal (igual que el código base)
EPOCHS = 50                 # épocas de entrenamiento (igual que el código base)
LR = 0.01                   # tasa de aprendizaje de Adam (igual que el código base)
HIDDEN_BASE = 16            # neuronas del estado oculto del modelo base
PROP_TRAIN = 0.8            # 80 % de las ventanas para entrenar y 20 % para evaluar
SEMILLAS = [0, 1, 2, 3, 4]  # repeticiones de cada experimento (media ± desviación)
DIR_RESULTADOS = Path(__file__).resolve().parent / "resultados"

# Paleta categórica apta para daltonismo; el color sigue al modelo en todas las gráficas
COLOR_RNN = "#2a78d6"       # azul
COLOR_LSTM = "#eb6834"      # naranja
COLOR_VARIANTE = "#1baf7a"  # aqua: segunda configuración de un mismo modelo
COLOR_REAL = "#52514e"      # gris: serie real


# ---------------------------------------------------------------------------
# 1. Generación de datos sintéticos (serie temporal de una onda senoidal)
# ---------------------------------------------------------------------------
def generar_datos(n_puntos=500, seq_len=10):
    """Corta la onda sin(t) en ventanas deslizantes de `seq_len` pasos.

    Cada ventana X[i] = serie[i : i+seq_len] tiene como objetivo el valor
    inmediatamente siguiente y[i] = serie[i+seq_len].
    """
    t = np.linspace(0, 50, n_puntos)
    serie = np.sin(t)

    X, y = [], []
    for i in range(len(serie) - seq_len):
        X.append(serie[i:i+seq_len])
        y.append(serie[i+seq_len])

    # Convertir a tensores de PyTorch con dimensiones tridimensionales (Batch, Seq_Len, Features).
    # np.array() evita el UserWarning de PyTorch al convertir una lista de arreglos.
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # [N, seq_len, 1]
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)  # [N, 1]
    return X, y


def dividir_cronologicamente(X, y, prop_train=PROP_TRAIN):
    """Separa entrenamiento y prueba respetando el orden temporal (sin mezclar).

    El modelo nunca ve durante el entrenamiento los valores objetivo del tramo
    final de la serie, así que la evaluación mide predicción sobre datos nuevos.
    """
    n_train = int(len(X) * prop_train)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def tiempos_objetivo(seq_len):
    """Instante t de cada valor objetivo y[i] (sirve como eje x de las predicciones)."""
    return np.linspace(0, 50, N_PUNTOS)[seq_len:]


# ---------------------------------------------------------------------------
# 2. Definición de la arquitectura RNN simple
# ---------------------------------------------------------------------------
class SimpleRNNModel(nn.Module):
    """RNN de una capa + capa lineal, paradigma Many-to-One.

    h_t = tanh(W_ih·x_t + b_ih + W_hh·h_(t-1) + b_hh)    (mismos pesos en cada paso)
    ŷ   = W_fc·h_T + b_fc                                (solo el último estado)
    """

    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size] -> out: [batch_size, seq_len, hidden_size]
        out, _ = self.rnn(x)
        # Extraemos exclusivamente el último estado temporal (Many-to-One): [batch_size, hidden_size]
        out_final = out[:, -1, :]
        pred = self.fc(out_final)  # [batch_size, output_size]
        return pred


# ---------------------------------------------------------------------------
# 2b. Reto avanzado: arquitectura LSTM (mismo esquema, nn.RNN -> nn.LSTM)
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """LSTM de una capa + capa lineal, paradigma Many-to-One.

    f_t = σ(W_f·[h_(t-1), x_t] + b_f)      compuerta de olvido
    i_t = σ(W_i·[h_(t-1), x_t] + b_i)      compuerta de entrada
    o_t = σ(W_o·[h_(t-1), x_t] + b_o)      compuerta de salida
    c_t = f_t ⊙ c_(t-1) + i_t ⊙ tanh(W_c·[h_(t-1), x_t] + b_c)
    h_t = o_t ⊙ tanh(c_t)
    """

    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)  # LSTM retorna estado oculto y estado de celda
        out_final = out[:, -1, :]
        pred = self.fc(out_final)
        return pred


# ---------------------------------------------------------------------------
# 3. Entrenamiento, evaluación y medición de la memoria del modelo
# ---------------------------------------------------------------------------
def fijar_semilla(semilla):
    """Hace reproducible la inicialización de pesos."""
    torch.manual_seed(semilla)
    np.random.seed(semilla)


def contar_parametros(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def norma_gradiente(model):
    """Norma L2 de todos los gradientes del modelo (después de loss.backward())."""
    return torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)).item()


def entrenar(model, X, y, epochs=EPOCHS, lr=LR, verbose=False):
    """Bucle de entrenamiento del código base (MSE + Adam, batch completo).

    Además de entrenar, registra la pérdida y la norma del gradiente en cada
    época para poder analizar la estabilidad del entrenamiento.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    perdidas, normas = [], []

    inicio = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        normas.append(norma_gradiente(model))
        optimizer.step()
        perdidas.append(loss.item())

        if verbose and (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    duracion = time.perf_counter() - inicio

    return {"perdida": np.array(perdidas), "norma_grad": np.array(normas),
            "ms_por_epoca": 1000 * duracion / epochs}


@torch.no_grad()
def evaluar(model, X, y):
    """Métricas de predicción sobre el conjunto de prueba."""
    model.eval()
    pred = model(X)
    mse = nn.functional.mse_loss(pred, y).item()
    mae = (pred - y).abs().mean().item()
    r2 = 1 - ((pred - y) ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()
    return {"mse": mse, "rmse": mse ** 0.5, "mae": mae, "r2": r2, "pred": pred.squeeze(-1).numpy()}


def sensibilidad_temporal(model, X, y):
    """Promedio de |∂L/∂x_t| para cada paso t de la ventana, normalizado a 1.

    Mide cuánta señal de gradiente llega a cada instante de la entrada. En BPTT
    esa señal se multiplica por la Jacobiana recurrente una vez por paso, de modo
    que un decaimiento exponencial hacia el pasado es la huella del gradiente
    desvaneciente ("amnesia matemática").
    """
    model.eval()
    X = X.clone().requires_grad_(True)
    loss = nn.functional.mse_loss(model(X), y)
    loss.backward()
    model.zero_grad()
    sens = X.grad.abs().mean(dim=(0, 2)).numpy()
    return sens / sens.max()


def analizar_memoria(sens):
    """Ajusta s_k ≈ C·ρ^k, con k = pasos hacia atrás desde la predicción.

    ρ es el factor por el que se reduce la señal en cada paso hacia el pasado y
    la vida media es el número de pasos que tarda en caer a la mitad.
    """
    s = sens[::-1]  # s[0] = paso más reciente, s[-1] = paso más antiguo
    k = np.arange(len(s))
    pendiente, _ = np.polyfit(k, np.log(s + 1e-12), 1)
    rho = float(np.exp(pendiente))
    return {
        "rho": rho,
        "vida_media_pasos": float(np.log(0.5) / np.log(rho)) if rho < 1 else float("inf"),
        "sensibilidad_paso_mas_antiguo": float(s[-1]),
        "fraccion_mas_alla_de_10_pasos": float(s[10:].sum() / s.sum()),
    }


@torch.no_grad()
def compuerta_olvido_media(model, X):
    """Valor medio de la compuerta de olvido f_t de una LSTMModel sobre las ventanas X.

    PyTorch guarda las compuertas en el orden [entrada, olvido, celda, salida].
    f_t ≈ 1 conserva la memoria de la celda y f_t ≈ 0 la borra.
    """
    lstm = model.lstm
    out, _ = lstm(X)
    h_prev = torch.cat([torch.zeros_like(out[:, :1]), out[:, :-1]], dim=1)  # h_(t-1), con h_(-1) = 0
    compuertas = X @ lstm.weight_ih_l0.T + lstm.bias_ih_l0 + h_prev @ lstm.weight_hh_l0.T + lstm.bias_hh_l0
    _, f, _, _ = compuertas.chunk(4, dim=-1)
    return torch.sigmoid(f).mean().item()


def hidden_equivalente(parametros_objetivo):
    """hidden_size con el que una SimpleRNNModel tiene casi los mismos parámetros que otro modelo."""
    return min(range(1, 129), key=lambda H: abs(contar_parametros(SimpleRNNModel(1, H, 1)) - parametros_objetivo))


_CACHE = {}


def repetir_experimento(clase_modelo, seq_len, hidden_size=HIDDEN_BASE):
    """Entrena un modelo con cada semilla de SEMILLAS bajo condiciones idénticas.

    Todas las corridas comparten datos, partición, épocas, tasa de aprendizaje y
    optimizador; solo cambia la semilla con la que se inicializan los pesos.
    """
    clave = (clase_modelo.__name__, seq_len, hidden_size)
    if clave in _CACHE:
        return _CACHE[clave]

    X, y = generar_datos(n_puntos=N_PUNTOS, seq_len=seq_len)
    X_train, y_train, X_test, y_test = dividir_cronologicamente(X, y)

    corridas = []
    for semilla in SEMILLAS:
        fijar_semilla(semilla)
        model = clase_modelo(input_size=1, hidden_size=hidden_size, output_size=1)
        historia = entrenar(model, X_train, y_train)
        corridas.append({
            "modelo": model,
            **historia,
            "test": evaluar(model, X_test, y_test),
            "sensibilidad": sensibilidad_temporal(model, X_test, y_test),
        })

    perdidas = np.stack([c["perdida"] for c in corridas])
    mse_test = np.array([c["test"]["mse"] for c in corridas])
    sensibilidad = np.mean([c["sensibilidad"] for c in corridas], axis=0)
    resultado = {
        "parametros": contar_parametros(corridas[0]["modelo"]),
        "perdidas": perdidas,
        "normas_grad": np.stack([c["norma_grad"] for c in corridas]),
        "sensibilidad": sensibilidad,
        "corridas": corridas,
        "X_test": X_test,
        "y_test": y_test.squeeze(-1).numpy(),
        "t_test": tiempos_objetivo(seq_len)[len(X_train):],
        "resumen": {
            "modelo": clase_modelo.__name__,
            "seq_len": seq_len,
            "hidden_size": hidden_size,
            "perdida_train_final_media": float(perdidas[:, -1].mean()),
            "perdida_train_final_std": float(perdidas[:, -1].std()),
            "mse_test_media": float(mse_test.mean()),
            "mse_test_std": float(mse_test.std()),
            "mse_test_por_semilla": [float(v) for v in mse_test],
            "rmse_test_media": float(np.mean([c["test"]["rmse"] for c in corridas])),
            "r2_test_media": float(np.mean([c["test"]["r2"] for c in corridas])),
            "oscilaciones_media": float(np.mean([(np.diff(p) > 0).sum() for p in perdidas])),
            "norma_grad_max_media": float(np.mean([c["norma_grad"].max() for c in corridas])),
            "ms_por_epoca_media": float(np.mean([c["ms_por_epoca"] for c in corridas])),
            **analizar_memoria(sensibilidad),
        },
    }
    resultado["resumen"]["parametros"] = resultado["parametros"]
    _CACHE[clave] = resultado
    return resultado


# ---------------------------------------------------------------------------
# Utilidades de salida (consola y gráficas)
# ---------------------------------------------------------------------------
def titulo(texto):
    print("\n" + "=" * 96)
    print(texto)
    print("=" * 96)


def _pm(media, std):
    return f"{media:.6f} ± {std:.6f}"


def imprimir_tabla(filas):
    """Tabla comparativa: media ± desviación estándar sobre las semillas."""
    print(f"\nResultados promediados sobre {len(SEMILLAS)} semillas (media ± desviación estándar):")
    print(f"{'Experimento':<24}{'Parám.':>8}{'Loss train final':>24}{'MSE test':>24}"
          f"{'Oscil.':>8}{'|∇| máx':>9}{'ms/ép.':>8}")
    for nombre, r in filas:
        s = r["resumen"]
        print(f"{nombre:<24}{r['parametros']:>8,}"
              f"{_pm(s['perdida_train_final_media'], s['perdida_train_final_std']):>24}"
              f"{_pm(s['mse_test_media'], s['mse_test_std']):>24}"
              f"{s['oscilaciones_media']:>8.1f}{s['norma_grad_max_media']:>9.2f}{s['ms_por_epoca_media']:>8.1f}")
    print("Oscil. = épocas en las que la pérdida subió respecto a la anterior (de 49 transiciones).")


def configurar_estilo():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "axes.titlecolor": "#0b0b0b",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.8,
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "font.size": 10,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def guardar(fig, nombre):
    ruta = DIR_RESULTADOS / nombre
    fig.savefig(ruta)
    print(f"Gráfica guardada: {ruta.relative_to(DIR_RESULTADOS.parent)}")


def graficar_curvas(ax, curvas, color, etiqueta):
    """Media de las corridas (línea) y rango mínimo-máximo entre semillas (banda)."""
    epocas = np.arange(1, curvas.shape[1] + 1)
    ax.fill_between(epocas, curvas.min(axis=0), curvas.max(axis=0), color=color, alpha=0.15, linewidth=0)
    ax.plot(epocas, curvas.mean(axis=0), color=color, label=etiqueta)


def graficar_sensibilidad(ax, r, color, etiqueta):
    """Sensibilidad por paso hacia atrás (k = 0 es el paso más reciente) en escala log."""
    s = r["sensibilidad"][::-1]
    ax.plot(np.arange(len(s)), s, "o-", color=color, markersize=4, linewidth=1.5, label=etiqueta)


# ---------------------------------------------------------------------------
# Actividad 1 · Inspección de dimensiones (tensores 3D)
# ---------------------------------------------------------------------------
def actividad_1_dimensiones():
    titulo("ACTIVIDAD 1 · Inspección de dimensiones (tensores 3D)")
    seq_len = 10
    X, y = generar_datos(n_puntos=N_PUNTOS, seq_len=seq_len)
    X_train, _, X_test, _ = dividir_cronologicamente(X, y)

    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    N, L, F = X.shape
    print(f"\nX.shape = {tuple(X.shape)} -> [Batch Size, Sequence Length, Features]")
    print(f"  eje 0 · Batch Size      = {N:>3}  ventanas deslizantes extraídas de la serie ({N_PUNTOS} - {L} = {N})")
    print(f"  eje 1 · Sequence Length = {L:>3}  pasos temporales consecutivos que la red lee antes de predecir")
    print(f"  eje 2 · Features        = {F:>3}  variables por paso: un único valor escalar sin(t)")
    print(f"y.shape = {tuple(y.shape)} -> [Batch Size, Output]: el valor siguiente a cada ventana")
    print(f"Partición cronológica 80/20: X_train {tuple(X_train.shape)} · X_test {tuple(X_test.shape)}")

    # Recorrido de las dimensiones dentro del modelo (Many-to-One)
    fijar_semilla(SEMILLAS[0])
    model = SimpleRNNModel(input_size=1, hidden_size=HIDDEN_BASE, output_size=1)
    with torch.no_grad():
        out, h_n = model.rnn(X)
        out_final = out[:, -1, :]
        pred = model.fc(out_final)
    print("\nFlujo de dimensiones dentro de SimpleRNNModel:")
    print(f"  x                      {str(tuple(X.shape)):<15} entrada")
    print(f"  out, h_n = rnn(x)      {str(tuple(out.shape)):<15} out: h_t de los {L} pasos (hidden_size = {HIDDEN_BASE})")
    print(f"                         {str(tuple(h_n.shape)):<15} h_n: último estado oculto [num_layers, Batch, Hidden]")
    print(f"  out[:, -1, :]          {str(tuple(out_final.shape)):<15} solo el último paso temporal (Many-to-One)")
    print(f"  pred = fc(out_final)   {str(tuple(pred.shape)):<15} una predicción por ventana")
    print(f"  ¿out[:, -1, :] es igual a h_n[0]? {torch.allclose(out_final, h_n[0])}")

    # Figura 01: la serie, la partición y una ventana de ejemplo
    t = np.linspace(0, 50, N_PUNTOS)
    serie = np.sin(t)
    inicio_test = t[len(X_train) + seq_len]
    i = 120  # ventana de ejemplo
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    ax1.plot(t, serie, color=COLOR_REAL, linewidth=1.5, label="sin(t)")
    ax1.axvspan(inicio_test, t[-1], color=COLOR_REAL, alpha=0.08, linewidth=0, label="tramo de prueba (20 %)")
    ax1.axvspan(t[i], t[i + seq_len], color=COLOR_RNN, alpha=0.25, linewidth=0, label="ventana de ejemplo")
    ax1.set(title="Serie temporal y partición cronológica", xlabel="t", ylabel="sin(t)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)

    ax2.plot(t[i - 3:i + seq_len + 4], serie[i - 3:i + seq_len + 4], color=COLOR_REAL, linewidth=1.2)
    ax2.plot(t[i:i + seq_len], serie[i:i + seq_len], "o", color=COLOR_RNN, markersize=7,
             label=f"X[{i}]: {seq_len} pasos de entrada")
    ax2.plot(t[i + seq_len], serie[i + seq_len], "D", color=COLOR_VARIANTE, markersize=9,
             label=f"y[{i}]: valor a predecir")
    ax2.set(title=f"Una muestra: tensor [1, {seq_len}, 1] -> [1, 1]", xlabel="t")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    guardar(fig, "01_serie_y_ventanas.png")

    return {
        "X": list(X.shape), "y": list(y.shape),
        "X_train": list(X_train.shape), "X_test": list(X_test.shape),
        "rnn_out": list(out.shape), "h_n": list(h_n.shape),
        "out_final": list(out_final.shape), "pred": list(pred.shape),
    }


# ---------------------------------------------------------------------------
# Sección 3 · Modelo base (RNN simple)
# ---------------------------------------------------------------------------
def modelo_base():
    titulo("SECCIÓN 3 · Modelo base: RNN simple (hidden_size = 16, SEQ_LEN = 10, 50 épocas)")
    SEQ_LEN = 10
    X, y = generar_datos(n_puntos=N_PUNTOS, seq_len=SEQ_LEN)
    X_train, y_train, X_test, y_test = dividir_cronologicamente(X, y)

    fijar_semilla(SEMILLAS[0])
    model = SimpleRNNModel(input_size=1, hidden_size=HIDDEN_BASE, output_size=1)
    historia = entrenar(model, X_train, y_train, verbose=True)
    print("¡Entrenamiento de la RNN Simple finalizado con éxito!")

    test = evaluar(model, X_test, y_test)
    print(f"Predicción sobre el tramo de prueba: MSE = {test['mse']:.6f} · RMSE = {test['rmse']:.4f}"
          f" · MAE = {test['mae']:.4f} · R² = {test['r2']:.4f}")

    t_test = tiempos_objetivo(SEQ_LEN)[len(X_train):]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    ax1.plot(np.arange(1, EPOCHS + 1), historia["perdida"], color=COLOR_RNN)
    ax1.set(title="Curva de pérdida (entrenamiento)", xlabel="Época", ylabel="MSE (escala log)", yscale="log")
    ax2.plot(t_test, y_test.squeeze(-1).numpy(), color=COLOR_REAL, linewidth=3, alpha=0.5, label="Serie real")
    ax2.plot(t_test, test["pred"], "--", color=COLOR_RNN, label=f"Predicción RNN (MSE = {test['mse']:.5f})")
    ax2.set(title="Predicción sobre el tramo de prueba", xlabel="t", ylabel="sin(t)")
    ax2.legend(loc="lower left")
    fig.suptitle("Modelo base · RNN simple (hidden_size = 16, SEQ_LEN = 10)", x=0.01, ha="left")
    fig.tight_layout()
    guardar(fig, "02_rnn_base.png")

    return {"perdida_final": float(historia["perdida"][-1]),
            **{k: v for k, v in test.items() if k != "pred"}}


# ---------------------------------------------------------------------------
# Actividad 2 · Modificación de hiperparámetros (hidden_size)
# ---------------------------------------------------------------------------
def actividad_2_hiperparametros():
    titulo("ACTIVIDAD 2 · Modificación de hiperparámetros (hidden_size 16 -> 64)")
    I, O = 1, 1
    conteos = {}
    for H in (16, 64):
        model = SimpleRNNModel(input_size=I, hidden_size=H, output_size=O)
        total = sum(p.numel() for p in model.parameters())
        formula = H * I + H * H + 2 * H + O * H + O
        print(f"\nSimpleRNNModel(hidden_size={H}): {total:,} parámetros entrenables")
        for nombre, p in model.named_parameters():
            print(f"  {nombre:<18}{str(tuple(p.shape)):<10}{p.numel():>7,}")
        print(f"  Fórmula H·I + H·H + 2H + O·H + O = {formula:,} "
              f"({'coincide' if formula == total else 'NO coincide'} con sum(p.numel()))")
        conteos[f"SimpleRNNModel_H{H}"] = total

    factor = conteos["SimpleRNNModel_H64"] / conteos["SimpleRNNModel_H16"]
    print(f"\nPasar de 16 a 64 neuronas multiplica los parámetros por {factor:.2f}: "
          f"el término recurrente H·H crece de {16 * 16} a {64 * 64} pesos (x16).")

    # La LSTM repite el bloque recurrente cuatro veces (una por compuerta: i, f, g, o)
    for H in (16, 64):
        total = contar_parametros(LSTMModel(input_size=I, hidden_size=H, output_size=O))
        formula = 4 * (H * I + H * H + 2 * H) + O * H + O
        print(f"LSTMModel(hidden_size={H}): {total:,} parámetros = 4·(H·I + H·H + 2H) + O·H + O = {formula:,}")
        conteos[f"LSTMModel_H{H}"] = total

    # Impacto en el entrenamiento: mismo problema (SEQ_LEN = 10), dos tamaños de estado oculto
    r16 = repetir_experimento(SimpleRNNModel, seq_len=10, hidden_size=16)
    r64 = repetir_experimento(SimpleRNNModel, seq_len=10, hidden_size=64)
    imprimir_tabla([("RNN · hidden_size=16", r16), ("RNN · hidden_size=64", r64)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    etiquetas = ["hidden_size = 16", "hidden_size = 64"]
    valores = [conteos["SimpleRNNModel_H16"], conteos["SimpleRNNModel_H64"]]
    barras = ax1.bar(etiquetas, valores, color=[COLOR_RNN, COLOR_VARIANTE], width=0.5)
    ax1.bar_label(barras, labels=[f"{v:,}" for v in valores], padding=3, color="#0b0b0b")
    ax1.set(title="Parámetros entrenables de la RNN simple", ylabel="Parámetros")
    ax1.grid(axis="x", visible=False)
    graficar_curvas(ax2, r16["perdidas"], COLOR_RNN, "hidden_size = 16")
    graficar_curvas(ax2, r64["perdidas"], COLOR_VARIANTE, "hidden_size = 64")
    ax2.set(title="Pérdida de entrenamiento (media y rango de 5 semillas)", xlabel="Época",
            ylabel="MSE (escala log)", yscale="log")
    ax2.legend()
    fig.tight_layout()
    guardar(fig, "03_hidden_size.png")

    return {"parametros": conteos, "factor_16_a_64": factor,
            "hidden_16": r16["resumen"], "hidden_64": r64["resumen"]}


# ---------------------------------------------------------------------------
# Actividad 3 · Análisis de ventanas temporales (SEQ_LEN)
# ---------------------------------------------------------------------------
def actividad_3_ventanas():
    titulo("ACTIVIDAD 3 · Análisis de ventanas temporales (SEQ_LEN 10 -> 50)")
    r10 = repetir_experimento(SimpleRNNModel, seq_len=10)
    r50 = repetir_experimento(SimpleRNNModel, seq_len=50)
    imprimir_tabla([("RNN · SEQ_LEN=10", r10), ("RNN · SEQ_LEN=50", r50)])

    print("\nMemoria efectiva de la RNN (sensibilidad |∂L/∂x_t| promedio, normalizada al paso más sensible):")
    for nombre, r in (("SEQ_LEN=10", r10), ("SEQ_LEN=50", r50)):
        s = r["resumen"]
        print(f"  {nombre}: ρ = {s['rho']:.3f} por paso -> vida media ≈ {s['vida_media_pasos']:.1f} pasos; "
              f"el paso más antiguo conserva {100 * s['sensibilidad_paso_mas_antiguo']:.3f} % de la señal")
    frac = r50["resumen"]["fraccion_mas_alla_de_10_pasos"]
    print(f"  Con SEQ_LEN=50, los 10 pasos más recientes concentran {100 * (1 - frac):.1f} % de la sensibilidad "
          f"total y los 40 más antiguos (80 % de la ventana) solo {100 * frac:.1f} %.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    graficar_curvas(ax1, r10["perdidas"], COLOR_VARIANTE, "SEQ_LEN = 10")
    graficar_curvas(ax1, r50["perdidas"], COLOR_RNN, "SEQ_LEN = 50")
    ax1.set(title="Pérdida de la RNN simple (media y rango de 5 semillas)", xlabel="Época",
            ylabel="MSE (escala log)", yscale="log")
    ax1.legend()
    graficar_sensibilidad(ax2, r10, COLOR_VARIANTE, "SEQ_LEN = 10")
    graficar_sensibilidad(ax2, r50, COLOR_RNN, "SEQ_LEN = 50")
    rho = r50["resumen"]["rho"]
    k = np.arange(50)
    ax2.plot(k, r50["sensibilidad"][::-1][0] * rho ** k, ":", color=COLOR_REAL, linewidth=1.5,
             label=f"ajuste ρ^k, ρ = {rho:.3f}")
    ax2.set(title="Amnesia: |∂L/∂x_t| según la antigüedad del paso", xlabel="Pasos hacia atrás desde la predicción (k)",
            ylabel="Sensibilidad relativa (log)", yscale="log")
    ax2.legend()
    fig.tight_layout()
    guardar(fig, "04_ventanas_rnn.png")

    return {"seq_len_10": r10["resumen"], "seq_len_50": r50["resumen"]}


# ---------------------------------------------------------------------------
# Reto avanzado · LSTM frente a RNN simple (SEQ_LEN = 50)
# ---------------------------------------------------------------------------
def reto_lstm():
    titulo("RETO AVANZADO · LSTM frente a RNN simple con secuencias largas (SEQ_LEN = 50)")
    L = 50
    rnn = repetir_experimento(SimpleRNNModel, seq_len=L)
    lstm = repetir_experimento(LSTMModel, seq_len=L)
    # Control de capacidad: la LSTM tiene ~4 veces más parámetros que la RNN con el mismo hidden_size
    h_eq = hidden_equivalente(lstm["parametros"])
    rnn_eq = repetir_experimento(SimpleRNNModel, seq_len=L, hidden_size=h_eq)
    modelos = [
        ("RNN · H=16", rnn, COLOR_RNN, SimpleRNNModel, HIDDEN_BASE),
        ("LSTM · H=16", lstm, COLOR_LSTM, LSTMModel, HIDDEN_BASE),
        (f"RNN · H={h_eq} (control)", rnn_eq, COLOR_VARIANTE, SimpleRNNModel, h_eq),
    ]
    imprimir_tabla([(nombre, r) for nombre, r, *_ in modelos])

    # Compuerta de olvido de la LSTM al inicializar y después de entrenar
    f_inicial, f_final = [], []
    for semilla, corrida in zip(SEMILLAS, lstm["corridas"]):
        fijar_semilla(semilla)
        f_inicial.append(compuerta_olvido_media(LSTMModel(1, HIDDEN_BASE, 1), lstm["X_test"]))
        f_final.append(compuerta_olvido_media(corrida["modelo"], lstm["X_test"]))
    f_inicial, f_final = float(np.mean(f_inicial)), float(np.mean(f_final))

    print("\nMemoria efectiva con SEQ_LEN=50 (sensibilidad |∂L/∂x_t|):")
    for nombre, r, *_ in modelos:
        s = r["resumen"]
        print(f"  {nombre:<20} ρ = {s['rho']:.3f} · vida media ≈ {s['vida_media_pasos']:.1f} pasos · "
              f"paso más antiguo = {s['sensibilidad_paso_mas_antiguo']:.2e}")
    print(f"  Compuerta de olvido media de la LSTM: {f_inicial:.3f} al inicializar -> {f_final:.3f} entrenada")

    # Respuesta a la pregunta de análisis de la guía
    s_rnn, s_lstm, s_eq = rnn["resumen"], lstm["resumen"], rnn_eq["resumen"]
    mejor = "LSTM" if s_lstm["mse_test_media"] < s_rnn["mse_test_media"] else "RNN simple"
    print("\nRespuesta a la pregunta de análisis:")
    print(f"  Con SEQ_LEN=50 y condiciones idénticas se ajusta mejor la {mejor}: MSE de prueba "
          f"{s_lstm['mse_test_media']:.6f} (LSTM) frente a {s_rnn['mse_test_media']:.6f} (RNN), "
          f"{max(s_rnn['mse_test_media'], s_lstm['mse_test_media']) / min(s_rnn['mse_test_media'], s_lstm['mse_test_media']):.1f}x de diferencia.")
    print(f"  Convergencia: {s_lstm['oscilaciones_media']:.1f} oscilaciones de la pérdida en la LSTM frente a "
          f"{s_rnn['oscilaciones_media']:.1f} en la RNN; desviación entre semillas {s_lstm['mse_test_std']:.6f} "
          f"frente a {s_rnn['mse_test_std']:.6f}.")
    print(f"  Control: una RNN con H={h_eq} ({rnn_eq['parametros']:,} parámetros, la LSTM tiene {lstm['parametros']:,}) "
          f"logra MSE {s_eq['mse_test_media']:.6f}, pero con {s_eq['oscilaciones_media']:.1f} oscilaciones y "
          f"|∇| máx {s_eq['norma_grad_max_media']:.2f} (LSTM: {s_lstm['norma_grad_max_media']:.2f}).")

    # Figura 05: convergencia de la pérdida y norma del gradiente
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    for nombre, r, color, *_ in modelos:
        graficar_curvas(ax1, r["perdidas"], color, nombre)
        graficar_curvas(ax2, r["normas_grad"], color, nombre)
    ax1.set(title="Pérdida de entrenamiento (media y rango de 5 semillas)", xlabel="Época",
            ylabel="MSE (escala log)", yscale="log")
    ax1.legend()
    ax2.set(title="Norma del gradiente por época", xlabel="Época", ylabel="‖∇L‖ (escala log)", yscale="log")
    ax2.legend()
    fig.suptitle("RNN simple vs LSTM · SEQ_LEN = 50, condiciones idénticas", x=0.01, ha="left")
    fig.tight_layout()
    guardar(fig, "05_rnn_vs_lstm_perdida.png")

    # Figura 06: predicción sobre el tramo de prueba (semilla 0)
    t_test, y_test = rnn["t_test"], rnn["y_test"]
    pred_rnn, pred_lstm = rnn["corridas"][0]["test"]["pred"], lstm["corridas"][0]["test"]["pred"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    ax1.plot(t_test, y_test, color=COLOR_REAL, linewidth=3, alpha=0.5, label="Serie real")
    ax1.plot(t_test, pred_rnn, "--", color=COLOR_RNN, label="RNN · H=16")
    ax1.plot(t_test, pred_lstm, "--", color=COLOR_LSTM, label="LSTM · H=16")
    ax1.set(title="Predicción sobre el tramo de prueba", xlabel="t", ylabel="sin(t)")
    ax1.legend(loc="upper left")
    ax2.plot(t_test, np.abs(pred_rnn - y_test), color=COLOR_RNN, label="RNN · H=16")
    ax2.plot(t_test, np.abs(pred_lstm - y_test), color=COLOR_LSTM, label="LSTM · H=16")
    ax2.set(title="Error absoluto |ŷ - y|", xlabel="t", ylabel="Error absoluto")
    ax2.legend()
    fig.tight_layout()
    guardar(fig, "06_rnn_vs_lstm_prediccion.png")

    # Figura 07: sensibilidad temporal de ambos modelos
    fig, ax = plt.subplots(figsize=(8, 3.8))
    graficar_sensibilidad(ax, rnn, COLOR_RNN, "RNN · H=16")
    graficar_sensibilidad(ax, lstm, COLOR_LSTM, f"LSTM · H=16 (compuerta de olvido media = {f_final:.2f})")
    ax.set(title="Sensibilidad |∂L/∂x_t| con SEQ_LEN = 50", xlabel="Pasos hacia atrás desde la predicción (k)",
           ylabel="Sensibilidad relativa (log)", yscale="log")
    ax.legend()
    fig.tight_layout()
    guardar(fig, "07_sensibilidad_rnn_vs_lstm.png")

    # Barrido de longitudes de secuencia: impacto de alargar la ventana
    longitudes = [10, 25, 50, 100]
    barrido = {nombre: [repetir_experimento(clase, seq_len=l, hidden_size=h) for l in longitudes]
               for nombre, _, _, clase, h in modelos}
    print("\nBarrido de SEQ_LEN · MSE de prueba (media ± desviación de 5 semillas) y ms por época:")
    print(f"{'SEQ_LEN':<9}" + "".join(f"{nombre:>34}" for nombre in barrido))
    for i, l in enumerate(longitudes):
        celdas = [f"{_pm(rs[i]['resumen']['mse_test_media'], rs[i]['resumen']['mse_test_std'])}"
                  f"  {rs[i]['resumen']['ms_por_epoca_media']:4.1f} ms" for rs in barrido.values()]
        print(f"{l:<9}" + "".join(f"{c:>34}" for c in celdas))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    for j, ((nombre, _, color, *_), rs) in enumerate(zip(modelos, barrido.values())):
        ax1.plot(longitudes, [r["resumen"]["mse_test_media"] for r in rs], "o-", color=color, markersize=6,
                 label=nombre)
        for l, r in zip(longitudes, rs):  # cada semilla, con un pequeño desplazamiento horizontal por modelo
            ax1.scatter([l + 2 * (j - 1)] * len(SEMILLAS), r["resumen"]["mse_test_por_semilla"], color=color,
                        s=14, alpha=0.4, linewidths=0)
        ax2.plot(longitudes, [r["resumen"]["sensibilidad_paso_mas_antiguo"] for r in rs], "o-", color=color,
                 markersize=6, label=nombre)
    ax1.set(title="MSE de prueba según SEQ_LEN (línea = media, puntos = semillas)", xlabel="SEQ_LEN",
            ylabel="MSE de prueba (escala log)", yscale="log", xticks=longitudes)
    ax1.legend()
    ax2.set(title="Señal que llega al paso más antiguo de la ventana", xlabel="SEQ_LEN",
            ylabel="Sensibilidad relativa (log)", yscale="log", xticks=longitudes)
    ax2.legend()
    fig.tight_layout()
    guardar(fig, "08_barrido_seq_len.png")

    return {
        "rnn_h16": s_rnn, "lstm_h16": s_lstm, f"rnn_h{h_eq}_control": s_eq,
        "compuerta_olvido_media": {"inicial": f_inicial, "entrenada": f_final},
        "barrido_seq_len": {nombre: [r["resumen"] for r in rs] for nombre, rs in barrido.items()},
    }


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Guía 06: RNN simple y LSTM para series temporales")
    parser.add_argument("--mostrar", action="store_true", help="abre las gráficas además de guardarlas")
    args = parser.parse_args()
    if not args.mostrar:
        plt.switch_backend("Agg")

    configurar_estilo()
    DIR_RESULTADOS.mkdir(exist_ok=True)
    inicio = time.perf_counter()

    metricas = {
        "configuracion": {"n_puntos": N_PUNTOS, "epochs": EPOCHS, "lr": LR, "hidden_base": HIDDEN_BASE,
                          "prop_train": PROP_TRAIN, "semillas": SEMILLAS, "torch": torch.__version__},
        "actividad_1_dimensiones": actividad_1_dimensiones(),
        "modelo_base": modelo_base(),
        "actividad_2_hiperparametros": actividad_2_hiperparametros(),
        "actividad_3_ventanas": actividad_3_ventanas(),
        "reto_lstm": reto_lstm(),
    }

    with open(DIR_RESULTADOS / "metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"\nMétricas guardadas en resultados/metricas.json · tiempo total: {time.perf_counter() - inicio:.1f} s")

    if args.mostrar:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
