"""ARIMA baseline, Whale Optimization Algorithm (WOA), LSTM and hybrid utilities."""
import numpy as np
import pandas as pd

def fit_arima_forecast(train_series, steps, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
    """Fit ARIMA/SARIMAX on a univariate series and forecast `steps` future values."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    series = pd.Series(train_series).astype(float).dropna()
    if len(series) < 10:
        raise ValueError("La série est trop courte pour entraîner ARIMA (minimum conseillé : 10 points).")
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=steps)
    return fitted, np.asarray(forecast.predicted_mean), forecast.conf_int()

def whale_optimization(objective, bounds, n_whales=8, max_iter=15, random_state=42):
    """
    Basic Whale Optimization Algorithm minimizer.
    bounds: sequence of (lower, upper). Returns best_position, best_score, history.
    """
    rng = np.random.default_rng(random_state)
    bounds = np.asarray(bounds, dtype=float)
    lower, upper = bounds[:, 0], bounds[:, 1]
    dim = len(bounds)
    positions = rng.uniform(lower, upper, size=(n_whales, dim))
    scores = np.array([objective(p) for p in positions], dtype=float)
    best_idx = int(np.argmin(scores))
    best_pos, best_score = positions[best_idx].copy(), float(scores[best_idx])
    history = [best_score]

    for iteration in range(max_iter):
        a = 2 - 2 * iteration / max(1, max_iter - 1)
        for i in range(n_whales):
            r1, r2 = rng.random(), rng.random()
            A, C = 2 * a * r1 - a, 2 * r2
            p = rng.random()
            if p < .5:
                if abs(A) < 1:
                    D = abs(C * best_pos - positions[i])
                    candidate = best_pos - A * D
                else:
                    peer = positions[rng.integers(n_whales)]
                    D = abs(C * peer - positions[i])
                    candidate = peer - A * D
            else:
                l = rng.uniform(-1, 1)
                D = abs(best_pos - positions[i])
                candidate = D * np.exp(l) * np.cos(2 * np.pi * l) + best_pos
            positions[i] = np.clip(candidate, lower, upper)
            scores[i] = float(objective(positions[i]))
        best_idx = int(np.argmin(scores))
        if scores[best_idx] < best_score:
            best_score, best_pos = float(scores[best_idx]), positions[best_idx].copy()
        history.append(best_score)
    return best_pos, best_score, history

def build_lstm(input_shape, units=32, dropout=.2, learning_rate=.001):
    """Build a Keras LSTM. TensorFlow is an optional dependency."""
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise ImportError("Installe TensorFlow pour utiliser le LSTM : pip install tensorflow") from exc
    model = Sequential([
        Input(shape=input_shape),
        LSTM(int(units)),
        Dropout(float(dropout)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=float(learning_rate)), loss="mse", metrics=["mae"])
    return model

def optimize_lstm_woa(x_train, y_train, x_val, y_val, n_whales=5, max_iter=5,
                      epochs=12, batch_size=32, random_state=42):
    """Tune LSTM units, dropout and learning rate with WOA using validation MAE."""
    from sklearn.metrics import mean_absolute_error
    def objective(position):
        units = int(np.clip(round(position[0] / 8) * 8, 8, 128))
        dropout = float(np.clip(position[1], 0.0, .5))
        lr = float(10 ** np.clip(position[2], -4, -2))
        model = build_lstm(x_train.shape[1:], units=units, dropout=dropout, learning_rate=lr)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                  verbose=0, shuffle=False)
        pred = model.predict(x_val, verbose=0).reshape(-1)
        return mean_absolute_error(y_val, pred)
    best, score, history = whale_optimization(
        objective, bounds=[(8, 128), (0.0, .5), (-4, -2)],
        n_whales=n_whales, max_iter=max_iter, random_state=random_state
    )
    params = {"units": int(np.clip(round(best[0] / 8) * 8, 8, 128)),
              "dropout": float(np.clip(best[1], 0, .5)),
              "learning_rate": float(10 ** np.clip(best[2], -4, -2))}
    return params, score, history

def make_sequences(X, y, sequence_length=7):
    """Convert tabular time-ordered arrays to 3D LSTM sequences."""
    X, y = np.asarray(X), np.asarray(y).reshape(-1)
    if len(X) <= sequence_length:
        raise ValueError("Pas assez d'observations pour construire les séquences LSTM.")
    xs, ys = [], []
    for end in range(sequence_length, len(X)):
        xs.append(X[end-sequence_length:end])
        ys.append(y[end])
    return np.asarray(xs), np.asarray(ys)

def hybrid_residual_forecast(arima_predictions, lstm_residual_predictions):
    """Hybrid point forecast = ARIMA forecast + predicted ARIMA residual correction."""
    a = np.asarray(arima_predictions).reshape(-1)
    r = np.asarray(lstm_residual_predictions).reshape(-1)
    if len(a) != len(r):
        raise ValueError("ARIMA et LSTM doivent produire le même nombre de prédictions.")
    return a + r
