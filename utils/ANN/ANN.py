import os
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_split(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Load CSV data, scale inputs and targets, split into train and test sets.
    """
    data = pd.read_csv(csv_path)
    X = data[['Vin', 'RL', 'Iin']].values
    y = data[['Eff', 'Pout']].values

    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    joblib.dump(x_scaler, 'x_scaler.save')

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    joblib.dump(y_scaler, 'y_scaler.save')

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_scaled,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def build_model(input_dim: int, output_dim: int) -> Model:
    """
    Construct and compile the neural network model.
    """
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='elu', kernel_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='elu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation='elu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    outputs = Dense(output_dim, activation=None)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


def train_model(csv_path: str):
    """
    Train the model on data from csv_path and return the trained model, training history, and test set.
    """
    X_train, X_test, y_train, y_test = load_and_split(csv_path)
    model = build_model(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

    checkpoint = ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-5
    )
    tensorboard = TensorBoard(log_dir='logs')

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=500,
        batch_size=32,
        callbacks=[checkpoint, early_stop, reduce_lr, tensorboard],
        verbose=1
    )

    return model, history, (X_test, y_test)


def evaluate_and_plot(model, X_test, y_test_scaled, y_scaler_path: str) -> None:
    """
    Evaluate model performance on test set and save Actual vs Predicted plots.
    """
    y_scaler = joblib.load(y_scaler_path)
    preds_scaled = model.predict(X_test)
    preds = y_scaler.inverse_transform(preds_scaled)
    y_true = y_scaler.inverse_transform(y_test_scaled)

    mse = mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)

    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R2 : {r2:.6f}")

    labels = ['Eff', 'Pout']
    for i, label in enumerate(labels):
        plt.figure()
        plt.scatter(y_true[:, i], preds[:, i], alpha=0.5)
        mn = min(y_true[:, i].min(), preds[:, i].min())
        mx = max(y_true[:, i].max(), preds[:, i].max())
        plt.plot([mn, mx], [mn, mx], linestyle='--')
        plt.xlabel(f"Actual {label}")
        plt.ylabel(f"Predicted {label}")
        plt.title(f"{label}: Actual vs Predicted")
        plt.savefig(f"{label}_actual_vs_predicted.png")
        plt.close()


def predict_batch(records, model_path: str, x_scaler_path: str, y_scaler_path: str):
    """
    Perform batch prediction on a list of record dicts and return a DataFrame of results.
    """
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    model = tf.keras.models.load_model(model_path)

    df = pd.DataFrame(records)
    X = df[['Vin', 'RL', 'Iin']].values
    X_scaled = x_scaler.transform(X)
    preds_scaled = model.predict(X_scaled)
    preds = y_scaler.inverse_transform(preds_scaled)

    return pd.DataFrame(preds, columns=['Eff', 'Pout'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate ANN for DAB converter optimization"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file containing training data"
    )
    args = parser.parse_args()

    set_seed(42)
    model, history, (X_test, y_test_scaled) = train_model(args.csv)
    evaluate_and_plot(model, X_test, y_test_scaled, y_scaler_path='y_scaler.save')
