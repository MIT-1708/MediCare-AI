from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SerializableKerasBinaryClassifier:
    """A lightweight, pickle-friendly wrapper around a Keras binary classifier."""

    input_dim: int
    hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 60
    batch_size: int = 32
    validation_split: float = 0.2
    patience: int = 8
    random_state: int = 42
    verbose: int = 0
    threshold: float = 0.5
    history_: Dict[str, List[float]] = field(default_factory=dict, init=False)
    model_json_: Optional[str] = field(default=None, init=False)
    weights_: Optional[List[np.ndarray]] = field(default=None, init=False)

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def _set_random_seeds(self) -> None:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.random_state)

    def _build_model(self):
        import tensorflow as tf

        inputs = tf.keras.Input(shape=(self.input_dim,), name="features")
        x = inputs
        for idx, units in enumerate(self.hidden_units):
            x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{idx + 1}")(x)
            if self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate, name=f"dropout_{idx + 1}")(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="risk")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tabular_binary_classifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )
        return model

    def fit(self, X, y):
        import tensorflow as tf

        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.float32)
        self._set_random_seeds()

        model = self._build_model()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            X_np,
            y_np,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
            callbacks=callbacks,
        )
        self.history_ = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        self.model_json_ = model.to_json()
        self.weights_ = [np.asarray(w) for w in model.get_weights()]
        return self

    def _restore_model(self):
        import tensorflow as tf

        if not self.model_json_ or self.weights_ is None:
            raise RuntimeError("Keras model has not been fitted yet.")
        model = tf.keras.models.model_from_json(self.model_json_)
        model.set_weights(self.weights_)
        return model

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    def _predict_scores_numpy(self, X_np: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Keras model has not been fitted yet.")

        activations = X_np
        # Keras stores Dense layer weights as kernel, bias pairs.
        dense_layer_count = len(self.hidden_units) + 1
        expected_arrays = dense_layer_count * 2
        if len(self.weights_) != expected_arrays:
            raise RuntimeError(
                "Unexpected Keras weight layout for numpy inference. "
                "Retrain the model if the architecture changed."
            )

        for layer_idx in range(dense_layer_count):
            kernel = np.asarray(self.weights_[layer_idx * 2], dtype=np.float32)
            bias = np.asarray(self.weights_[layer_idx * 2 + 1], dtype=np.float32)
            activations = activations @ kernel + bias
            if layer_idx < dense_layer_count - 1:
                activations = self._relu(activations)

        return self._sigmoid(activations.reshape(-1))

    def predict_proba(self, X):
        X_np = self._to_numpy(X)
        scores = self._predict_scores_numpy(X_np)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)
