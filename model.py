from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Normalization
from tensorflow import keras

def init_model(input_shape,X_train_resampled):
    """Initialize the neural network model."""

    metrics = [
    keras.metrics.Recall(name='recall'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve]
    ]

    normalizer = Normalization()
    normalizer.adapt(X_train_resampled)

    model = models.Sequential([
        normalizer,
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(8, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=metrics)
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=100):
    """Train the neural network model."""
    es = EarlyStopping(patience=10, monitor='val_recall', restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[es],
                        shuffle=True)
    return history
