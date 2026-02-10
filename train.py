"""
Train LSTM sentiment model on IMDB dataset.
Saves model and tokenizer for use by the Flask API.
"""
import os
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

# Config
MAX_FEATURES = 10000
MAXLEN = 256
EMBEDDING_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 4
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_preprocess():
    """Load IMDB and pad sequences."""
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAXLEN, padding="post", truncating="post")
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=MAXLEN, padding="post", truncating="post")
    return (x_train, y_train), (x_test, y_test), imdb.get_word_index()


def build_model():
    model = keras.Sequential([
        layers.Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAXLEN),
        layers.LSTM(LSTM_UNITS, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("Loading IMDB dataset...")
    (x_train, y_train), (x_test, y_test), word_index = load_and_preprocess()

    # Invert word_index so we can save {word: id} for tokenizer at inference
    # Keras imdb uses 1-indexed (0 = padding), we keep as-is for encoding
    word_index_path = os.path.join(MODEL_DIR, "word_index.json")
    with open(word_index_path, "w") as f:
        json.dump(word_index, f)

    config = {
        "max_features": MAX_FEATURES,
        "maxlen": MAXLEN,
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(config, f)

    print("Building model...")
    model = build_model()
    model.summary()

    print("Training...")
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

    model.save(os.path.join(MODEL_DIR, "sentiment_lstm.keras"))
    print(f"Model and config saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
