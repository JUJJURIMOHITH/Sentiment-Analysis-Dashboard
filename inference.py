"""
Inference for sentiment model: load model + tokenizer, map text -> pos/neg/neutral and scores.
"""
import os
import json
import re
import numpy as np
from tensorflow import keras

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_lstm.keras")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
WORD_INDEX_PATH = os.path.join(MODEL_DIR, "word_index.json")

# Neutral when max probability is below this (so we get 3-way from binary model)
NEUTRAL_THRESHOLD = 0.55

_model = None
_word_index = None
_maxlen = None


def _load_assets():
    global _model, _word_index, _maxlen
    if _model is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python train.py"
        )
    _model = keras.models.load_model(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    _maxlen = config["maxlen"]
    with open(WORD_INDEX_PATH) as f:
        _word_index = json.load(f)
    # Keras IMDB uses 1-indexed; 0 is reserved for padding
    # word_index: word -> 1-based index (1..max_features), we use first 10000


def _text_to_sequence(text: str) -> list:
    """Convert raw text to list of word indices using IMDB word_index."""
    text = re.sub(r"[^a-zA-Z0-9'\s]", " ", text.lower())
    words = text.split()
    # word_index is 1-indexed; keys are lowercased words
    seq = []
    for w in words:
        idx = _word_index.get(w)
        if idx is not None and idx < 10000:  # stay within max_features
            seq.append(idx)
    return seq


def predict(text: str) -> dict:
    """
    Predict sentiment and confidence scores.
    Returns: {
        "sentiment": "positive" | "negative" | "neutral",
        "confidence": float,
        "scores": { "positive": float, "negative": float, "neutral": float }
    }
    """
    _load_assets()
    seq = _text_to_sequence(text)
    if not seq:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "scores": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
        }
    x = keras.preprocessing.sequence.pad_sequences(
        [seq], maxlen=_maxlen, padding="post", truncating="post", value=0
    )
    prob_positive = float(_model.predict(x, verbose=0)[0, 0])
    print("DEBUG → prob_positive:", prob_positive)
    prob_negative = 1.0 - prob_positive

    if prob_positive >= NEUTRAL_THRESHOLD:
        sentiment = "positive"
        confidence = prob_positive
        neutral = 1.0 - max(prob_positive, prob_negative)
    elif prob_negative >= NEUTRAL_THRESHOLD:
        sentiment = "negative"
        confidence = prob_negative
        neutral = 1.0 - max(prob_positive, prob_negative)
    else:
        sentiment = "neutral"
        neutral = 1.0 - abs(prob_positive - 0.5) * 2  # how "middle" we are
        confidence = min(prob_positive, prob_negative) + neutral * 0.5

    # Normalize scores for display (positive + negative + neutral = 1)
    if sentiment == "neutral":
        s_pos = prob_positive
        s_neg = prob_negative
        s_neu = 1.0 - abs(prob_positive - 0.5) * 2
        total = s_pos + s_neg + s_neu
        s_pos, s_neg, s_neu = s_pos / total, s_neg / total, s_neu / total
    else:
        s_neu = 1.0 - confidence
        if sentiment == "positive":
            s_pos, s_neg = confidence, 0.0
        else:
            s_pos, s_neg = 0.0, confidence
        total = s_pos + s_neg + s_neu
        s_pos, s_neg, s_neu = s_pos / total, s_neg / total, s_neu / total

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "scores": {
            "positive": round(s_pos, 4),
            "negative": round(s_neg, 4),
            "neutral": round(s_neu, 4),
        },
    }
