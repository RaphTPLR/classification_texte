"""Chargement et inférence du modèle BiLSTM de sentiment."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bilstm.keras"
VOCAB_PATH = BASE_DIR / "lstm_word2idx.json"
PARAMS_PATH = BASE_DIR / "preprocessing_params.json"

PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"

# Un mot suivi d'une apostrophe reste un token (l', d', qu', aujourd'),
# tandis que le reste de la ponctuation est séparé.
_LSTM_TOKEN_RE = re.compile(r"\w+['’]?|[^\w\s]", flags=re.UNICODE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Prediction:
    label: str
    positive_probability: float
    negative_probability: float
    token_count: int
    used_token_count: int
    unknown_token_count: int
    was_truncated: bool


def preprocess_common(text: str) -> str:
    """Applique la normalisation commune utilisée avant la tokenisation.

    L'ordre est volontairement stable : décodage HTML, minuscules,
    uniformisation des apostrophes, retrait des balises, puis espaces.
    """
    normalized = html.unescape(str(text))
    normalized = normalized.lower()
    normalized = normalized.replace("’", "'").replace("`", "'").replace("´", "'")
    normalized = _HTML_TAG_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def tokenize_lstm(text: str) -> list[str]:
    """Découpe le texte en mots/préfixes apostrophés et ponctuation."""
    return _LSTM_TOKEN_RE.findall(text)


def load_resources() -> tuple[dict[str, int], dict[str, Any]]:
    """Charge le vocabulaire et les paramètres exportés par le notebook."""
    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        word2idx = json.load(file)
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        params = json.load(file)

    if word2idx.get(PAD_TOKEN) != 0 or word2idx.get(OOV_TOKEN) != 1:
        raise ValueError("Vocabulaire invalide : indices <PAD>/<OOV> inattendus.")
    if int(params.get("MAX_LEN_LSTM", 0)) <= 0:
        raise ValueError("Paramètre MAX_LEN_LSTM absent ou invalide.")

    return word2idx, params


def encode_lstm(
    text: str,
    word2idx: dict[str, int],
    max_len: int,
) -> tuple[np.ndarray, list[str], int]:
    """Transforme un texte en une ligne de ``max_len`` indices.

    Les premiers tokens sont conservés. Le padding est ajouté à droite,
    comme lors de l'entraînement.
    """
    clean_text = preprocess_common(text)
    tokens = tokenize_lstm(clean_text)
    oov_index = word2idx[OOV_TOKEN]
    pad_index = word2idx[PAD_TOKEN]

    token_ids = [word2idx.get(token, oov_index) for token in tokens[:max_len]]
    unknown_count = sum(token_id == oov_index for token_id in token_ids)
    token_ids.extend([pad_index] * (max_len - len(token_ids)))

    encoded = np.asarray([token_ids], dtype=np.int32)
    return encoded, tokens, unknown_count


def load_bilstm_model():
    """Charge le modèle Keras sans restaurer son état d'entraînement."""
    import keras

    return keras.models.load_model(MODEL_PATH, compile=False)


def predict_bilstm(
    text: str,
    model: Any,
    word2idx: dict[str, int],
    max_len: int,
    threshold: float = 0.5,
) -> Prediction:
    """Prédit le sentiment et renvoie les informations utiles à l'UI."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Le texte ne peut pas être vide.")

    encoded, tokens, unknown_count = encode_lstm(text, word2idx, max_len)
    probability = float(np.asarray(model.predict(encoded, verbose=0)).reshape(-1)[0])
    probability = min(max(probability, 0.0), 1.0)

    return Prediction(
        label="positif" if probability >= threshold else "négatif",
        positive_probability=probability,
        negative_probability=1.0 - probability,
        token_count=len(tokens),
        used_token_count=min(len(tokens), max_len),
        unknown_token_count=unknown_count,
        was_truncated=len(tokens) > max_len,
    )


def decode_text_file(file_bytes: bytes) -> tuple[str, str]:
    """Décode un fichier texte en UTF-8, puis en cp1252 si nécessaire."""
    try:
        return file_bytes.decode("utf-8"), "UTF-8"
    except UnicodeDecodeError:
        return file_bytes.decode("cp1252"), "Windows-1252"

