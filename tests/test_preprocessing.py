import numpy as np

from inference import encode_lstm, preprocess_common, tokenize_lstm


def test_preprocess_normalizes_apostrophes_html_and_spaces():
    text = "  L’histoire&nbsp; est <b>TRÈS</b> bien !  "
    assert preprocess_common(text) == "l'histoire est très bien !"


def test_preprocess_keeps_urls_seen_during_training():
    assert preprocess_common("Voir HTTPS://EXEMPLE.FR/test") == (
        "voir https://exemple.fr/test"
    )


def test_tokenizer_preserves_french_apostrophe_prefixes():
    assert tokenize_lstm("l'histoire qu'on aime !") == [
        "l'",
        "histoire",
        "qu'",
        "on",
        "aime",
        "!",
    ]


def test_encoding_truncates_and_pads_on_the_right():
    vocab = {"<PAD>": 0, "<OOV>": 1, "bon": 2, "!": 3}
    encoded, tokens, unknown_count = encode_lstm("Bon ! inconnu", vocab, max_len=5)
    np.testing.assert_array_equal(encoded, [[2, 3, 1, 0, 0]])
    assert tokens == ["bon", "!", "inconnu"]
    assert unknown_count == 1


def test_encoding_keeps_first_tokens_when_too_long():
    vocab = {"<PAD>": 0, "<OOV>": 1, "bon": 2}
    encoded, _, _ = encode_lstm("bon bon bon bon", vocab, max_len=3)
    np.testing.assert_array_equal(encoded, [[2, 2, 2]])

