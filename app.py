"""Interface Streamlit de classification de critiques de films."""

from __future__ import annotations

import streamlit as st

from inference import decode_text_file, load_bilstm_model, load_resources, predict_bilstm


st.set_page_config(
    page_title="FilmSentiment — Analyse de commentaires",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def get_inference_resources():
    word2idx, params = load_resources()
    return load_bilstm_model(), word2idx, params


st.markdown(
    """
    <style>
    :root {
        --bg: #08090B;
        --surface: #111318;
        --surface-2: #17191F;
        --white: #F5F6F8;
        --text: #B8BBC3;
        --muted: #7D818B;
        --line: #30333B;
        --line-light: #555A65;
        --positive: #4EE6A5;
        --positive-soft: rgba(78, 230, 165, .13);
        --negative: #FF6565;
        --negative-soft: rgba(255, 101, 101, .13);
    }

    html, body, .stApp, .stApp *, button, input, textarea {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
    }
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
    }
    .stApp {
        color: var(--text);
        background:
            radial-gradient(circle at 82% 20%, rgba(73, 78, 91, .16), transparent 30%),
            radial-gradient(circle at 18% 82%, rgba(52, 57, 67, .12), transparent 30%),
            var(--bg);
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display:none !important; }
    .block-container { max-width: 1080px; padding: 1.15rem 1.25rem 4rem; }

    .nav { display:flex; align-items:center; justify-content:space-between; height:52px; }
    .stApp .brand { color:var(--white); font-size:1.13rem; font-weight:400; letter-spacing:-.045em; }
    .stApp .brand strong { font-weight:750; }
    .nav-links { display:flex; align-items:center; gap:.35rem; }
    .nav-link { padding:.58rem .85rem; border-radius:999px; color:#A9ADB6; font-size:.76rem; font-weight:500; }
    .nav-link.active { background:#1B1D23; color:var(--white); }
    .nav-pill { padding:.58rem .95rem; border:1px solid var(--line-light); border-radius:999px; color:var(--white); font-size:.74rem; font-weight:600; }

    .hero { padding:5.5rem 0 2.8rem; text-align:center; }
    .eyebrow { margin-bottom:1.1rem; color:#A6AAB3; font-size:.67rem; font-weight:600; letter-spacing:.28em; text-transform:uppercase; }
    .stApp .hero-title { width:min(820px,100%); max-width:820px; margin:0 auto !important; color:var(--white) !important; font-size:clamp(3rem,5.1vw,4.35rem) !important; font-weight:750 !important; line-height:1.04 !important; letter-spacing:-.058em !important; }
    .stApp .hero-copy { width:min(610px,100%); max-width:610px; margin:1.25rem auto 0 !important; color:#B3B6BE; font-size:.98rem; line-height:1.65; }

    .input-label { max-width:900px; margin:0 auto .65rem; color:#878B95; font-size:.66rem; font-weight:650; letter-spacing:.11em; text-transform:uppercase; }
    [data-testid="stForm"] {
        max-width:900px;
        margin:0 auto;
        padding:.48rem .5rem .48rem 1.25rem;
        border:1px solid #555A65;
        border-radius:999px;
        background:rgba(21,23,28,.92);
        box-shadow:0 18px 55px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.035);
    }
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] { align-items:center; gap:.35rem; }
    [data-testid="stForm"] [data-testid="column"] { padding:0 !important; }
    [data-testid="stForm"] [data-testid="stTextInput"] > div { margin:0; }
    [data-testid="stForm"] [data-baseweb="input"] {
        min-height:48px;
        border:0 !important;
        background:transparent !important;
        box-shadow:none !important;
    }
    [data-testid="stForm"] input {
        padding:0 !important;
        background:transparent !important;
        color:var(--white) !important;
        font-size:.98rem !important;
        caret-color:var(--white);
    }
    [data-testid="stForm"] input::placeholder { color:#858994 !important; opacity:1; }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] { display:flex; justify-content:flex-end; }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
        width:48px;
        min-width:48px;
        height:48px;
        min-height:48px;
        padding:0;
        border:0;
        border-radius:50%;
        background:var(--white);
        color:#0B0C0F;
        font-size:1.35rem;
        font-weight:500;
        box-shadow:none;
        transition:transform .15s ease, background .15s ease;
    }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover { background:#DADCE1; color:#0B0C0F; transform:scale(1.03); }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button:focus { box-shadow:0 0 0 3px rgba(255,255,255,.14); }

    .file-row { max-width:900px; margin:.85rem auto 0; }
    [data-testid="stExpander"] { max-width:900px; margin:.85rem auto 0; border:1px solid #292C33; border-radius:12px; background:rgba(17,19,24,.75); }
    [data-testid="stExpander"] summary { color:#9498A2; font-size:.75rem; }
    div[data-testid="stFileUploader"] section { border:1px dashed #40444E; border-radius:10px; background:#111318; }
    div[data-testid="stFileUploader"] section button { border-color:#484C56; background:#1B1E24; color:var(--white); }

    .result-shell {
        max-width:900px;
        margin:2.5rem auto 0;
        padding:1.5rem 1.65rem;
        border:1px solid var(--line);
        border-radius:18px;
        background:linear-gradient(145deg, rgba(20,22,27,.96), rgba(12,13,16,.96));
        box-shadow:0 20px 55px rgba(0,0,0,.22);
    }
    .result-grid { display:grid; grid-template-columns:minmax(260px,.8fr) 1px minmax(360px,1.2fr); gap:2rem; align-items:center; }
    .result-divider { width:1px; height:118px; background:var(--line); }
    .result-kicker { margin-bottom:.8rem; color:#8F939D; font-size:.65rem; font-weight:600; letter-spacing:.18em; text-transform:uppercase; }
    .verdict-row { display:flex; align-items:center; gap:1rem; }
    .verdict-icon { display:grid; place-items:center; width:46px; height:46px; border-radius:50%; font-size:1.15rem; font-weight:700; }
    .verdict-positive .verdict-icon { background:var(--positive-soft); color:var(--positive); }
    .verdict-negative .verdict-icon { background:var(--negative-soft); color:var(--negative); }
    .verdict-label { color:var(--white); font-size:2.8rem; font-weight:750; line-height:1; letter-spacing:-.055em; }
    .score-block + .score-block { margin-top:1.15rem; }
    .score-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:.48rem; }
    .score-name { color:var(--white); font-size:.82rem; font-weight:600; }
    .score-value { color:var(--white); font-size:.84rem; font-weight:700; font-variant-numeric:tabular-nums; }
    .score-track { height:7px; overflow:hidden; border-radius:99px; background:#383B43; }
    .score-fill { height:100%; border-radius:99px; }
    .score-positive { background:var(--positive); box-shadow:0 0 15px rgba(78,230,165,.23); }
    .score-negative { background:var(--negative); box-shadow:0 0 15px rgba(255,101,101,.2); }
    .details { max-width:900px; margin:.8rem auto 0; }
    .footer { max-width:900px; margin:2.8rem auto 0; color:#555963; font-size:.65rem; text-align:center; }

    [data-testid="stAlert"] { max-width:940px; margin-left:auto; margin-right:auto; background:#17191F; color:var(--white); border-color:#3A3D45; }
    div[data-testid="stStatusWidget"] { color:var(--white); }

    @media (max-width:760px) {
        .block-container { padding:1rem 1rem 3rem; }
        .nav-links { display:none; }
        .hero { padding:3.5rem 0 2.2rem; }
        .stApp .hero-title { font-size:2.65rem !important; line-height:1 !important; }
        .hero-title br { display:none; }
        .hero-copy { font-size:.9rem; }
        [data-testid="stForm"] { padding:.4rem .42rem .4rem 1rem; }
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] { flex-wrap:nowrap !important; }
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-of-type { flex:1 1 0 !important; width:auto !important; min-width:0 !important; }
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-of-type { flex:0 0 48px !important; width:48px !important; min-width:48px !important; max-width:48px !important; }
        [data-testid="stForm"] input { font-size:.88rem !important; }
        .result-grid { grid-template-columns:1fr; gap:1.4rem; }
        .result-divider { width:100%; height:1px; }
        .verdict-label { font-size:2.35rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_result(prediction) -> None:
    is_positive = prediction.label == "positif"
    verdict_class = "verdict-positive" if is_positive else "verdict-negative"
    icon = "✓" if is_positive else "×"
    positive_width = max(0, min(100, prediction.positive_probability * 100))
    negative_width = max(0, min(100, prediction.negative_probability * 100))

    st.markdown(
        f"""
        <section class="result-shell">
            <div class="result-grid">
                <div>
                    <div class="result-kicker">Résultat de l’analyse</div>
                    <div class="verdict-row {verdict_class}">
                        <div class="verdict-icon">{icon}</div>
                        <div class="verdict-label">{prediction.label.capitalize()}</div>
                    </div>
                </div>
                <div class="result-divider"></div>
                <div>
                    <div class="score-block">
                        <div class="score-head"><span class="score-name">Positif</span><span class="score-value">{prediction.positive_probability:.1%}</span></div>
                        <div class="score-track"><div class="score-fill score-positive" style="width:{positive_width:.1f}%"></div></div>
                    </div>
                    <div class="score-block">
                        <div class="score-head"><span class="score-name">Négatif</span><span class="score-value">{prediction.negative_probability:.1%}</span></div>
                        <div class="score-track"><div class="score-fill score-negative" style="width:{negative_width:.1f}%"></div></div>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="details">', unsafe_allow_html=True)
    with st.expander("Détails techniques"):
        st.write(f"Tokens détectés : **{prediction.token_count}**")
        st.write(f"Tokens utilisés : **{prediction.used_token_count} sur 310**")
        st.write(f"Mots absents du vocabulaire : **{prediction.unknown_token_count}**")
        if prediction.was_truncated:
            st.warning("Le texte dépasse 310 tokens. Seule sa première partie a été analysée.")
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    """
    <nav class="nav">
        <div class="brand"><strong>Film</strong>Sentiment</div>
        <div class="nav-links">
            <span class="nav-link active">Analyse</span>
            <span class="nav-link">Fonctionnement</span>
            <span class="nav-link">À propos</span>
        </div>
        <div class="nav-pill">BiLSTM · FR</div>
    </nav>
    <header class="hero">
        <div class="eyebrow">IA · Analyse de sentiment</div>
        <h1 class="hero-title">Analysez le sentiment <br>des commentaires de films</h1>
        <p class="hero-copy">Le modèle classe un commentaire en positif ou négatif et affiche les deux scores de confiance.</p>
    </header>
    <div class="input-label">Commentaire à analyser</div>
    """,
    unsafe_allow_html=True,
)

with st.form("sentiment_form", clear_on_submit=False, border=True):
    text_col, action_col = st.columns([12, 1])
    with text_col:
        manual_text = st.text_input(
            "Commentaire à analyser",
            placeholder="Écrivez votre commentaire sur un film…",
            label_visibility="collapsed",
            key="manual_text",
        )
    with action_col:
        analyze = st.form_submit_button("→", type="primary", help="Analyser le commentaire")

st.markdown('<div class="file-row">', unsafe_allow_html=True)
with st.expander("Ou importer un fichier .txt"):
    uploaded_file = st.file_uploader(
        "Fichier texte",
        type=["txt"],
        label_visibility="collapsed",
        key="review_file",
    )
    uploaded_text = ""
    if uploaded_file is not None:
        try:
            uploaded_text, encoding = decode_text_file(uploaded_file.getvalue())
            st.success(f"Fichier chargé · {len(uploaded_text):,} caractères · {encoding}")
        except UnicodeDecodeError:
            st.error("Le fichier n’est lisible ni en UTF-8 ni en Windows-1252.")
st.markdown('</div>', unsafe_allow_html=True)

text_to_analyze = uploaded_text if uploaded_file is not None else manual_text

if analyze:
    if not text_to_analyze.strip():
        st.error("Écrivez un commentaire ou importez un fichier avant de lancer l’analyse.")
    else:
        try:
            with st.spinner("Analyse en cours…"):
                model, word2idx, params = get_inference_resources()
                prediction = predict_bilstm(
                    text_to_analyze,
                    model=model,
                    word2idx=word2idx,
                    max_len=int(params["MAX_LEN_LSTM"]),
                )
            st.session_state["last_prediction"] = prediction
        except FileNotFoundError as error:
            st.error(f"Fichier du modèle introuvable : {error.filename}")
        except Exception as error:
            st.error(f"L’analyse a échoué : {error}")

if "last_prediction" in st.session_state:
    render_result(st.session_state["last_prediction"])

st.markdown(
    '<div class="footer">Analyse limitée aux 310 premiers tokens · Seuil de décision : 0,5</div>',
    unsafe_allow_html=True,
)
