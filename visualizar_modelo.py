import joblib
import spacy
import streamlit as st

MODEL_PATH = "modelo_chamados.pkl"

st.set_page_config(page_title="Classificador de Chamados", page_icon="📊")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --piano-black: #070707;
        --piano-panel: #111111;
        --piano-edge: #2b2b2b;
        --red: #e50914;
        --red-soft: #ff6b6b;
        --paper: #f4f0e6;
        --muted: #a8a49a;
    }

    .stApp {
        color: var(--paper);
        background:
            radial-gradient(circle at 86% 4%, rgba(229, 9, 20, 0.16), transparent 28rem),
            linear-gradient(115deg, rgba(255, 255, 255, 0.035) 0 1px, transparent 1px 9px),
            linear-gradient(180deg, #171717 0%, var(--piano-black) 42%, #020202 100%);
        background-size: auto, 9px 100%, auto;
    }

    .block-container {
        max-width: 980px;
        padding: 4.5rem 2rem 5rem;
    }

    .brand-kicker {
        color: var(--red-soft);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .brand-title {
        color: var(--paper);
        font-family: 'UnifrakturCook', 'Old English Text MT', Georgia, serif;
        font-size: clamp(3rem, 8vw, 6.5rem);
        line-height: 0.92;
        margin: 0;
        text-shadow: 0 2px 0 #000, 0 0 28px rgba(229, 9, 20, 0.24);
    }

    .brand-rule {
        background: linear-gradient(90deg, var(--red), transparent);
        height: 2px;
        margin: 1.5rem 0 2.8rem;
        width: 100%;
    }

    [data-testid="stMarkdownContainer"], [data-testid="stText"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    [data-testid="stTextArea"] textarea {
        background: linear-gradient(145deg, #1c1c1c, #080808);
        border: 1px solid var(--piano-edge);
        border-radius: 2px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 12px 30px rgba(0,0,0,0.28);
        color: var(--paper);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1rem;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--red);
        box-shadow: 0 0 0 1px var(--red), inset 0 1px 0 rgba(255,255,255,0.08);
    }

    [data-testid="stTextArea"] label, .stSubheader {
        color: var(--red-soft) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }

    .stButton > button {
        background: linear-gradient(135deg, #ff5a61 0%, var(--red) 48%, #8f0008 100%);
        border: 1px solid #ff8a8f;
        border-radius: 2px;
        box-shadow: 0 8px 22px rgba(229, 9, 20, 0.28), inset 0 1px 0 rgba(255,255,255,0.55);
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        letter-spacing: 0.04em;
        min-height: 2.8rem;
        transition: transform 160ms ease, box-shadow 160ms ease;
    }

    .stButton > button:hover {
        box-shadow: 0 12px 30px rgba(229, 9, 20, 0.45), inset 0 1px 0 rgba(255,255,255,0.7);
        color: #fff;
        transform: translateY(-2px);
    }

    [data-testid="stAlert"] {
        background: #1b0c0d;
        border: 1px solid var(--red);
        border-radius: 2px;
        color: var(--paper);
    }

    [data-testid="stMetricValue"], code {
        color: var(--red-soft) !important;
    }

    [data-testid="stVegaLiteChart"] {
        background: linear-gradient(145deg, rgba(28,28,28,0.95), rgba(7,7,7,0.95));
        border: 1px solid var(--piano-edge);
        border-radius: 2px;
        padding: 0.7rem;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def carregar_recursos():
    modelo = joblib.load(MODEL_PATH)
    nlp = spacy.load("pt_core_news_sm")
    return modelo, nlp


def preprocessar_texto(texto, nlp):
    doc = nlp(texto)
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if not token.is_punct
    )


st.markdown(
    """
    <div class="brand-kicker">Central de operações · modelo ativo</div>
    <h1 class="brand-title">Classificador de Chamados</h1>
    <div class="brand-rule"></div>
    """,
    unsafe_allow_html=True,
)
st.write("Consulte o modelo treinado e classifique um novo chamado.")

try:
    modelo, nlp = carregar_recursos()
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo: {erro}")
    st.stop()

st.subheader("Modelo carregado")
st.write(f"Pipeline: `{type(modelo).__name__}`")
st.write(f"Etapas: `{', '.join(modelo.named_steps)}`")

texto = st.text_area(
    "Descrição do chamado",
    placeholder="Ex.: O Teams não abre no computador.",
    height=120,
)

if st.button("Classificar chamado", type="primary"):
    if not texto.strip():
        st.warning("Digite uma descrição antes de classificar.")
    else:
        texto_processado = preprocessar_texto(texto, nlp)
        categoria = modelo.predict([texto_processado])[0]
        probabilidades = modelo.predict_proba([texto_processado])[0]
        classes = modelo.classes_

        st.success(f"Categoria prevista: {categoria}")
        st.subheader("Probabilidades")
        st.bar_chart(
            {classe: float(probabilidade) for classe, probabilidade in zip(classes, probabilidades)},
            horizontal=True,
        )
