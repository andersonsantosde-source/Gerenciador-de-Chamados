import joblib
import pandas as pd
import plotly.graph_objects as go
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
        max-width: 1180px;
        padding: 4.5rem 2rem 5rem;
    }

    .console-line {
        align-items: center;
        border-bottom: 1px solid #242424;
        border-top: 1px solid #242424;
        color: #777;
        display: flex;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.68rem;
        justify-content: space-between;
        letter-spacing: 0.12em;
        margin: 0.8rem 0 2rem;
        padding: 0.7rem 0;
        text-transform: uppercase;
    }

    .status-dot {
        background: #ff3b45;
        border-radius: 50%;
        box-shadow: 0 0 12px #e50914;
        display: inline-block;
        height: 7px;
        margin-right: 0.45rem;
        width: 7px;
    }

    .result-card {
        background: linear-gradient(145deg, rgba(30,30,30,0.92), rgba(7,7,7,0.96));
        border: 1px solid #3a191b;
        border-left: 3px solid var(--red);
        box-shadow: 0 18px 45px rgba(0,0,0,0.32), inset 0 1px 0 rgba(255,255,255,0.05);
        padding: 1.15rem 1.3rem;
    }

    .result-label {
        color: #888;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .result-value {
        color: #ff5a61;
        font-family: 'UnifrakturCook', 'Old English Text MT', Georgia, serif;
        font-size: 2.8rem;
        line-height: 1.1;
        margin-top: 0.35rem;
    }

    .result-meta {
        color: var(--muted);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        margin-top: 0.6rem;
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

    [data-testid="stPlotlyChart"] {
        background: radial-gradient(circle at 50% 42%, rgba(100, 9, 16, 0.18), transparent 48%), #080808;
        border: 1px solid #2e2021;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 45px rgba(0,0,0,0.28);
        padding: 0.35rem;
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
    <div class="console-line">
        <span><span class="status-dot"></span>Neural routing online</span>
        <span>TF-IDF / MULTINOMIAL NB · v1.0</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("Consulte o modelo treinado e classifique um novo chamado.")

try:
    modelo, nlp = carregar_recursos()
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo: {erro}")
    st.stop()

with st.expander("Detalhes do núcleo de classificação", expanded=False):
    st.write(f"Pipeline: `{type(modelo).__name__}`")
    st.write(f"Etapas: `{', '.join(modelo.named_steps)}`")

texto = st.text_area(
    "Descrição do chamado",
    placeholder="Ex.: O Teams não abre no computador.",
    height=120,
)

if st.button("Classificar chamado", type="primary", use_container_width=True):
    if not texto.strip():
        st.warning("Digite uma descrição antes de classificar.")
    else:
        texto_processado = preprocessar_texto(texto, nlp)
        categoria = modelo.predict([texto_processado])[0]
        probabilidades = modelo.predict_proba([texto_processado])[0]
        classes = modelo.classes_

        indice_maior_probabilidade = probabilidades.argmax()
        maior_probabilidade = float(probabilidades[indice_maior_probabilidade])
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Diagnóstico dominante</div>
                <div class="result-value">{categoria}</div>
                <div class="result-meta">Confiança do modelo: {maior_probabilidade:.1%} · análise concluída agora</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Mapa 3D de confiança")
        st.caption("Gire, aproxime e explore as torres para comparar a leitura do classificador.")

        cores = ["#e50914", "#ff4b55", "#a70710", "#ff858a", "#6e050b", "#d9232e"]
        x_positions = list(range(len(classes)))
        figura = go.Figure()

        for indice, (classe, probabilidade) in enumerate(zip(classes, probabilidades)):
            probabilidade = float(probabilidade)
            figura.add_trace(
                go.Scatter3d(
                    x=[indice, indice],
                    y=[0, 0],
                    z=[0, probabilidade],
                    mode="lines+markers+text",
                    text=["", f"{probabilidade:.0%}"],
                    textposition="top center",
                    textfont={"color": "#f4f0e6", "size": 13},
                    line={"color": cores[indice % len(cores)], "width": 14},
                    marker={
                        "color": cores[indice % len(cores)],
                        "size": [4, 14 + probabilidade * 24],
                        "line": {"color": "#ffb3b6", "width": 1},
                    },
                    name=classe,
                    hovertemplate=f"<b>{classe}</b><br>Confiança: {probabilidade:.1%}<extra></extra>",
                )
            )

        figura.update_layout(
            height=520,
            margin={"l": 0, "r": 0, "t": 20, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#a8a49a", "family": "Space Grotesk, sans-serif"},
            legend={"orientation": "h", "y": 0.02, "x": 0.02, "font": {"size": 11}},
            scene={
                "bgcolor": "rgba(0,0,0,0)",
                "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.1}},
                "xaxis": {
                    "tickvals": x_positions,
                    "ticktext": classes,
                    "showgrid": True,
                    "gridcolor": "#321a1c",
                    "zerolinecolor": "#572126",
                    "title": "Categoria",
                },
                "yaxis": {"visible": False, "range": [-0.3, 0.3]},
                "zaxis": {
                    "range": [0, 1.08],
                    "tickformat": ".0%",
                    "showgrid": True,
                    "gridcolor": "#321a1c",
                    "zerolinecolor": "#572126",
                    "title": "Confiança",
                },
            },
            showlegend=True,
        )
        st.plotly_chart(figura, use_container_width=True, config={"displaylogo": False})
