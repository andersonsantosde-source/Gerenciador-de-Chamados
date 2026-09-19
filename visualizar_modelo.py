from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.graph_objects as go
import spacy
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "modelo_chamados.pkl"
MIN_TEXT_LENGTH = 8
EXAMPLES = {
    "Teams não abre": "O Teams não abre no computador e fecha sozinho.",
    "Notebook superaquecendo": "Meu notebook está superaquecendo e desligando durante o trabalho.",
    "Acesso ao AD": "Preciso de permissão de administrador no AD para instalar um aplicativo.",
    "Rede instável": "A conexão Wi-Fi está com alta latência e cai durante as reuniões.",
}

st.set_page_config(
    page_title="Gerenciador de Chamados Inteligente",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #08090b;
        --panel: #111317;
        --panel-2: #171a20;
        --line: #292e38;
        --text: #f4f5f7;
        --muted: #9ba3b2;
        --red: #e50914;
        --red-bright: #ff5a63;
        --cyan: #37d5ff;
        --amber: #ffb000;
        --violet: #b784ff;
    }

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp {
        color: var(--text);
        background: radial-gradient(circle at 90% 0%, rgba(229, 9, 20, .13), transparent 27rem),
            linear-gradient(180deg, #15171b 0%, var(--bg) 34%, #050506 100%);
    }
    .block-container { max-width: 1280px; padding: 2.7rem 2rem 4rem; }
    .hero { border-bottom: 1px solid var(--line); padding-bottom: 1.6rem; }
    .eyebrow { color: var(--red-bright); font-size: .72rem; font-weight: 700; letter-spacing: .2em; text-transform: uppercase; }
    .hero h1 { color: var(--text); font-size: clamp(2.1rem, 5vw, 4.6rem); letter-spacing: -.04em; line-height: .95; margin: .55rem 0 .8rem; }
    .hero p { color: var(--muted); font-size: 1rem; margin: 0; max-width: 700px; }
    .status-line { align-items: center; color: var(--muted); display: flex; font-size: .73rem; gap: .55rem; letter-spacing: .1em; margin-top: 1.35rem; text-transform: uppercase; }
    .status-dot { background: #36db8b; border-radius: 50%; box-shadow: 0 0 12px #36db8b; height: 8px; width: 8px; }
    .section-label { color: var(--muted); font-size: .72rem; font-weight: 700; letter-spacing: .15em; margin: 1.9rem 0 .75rem; text-transform: uppercase; }
    .metric-card { background: linear-gradient(145deg, rgba(29,32,39,.96), rgba(13,14,17,.96)); border: 1px solid var(--line); border-top: 2px solid var(--accent, var(--red)); min-height: 100px; padding: 1rem 1.1rem; }
    .metric-label { color: var(--muted); font-size: .73rem; letter-spacing: .08em; text-transform: uppercase; }
    .metric-value { color: var(--text); font-size: 1.55rem; font-weight: 700; margin-top: .45rem; overflow-wrap: anywhere; }
    .metric-note { color: var(--muted); font-size: .74rem; margin-top: .2rem; }
    .input-panel { background: rgba(17,19,23,.75); border: 1px solid var(--line); padding: 1rem 1.1rem .65rem; }
    [data-testid="stTextArea"] textarea { background: #0a0c0f; border: 1px solid #343a46; border-radius: 3px; color: var(--text); font-size: 1rem; line-height: 1.5; }
    [data-testid="stTextArea"] textarea:focus { border-color: var(--red-bright); box-shadow: 0 0 0 1px var(--red); }
    [data-testid="stTextArea"] label, [data-testid="stSelectbox"] label { color: var(--text) !important; font-weight: 600; }
    .stButton > button { border-radius: 3px; font-weight: 700; min-height: 2.75rem; }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, #ff4650, var(--red) 55%, #940009); border: 1px solid #ff7c82; color: white; }
    .stButton > button:hover { border-color: var(--red-bright); color: white; }
    .result-panel { background: linear-gradient(135deg, rgba(74,10,14,.42), rgba(18,20,25,.96) 55%); border: 1px solid #633039; border-left: 4px solid var(--red); padding: 1.2rem 1.3rem; }
    .result-kicker { color: var(--red-bright); font-size: .7rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase; }
    .result-category { color: var(--text); font-size: clamp(1.8rem, 4vw, 3rem); font-weight: 700; line-height: 1.05; margin-top: .45rem; }
    .result-description { color: var(--muted); font-size: .86rem; margin-top: .55rem; }
    [data-testid="stAlert"] { border-radius: 3px; }
    [data-testid="stPlotlyChart"] { background: #0a0c0f; border: 1px solid var(--line); padding: .2rem; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); }
    code { color: var(--red-bright) !important; }
    @media (max-width: 700px) {
        .block-container { padding: 1.5rem 1rem 3rem; }
        .hero h1 { font-size: 2.4rem; }
        .status-line { align-items: flex-start; flex-direction: column; gap: .3rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Carregando modelo e processamento de linguagem...")
def carregar_recursos() -> tuple[Any, Any]:
    """Carrega uma única vez o pipeline persistido e o modelo spaCy."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH.name}.")
    try:
        modelo = joblib.load(MODEL_PATH)
    except Exception as erro:
        raise RuntimeError("O arquivo do modelo está corrompido ou é incompatível.") from erro
    try:
        nlp = spacy.load("pt_core_news_sm")
    except OSError as erro:
        raise RuntimeError("O modelo spaCy pt_core_news_sm não está instalado.") from erro
    if not hasattr(modelo, "predict") or not hasattr(modelo, "predict_proba"):
        raise TypeError("O arquivo carregado não possui uma interface de classificação compatível.")
    return modelo, nlp


def preprocessar_texto(texto: str, nlp: Any) -> str:
    """Mantém o mesmo lematizador usado no treinamento do modelo."""
    doc = nlp(texto)
    return " ".join(token.lemma_.lower() for token in doc if not token.is_punct)


def obter_classes(modelo: Any) -> list[str]:
    classes = getattr(modelo, "classes_", None)
    if classes is None and hasattr(modelo, "named_steps"):
        ultimo_estagio = list(modelo.named_steps.values())[-1]
        classes = getattr(ultimo_estagio, "classes_", None)
    if classes is None:
        raise ValueError("Não foi possível identificar as categorias do modelo.")
    return [str(classe) for classe in classes]


def classificar_chamado(texto: str, modelo: Any, nlp: Any) -> pd.DataFrame:
    texto_processado = preprocessar_texto(texto, nlp)
    categoria = str(modelo.predict([texto_processado])[0])
    probabilidades = modelo.predict_proba([texto_processado])[0]
    classes = obter_classes(modelo)
    resultado = pd.DataFrame({"Categoria": classes, "Probabilidade": [float(valor) for valor in probabilidades]})
    resultado = resultado.sort_values("Probabilidade", ascending=False).reset_index(drop=True)
    resultado["Percentual"] = resultado["Probabilidade"].map(lambda valor: f"{valor:.2%}")
    resultado.attrs["categoria"] = categoria
    return resultado


def criar_grafico_3d(resultado: pd.DataFrame) -> go.Figure:
    """Cria colunas 3D reais com eixo Y visível e escala de 0 a 100%."""
    paleta = ["#e50914", "#ffb000", "#37d5ff", "#b784ff", "#36db8b", "#ff7a45"]
    figura = go.Figure()
    largura = 0.58
    profundidade = 0.48

    def adicionar_coluna(indice: int, altura: float, cor: str, categoria: str) -> None:
        x0, x1 = indice - largura / 2, indice + largura / 2
        y0, y1 = -profundidade / 2, profundidade / 2
        vertices = [(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0), (x0, y0, altura), (x1, y0, altura), (x1, y1, altura), (x0, y1, altura)]
        indices_x, indices_y, indices_z = zip(*vertices)
        faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
        face_i, face_j, face_k = zip(*faces)
        figura.add_trace(go.Mesh3d(x=indices_x, y=indices_y, z=indices_z, i=face_i, j=face_j, k=face_k, color=cor, opacity=.92, flatshading=True, name=categoria, hovertemplate=f"<b>{categoria}</b><br>Probabilidade estimada: {altura:.2%}<extra></extra>"))
        figura.add_trace(go.Scatter3d(x=[indice], y=[0], z=[min(altura + .045, 1.04)], mode="text", text=[f"{altura:.1%}"], textfont={"color": "#f4f5f7", "size": 12}, showlegend=False, hoverinfo="skip"))

    for indice, linha in resultado.iterrows():
        adicionar_coluna(indice, float(linha["Probabilidade"]), paleta[indice % len(paleta)], str(linha["Categoria"]))

    nomes = resultado["Categoria"].tolist()
    abreviados = [nome if len(nome) <= 13 else f"{nome[:11]}…" for nome in nomes]
    figura.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c9ced8", "family": "Space Grotesk, sans-serif"},
        legend={"orientation": "h", "y": 1.02, "x": 0, "font": {"size": 11}},
        scene={
            "bgcolor": "rgba(0,0,0,0)",
            "aspectmode": "manual",
            "aspectratio": {"x": 1.7, "y": .75, "z": 1.1},
            "camera": {"eye": {"x": 1.65, "y": 1.55, "z": 1.1}},
            "xaxis": {"title": "Categoria", "tickvals": list(range(len(nomes))), "ticktext": abreviados, "showgrid": True, "gridcolor": "#303642", "zerolinecolor": "#59616e"},
            "yaxis": {"title": "Profundidade", "range": [-.55, .55], "showgrid": True, "gridcolor": "#252a33", "zerolinecolor": "#59616e", "showticklabels": False},
            "zaxis": {"title": "Probabilidade estimada", "range": [0, 1.08], "tickformat": ".0%", "showgrid": True, "gridcolor": "#303642", "zerolinecolor": "#59616e"},
        },
    )
    return figura


def criar_grafico_2d(resultado: pd.DataFrame) -> go.Figure:
    cores = ["#e50914"] + ["#59616e"] * max(len(resultado) - 1, 0)
    figura = go.Figure(go.Bar(x=resultado["Probabilidade"], y=resultado["Categoria"], orientation="h", marker_color=cores, text=resultado["Percentual"], textposition="outside", hovertemplate="<b>%{y}</b><br>Probabilidade estimada: %{x:.2%}<extra></extra>"))
    figura.update_layout(height=280, margin={"l": 0, "r": 35, "t": 10, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis={"range": [0, 1.12], "tickformat": ".0%", "gridcolor": "#303642", "title": "Probabilidade estimada"}, yaxis={"autorange": "reversed", "title": None}, font={"color": "#c9ced8", "family": "Space Grotesk, sans-serif"})
    return figura


def exportar_resultado(resultado: pd.DataFrame) -> bytes:
    return resultado[["Categoria", "Probabilidade", "Percentual"]].to_csv(index=False).encode("utf-8-sig")


def carregar_exemplo(nome: str) -> None:
    if nome in EXAMPLES:
        st.session_state.descricao = EXAMPLES[nome]


def limpar_estado() -> None:
    st.session_state.descricao = ""
    st.session_state.resultado = None


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Service desk · inteligência operacional</div>
        <h1>Gerenciador de Chamados Inteligente</h1>
        <p>Classificação automática de chamados de TI com leitura transparente das probabilidades estimadas pelo modelo.</p>
        <div class="status-line"><span class="status-dot"></span><span>Modelo operacional · TF-IDF + MultinomialNB · versão não informada no artefato</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)

if "classificacoes" not in st.session_state:
    st.session_state.classificacoes = 0
if "resultado" not in st.session_state:
    st.session_state.resultado = None

try:
    modelo, nlp = carregar_recursos()
    categorias = obter_classes(modelo)
    modelo_status = "Operacional"
except Exception as erro:
    modelo = nlp = None
    categorias = []
    modelo_status = "Indisponível"
    st.error(f"Não foi possível preparar o classificador: {erro}")

resultado_atual = st.session_state.resultado
categoria_atual = str(resultado_atual.iloc[0]["Categoria"]) if resultado_atual is not None else "Aguardando"
confianca_atual = resultado_atual.iloc[0]["Percentual"] if resultado_atual is not None else "—"

st.markdown('<div class="section-label">Visão geral da sessão</div>', unsafe_allow_html=True)
metricas = st.columns(4)
metricas[0].markdown(f'<div class="metric-card" style="--accent:#36db8b"><div class="metric-label">Status do modelo</div><div class="metric-value">{modelo_status}</div><div class="metric-note">Pipeline carregado</div></div>', unsafe_allow_html=True)
metricas[1].markdown(f'<div class="metric-card" style="--accent:#e50914"><div class="metric-label">Categoria identificada</div><div class="metric-value">{categoria_atual}</div><div class="metric-note">Última análise</div></div>', unsafe_allow_html=True)
metricas[2].markdown(f'<div class="metric-card" style="--accent:#ffb000"><div class="metric-label">Probabilidade estimada</div><div class="metric-value">{confianca_atual}</div><div class="metric-note">Não é garantia de acerto</div></div>', unsafe_allow_html=True)
metricas[3].markdown(f'<div class="metric-card" style="--accent:#37d5ff"><div class="metric-label">Categorias disponíveis</div><div class="metric-value">{len(categorias) or "—"}</div><div class="metric-note">Classificações conhecidas</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Novo chamado</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="input-panel">', unsafe_allow_html=True)
    exemplo = st.selectbox("Exemplos para teste", ["Selecione um exemplo"] + list(EXAMPLES), label_visibility="collapsed")
    acoes = st.columns([1, 1, 4])
    acoes[0].button("Usar exemplo", use_container_width=True, disabled=exemplo == "Selecione um exemplo", on_click=carregar_exemplo, args=(exemplo,))
    acoes[1].button("Limpar", use_container_width=True, on_click=limpar_estado)
    texto = st.text_area("Descrição do chamado", key="descricao", height=145, placeholder="Ex.: O Teams fecha sozinho ao abrir e preciso participar de uma reunião.")
    classificar = st.button("Classificar chamado", type="primary", use_container_width=True, disabled=modelo is None)
    st.markdown('</div>', unsafe_allow_html=True)

if classificar:
    if len(texto.strip()) < MIN_TEXT_LENGTH:
        st.warning(f"Descreva o problema com pelo menos {MIN_TEXT_LENGTH} caracteres para gerar uma análise útil.")
    else:
        try:
            with st.spinner("Analisando descrição e calculando probabilidades..."):
                st.session_state.resultado = classificar_chamado(texto.strip(), modelo, nlp)
            st.session_state.classificacoes += 1
            st.rerun()
        except Exception as erro:
            st.error(f"Não foi possível classificar o chamado: {erro}")

resultado_atual = st.session_state.resultado
if resultado_atual is not None:
    categoria = str(resultado_atual.iloc[0]["Categoria"])
    probabilidade = float(resultado_atual.iloc[0]["Probabilidade"])
    segunda = resultado_atual.iloc[1] if len(resultado_atual) > 1 else None
    diferenca = probabilidade - float(segunda["Probabilidade"]) if segunda is not None else None
    st.markdown('<div class="section-label">Resultado da análise</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-panel"><div class="result-kicker">Categoria mais provável</div><div class="result-category">{categoria}</div><div class="result-description">Probabilidade estimada: <strong>{probabilidade:.2%}</strong> · esta medida representa a saída do modelo e não uma garantia de acerto.</div></div>', unsafe_allow_html=True)
    if probabilidade < .60:
        st.warning("A probabilidade estimada é baixa. Revise a descrição ou encaminhe o chamado para triagem manual.")
    elif segunda is not None:
        st.info(f"A primeira categoria supera {segunda['Categoria']} em {diferenca:.2%} de probabilidade estimada.")

    esquerda, direita = st.columns([1.55, 1], gap="large")
    with esquerda:
        st.markdown("#### Mapa 3D de probabilidades")
        st.caption("Arraste para girar · scroll para aproximar · use os controles do gráfico para explorar.")
        st.plotly_chart(criar_grafico_3d(resultado_atual), use_container_width=True, config={"displaylogo": False, "responsive": True})
    with direita:
        st.markdown("#### Comparação rápida")
        st.plotly_chart(criar_grafico_2d(resultado_atual), use_container_width=True, config={"displaylogo": False, "responsive": True})
        st.download_button("Baixar probabilidades em CSV", exportar_resultado(resultado_atual), "resultado_chamado.csv", "text/csv", use_container_width=True)

    st.markdown("#### Detalhamento por categoria")
    tabela = resultado_atual[["Categoria", "Percentual"]].rename(columns={"Categoria": "Categoria", "Percentual": "Probabilidade estimada"})
    st.dataframe(tabela, hide_index=True, use_container_width=True)

with st.expander("Detalhes técnicos", expanded=False):
    st.write(f"Sessões classificadas nesta execução: `{st.session_state.classificacoes}`")
    st.write("Pré-processamento: lematização em português com `pt_core_news_sm`, igual ao fluxo de treinamento.")
