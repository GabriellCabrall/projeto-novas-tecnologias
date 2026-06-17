"""
Objetivo 3 — Dashboard de Representatividade Política
======================================================
Interface Streamlit com visualizações Matplotlib para análise
de representatividade de gênero, raça e perfil demográfico dos
candidatos nas eleições brasileiras de 2014, 2018 e 2022.

Execução:
    streamlit run src/03_dashboard.py
"""

import os
import unicodedata
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem janela — obrigatório para Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Representatividade Política BR",
    layout="wide",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS_PATH = os.path.join(ROOT, "dados_tratados", "candidatos_tratados.csv")

GENEROS_VALIDOS = ["FEMININO", "MASCULINO"]
RACAS_VALIDAS   = ["Branca", "Negra", "Outras"]
REGIOES_VALIDAS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

ORDEM_FAIXAS = ["ate 29", "30-39", "40-49", "50-59", "60-69", "70+"]

COR_FEMININO  = "#E91E8C"
COR_MASCULINO = "#1565C0"
CORES_RACA    = {"Branca": "#FFB74D", "Negra": "#5D4037", "Outras": "#78909C"}
CORES_REGIAO  = ["#FF7043", "#FFCA28", "#26A69A", "#5C6BC0", "#EC407A"]

# Normaliza string removendo acentos — usado para comparar DS_GRAU_INSTRUCAO
def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").upper().strip()

MAPA_INSTRUCAO = {
    _norm("Lê e Escreve"):                  "Lê e Escreve",
    _norm("Ensino Fundamental Incompleto"): "Fund. Incompleto",
    _norm("Ensino Fundamental Completo"):   "Fund. Completo",
    _norm("Ensino Médio Incompleto"):       "Médio Incompleto",
    _norm("Ensino Médio Completo"):         "Médio Completo",
    _norm("Superior Incompleto"):           "Superior Inc.",
    _norm("Superior Completo"):             "Superior Comp.",
}
ORDEM_INSTRUCAO = list(MAPA_INSTRUCAO.values())

CARGOS_FOCO = [
    "DEPUTADO ESTADUAL", "DEPUTADO FEDERAL",
    "DEPUTADO DISTRITAL", "SENADOR", "GOVERNADOR",
]

N_TOP_PARTIDOS  = 20
FEATURES_MODELO = [
    "DS_GENERO", "COR_RACA_AGRUPADA", "FAIXA_ETARIA",
    "INSTRUCAO", "REGIAO", "DS_CARGO", "PARTIDO_AGRUP",
]

# ---------------------------------------------------------------------------
# CARREGAMENTO E CACHE
# ---------------------------------------------------------------------------

@st.cache_data
def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(DADOS_PATH, encoding="utf-8-sig", low_memory=False)
    df["DS_GENERO"] = df["DS_GENERO"].str.strip()
    df["FAIXA_ETARIA"] = pd.Categorical(
        df["FAIXA_ETARIA"], categories=ORDEM_FAIXAS, ordered=True
    )
    return df

# ---------------------------------------------------------------------------
# FUNÇÕES DE GRÁFICO
# ---------------------------------------------------------------------------

def fig_evolucao_genero(df: pd.DataFrame) -> plt.Figure:
    """% de mulheres entre candidatas e eleitas por ano eleitoral."""
    anos = sorted(df["ANO_ELEICAO"].unique())
    sub  = df[df["DS_GENERO"].isin(GENEROS_VALIDOS)]

    pct_cand, pct_eleitas = [], []
    for ano in anos:
        s = sub[sub["ANO_ELEICAO"] == ano]
        pct_cand.append(100 * (s["DS_GENERO"] == "FEMININO").mean())
        sel = s[s["ELEITO"] == 1]
        pct_eleitas.append(
            100 * (sel["DS_GENERO"] == "FEMININO").mean() if len(sel) else 0
        )

    fig, ax = plt.subplots(figsize=(7, 4))
    x, w = np.arange(len(anos)), 0.35
    b1 = ax.bar(x - w / 2, pct_cand,   w, label="Candidatas", color=COR_FEMININO, alpha=0.75)
    b2 = ax.bar(x + w / 2, pct_eleitas, w, label="Eleitas",    color="#880E4F",    alpha=0.90)
    ax.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in anos])
    ax.set_ylabel("% de Mulheres")
    ax.set_title("Participação Feminina: Candidatas vs. Eleitas", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, max(max(pct_cand), max(pct_eleitas), 1) * 1.30)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_taxa_eleicao_genero_raca(df: pd.DataFrame) -> plt.Figure:
    """Taxa de eleição (% eleitos) por gênero e raça — análise cruzada."""
    sub = df[df["DS_GENERO"].isin(GENEROS_VALIDOS) & df["COR_RACA_AGRUPADA"].isin(RACAS_VALIDAS)]
    pivot = (
        sub.groupby(["DS_GENERO", "COR_RACA_AGRUPADA"])["ELEITO"]
        .mean()
        .mul(100)
        .unstack("COR_RACA_AGRUPADA")
        .reindex(index=GENEROS_VALIDOS, columns=RACAS_VALIDAS)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    x, w = np.arange(len(RACAS_VALIDAS)), 0.35
    b_f = ax.bar(x - w / 2, pivot.loc["FEMININO"],  w, label="Feminino",  color=COR_FEMININO,  alpha=0.85)
    b_m = ax.bar(x + w / 2, pivot.loc["MASCULINO"], w, label="Masculino", color=COR_MASCULINO, alpha=0.85)
    ax.bar_label(b_f, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(b_m, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(RACAS_VALIDAS)
    ax.set_ylabel("Taxa de Eleição (%)")
    ax.set_title("Taxa de Eleição por Gênero e Raça", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, pivot.values.max() * 1.30)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_representatividade_racial(df: pd.DataFrame) -> plt.Figure:
    """% de cada grupo racial entre candidatos vs eleitos."""
    sub_cand   = df[df["COR_RACA_AGRUPADA"].isin(RACAS_VALIDAS)]
    sub_eleito = sub_cand[sub_cand["ELEITO"] == 1]

    pct_c = [100 * (sub_cand["COR_RACA_AGRUPADA"] == g).mean()   for g in RACAS_VALIDAS]
    pct_e = [100 * (sub_eleito["COR_RACA_AGRUPADA"] == g).mean() for g in RACAS_VALIDAS]
    cores = [CORES_RACA[g] for g in RACAS_VALIDAS]

    fig, ax = plt.subplots(figsize=(7, 4))
    x, w = np.arange(len(RACAS_VALIDAS)), 0.35
    b1 = ax.bar(x - w / 2, pct_c, w, label="Candidatos", color=cores, alpha=0.65)
    b2 = ax.bar(x + w / 2, pct_e, w, label="Eleitos",    color=cores, alpha=1.00,
                edgecolor="black", linewidth=0.8)
    ax.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(RACAS_VALIDAS)
    ax.set_ylabel("% do Total")
    ax.set_title("Representação Racial: Candidatos vs. Eleitos", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, max(max(pct_c), max(pct_e)) * 1.25)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_evolucao_racial(df: pd.DataFrame) -> plt.Figure:
    """Evolução do % de candidatos e eleitos negros (preta + parda) por ano."""
    anos = sorted(df["ANO_ELEICAO"].unique())
    sub  = df[df["COR_RACA_AGRUPADA"].isin(["Branca", "Negra"])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (titulo, mask_eleito) in zip(axes, [("Candidatos", None), ("Eleitos", 1)]):
        base = sub if mask_eleito is None else sub[sub["ELEITO"] == mask_eleito]
        pcts = []
        for ano in anos:
            s = base[base["ANO_ELEICAO"] == ano]
            pcts.append(100 * (s["COR_RACA_AGRUPADA"] == "Negra").mean() if len(s) else 0)

        ax.plot([str(a) for a in anos], pcts, marker="o", linewidth=2.5,
                color="#5D4037", markersize=9)
        for ano, pct in zip(anos, pcts):
            ax.annotate(f"{pct:.1f}%", (str(ano), pct),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=10, fontweight="bold")
        ax.set_title(f"% de Negros entre {titulo}", fontweight="bold")
        ax.set_ylabel("% (Preta + Parda)")
        ax.set_ylim(0, max(pcts, default=1) * 1.40)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    fig.suptitle("Evolução da Representatividade Negra (2014–2022)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig


def fig_taxa_eleicao_faixa_genero(df: pd.DataFrame) -> plt.Figure:
    """Taxa de eleição por faixa etária e gênero — mostra se a idade afeta as chances."""
    sub = df[df["DS_GENERO"].isin(GENEROS_VALIDOS)]
    pivot = (
        sub.groupby(["FAIXA_ETARIA", "DS_GENERO"])["ELEITO"]
        .mean().mul(100)
        .unstack("DS_GENERO")
        .reindex(ORDEM_FAIXAS)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    x, w = np.arange(len(ORDEM_FAIXAS)), 0.35
    b_f = ax.bar(x - w / 2, pivot["FEMININO"],  w, label="Feminino",  color=COR_FEMININO,  alpha=0.85)
    b_m = ax.bar(x + w / 2, pivot["MASCULINO"], w, label="Masculino", color=COR_MASCULINO, alpha=0.85)
    ax.bar_label(b_f, fmt="%.1f%%", padding=3, fontsize=8)
    ax.bar_label(b_m, fmt="%.1f%%", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDEM_FAIXAS)
    ax.set_ylabel("Taxa de Eleição (%)")
    ax.set_title("Taxa de Eleição por Faixa Etária e Gênero", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, pivot.values.max() * 1.30)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_genero_por_cargo(df: pd.DataFrame) -> plt.Figure:
    """% feminino entre candidatos vs eleitos nos 5 principais cargos."""
    sub = df[df["DS_GENERO"].isin(GENEROS_VALIDOS) & df["DS_CARGO"].isin(CARGOS_FOCO)]

    pct_cand, pct_eleitas = [], []
    for cargo in CARGOS_FOCO:
        s = sub[sub["DS_CARGO"] == cargo]
        pct_cand.append(100 * (s["DS_GENERO"] == "FEMININO").mean() if len(s) else 0)
        sel = s[s["ELEITO"] == 1]
        pct_eleitas.append(100 * (sel["DS_GENERO"] == "FEMININO").mean() if len(sel) else 0)

    # Rótulos compactos para o eixo Y
    rotulos = [c.replace("DEPUTADO ", "DEP. ") for c in CARGOS_FOCO]

    fig, ax = plt.subplots(figsize=(7, 4))
    y, h = np.arange(len(CARGOS_FOCO)), 0.35
    b1 = ax.barh(y + h / 2, pct_cand,   h, label="Candidatas", color=COR_FEMININO, alpha=0.65)
    b2 = ax.barh(y - h / 2, pct_eleitas, h, label="Eleitas",    color="#880E4F",    alpha=0.90)
    ax.bar_label(b1, fmt="%.1f%%", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(rotulos)
    ax.set_xlabel("% de Mulheres")
    ax.set_title("Representação Feminina por Cargo", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, max(max(pct_cand), max(pct_eleitas), 1) * 1.30)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_instrucao_taxa_eleicao(df: pd.DataFrame) -> plt.Figure:
    """Taxa de eleição por grau de instrução — escolaridade como fator de sucesso."""
    df2 = df.copy()
    df2["_instrucao_norm"] = df2["DS_GRAU_INSTRUCAO"].apply(
        lambda v: MAPA_INSTRUCAO.get(_norm(str(v))) if pd.notna(v) else None
    )
    sub = df2[df2["_instrucao_norm"].isin(ORDEM_INSTRUCAO)]
    taxas = (
        sub.groupby("_instrucao_norm")["ELEITO"]
        .mean().mul(100)
        .reindex(ORDEM_INSTRUCAO)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    cores = plt.cm.Blues(np.linspace(0.35, 0.85, len(ORDEM_INSTRUCAO)))
    bars = ax.barh(ORDEM_INSTRUCAO, taxas.values, color=cores)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlabel("Taxa de Eleição (%)")
    ax.set_title("Taxa de Eleição por Grau de Instrução", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, taxas.max() * 1.30)
    fig.tight_layout()
    return fig


def fig_taxa_eleicao_regiao(df: pd.DataFrame) -> plt.Figure:
    """Taxa de eleição por região geográfica."""
    sub   = df[df["REGIAO"].isin(REGIOES_VALIDAS)]
    taxas = sub.groupby("REGIAO")["ELEITO"].mean().mul(100).reindex(REGIOES_VALIDAS)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(REGIOES_VALIDAS, taxas.values, color=CORES_REGIAO, alpha=0.85)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    ax.set_xlabel("Taxa de Eleição (%)")
    ax.set_title("Taxa de Eleição por Região", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, taxas.max() * 1.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MODELO DE ML — SIMULADOR DE CHANCES DE ELEIÇÃO
# ---------------------------------------------------------------------------

@st.cache_resource
def treinar_modelo():
    """
    Treina Regressão Logística para prever ELEITO a partir do perfil do candidato.
    Decorado com cache_resource: treina uma única vez por sessão do servidor.
    """
    df = pd.read_csv(DADOS_PATH, encoding="utf-8-sig", low_memory=False)
    df["DS_GENERO"] = df["DS_GENERO"].str.strip()
    df["INSTRUCAO"] = df["DS_GRAU_INSTRUCAO"].apply(
        lambda v: MAPA_INSTRUCAO.get(_norm(str(v))) if pd.notna(v) else None
    )
    top_p = df["SG_PARTIDO"].value_counts().head(N_TOP_PARTIDOS).index.tolist()
    df["PARTIDO_AGRUP"] = df["SG_PARTIDO"].apply(
        lambda p: p if p in top_p else "OUTROS"
    )

    sub = df[FEATURES_MODELO + ["ELEITO"]].dropna().copy()
    for col in FEATURES_MODELO:
        sub[col] = sub[col].astype(str)

    X = pd.get_dummies(sub[FEATURES_MODELO])
    y = sub["ELEITO"]
    feature_cols = X.columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    modelo = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    modelo.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, modelo.predict(X_te))
    auc = roc_auc_score(y_te, modelo.predict_proba(X_te)[:, 1])

    return modelo, feature_cols, top_p, acc, auc, len(X_tr)


def _perfil_para_x(genero, raca, faixa, instrucao, regiao, cargo, partido, feature_cols):
    """Constrói um vetor de features para uma linha do formulário."""
    row = {
        "DS_GENERO": genero, "COR_RACA_AGRUPADA": raca,
        "FAIXA_ETARIA": faixa, "INSTRUCAO": instrucao,
        "REGIAO": regiao, "DS_CARGO": cargo, "PARTIDO_AGRUP": partido,
    }
    X = pd.get_dummies(pd.DataFrame([row]).astype(str))
    return X.reindex(columns=feature_cols, fill_value=0)


def fig_probabilidades_cargos(probs: dict, genero: str, raca: str) -> plt.Figure:
    """Barras horizontais com a probabilidade prevista para cada cargo."""
    ordem = sorted(probs, key=probs.get, reverse=True)
    valores = [probs[c] for c in ordem]
    rotulos = [c.replace("DEPUTADO ", "DEP. ") for c in ordem]

    fig, ax = plt.subplots(figsize=(8, 4))
    cores = plt.cm.RdYlGn(np.linspace(0.25, 0.75, len(valores)))
    bars = ax.barh(rotulos, valores, color=cores)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)
    ax.set_xlabel("Probabilidade de Eleição (%)")
    ax.set_title(
        f"Probabilidade de Eleição por Cargo\n({genero.capitalize()}, {raca})",
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlim(0, max(valores, default=1) * 1.40)
    fig.tight_layout()
    return fig


def fig_contexto_historico(df_full: pd.DataFrame, probs: dict,
                            genero: str, raca: str) -> plt.Figure:
    """
    Compara a previsão do modelo com as taxas históricas reais:
    - taxa do grupo (mesmo gênero + raça)
    - taxa geral de todos os candidatos
    """
    sub   = df_full[df_full["DS_CARGO"].isin(CARGOS_FOCO) &
                    df_full["DS_GENERO"].isin(GENEROS_VALIDOS)]
    grupo = sub[(sub["DS_GENERO"] == genero) & (sub["COR_RACA_AGRUPADA"] == raca)]

    t_grupo = sub.groupby("DS_CARGO")["ELEITO"].mean().mul(100)
    t_geral = grupo.groupby("DS_CARGO")["ELEITO"].mean().mul(100)

    fig, ax = plt.subplots(figsize=(10, 4))
    x, w = np.arange(len(CARGOS_FOCO)), 0.26
    rotulos = [c.replace("DEPUTADO ", "DEP. ") for c in CARGOS_FOCO]

    v_modelo = [probs.get(c, 0) for c in CARGOS_FOCO]
    v_grupo  = [t_geral.get(c, 0) for c in CARGOS_FOCO]
    v_geral  = [t_grupo.get(c, 0) for c in CARGOS_FOCO]

    cor_perfil = COR_FEMININO if genero == "FEMININO" else COR_MASCULINO
    b1 = ax.bar(x - w,     v_modelo, w, label="Modelo (previsão)",
                color="#7B1FA2", alpha=0.90)
    b2 = ax.bar(x,         v_grupo,  w, label=f"Histórico ({genero[:3].title()}. {raca})",
                color=cor_perfil, alpha=0.75)
    b3 = ax.bar(x + w,     v_geral,  w, label="Histórico geral",
                color="#90A4AE", alpha=0.75)

    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(rotulos)
    ax.set_ylabel("Taxa / Probabilidade (%)")
    ax.set_title("Previsão do Modelo vs. Taxas Históricas Reais", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, max(max(v_modelo), max(v_grupo), max(v_geral), 1) * 1.40)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# LAYOUT PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    st.title("Representatividade Política nas Eleições Brasileiras")
    st.markdown("**Eleições Gerais de 2014, 2018 e 2022 — Dados do TSE**")
    st.markdown("---")

    df_full = carregar_dados()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filtros")

        anos_sel = st.multiselect(
            "Ano da Eleição",
            options=sorted(df_full["ANO_ELEICAO"].unique()),
            default=sorted(df_full["ANO_ELEICAO"].unique()),
        )

        cargos_sel = st.multiselect(
            "Cargo",
            options=sorted(df_full["DS_CARGO"].dropna().unique()),
            default=sorted(df_full["DS_CARGO"].dropna().unique()),
        )

        regioes_sel = st.multiselect(
            "Região",
            options=REGIOES_VALIDAS,
            default=REGIOES_VALIDAS,
        )

        st.markdown("---")
        st.caption("Fonte: TSE — Repositório de Dados Eleitorais")
        st.caption("Disciplina: Novas Tecnologias — UCB")

    # Aplica filtros globais
    df = df_full[
        df_full["ANO_ELEICAO"].isin(anos_sel) &
        df_full["DS_CARGO"].isin(cargos_sel) &
        df_full["REGIAO"].isin(regioes_sel)
    ].copy()

    if df.empty:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
        return

    # ── MÉTRICAS DE TOPO ─────────────────────────────────────────────────────
    total       = len(df)
    eleitos     = int(df["ELEITO"].sum())
    pct_fem     = 100 * (df["DS_GENERO"] == "FEMININO").sum() / total
    pct_neg_el  = (
        100 * ((df["ELEITO"] == 1) & (df["COR_RACA_AGRUPADA"] == "Negra")).sum()
        / max(eleitos, 1)
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Candidatos", f"{total:,}")
    c2.metric("Total de Eleitos",    f"{eleitos:,}")
    c3.metric("% Candidatas Mulheres", f"{pct_fem:.1f}%")
    c4.metric("% Eleitos Negros",     f"{pct_neg_el:.1f}%")

    st.markdown("---")

    # ── ABAS ─────────────────────────────────────────────────────────────────
    tab_genero, tab_raca, tab_perfil, tab_simulador = st.tabs(
        ["Gênero", "Raça", "Perfil Demográfico", "Simulador"]
    )

    with tab_genero:
        st.subheader("Representatividade de Gênero")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_evolucao_genero(df))
        with col2:
            st.pyplot(fig_taxa_eleicao_genero_raca(df))

        st.info(
            "**Leitura:** A diferença entre % de candidatas e % de eleitas indica "
            "a sub-representação feminina. A análise cruzada (gênero × raça) mostra "
            "que mulheres negras enfrentam a maior barreira de eleição."
        )

    with tab_raca:
        st.subheader("Representatividade Racial")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_representatividade_racial(df))
        with col2:
            st.pyplot(fig_taxa_eleicao_regiao(df))

        st.pyplot(fig_evolucao_racial(df))

        st.info(
            "**Leitura:** A comparação candidatos vs. eleitos revela se grupos raciais "
            "são proporcionalmente representados nos mandatos. O gráfico de evolução "
            "mostra a tendência histórica de 2014 a 2022."
        )

    with tab_perfil:
        st.subheader("Perfil Demográfico dos Candidatos")

        st.pyplot(fig_taxa_eleicao_faixa_genero(df))
        st.info(
            "**Leitura:** A taxa de eleição cresce com a idade para ambos os gêneros — "
            "candidatos mais velhos têm mais capital político acumulado. "
            "A diferença entre masculino e feminino se mantém em todas as faixas, "
            "indicando que a barreira de gênero não depende da idade."
        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_genero_por_cargo(df))
            st.info(
                "**Leitura:** Compara o percentual de candidatas com o de eleitas "
                "em cada cargo. Uma barra de eleitas menor que a de candidatas indica "
                "que mulheres convertem menos em vitórias naquele cargo."
            )
        with col2:
            st.pyplot(fig_instrucao_taxa_eleicao(df))
            st.info(
                "**Leitura:** Candidatos com superior completo têm taxa de eleição "
                "substancialmente maior. Reflete tanto exigências do cargo quanto "
                "acesso desigual ao capital político e financeiro de campanha."
            )


    with tab_simulador:
        st.subheader("Simulador de Chances de Eleição")
        st.markdown(
            "Preencha seu perfil e veja a probabilidade estimada de eleição "
            "em cada cargo, com base nos padrões históricos de 2014–2022."
        )
        st.warning(
            "**Atenção:** O modelo aprendeu padrões históricos dos dados do TSE. "
            "Probabilidades baixas para certos grupos refletem desigualdades "
            "estruturais do sistema eleitoral — não são uma sentença sobre potencial individual."
        )

        with st.spinner("Treinando modelo de Regressão Logística..."):
            modelo, feature_cols, top_partidos, acc, auc, n_treino = treinar_modelo()

        with st.expander("Métricas do modelo"):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Acurácia (teste 20%)", f"{acc * 100:.1f}%")
            mc2.metric("AUC-ROC",              f"{auc:.3f}")
            mc3.metric("Registros no treino",  f"{n_treino:,}")
            st.caption(
                "Regressão Logística treinada em 80% dos dados e avaliada nos 20% restantes. "
                "AUC-ROC: 0.5 = aleatório · 1.0 = perfeito."
            )

        st.markdown("---")
        st.markdown("#### Seu perfil")

        col1, col2, col3 = st.columns(3)
        with col1:
            genero    = st.selectbox("Gênero",            GENEROS_VALIDOS)
            raca      = st.selectbox("Cor/Raça",           RACAS_VALIDAS)
        with col2:
            faixa     = st.selectbox("Faixa Etária",       ORDEM_FAIXAS, index=2)
            instrucao = st.selectbox("Grau de Instrução",  ORDEM_INSTRUCAO, index=6)
        with col3:
            regiao    = st.selectbox("Região",             REGIOES_VALIDAS)
            partido   = st.selectbox("Partido",
                                     sorted(top_partidos) + ["OUTROS"])

        st.markdown("---")

        # Calcula probabilidade para cada cargo mantendo o perfil fixo
        probs = {
            cargo: round(
                modelo.predict_proba(
                    _perfil_para_x(genero, raca, faixa, instrucao,
                                   regiao, cargo, partido, feature_cols)
                )[0, 1] * 100,
                2,
            )
            for cargo in CARGOS_FOCO
        }

        melhor = max(probs, key=probs.get)
        st.markdown(
            f"#### Maior chance estimada: **{melhor}** — **{probs[melhor]:.2f}%**"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.pyplot(fig_probabilidades_cargos(probs, genero, raca))
        with col_b:
            st.pyplot(fig_contexto_historico(df_full, probs, genero, raca))

        st.info(
            "**Como ler o segundo gráfico:** Roxo = o que o modelo prevê para você. "
            "Colorido = taxa histórica real de candidatos com o mesmo gênero e raça. "
            "Cinza = taxa histórica de todos os candidatos para aquele cargo. "
            "Quando roxo ≈ colorido, o modelo está bem calibrado para o seu perfil."
        )


if __name__ == "__main__":
    main()
