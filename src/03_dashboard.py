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
    tab_genero, tab_raca, tab_perfil = st.tabs(
        ["Gênero", "Raça", "Perfil Demográfico"]
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


if __name__ == "__main__":
    main()
