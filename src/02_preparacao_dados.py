"""
Objetivo 2 — Obtenção e Preparação dos Dados
=============================================
Carrega os dados brutos do TSE (2014, 2018, 2022), aplica todas as
transformações necessárias e salva o DataFrame limpo em dados_tratados/.

Etapas:
  1. Carregamento e unificação dos três anos
  2. Remoção de duplicatas
  3. Tratamento do marcador #NULO# (string falsa de nulo)
  4. Correção de tipos de dados
  5. Criação de novas features
  6. Exportação do dataset tratado
"""

import os
import glob
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "base_de_dados")
OUTPUT_DIR = os.path.join(ROOT, "dados_tratados")

READ_PARAMS = dict(
    sep=";",
    encoding="latin-1",
    on_bad_lines="skip",
    low_memory=False,
)

ANOS = [2014, 2018, 2022]

# Mapeamento de UF para região geográfica
REGIOES = {
    "AC": "Norte",     "AM": "Norte",     "AP": "Norte",
    "PA": "Norte",     "RO": "Norte",     "RR": "Norte",     "TO": "Norte",
    "AL": "Nordeste",  "BA": "Nordeste",  "CE": "Nordeste",
    "MA": "Nordeste",  "PB": "Nordeste",  "PE": "Nordeste",
    "PI": "Nordeste",  "RN": "Nordeste",  "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste",
    "MS": "Centro-Oeste", "MT": "Centro-Oeste",
    "ES": "Sudeste",   "MG": "Sudeste",   "RJ": "Sudeste",   "SP": "Sudeste",
    "PR": "Sul",       "RS": "Sul",       "SC": "Sul",
    "BR": "Nacional",  # presidente e vice-presidente
}

# Valores que o TSE usa como marcador de ausência (não são NaN reais)
VALORES_NULOS_TSE = ["#NULO#", "#NULO", "NAO DIVULGAVEL", "NAO INFORMADO"]

# Categorias de resultado que significam eleição bem-sucedida
RESULTADOS_ELEITO = ["ELEITO", "ELEITO POR QP", "ELEITO POR MEDIA"]


# ---------------------------------------------------------------------------
# ETAPA 1 — CARREGAMENTO E UNIFICAÇÃO
# ---------------------------------------------------------------------------

def carregar_ano(ano: int) -> pd.DataFrame:
    padrao = os.path.join(DATA_DIR, str(ano), "**", f"consulta_cand_{ano}_*.csv")
    arquivos = glob.glob(padrao, recursive=True)
    partes = []
    for caminho in arquivos:
        try:
            partes.append(pd.read_csv(caminho, **READ_PARAMS))
        except Exception as e:
            print(f"  AVISO [{ano}]: erro ao ler {os.path.basename(caminho)}: {e}")
    return pd.concat(partes, ignore_index=True)


def unificar_anos(anos: list) -> pd.DataFrame:
    """
    Carrega os três anos e mantém apenas as colunas que existem em TODOS eles.

    O motivo de fazer a interseção: 2022 tem 12 colunas a mais que 2014/2018.
    Se fizéssemos concat direto, essas colunas ficariam cheias de NaN para
    os outros anos, sujando a análise temporal.
    """
    dfs = {}
    for ano in anos:
        print(f"  Carregando {ano}...")
        dfs[ano] = carregar_ano(ano)
        print(f"    {dfs[ano].shape[0]:,} linhas x {dfs[ano].shape[1]} colunas")

    # Interseção das colunas: só o que existe nos três anos
    colunas_comuns = set(dfs[anos[0]].columns)
    for ano in anos[1:]:
        colunas_comuns &= set(dfs[ano].columns)

    colunas_comuns = sorted(colunas_comuns)
    print(f"\n  Colunas comuns entre os tres anos: {len(colunas_comuns)}")

    # Filtra cada DataFrame para manter só as colunas comuns e empilha
    partes = [dfs[ano][colunas_comuns] for ano in anos]
    df = pd.concat(partes, ignore_index=True)
    print(f"  Dataset unificado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
    return df


# ---------------------------------------------------------------------------
# ETAPA 2 — REMOÇÃO DE DUPLICATAS
# ---------------------------------------------------------------------------

def remover_duplicatas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Um candidato pode aparecer mais de uma vez quando há segundo turno:
    ex. governadores e presidentes têm uma linha por turno disputado.

    SQ_CANDIDATO é o código único por candidato dentro de uma eleição.
    Mantemos a linha do ÚLTIMO turno (NR_TURNO maior), que traz o resultado
    final. Para candidatos que só foram ao 1º turno, não há diferença.

    Além disso, removemos linhas completamente idênticas (duplicatas brutas).
    """
    antes = len(df)

    # Remove linhas 100% iguais primeiro
    df = df.drop_duplicates()

    # Para candidatos com mais de um turno: mantém o turno mais alto
    df = (
        df.sort_values("NR_TURNO", ascending=False)
          .drop_duplicates(subset=["ANO_ELEICAO", "SQ_CANDIDATO"], keep="first")
          .sort_index()
          .reset_index(drop=True)
    )

    depois = len(df)
    print(f"  Duplicatas removidas: {antes - depois:,} linhas ({antes:,} -> {depois:,})")
    return df


# ---------------------------------------------------------------------------
# ETAPA 3 — TRATAMENTO DO #NULO# (nulo falso do TSE)
# ---------------------------------------------------------------------------

def tratar_nulos_tse(df: pd.DataFrame) -> pd.DataFrame:
    """
    O TSE preenche campos sem informação com a string "#NULO#" em vez de
    deixar a célula vazia. Para o pandas, isso parece um valor válido —
    mas é na prática um dado ausente.

    Convertemos essas strings para NaN (o nulo real do pandas) e também
    substituímos -1 em colunas numéricas, que o TSE usa com o mesmo sentido.
    """
    antes = df.isnull().sum().sum()

    # Substitui as strings marcadoras por NaN em todo o DataFrame
    df = df.replace(VALORES_NULOS_TSE, np.nan)

    # Remove espaços extras e padroniza strings antes de checar novamente
    colunas_texto = df.select_dtypes(include="str").columns
    for col in colunas_texto:
        df[col] = df[col].str.strip()

    # Teto de campanha: -1 significa "não se aplica" (candidato inapto)
    if "VR_DESPESA_MAX_CAMPANHA" in df.columns:
        df["VR_DESPESA_MAX_CAMPANHA"] = df["VR_DESPESA_MAX_CAMPANHA"].replace(-1, np.nan)

    depois = df.isnull().sum().sum()
    print(f"  Nulos antes: {antes:,}  |  Nulos depois: {depois:,}  (+{depois - antes:,} NaN reais revelados)")
    return df


# ---------------------------------------------------------------------------
# ETAPA 4 — CORREÇÃO DE TIPOS
# ---------------------------------------------------------------------------

def corrigir_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Após a leitura, o pandas infere tipos automaticamente. Alguns precisam
    de ajuste:

    - VR_DESPESA_MAX_CAMPANHA: pode ter sido lida como object se havia
      vírgulas. Convertemos para float.
    - NR_IDADE_DATA_POSSE: deve ser inteiro, mas NaN força float no pandas
      (limitação histórica). Usamos Int64 (com I maiúsculo), que suporta NaN.
    - DT_NASCIMENTO / DT_ELEICAO: lidas como texto, convertemos para datetime.
    """
    # Teto de gastos: garante float numérico
    if "VR_DESPESA_MAX_CAMPANHA" in df.columns:
        df["VR_DESPESA_MAX_CAMPANHA"] = pd.to_numeric(
            df["VR_DESPESA_MAX_CAMPANHA"], errors="coerce"
        )

    # Idade: inteiro nullable (Int64 com I maiúsculo aceita NaN, int64 não aceita)
    if "NR_IDADE_DATA_POSSE" in df.columns:
        df["NR_IDADE_DATA_POSSE"] = pd.to_numeric(
            df["NR_IDADE_DATA_POSSE"], errors="coerce"
        ).astype("Int64")

    # Datas: converte texto para datetime, erros viram NaT (nulo de data)
    for col_data in ["DT_NASCIMENTO", "DT_ELEICAO"]:
        if col_data in df.columns:
            df[col_data] = pd.to_datetime(df[col_data], dayfirst=True, errors="coerce")

    print("  Tipos corrigidos: VR_DESPESA_MAX_CAMPANHA, NR_IDADE_DATA_POSSE, datas")
    return df


# ---------------------------------------------------------------------------
# ETAPA 5 — CRIAÇÃO DE NOVAS FEATURES
# ---------------------------------------------------------------------------

def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Novas colunas derivadas dos dados existentes. Chamamos de 'features'
    porque serão as variáveis usadas diretamente na análise e, se houver
    modelo de ML, no treinamento.
    """

    # -- ELEITO (0 ou 1) -------------------------------------------------------
    # DS_SIT_TOT_TURNO tem vários valores possíveis. Simplificamos:
    # 1 = qualquer forma de eleição; 0 = todo o resto.
    # Normalizamos antes de comparar para não depender de acentuação exata.
    resultado = df["DS_SIT_TOT_TURNO"].fillna("").str.upper().str.strip()
    resultado = resultado.str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
    df["ELEITO"] = resultado.isin(
        ["ELEITO", "ELEITO POR QP", "ELEITO POR MEDIA"]
    ).astype(int)

    # -- COR_RACA_AGRUPADA -----------------------------------------------------
    # Para análise de representatividade racial, a categoria mais usada é
    # "negra" = preta + parda (conforme classificação do IBGE).
    mapa_raca = {
        "BRANCA":   "Branca",
        "PARDA":    "Negra",
        "PRETA":    "Negra",
        "AMARELA":  "Outras",
        "INDIGENA": "Outras",
    }
    raca_normalizada = (
        df["DS_COR_RACA"]
        .fillna("")
        .str.upper()
        .str.strip()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
    )
    df["COR_RACA_AGRUPADA"] = raca_normalizada.map(mapa_raca).fillna("Nao informado")

    # -- REGIAO ----------------------------------------------------------------
    # Deriva a região geográfica a partir da sigla do estado (SG_UF).
    df["REGIAO"] = df["SG_UF"].map(REGIOES).fillna("Nao identificado")

    # -- FAIXA_ETARIA ----------------------------------------------------------
    # Agrupa idades em faixas para facilitar visualizações.
    bins   = [0,  29, 39, 49, 59, 69, 200]
    labels = ["ate 29", "30-39", "40-49", "50-59", "60-69", "70+"]
    df["FAIXA_ETARIA"] = pd.cut(
        df["NR_IDADE_DATA_POSSE"], bins=bins, labels=labels, right=True
    )

    print("  Features criadas: ELEITO, COR_RACA_AGRUPADA, REGIAO, FAIXA_ETARIA")
    return df


# ---------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PREPARACAO DOS DADOS TSE (2014 / 2018 / 2022)")
    print("=" * 60)

    # Etapa 1
    print("\n[1] Carregando e unificando anos...")
    df = unificar_anos(ANOS)

    # Etapa 2
    print("\n[2] Removendo duplicatas...")
    df = remover_duplicatas(df)

    # Etapa 3
    print("\n[3] Tratando valores nulos falsos (#NULO#)...")
    df = tratar_nulos_tse(df)

    # Etapa 4
    print("\n[4] Corrigindo tipos de dados...")
    df = corrigir_tipos(df)

    # Etapa 5
    print("\n[5] Criando novas features...")
    df = criar_features(df)

    # Sumario final antes de salvar
    print("\n" + "=" * 60)
    print("DATASET FINAL")
    print("=" * 60)
    print(f"  Linhas       : {df.shape[0]:,}")
    print(f"  Colunas      : {df.shape[1]}")
    print(f"  Nulos totais : {df.isnull().sum().sum():,}")
    print(f"\n  Distribuicao por ano:")
    print(df["ANO_ELEICAO"].value_counts().sort_index().to_string())
    print(f"\n  Distribuicao ELEITO:")
    print(df["ELEITO"].value_counts().to_string())
    print(f"\n  Distribuicao COR_RACA_AGRUPADA:")
    print(df["COR_RACA_AGRUPADA"].value_counts().to_string())

    # Etapa 6 — salva
    caminho_saida = os.path.join(OUTPUT_DIR, "candidatos_tratados.csv")
    df.to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    print(f"\n  Salvo em: {caminho_saida}")
    print("=" * 60)


if __name__ == "__main__":
    main()
