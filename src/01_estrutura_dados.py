"""
Objetivo 1 — Análise de Estrutura dos Dados
============================================
Este script carrega uma amostra representativa dos dados do TSE (2014, 2018 e 2022)
e registra em DataFrames a estrutura, tipos de colunas, valores nulos e
distribuição de colunas categóricas de interesse para a análise de representatividade.

Todos os resultados são salvos em outputs/dados_processados/.
"""

import os
import glob
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------

# Raiz do projeto — sobe um nível a partir de /src
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "base_de_dados")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "dados_processados")

# Parâmetros de leitura dos CSVs do TSE
# sep=';'    → separador padrão dos arquivos do TSE
# encoding   → Latin-1 / ISO-8859-1, padrão dos sistemas legados brasileiros
# on_bad_lines → ignora linhas malformadas sem interromper a leitura
READ_PARAMS = dict(
    sep=";",
    encoding="latin-1",
    on_bad_lines="skip",
    low_memory=False,   # evita inferência incorreta de tipos em arquivos grandes
)

ANOS = [2014, 2018, 2022]

# Colunas que nos interessam para a análise de representatividade
# (serão usadas nas etapas seguintes; aqui servem para avaliar qualidade)
COLUNAS_FOCO = [
    "ANO_ELEICAO",
    "SG_UF",
    "DS_CARGO",
    "DS_GENERO",
    "DS_COR_RACA",
    "DS_GRAU_INSTRUCAO",
    "DS_ESTADO_CIVIL",
    "NR_IDADE_DATA_POSSE",
    "DS_OCUPACAO",
    "SG_PARTIDO",
    "DS_SIT_TOT_TURNO",      # resultado: eleito, não eleito, suplente...
    "ST_REELEICAO",
    "VR_DESPESA_MAX_CAMPANHA",
    "DS_SITUACAO_CANDIDATURA",
]


# ---------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ---------------------------------------------------------------------------

def carregar_arquivos_ano(ano: int) -> pd.DataFrame:
    """
    Carrega e concatena todos os CSVs de um determinado ano eleitoral.

    Para 2022 existe um arquivo nacional (consulta_cand_2022_BR.csv); nesse
    caso usamos ele diretamente para evitar duplicatas.
    Para 2014 e 2018 os dados estão divididos por estado — carregamos todos
    e empilhamos num único DataFrame.
    """
    # Arquivo nacional do TSE (existe apenas em 2022)
    arquivo_nacional = os.path.join(DATA_DIR, str(ano), f"consulta_cand_{ano}_BR.csv")

    if os.path.exists(arquivo_nacional):
        print(f"  [{ano}] Carregando arquivo nacional...")
        df = pd.read_csv(arquivo_nacional, **READ_PARAMS)
    else:
        # Busca todos os CSVs dentro da pasta do ano recursivamente
        padrao = os.path.join(DATA_DIR, str(ano), "**", f"consulta_cand_{ano}_*.csv")
        arquivos = glob.glob(padrao, recursive=True)

        print(f"  [{ano}] {len(arquivos)} arquivos estaduais encontrados — concatenando...")

        partes = []
        for caminho in arquivos:
            try:
                partes.append(pd.read_csv(caminho, **READ_PARAMS))
            except Exception as e:
                print(f"    AVISO: erro ao ler {caminho}: {e}")

        df = pd.concat(partes, ignore_index=True)

    print(f"  [{ano}] Shape: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    return df


def resumo_estrutura(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    """
    Gera um DataFrame com a estrutura de cada coluna:
    - nome da coluna
    - tipo de dado inferido pelo Pandas
    - quantidade de valores nulos
    - percentual de nulos
    - quantidade de valores únicos

    Isso nos permite identificar rapidamente:
    * colunas com muitos nulos (podem precisar de imputação ou remoção)
    * colunas categóricas vs numéricas
    * colunas que têm valor único (sem variância — inúteis para análise)
    """
    resumo = pd.DataFrame({
        "coluna": df.columns,
        "tipo_pandas": df.dtypes.values.astype(str),
        "nulos": df.isnull().sum().values,
        "pct_nulos": (df.isnull().mean().values * 100).round(2),
        "valores_unicos": df.nunique().values,
    })
    resumo.insert(0, "ano", ano)
    return resumo


def distribuicao_categorica(df: pd.DataFrame, coluna: str, ano: int) -> pd.DataFrame:
    """
    Conta quantas vezes cada valor aparece em uma coluna categórica.
    Útil para entender o 'universo' de cada variável antes de qualquer análise.

    Por exemplo: quais valores existem em DS_GENERO? Há apenas "MASCULINO"
    e "FEMININO" ou existem outros valores / inconsistências?
    """
    contagem = (
        df[coluna]
        .value_counts(dropna=False)   # dropna=False para ver nulos também
        .reset_index()
    )
    contagem.columns = ["valor", "contagem"]
    contagem.insert(0, "ano", ano)
    contagem.insert(1, "coluna", coluna)
    contagem["pct"] = (contagem["contagem"] / contagem["contagem"].sum() * 100).round(2)
    return contagem


def comparar_colunas_entre_anos(dfs: dict) -> pd.DataFrame:
    """
    Compara quais colunas existem em cada ano.

    Como o TSE adicionou colunas em 2022 (ex.: federações partidárias),
    precisamos saber exatamente o que muda entre os anos antes de
    unificar os dados numa análise temporal.
    """
    todos_anos = list(dfs.keys())
    todas_colunas = set()
    for df in dfs.values():
        todas_colunas.update(df.columns)

    linhas = []
    for col in sorted(todas_colunas):
        linha = {"coluna": col}
        for ano in todos_anos:
            linha[str(ano)] = "SIM" if col in dfs[ano].columns else "---"
        linhas.append(linha)

    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("ANÁLISE DE ESTRUTURA — DADOS TSE (2014 / 2018 / 2022)")
    print("=" * 60)

    # ── 1. CARREGAMENTO ─────────────────────────────────────────────────────
    print("\n[1] Carregando arquivos...\n")
    dfs = {}
    for ano in ANOS:
        dfs[ano] = carregar_arquivos_ano(ano)

    # ── 2. COMPARAÇÃO DE COLUNAS ENTRE ANOS ─────────────────────────────────
    print("\n[2] Comparando colunas entre anos...")
    df_colunas = comparar_colunas_entre_anos(dfs)

    # Filtra só as colunas que mudam (não existem em todos os anos)
    mask_muda = df_colunas[["2014", "2018", "2022"]].nunique(axis=1) > 1
    df_colunas_divergentes = df_colunas[mask_muda].reset_index(drop=True)

    print(f"  Colunas presentes em todos os anos: "
          f"{(~mask_muda).sum()}")
    print(f"  Colunas que diferem entre algum ano: "
          f"{mask_muda.sum()}")

    df_colunas.to_csv(
        os.path.join(OUTPUT_DIR, "comparacao_colunas_por_ano.csv"),
        index=False, encoding="utf-8-sig"
    )
    df_colunas_divergentes.to_csv(
        os.path.join(OUTPUT_DIR, "colunas_divergentes_entre_anos.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("  → Salvo: comparacao_colunas_por_ano.csv")
    print("  → Salvo: colunas_divergentes_entre_anos.csv")

    # ── 3. RESUMO DE ESTRUTURA POR ANO ──────────────────────────────────────
    print("\n[3] Gerando resumo de estrutura por ano...")
    resumos = [resumo_estrutura(df, ano) for ano, df in dfs.items()]
    df_resumo = pd.concat(resumos, ignore_index=True)

    df_resumo.to_csv(
        os.path.join(OUTPUT_DIR, "estrutura_colunas_por_ano.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("  → Salvo: estrutura_colunas_por_ano.csv")

    # ── 4. RESUMO DAS COLUNAS DE FOCO ───────────────────────────────────────
    print("\n[4] Verificando qualidade das colunas de foco (análise de representatividade)...")
    colunas_presentes = [c for c in COLUNAS_FOCO if any(c in df.columns for df in dfs.values())]

    linhas_foco = []
    for ano, df in dfs.items():
        for col in colunas_presentes:
            if col in df.columns:
                linhas_foco.append({
                    "ano": ano,
                    "coluna": col,
                    "tipo_pandas": str(df[col].dtype),
                    "nulos": int(df[col].isnull().sum()),
                    "pct_nulos": round(df[col].isnull().mean() * 100, 2),
                    "valores_unicos": int(df[col].nunique()),
                    "presente": "SIM",
                })
            else:
                linhas_foco.append({
                    "ano": ano, "coluna": col,
                    "tipo_pandas": "—", "nulos": "—",
                    "pct_nulos": "—", "valores_unicos": "—",
                    "presente": "NÃO",
                })

    df_foco = pd.DataFrame(linhas_foco)
    df_foco.to_csv(
        os.path.join(OUTPUT_DIR, "qualidade_colunas_foco.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("  → Salvo: qualidade_colunas_foco.csv")

    # ── 5. DISTRIBUIÇÃO DAS VARIÁVEIS CATEGÓRICAS DE INTERESSE ──────────────
    print("\n[5] Mapeando valores únicos das variáveis categóricas de interesse...")
    categoricas = ["DS_GENERO", "DS_COR_RACA", "DS_GRAU_INSTRUCAO",
                   "DS_SIT_TOT_TURNO", "DS_CARGO", "ST_REELEICAO"]

    dist_frames = []
    for ano, df in dfs.items():
        for col in categoricas:
            if col in df.columns:
                dist_frames.append(distribuicao_categorica(df, col, ano))

    df_distribuicoes = pd.concat(dist_frames, ignore_index=True)
    df_distribuicoes.to_csv(
        os.path.join(OUTPUT_DIR, "distribuicao_categoricas.csv"),
        index=False, encoding="utf-8-sig"
    )
    print("  → Salvo: distribuicao_categoricas.csv")

    # ── 6. IMPRESSÃO DE SUMÁRIO NO CONSOLE ──────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMÁRIO FINAL")
    print("=" * 60)
    for ano, df in dfs.items():
        print(f"\n  ANO {ano}")
        print(f"    Total de candidatos : {df.shape[0]:>10,}")
        print(f"    Total de colunas    : {df.shape[1]:>10}")
        nulos_totais = df.isnull().sum().sum()
        pct_nulos = nulos_totais / (df.shape[0] * df.shape[1]) * 100
        print(f"    Células nulas       : {nulos_totais:>10,}  ({pct_nulos:.2f}%)")

    print(f"\nTodos os arquivos foram salvos em:\n  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
