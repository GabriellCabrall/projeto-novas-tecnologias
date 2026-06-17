# Representatividade Política nas Eleições Brasileiras

Projeto da disciplina **Novas Tecnologias** — Universidade Católica de Brasília.

Análise de representatividade de gênero, raça e perfil demográfico dos candidatos
nas eleições gerais de **2014, 2018 e 2022**, com base nos dados abertos do TSE.

---

## Pré-requisitos

- [Python 3.11](https://www.python.org/downloads/) ou superior
- Git

Para verificar se o Python está instalado corretamente:

```bash
python --version
```

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/GabriellCabrall/projeto-novas-tecnologias.git
cd projeto-novas-tecnologias
```

### 2. Crie e ative o ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

> Se o PowerShell bloquear a execução de scripts, rode antes:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## Como rodar

### Dashboard principal

O dataset já está incluído no repositório (`dados_tratados/candidatos_tratados.csv`).
Basta rodar diretamente:

```bash
streamlit run src/03_dashboard.py
```

O Streamlit abrirá automaticamente no navegador em `http://localhost:8501`.

---

## Estrutura do projeto

```
projeto-novas-tecnologias/
│
├── base_de_dados/              # Dados brutos do TSE (CSVs por estado e ano)
│   ├── 2014/
│   ├── 2018/
│   └── 2022/
│
├── dados_tratados/
│   └── candidatos_tratados.csv # Dataset limpo e unificado (pronto para uso)
│
├── src/
│   ├── 01_estrutura_dados.py   # Análise exploratória dos dados brutos
│   ├── 02_preparacao_dados.py  # Pipeline de limpeza e engenharia de features
│   └── 03_dashboard.py         # Interface Streamlit + modelo de ML
│
├── requirements.txt
└── README.md
```

---

## Reprocessar os dados (opcional)

O dataset tratado já está commitado. Os scripts abaixo só precisam ser executados
se você quiser reprocessar os dados brutos do zero.

**Passo 1 — análise exploratória dos dados brutos:**
```bash
python src/01_estrutura_dados.py
```
Gera arquivos de diagnóstico em `outputs/dados_processados/` (pasta ignorada pelo git).

**Passo 2 — limpeza e preparação:**
```bash
python src/02_preparacao_dados.py
```
Substitui o arquivo `dados_tratados/candidatos_tratados.csv`.

---

## Funcionalidades do dashboard

| Aba | Conteúdo |
|---|---|
| Gênero | Evolução da participação feminina e taxa de eleição por gênero × raça |
| Raça | Representação racial entre candidatos e eleitos, evolução 2014–2022 |
| Perfil Demográfico | Taxa de eleição por faixa etária, cargo e grau de instrução |
| Simulador | Formulário de perfil com previsão de chances por cargo via Regressão Logística |

A sidebar permite filtrar os dados por **ano**, **cargo** e **região** — os filtros se aplicam às três primeiras abas. A aba Simulador usa sempre o dataset completo para o treino do modelo.

---

## Fonte dos dados

Repositório de Dados Eleitorais do TSE:

- [Candidatos 2022](https://dadosabertos.tse.jus.br/dataset/candidatos-2022)
- [Candidatos 2018](https://dadosabertos.tse.jus.br/dataset/candidatos-2018)
- [Candidatos 2014](https://dadosabertos.tse.jus.br/dataset/candidatos-2014)
