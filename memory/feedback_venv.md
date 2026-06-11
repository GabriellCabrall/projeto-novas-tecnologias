---
name: feedback-venv
description: Sempre criar ambiente virtual antes de instalar bibliotecas Python neste projeto
metadata:
  type: feedback
---

Sempre criar um ambiente virtual Python (venv) antes de instalar qualquer biblioteca com pip.

**Why:** O usuário quer isolamento de dependências por projeto — instalar globalmente foi explicitamente rejeitado.

**How to apply:** Antes de qualquer `pip install`, verificar se existe um venv ativo. Se não existir, criar com `python -m venv venv` e ativar antes de instalar.
