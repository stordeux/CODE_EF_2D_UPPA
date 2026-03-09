# Installation de `mes_packages`

Cette page explique comment installer la bibliothèque **mes_packages** et configurer votre environnement de travail.  
Nous vous recommandons d’utiliser un **environnement virtuel** afin d’isoler les dépendances de ce projet:contentReference[oaicite:0]{index=0}.

## Clone the repository

Using SSH (recommended):

    git clone git@git.univ-pau.fr:stordeux/ef_ed_upp.git
    cd CODE_EF_2D_UPPA

Or via GitHub:

    git clone https://github.com/stordeux/CODE_EF_2D_UPPA.git
    cd CODE_EF_2D_UPPA

---

## Create a virtual environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

Linux / macOS:

    python -m venv .venv
    source .venv/bin/activate

---

## Install the finite element library

python -m pip install -e .

This installs the package "mes_packages" in editable mode.

---

## RUNNING TESTS

    pytest

---

## USING THE NOTEBOOKS

After installation, notebooks in "examples/" and "notebooks/" can import the library with:

    from mes_packages import *

Make sure the Jupyter kernel uses the virtual environment ".venv".

---
