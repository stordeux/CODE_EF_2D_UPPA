CODE_EF_2D_UPPA

Finite Element Methods (2D) — Teaching Repository
Université de Pau et des Pays de l'Adour (UPPA)

---

DESCRIPTION

This repository contains teaching material for the 2D Finite Element course:

- Core finite element Python library (src/mes_packages)
- Unit tests
- Jupyter notebooks (examples and demonstrations)
- Static FEM implementation
- Continuous Galerkin (CG) and Discontinuous Galerkin (DG) methods

The repository is structured for pedagogical use and reproducible scientific computing.

---

REPOSITORY HOSTING

This project is hosted on two platforms.

GitHub (Main Repository)
Primary development repository and external backup.

https://github.com/stordeux/CODE_EF_2D_UPPA

GitLab UPPA (Institutional Repository)
Institutional hosting and student access.

git@git.univ-pau.fr:stordeux/ef_ed_upp.git

Both repositories contain identical content and are synchronized manually.

---

STUDENT ACCESS

Students have read-only access via GitLab UPPA.

Permissions:

- Clone the repository
- Download the source code
- No push access
- No modification rights

The repository must not be redistributed.

---

INSTALLATION

1. Clone the repository

Using SSH (recommended):

    git clone git@git.univ-pau.fr:stordeux/ef_ed_upp.git
    cd CODE_EF_2D_UPPA

Or via GitHub:

    git clone https://github.com/stordeux/CODE_EF_2D_UPPA.git
    cd CODE_EF_2D_UPPA

---

2. Create a virtual environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

Linux / macOS:

    python -m venv .venv
    source .venv/bin/activate

---

3. Install the finite element library

   pip install -e .

This installs the package "mes_packages" in editable mode.

---

RUNNING TESTS

    pytest

---

USING THE NOTEBOOKS

After installation, notebooks in "examples/" and "notebooks/" can import the library with:

    from mes_packages import *

Make sure the Jupyter kernel uses the virtual environment ".venv".

---

REPOSITORY STRUCTURE

    project_root/
    │
    ├── pyproject.toml          # Package configuration (PEP 517/518)
    ├── README.md               # Project documentation
    ├── LICENSE                 # License file
    │
    ├── src/                    # Source layout (import isolation)
    │   └── mes_packages/       # Finite element library
    │       ├── __init__.py
    │       ├── base.py
    │       ├── calcul_symbolique.py
    │       ├── matrice_reference.py
    │       ├── mesh.py
    │       ├── methode_CG.py
    │       ├── methode_DG.py
    │       ├── methode_hyperbolique.py
    │       ├── quadrature.py
    │       └── sparse.py
    │
    ├── tests/                  # Unit tests (pytest)
    │   ├── test_base.py
    │   ├── test_calcul_symbolique.py
    │   ├── test_matrice_reference.py
    │   ├── test_mesh.py
    │   ├── test_methode_CG.py
    │   ├── test_methode_DG.py
    │   ├── test_quadrature.py
    │   └── test_sparse.py
    │
    ├── examples/               # Minimal usage scripts
    │
    ├── notebooks/              # Teaching / research notebooks
    │
    └── .gitignore              # Ignored files (venv, cache, build, etc.)


---

DEVELOPMENT WORKFLOW

# Git — Workflow rapide

A taper dans un powershell

    git pull # récupérer dernières modifications
    git status # voir les fichiers modifiés
    git add . # ajouter les modifications
    git commit -m "message" # enregistrer les changements
    git push && git pushall # envoyer vers GitHub (+ GitLab optionnel)

---

ACADEMIC INTEGRITY

This repository is provided strictly for educational use.

Students must:

- Work independently
- Not redistribute solutions
- Not publish modified versions publicly

---

Acknowledgments

AI-assisted development tools (ChatGPT and GitHub Copilot) were used to facilitate parts of the implementation and documentation.

---

AUTHOR

Sébastien Tordeux
EPC Makutu, UPPA, INRIA, LMAP UMR CNRS 5142

Do not hesitate to report bug at sebastien.tordeux@univ-pau.fr
