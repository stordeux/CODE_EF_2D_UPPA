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

src/
mes_packages/ finite element library

tests/ unit tests
examples/ usage examples
notebooks/ teaching notebooks

pyproject.toml package configuration
README.md
LICENSE

---

DEVELOPMENT WORKFLOW

Push to GitHub:

    git push

Push to GitLab:

    git push gitlab main

---

ACADEMIC INTEGRITY

This repository is provided strictly for educational use.

Students must:

- Work independently
- Not redistribute solutions
- Not publish modified versions publicly

---

AUTHOR

Sébastien Tordeux
UPPA — Applied Mathematics
