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

   python -m pip install -e .

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
├── mkdocs.yml                # configuration MkDocs pour générer la documentation
├── docs/                     # sources Markdown de la documentation
│   └── exemple_CG2.md        # page d’exemple pour la méthode CG
├── site/                     # documentation HTML générée par MkDocs
│   ├── 404.html              # page 404 par défaut
│   ├── sitemap.xml           # plan du site
│   ├── sitemap.xml.gz        # version compressée du plan du site
│   └── search/               # index de recherche pour la doc
│
├── pyproject.toml            # configuration du paquet (PEP 517/518)
├── README.md                 # présentation et instructions
├── LICENSE                   # licence du projet
│
├── src/                      # code source du paquet
│   ├── bib_code_ef.egg-info/ # métadonnées générées pour le packaging
│   └── mes_packages/         # bibliothèque éléments finis
│       ├── __init__.py
│       ├── assemblage_general.py    # assemblage DG et CG
│       ├── base.py                  # classes et outils de base
│       ├── calcul_symbolique.py     # calculs symboliques
│       ├── matrice_reference.py     # matrices de référence
│       ├── mesh.py                  # génération et outils de maillage
│       ├── methode_CG.py            # routines Galerkin continu
│       ├── methode_DG.py            # routines Galerkin discontinu
│       ├── methode_hyperbolique.py  # équations hyperboliques
│       ├── quadrature.py            # routines de quadrature/intégration
│       └── sparse.py                # opérations sur matrices creuses
│
├── tests/                    # tests unitaires (pytest)
│   ├── test_base.py
│   ├── test_calcul_symbolique.py
│   ├── test_matrice_reference.py
│   ├── test_mesh.py
│   ├── test_methode_CG.py
│   ├── test_methode_DG.py
│   ├── test_quadrature.py
│   ├── test_sparse.py
│   ├── test_assemble_surface_rhs.py
│   ├── test_hyperbo_acoustic.py
│   ├── test_skeleton.py
│   ├── test_skeleton_new.py
│   ├── test_terme_source.py
│   └── test_assemblage_general.py
│
├── examples/                 # notebooks et scripts d’exemple
│   ├── exemple_CG.ipynb
│   ├── exemple_DG.ipynb
│   ├── exemple_hyperbo.ipynb
│   ├── exemple_mesh.ipynb
│   ├── exemple_normale_triange.ipynb
│   ├── exemple_quadrature.ipynb
│   ├── exemple_reference.ipynb
│   ├── exemple_sparse.ipynb
│   ├── exemple_matrices_locales.ipynb
│   ├── exemple_matrice_face_dun_triangle_DG.ipynb
│   ├── exemple_matrice_saut_DG.ipynb
│   ├── exemple_CG_2.py
│   ├── exemple_CG_aeroac.py
│   ├── exemple_CG_diffraction_all.py
│   ├── exemple_CG_diffraction_fourier.py
│   ├── exemple_hyperbo_2.py
│   ├── exemple_mesh_marquage_frontière.ipynb
│   ├── exemple_SIPDG.py
│   ├── exemple_assemblage_masse_et_rigidite.py
│   ├── exemple_assemblage_surface.py
│   ├── exemple_assemblage_terme_source_volumique.py
│   ├── exemple_box.py
│   ├── exemple.visu.ipynb
│   └── …                     # autres exemples et scripts
│
├── notebooks/               # (optionnel) notebooks pédagogiques ou de recherche
│
├── .vscode/                 # configuration pour VS Code
├── .pytest_cache/           # cache de pytest (généré automatiquement)
├── .venv/                   # environnement virtuel local
├── test.py                  # script utilitaire
├── EF_2D_statique.ipynb     # exemple statique d’éléments finis
└── .gitignore               # fichiers/directoires ignorés par Git
---

DOCUMENTATION

The project includes a complete documentation written in Markdown and built
using **MkDocs**.

The documentation explains:

- the mathematical formulation of the methods
- the structure of the finite element library
- the Continuous Galerkin (CG) and Discontinuous Galerkin (DG) implementations
- practical examples from the course

The documentation source files are located in:

    docs/

and configured through:

    mkdocs.yml

---

BUILDING THE DOCUMENTATION

After installing the library in the virtual environment, the documentation
can be served locally.

Start the documentation server:

    mkdocs serve

Then open the following address in your web browser:

    http://127.0.0.1:8000

The documentation will automatically update when Markdown files in `docs/`
are modified.

---

GENERATING THE STATIC DOCUMENTATION

To generate the static HTML version of the documentation:

    mkdocs build

This command creates the directory:

    site/

which contains the full HTML documentation that can be opened locally or
deployed on a web server (for example GitHub Pages).


## Documentation

The full documentation is available here:

https://stordeux.github.io/CODE_EF_2D_UPPA/

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

---

License

This project is licensed under the MIT License.

Copyright (c) 2026 Sébastien Tordeux
