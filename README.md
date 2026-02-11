# CODE_EF_2D_UPPA

Finite Element Methods (2D) --- Teaching Repository\
Université de Pau et des Pays de l'Adour (UPPA)

---

## 📚 Description

This repository contains the teaching material for the 2D Finite Element
course:

- Core finite element library (`BIB/mes_packages`)
- Unit tests
- Jupyter notebooks (examples and demonstrations)
- Static FEM implementation
- Continuous Galerkin (CG) and Discontinuous Galerkin (DG) methods

The repository is structured to allow both pedagogical use and
structured code experimentation.

---

## 🔁 Repository Hosting

This project is hosted on **two platforms**:

### 🌍 GitHub (Main Repository)

Used as the primary development repository and external backup.

https://github.com/stordeux/CODE_EF_2D_UPPA

### 🏫 GitLab UPPA (Institutional Repository)

Used for institutional hosting and student access.

git@git.univ-pau.fr:stordeux/ef_ed_upp.git

Both repositories contain identical content and are synchronized
manually.

---

## 👩‍🎓 Student Access

Students are granted **read-only access** via GitLab UPPA.

Permissions: - ✔ Clone the repository - ✔ Download the source code - ✘
No push access - ✘ No modification rights

The repository must not be redistributed.

---

## 💻 Installation

### Clone the repository

Using SSH (recommended):

    git clone git@git.univ-pau.fr:stordeux/ef_ed_upp.git

Or via GitHub:

    git clone https://github.com/stordeux/CODE_EF_2D_UPPA.git

---

### Create a virtual environment

Windows:

    python -m venv .venv
    .venv\Scripts\activate

Linux / macOS:

    python -m venv .venv
    source .venv/bin/activate

---

### Install the internal package

    pip install -e BIB

---

## 🧪 Running Tests

    pytest BIB/tests

---

## 📂 Repository Structure

    BIB/
        mes_packages/
        tests/
    examples/
    notebooks/
    EF_2D_statique.ipynb

---

## 🛠 Development Workflow

Push to GitHub (main):

    git push

Push to GitLab:

    git push gitlab main

---

## 📜 Academic Integrity

This repository is provided strictly for educational use.

Students must: - Work independently - Not redistribute solutions - Not
publish modified versions publicly

---

## 👤 Author

Sébastien Tordeux\
UPPA -- Applied Mathematics
