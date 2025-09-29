# Edge-Enriched-Mesh-Representation

This repository accompanies the scientific paper "Edge-Enriched Mesh Representation for Protein Surface Classification", published at ICTAI 2025.

---

## 📂 Project structure
- scripts/get_data.sh – download the dataset automatically.
- scripts/main.sh – train and test the model (see usage below).
- requirements.txt – required Python dependencies.
- models.py, tools.py, main.py – core implementation.
- results/ – folder where results will be stored.

---

## 📥 Dataset
To download the dataset used in this work:
bash ./scripts/get_data.sh

If you want to inspect the original dataset directly, you can use the following link:
👉 Dataset link: link.dataset

---

## 🚀 Training and Testing
To launch training with default parameters:
bash ./scripts/main.sh

To explore all available options (including testing the best model on the test set), run:
python3 main.py --help

---

## 📄 Citation
If you use this code or dataset, please cite the following paper:

"Edge-Enriched Mesh Representation for Protein Surface Classification".
ICTAI 2025.

BibTeX:
@inproceedings{eemr2025,
  title={Edge-Enriched Mesh Representation for Protein Surface Classification},
  author={Abderrahim MECHACHE and Hamamache KHEDDOUCI},
  booktitle={Proceedings of the IEEE International Conference on Tools with Artificial Intelligence (ICTAI)},
  year={2025}
}

---

## ⚙️ Requirements
Install dependencies with:
pip install -r requirements.txt