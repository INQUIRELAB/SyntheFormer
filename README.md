
# 🧪 Synthesizability Prediction of Crystalline Materials with a Hierarchical Transformer and Uncertainty Quantification

## 👤 Authors
- **Danial Ebrahimzadeh** — School of Electrical & Computer Engineering, University of Oklahoma (OU), Norman, OK, USA — *danial.ebrahimzadeh@ou.edu*  
- **Sarah Sharif** — School of Electrical & Computer Engineering, OU, Norman, OK, USA — *s.sh@ou.edu*  
- **Yaser “Mike” Banad*** (Corresponding Author) — School of Electrical & Computer Engineering, OU, Norman, OK, USA — *bana@ou.edu*

---

## 🧾 Abstract
*Predicting which hypothetical inorganic crystals can be experimentally realized remains a central challenge in accelerating materials discovery. **SyntheFormer** is a positive–unlabeled (PU) framework that learns synthesizability directly from crystal structure, combining a Fourier-Transformed Crystal Properties (FTCP) representation with hierarchical self-supervised feature extraction across six structural blocks. The pipeline concatenates these block-wise features into a compact 2048-D descriptor, applies classical feature selection where appropriate, and trains lightweight discriminative models with multi-threshold calibration for high-recall screening. This approach emphasizes temporal generalization under severe class imbalance, prioritizing practical lab triage and the recovery of promising metastable candidates that stability-only screens can miss.*

---

## 📚 Project Overview
SyntheFormer unifies **composition**, **real-space**, and **reciprocal-space** signals via FTCP and specialized learning heads, then aggregates them for PU-aware prediction and uncertainty-aware decision rules.

---
> Download the FTCP dataset and place it in `Data_splitting/data/`:
> **[ftcp_data.h5 (Hugging Face)](https://huggingface.co/datasets/danial199472/FTCP_Synth/resolve/main/ftcp_data.h5)**
---

## 🗂️ Repository Structure
```

Data_splitting/
├─ Step1_Data_Preprocessing_And_Splitting.py
└─ data/mp_structures_with_dates.xlsx
└─ ftcp_data.h5 ← place the downloaded file here
feature extraction/
├─ concatenate_final_features.py
├─ Step2_Block1_ElementNet.py
├─ step2_block2_improved.py
├─ Step2_Block3_atomic_sites_fixed.py
├─ Step2_Block4_site_occupancy.py
├─ Step2_Block5_reciprocal_space.py
└─ Step2_Block6_structure_factors.py
Prediction/
├─ Advanced_Models_Training.py
├─ Advanced_Multi_Threshold_Optimization.py
└─ Fix_Model_Architecture.py

````

---

## 🛠️ Installation
```bash
# Python >= 3.12
# CUDA >= 12.1


pip install -r requirements.txt

````

---

## 📥 Data Download (FTCP Representation)

Please download the FTCP-formatted dataset from a Hugging Face repository and place it under `Data_splitting/data/` **before** running any scripts.

**Sample Hugging Face link (replace with your actual dataset URL):**
🔗 `https://huggingface.co/datasets/your-org/syntheformer-ftcp`

### Option A — Using `huggingface_hub` (Python)

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="your-org/syntheformer-ftcp",   # ← replace with real repo
    repo_type="dataset",
    local_dir="Data_splitting/data",
    local_dir_use_symlinks=False
)
print("Downloaded FTCP dataset to Data_splitting/data")
PY
```

### Option B — Using `git lfs`

```bash
# If needed:
# sudo apt-get install git-lfs && git lfs install

mkdir -p Data_splitting/data
cd Data_splitting/data
git lfs install
git clone https://huggingface.co/datasets/your-org/syntheformer-ftcp ftcp_data
# Move/merge contents if needed:
# mv ftcp_data/* . && rm -rf ftcp_data
cd ../../
```

> After download, ensure the .h5 file reside directly in `Data_splitting/data/`.


---

## 🤝 License

This project is licensed under the **MIT License**. See `LICENSE` for details.


