
# 🧪 Synthesizability Prediction of Crystalline Structures with Structure-Aware Feature Learning and Uncertainty Quantification

## 👤 Authors
- **Danial Ebrahimzadeh** — School of Electrical & Computer Engineering, University of Oklahoma (OU), Norman, OK, USA — *danial.ebrahimzadeh@ou.edu*  
- **Sarah Sharif** — School of Electrical & Computer Engineering, OU, Norman, OK, USA — *s.sh@ou.edu*  
- **Yaser “Mike” Banad*** (Corresponding Author) — School of Electrical & Computer Engineering, OU, Norman, OK, USA — *bana@ou.edu*

---

## 🧾 Abstract
*Predicting which hypothetical inorganic crystals can be experimentally realized remains a central challenge in accelerating materials discovery. SyntheFormer is a positive-unlabeled framework that learns synthesizability directly from crystal structure, combining Fourier-transformed crystal properties (FTCP) representation with structure-aware feature extraction, Random-Forest feature selection, and a compact deep MLP classifier. The model is trained on historical data from 2011 through 2018 and evaluated prospectively on future years from 2019 to 2025, where the positive class constitutes only 1.02 per cent of samples. Under this temporally separated evaluation, SyntheFormer achieves a test AUC of 0.735, AUPRC of 0.099 and, 97.6 percent recall at 94.2 percent coverage with dual-threshold calibration. Direct prospective validation supports this result as two materials that were unlabeled at the time of data curation, Y$_6$Fe(SiS$_7$)$_2$ and BaYb$_2$F$_8$, were assigned high SyntheFormer scores (0.961 and 0.753, respectively) and were subsequently reported experimentally. Crucially, the model recovers experimentally confirmed metastable compounds that lie far from the convex hull and simultaneously assigns low scores to many thermodynamically stable yet unsynthesized candidates, demonstrating that stability alone is insufficient to predict experimental attainability.*

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

## 🔏 Patent & Access Notice

> **This project is the subject of a U.S. patent filing.**   

> - Yaser Mike Banad — *bana@ou.edu*  
> - Danial Ebrahimzadeh — *danial.ebrahimzadeh@ou.edu*

---


## 🤝 License

This project is licensed under the **MIT License**. See `LICENSE` for details.


