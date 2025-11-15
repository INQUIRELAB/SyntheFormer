# 🧪 Synthesizability Prediction of Crystalline Materials with a Hierarchical Transformer and Uncertainty Quantification

## 👤 Authors
- **Danial Ebrahimzadeh** — School of Electrical & Computer Engineering, University of Oklahoma (OU), Norman, OK, USA — *danial.ebrahimzadeh@ou.edu*  
- **Sarah S. Sharif** — School of Electrical & Computer Engineering, OU, Norman, OK, USA — *s.sh@ou.edu*  
- **Yaser “Mike” Banad*** (Corresponding Author) — School of Electrical & Computer Engineering, OU, Norman, OK, USA — *bana@ou.edu*

---

## 🔏 Patent & Access Notice

> **This project is the subject of a U.S. patent filing.**  
> To protect the IP, certain files and implementation details are **withheld from public release**.  
> **Reviewers** or collaborators who require access to specific artifacts **that do not jeopardize the patent** may request them from the authors:
>
> - Yaser Mike Banad — *bana@ou.edu*  
> - Danial Ebrahimzadeh — *danial.ebrahimzadeh@ou.edu*
>
> Please include a short justification for your request and the exact files you need.

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

```text
Data_splitting/
├─ Step1_Data_Preprocessing_And_Splitting.py
└─ data/
   ├─ mp_structures_with_dates.xlsx
   └─ ftcp_data.h5  ← place/download here

feature extraction/   [WITHHELD PENDING PATENT REVIEW — AVAILABLE BY REQUEST]
├─ concatenate_final_features.py
├─ Step2_Block1_ElementNet.py
├─ step2_block2_improved.py
├─ Step2_Block3_atomic_sites_fixed.py
├─ Step2_Block4_site_occupancy.py
├─ Step2_Block5_reciprocal_space.py
└─ Step2_Block6_structure_factors.py

Prediction/           [WITHHELD PENDING PATENT REVIEW — AVAILABLE BY REQUEST]
├─ Advanced_Models_Training.py
├─ Advanced_Multi_Threshold_Optimization.py
└─ Fix_Model_Architecture.py

````

---

## 🤝 License

This project is licensed under the **MIT License**. See `LICENSE` for details.
