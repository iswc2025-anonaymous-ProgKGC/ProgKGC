# ProgKGC: Progressive Structure-Enhanced Semantic Framework for Knowledge Graph Completion

> 🔍 This repository contains the official implementation of **ProgKGC**, a progressive structure-enhanced semantic framework for knowledge graph completion. The model incorporates a progressive training strategy and a bidirectional neighbor integration mechanism to better leverage both textual and structural information.

---

## 📂 Overview

- Progressive training: semantic encoder first, followed by structural encoder  
- Bidirectional neighbor aggregation: integrates head and tail entity neighborhood information  
---

## 🛠 Installation

```bash
git clone https://github.com/iswc2025-ProgKGC-anonaymous/ProgKGC.git
cd ProgKGC
pip install -r requirements.txt
```

---

## 🚀 Usage
It involves 3 steps: dataset preprocessing, model training, and model evaluation.
### WN18RR dataset

**Step 1**, preprocess the dataset:

```bash
bash scripts/preprocess.sh WN18RR
```

**Step 2**, train the model and (optionally) specify the output directory:

```bash
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh
```

**Step 3**, evaluate a trained model:

```bash
bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR
```

Feel free to change the output directory to any path you think appropriate.

---

### FB15k-237 dataset

**Step 1**, preprocess the dataset:

```bash
bash scripts/preprocess.sh FB15k237
```

**Step 2**, train the model and (optionally) specify the output directory:

```bash
OUTPUT_DIR=./checkpoint/fb15k237/ bash scripts/train_fb.sh
```

**Step 3**, evaluate a trained model:

```bash
bash scripts/eval.sh ./checkpoint/fb15k237/model_last.mdl FB15k237
```

---
