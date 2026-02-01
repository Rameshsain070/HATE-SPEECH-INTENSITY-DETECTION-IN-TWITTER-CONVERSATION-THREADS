# Hate Speech Intensity Prediction 

This repository contains code for studying **hate speech intensity and escalation**
in Twitter reply chains using temporal, contextual, and graph-based representations.

The focus of this project is **not binary hate classification**, but understanding how hate
*emerges and intensifies over the course of a conversation*.

This codebase was developed as part of an academic project and primarily serves as a
**research and learning-oriented implementation**, rather than a polished production system.


#  What this project tries to do

Given a tweet and its early replies, the goal is to model and forecast how hateful
the *future replies* in the conversation might become.

Instead of predicting weather the tweet is hateful or not it predicts the probability 
of conversation evolving into toxic content, it analyzes the conversation to detect early signs 
of hate

# 🧠High level idea
The model treats the conversation tweets as a time series data.
The pipeline broadly follows these steps:
1. Convert reply chains into windowed hate intensity sequences
2. Encode historical and future trends into latent representations
3. Group similar conversation trajectories using soft clustering
4. Use historical context + prior cluster information to predict future trends


## Repository Structure

```text
hate-intensity-experiments/
├── src/
│   ├── models/        # Neural architectures (encoders, predictors, graph layers)
│   ├── clustering/    # Soft clustering utilities (fuzzy clustering)
│   └── utils/         # Metrics and helper functions
│
├── scripts/           # Entry-point scripts for running experiments
├── data/              # Dataset 
├── docs/              # Informal experiment notes
├── requirements.txt
└── README.md
```


# 📊 Models and architectural varients

Multiple architectural variants were explored during development, including:

- Patch-based Transformer encoders for temporal modeling
- Inception-style CNN encoders for capturing short-term patterns
- Graph-based components to encode reply-chain structure
- Prior-knowledge injection using soft cluster memberships
Not all variants are guaranteed to be optimized or directly comparable.
The code reflects experimentation rather than a single finalized model.

# 📁 Dataset

The dataset used in this project originates from prior research on hate intensity
prediction in Twitter reply chains.

Due to Twitter data redistribution restrictions and licensing constraints,
the dataset is **not included** in this repository.

This repository provides only the modeling and experimentation code.
Users interested in the dataset should consult the original research work
and follow the authors’ instructions for access.

# ▶️ How to use this repository

**Usage**
Install dependencies

```bash
pip install -r requirements.txt
```

```bash
python scripts/run_patch_variant.py
```

Note:
Dataset paths and preprocessing steps must be configured manually.


# 📈 Evaluation metrics

Performance is evaluated using standard regression-style metrics commonly
used for trend prediction:

1. Pearson Correlation Coefficient (PCC)
2. Mean Squared Error (MSE)
3. Root Mean Squared Error (RMSE)


# ⚠️ Important notes and limitations
1. This codebase reflects exploratory research work and iterative experimentation.
2. Some configurations and experiments are not fully reproducible..
3. Hyperparameters and model choices were tuned empirically.
4. The repository is not intended as a benchmark or production-ready system.
