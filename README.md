# Unsupervised Vulnerability Detection in GitHub Commits

This project focuses on identifying vulnerability-introducing commits in GitHub repositories using unsupervised machine learning and anomaly detection techniques. It was developed as a part of the Data Mining course project at Texas A&M University.

## 🚀 Key Features

- **Data Mining Pipeline:** Automated git log parsing and feature extraction from CVE-linked datasets.
- **Feature Engineering:** Extracts metadata features (e.g., line churn, message entropy, commit timing) and semantic features using **CodeBERT**.
- **Anomaly Detection Models:** Implements and compares multiple approaches:
  - Base Isolation Forest (Statistical features)
  - Weighted Security Features
  - Code Embedding-based detection
  - **Hybrid Ensemble Method** (Combining all the above)
- **Evaluation:** Performance analysis across various contamination levels using curated Java vulnerability datasets.

## 🛠 Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, XGBoost
- **Models:** Isolation Forest, Random Forest, CodeBERT (Transformers)

## 📊 Methodology

1. **Phase 1 (Setup):** Collects CVE data and clones relevant GitHub repositories.
2. **Phase 2 (Engineer):** Parses git logs to extract statistical metadata and generates code embeddings.
3. **Phase 3 (Detect):** Trains unsupervised models to score commits based on their "anomaly" level.
4. **Phase 4 (Evaluate):** Reports detection rates and F1 scores against known vulnerability labels.

## 📈 Results

The Hybrid Ensemble approach showed superior performance, particularly at a 10% contamination level, demonstrating the effectiveness of combining metadata with semantic code understanding.

---

**Collaborators:** Kyle Kim, Harshith Reddy
