# Network Intrusion Detection System (NIDS)

An AI-driven cybersecurity analytics platform engineered to ingest high-dimensional network traffic log data, execute automated data preparation pipelines, and classify connections as **Normal** or **Attack** using a Random Forest architecture. 

This repository implements a robust binary classification model capable of distinguishing legitimate network activities from potential cyber threats (such as DDoS, anomalies, and unauthorized access attempts).

---

## 🚀 Key Features

- **Automated Data Preparation:** Seamlessly handles multi-type data flows by isolating numerical and categorical features, applying runtime One-Hot Encoding, and normalizing features using `StandardScaler` to prevent data leakage.
- **Dual Validation Modes:** Supports stratified train-test splitting (80/20) on known attacks or evaluating against explicit holdout test sets containing unseen/novel attack categories.
- **High-Performance Architecture:** Utilizes Scikit-Learn's `RandomForestClassifier` configured for parallel multi-core processing (`n_jobs=-1`) to maximize computation speeds during training.
- **Granular Security Analytics:** Outputs detailed evaluation matrices, including Precision, Recall, F1-Scores, and an explicit breakdown of the Confusion Matrix (True Negatives, False Positives, False Negatives, True Positives) for actionable risk mitigation.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-Learn (Sklearn)

---

## 📊 Dataset Structure

The system is optimized for the industry-standard **NSL-KDD dataset**, an enhanced benchmark dataset for network intrusion detection. The pipeline maps 41 feature attributes (e.g., duration, protocol_type, service, src_bytes, dst_bytes) alongside an operational metadata attribute:

- `KDDTrain+.txt` : Primary training records containing known normal and attack behaviors.
- `KDDTest+.txt` : Holdout testing records containing unique, novel attacks to evaluate model generalization.

---

## 💻 Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Network-Intrusion-Detection-System.git](https://github.com/your-username/Network-Intrusion-Detection-System.git)
cd Network-Intrusion-Detection-System
