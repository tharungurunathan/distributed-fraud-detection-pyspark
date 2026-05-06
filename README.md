# Distributed Fraud Detection using PySpark

<p align="center">
  <img src="Pipeline.jpeg" width="900">
</p>

<p align="center">
  <b>Distributed Machine Learning Pipeline for Financial Fraud Detection using Apache Spark</b>
</p>

<p align="center">
  PySpark • Spark MLlib • Distributed SMOTE • Tomek Links • EasyEnsemble • Random Forest
</p>

---

# Overview

This project presents a distributed fraud detection pipeline developed using **PySpark** for large-scale financial transaction analysis on the **PaySim dataset**.

The study investigates how different sampling strategies handle severe class imbalance in fraud detection using distributed machine learning techniques.

Experiments were conducted using **Databricks** and **Apache Spark** on a large-scale dataset containing over **6.36 million financial transactions**.

The project compares multiple imbalance-handling strategies using:

- Random Forest (RF)
- Logistic Regression (LR)

across different dataset scales and distributed processing configurations.

---

# Project Objectives

The project focuses on answering the following research questions:

- Which sampling strategy best handles extreme class imbalance in fraud detection?
- How do Random Forest and Logistic Regression behave under different sampling strategies?
- Can distributed oversampling techniques scale efficiently on large datasets using Spark?

---

# Dataset Information

| Metric | Value |
|---|---|
| Dataset | PaySim |
| Total Transactions | 6.36 Million |
| Fraud Cases | 8,213 |
| Fraud Rate | 0.13% |
| Imbalance Ratio | 769:1 |

The PaySim dataset simulates mobile money transactions and contains highly imbalanced fraud labels, making it suitable for evaluating distributed fraud detection systems.

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| PySpark | Distributed Data Processing |
| Apache Spark | Cluster Computing |
| Spark MLlib | Machine Learning |
| NumPy | Numerical Computation |
| Pandas | Data Processing |
| Jupyter Notebook | Experimentation |
| Databricks | Distributed Execution Environment |
| Matplotlib | Visualization |

---

# Project Structure

```bash
distributed-fraud-detection-pyspark/
│
├── README.md
├── requirements.txt
├── paysim.ipynb
├── Presentation.pdf
├── Pipeline.jpeg
│
└── images/
```

---

# Distributed Pipeline Architecture

The fraud detection workflow follows a distributed Spark pipeline:

1. Data Loading
2. Feature Engineering
3. Feature Scaling
4. Sampling Strategy Application
5. Distributed Model Training
6. Model Evaluation
7. Metrics Comparison

---

# Feature Engineering

Selected Features:

- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

Removed Features:

- `nameOrig`
- `nameDest`

Identifier columns were removed because they do not provide meaningful predictive information and may introduce noise into the learning process.

---

# Sampling Strategies

The project evaluates five sampling approaches:

| Strategy | Description |
|---|---|
| S-00 | Baseline (No Resampling) |
| S-01 | Random Undersampling |
| S-02 | Distributed SMOTE |
| S-03 | SMOTE + Tomek Links |
| S-04 | EasyEnsemble |

---

# Distributed SMOTE Implementation

A custom distributed implementation of SMOTE was developed using PySpark.

## Workflow

- Minority fraud samples collected from Spark partitions
- Broadcast variables distribute minority samples across workers
- KNN-based interpolation generates synthetic fraud samples
- Synthetic samples merged using distributed Spark operations

## Key Advantages

- Reduces driver bottlenecks
- Minimizes expensive shuffle operations
- Scales efficiently on large datasets
- Fully integrated into distributed Spark processing

---

# Models Used

## Random Forest

Parameters:

```python
numTrees = 10
maxDepth = 5
seed = 42
```

Advantages:
- Handles non-linear fraud patterns
- Robust against noisy synthetic samples
- Best overall performance across experiments

---

## Logistic Regression

Parameters:

```python
maxIter = 20
regParam = 0.01
```

Advantages:
- Fast distributed training
- Lightweight baseline model
- Useful for comparison against ensemble methods

---

# Experimental Evaluation

Models were evaluated using:

- Precision
- Recall
- F1 Score
- AUC-ROC
- Training Time

Experiments were conducted across multiple dataset scales to evaluate both performance and scalability.

---

# Full Results (100% Dataset Scale)

## Random Forest Results

| Strategy | Recall | Precision | F1 Score | AUC-ROC | Time (s) |
|---|---|---|---|---|---|
| Baseline | 0.640 | 0.991 | 0.778 | 0.962 | 93 |
| Undersampling | 0.977 | 0.028 | 0.054 | 0.996 | 50 |
| Distributed SMOTE | 0.729 | 0.972 | 0.833 | 0.986 | 413 |
| SMOTE + Tomek | 0.727 | 0.970 | 0.831 | 0.989 | 3338 |
| EasyEnsemble | 0.981 | 0.027 | 0.053 | 0.996 | 411 |

---

# Key Findings

- Distributed SMOTE achieved the best balance between Recall and Precision.
- Random Forest consistently outperformed Logistic Regression.
- SMOTE + Tomek Links improved boundary cleaning but significantly increased runtime.
- EasyEnsemble achieved very high Recall but extremely poor Precision.
- F1 Score proved to be the most reliable metric for fraud detection evaluation.

---

# Scalability

The pipeline was evaluated using different dataset scales:

- 10% dataset scale
- 30% dataset scale
- 100% dataset scale

Results demonstrate that distributed Spark processing enables scalable experimentation on highly imbalanced big data workloads.

---

# How to Run

## 1. Clone Repository

```bash
git clone https://github.com/tharungurunathan/distributed-fraud-detection-pyspark.git
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Launch Notebook

```bash
jupyter notebook
```

Open:

```bash
paysim.ipynb
```

---

# Requirements

```txt
pyspark
numpy
pandas
scikit-learn
matplotlib
jupyter
```

---

# Future Improvements

Potential future enhancements include:

- Hyperparameter optimization
- Gradient Boosting / XGBoost integration
- Real-time fraud detection with Spark Streaming
- Graph-based fraud ring analysis
- Cost-sensitive learning techniques
- Advanced distributed ensemble methods

---

# Research Contribution

This project demonstrates:

- Distributed handling of imbalanced financial datasets
- Scalable oversampling using Spark
- Comparative evaluation of sampling techniques
- Large-scale fraud detection using distributed machine learning

---

# Academic Context

This project was developed as part of a group research study focused on big data learning and distributed machine learning techniques.

The presentation and implementation contributions were individually prepared and evaluated.

---

# Author

## Tharun Gurunathan

School of Computer Science  
University of Nottingham

GitHub:
https://github.com/tharungurunathan

---

# License

This project is intended for academic, research, and educational purposes.
