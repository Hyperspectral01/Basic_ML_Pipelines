---

# ML Pipelines — From Scikit-Learn to PyTorch

A compact yet powerful exploration of how Machine Learning pipelines are built, automated, tuned, and deployed — moving from classical ML approaches to fully custom deep learning workflows.

---

## What This Project Covers

This repository demonstrates three core pipeline strategies:

### 1. Scikit-Learn Pipeline

A clean end-to-end workflow that includes:

* Data preprocessing using StandardScaler
* Model training using GaussianNB
* Evaluation and inference
* Model persistence using pickle

This shows how pipelines standardize training and deployment without rewriting preprocessing steps.

---

### 2. Hyperparameter Search (Grid Search)

Extends the pipeline with automated experimentation using GridSearchCV across multiple parameter options. The process selects the best configuration based on validation performance rather than manual guessing.

---

### 3. PyTorch-Based Custom Pipeline

A modular deep learning workflow that includes:

* Custom neural network architecture
* Feature preprocessing inside the pipeline
* Training loop with backpropagation
* Saving and loading trained pipelines
* Automated hyperparameter tuning (hidden layer size, learning rate, epoch count)

This is closer to real-world deep learning development where structure and automation matter as much as accuracy.

---

## Dataset

All implementations use the Iris dataset, chosen to keep the focus on pipeline engineering rather than large compute.

---

## Why This Project Matters

This project answers a practical engineering question:

**How do we automate ML workflows so models can be trained, tuned, and deployed reliably?**

It demonstrates how the same problem can be approached through:

| Concept                    | Implementation               |
| -------------------------- | ---------------------------- |
| Classical Machine Learning | Scikit-Learn Pipeline        |
| Automated Model Search     | Grid Search                  |
| Deep Learning Engineering  | PyTorch Pipeline with tuning |

---

## Outcome

By the end of this project, you will have:

* A working understanding of scalable ML training workflows
* Reusable and modular pipeline code
* A foundation to extend these concepts into production environments

---

## Tech Stack

Python • Scikit-Learn • PyTorch • NumPy • Pandas

---

The original credit of the codes goes to our professor [Dr. Priyank Thakkar](https://www.kaggle.com/priyankdl)
