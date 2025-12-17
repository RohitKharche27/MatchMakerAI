# MatchMakerAI


## Overview

**MatchmakerAI** is a machine learning project that predicts whether a person is **single** or **not single** using a combination of **textual self-description (essay)** and **age**.

The project demonstrates a **complete ML workflow**, with special focus on:

* Handling **class imbalance** correctly
* Avoiding **data leakage**
* Comparing model performance **before and after resampling**
* Deploying the final model using **Streamlit**

---

## Problem Statement

Given:

* A short **essay** describing a person
* Their **age**

The goal is to predict:

* `single`
* `not_single`

This is formulated as a **binary classification problem**.

---

## Dataset

* The dataset contains user essays, age, and relationship status.
* The data was initially **imbalanced**, with one class dominating the others.

📌 **Dataset Link:**

> 👉 [https://drive.google.com/drive/u/1/folders/1dkxyM9rPDn7y_4uhfCl3AGxlaP5UTZHq](https://drive.google.com/file/d/1g2FrbjYpAZQLz8zop7rcIaWQMHTbYAxA/view?usp=sharing)

---

## Features Used

* **Essay (Text Feature)** – main source of semantic information
* **Age (Numeric Feature)** – provides useful demographic signal



---

## Methodology

### 1. Train–Test Split

* Data was split into training and testing sets **before any resampling**.
* The test set was kept **completely untouched**.

### 2. Resampling Strategy

* **Upsampling** was applied **only on the training set** to handle class imbalance.
* Random sampling with replacement was used.

> This prevents data leakage and ensures honest evaluation.

### 3. Model Pipeline

* Text preprocessing for essays
* Standard scaling for numeric features
* Logistic Regression classifier

---

## Experiments

Two experiments were conducted:

### 🔹 Experiment 1: Before Resampling

* Trained on imbalanced data
* Achieved high accuracy, but poor minority-class performance

### 🔹 Experiment 2: After Resampling

* Trained on balanced training data
* Lower accuracy, but significantly improved fairness and recall

> The second model was selected as the final model.

---

## Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

**F1-score** was emphasized due to class imbalance.

---

## Streamlit Application 🚀

A Streamlit web application was built to demonstrate the model interactively.
<img width="1082" height="810" alt="Screenshot 2025-12-12 160635" src="https://github.com/user-attachments/assets/8f8216c4-43dd-41b7-a1de-288a16520573" />

### App Inputs:

* Essay (text)
* Age (numeric)

### App Output:

* Predicted relationship status (Single / Not Single)


---

## How to Run Locally

```bash
# Install required libraries
pip install streamlit scikit-learn pandas numpy

# Run the Streamlit app
streamlit run app.py

```

---



## Key Takeaways

* High accuracy on imbalanced data can be misleading
* Resampling must be applied **only to the training set**
* Simpler models with relevant features generalize better
* Deployment should reflect the exact training pipeline

---

## Disclaimer

This project is for **educational and research purposes only**. Predictions are probabilistic and should not be interpreted as definitive judgments about individuals.

---

## Author

**Rohit**
Computer Science Student | Machine Learning Enthusiast
