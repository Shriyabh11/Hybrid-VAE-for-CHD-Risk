# Improving CHD Risk Prediction with a Hybrid VAE

A project using a hybrid VAE model to generate synthetic data for an imbalanced medical dataset, improving Coronary Heart Disease (CHD) risk prediction for under-represented age groups. Includes an interactive Streamlit UI for analysis and live predictions.

---

## The Problem: Predicting for the Under-Represented

Standard machine learning models often fail when a dataset is **imbalanced**—that is, when one outcome is far rarer than another. This project addresses this challenge in the context of the well-known Framingham Heart Study dataset.

While the dataset is large, the number of younger patients (age < 50) who develop Coronary Heart Disease (CHD) within 10 years is very small. This data scarcity makes it extremely difficult for a standard classifier to learn the specific patterns that lead to early-onset CHD. A model trained on this imbalanced data will be biased towards predicting "No CHD" and will fail to identify at-risk younger individuals, a critical failure for a medical diagnostic tool.

---

## My Solution: A Hybrid Data Augmentation Pipeline

As my primary contribution to this group project, I designed and built the core machine learning pipeline to solve this data imbalance problem. The solution was to generate high-quality, realistic synthetic data for the under-represented "CHD" class.

A single generative model (like a standard VAE) struggles to handle the mix of different data types (continuous, categorical, skewed) found in real-world medical data. Therefore, I developed a **Hybrid Generative Model** that uses the best technique for each feature type:

- **Variational Autoencoder (VAE):** For "well-behaved" continuous features (age, BMI, sysBP), a PyTorch-based VAE was used to capture the complex, multi-dimensional correlations between them.

- **Kernel Density Estimation (KDE):** For "problematic" continuous features with unusual distributions (like cigsPerDay), a statistical KDE model was used to accurately model and sample from their unique probability distributions.

- **Proportional Sampling:** For binary categorical features (male, prevalentHyp), a simple and robust sampling method was used to perfectly replicate the original data's proportions.

This hybrid approach produces a much higher-fidelity synthetic dataset than a single model could, leading to a more effective final classifier.

---

## Results: A Drastic Improvement in Detection

By training a Random Forest classifier on the newly balanced dataset, the model's ability to correctly identify at-risk patients (**Minority Class Recall**) on the unseen test set saw a dramatic improvement.

| Model                              | Minority Class Recall |
|-------------------------------------|----------------------|
| Baseline Model (on imbalanced data) | 0.0%                 |
| Augmented Model (on balanced data)  | 20.8%                |

This demonstrates the powerful impact of using a sophisticated data augmentation strategy to overcome the limitations of an imbalanced dataset.

---

## Interactive Dashboard
https://shriyabh11-hybrid-vae-for-chd-risk-app-4xvgby.streamlit.app/ 

To demonstrate this pipeline, I developed an interactive web application using Streamlit. The dashboard allows users to:

- Visualize the initial class imbalance.
- Compare the distributions of real vs. synthetic data for any feature.
- See the side-by-side performance comparison of the baseline and augmented models.
- Use a **Live Prediction Simulator** to input hypothetical patient data and get a risk score from the final, improved model.

---

## Project Structure

The code is organized into a clean, modular structure for maintainability and clarity:

- **app.py**: Contains all the Streamlit code for the user interface.
- **model.py**: Defines the core machine learning models (BoostedVAE) and data generation logic.
- **utils.py**: Handles all data loading, preprocessing, and the orchestration of the training pipeline.
- **requirements.txt**: A list of all necessary Python libraries.

---

## How to Run This Project

### 1. Prerequisites

- Python 3.8+
- An NVIDIA GPU with CUDA installed is recommended for faster VAE training, but the code will default to CPU if one is not available.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/Hybrid-VAE-for-CHD-Risk.git
cd Hybrid-VAE-for-CHD-Risk
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Data

This repository does **not** include the `framingham.csv` dataset. You must download it from its source on Kaggle:

- Download the dataset [here](https://www.kaggle.com/datasets/amanajmera1/framingham-heart-study-dataset)
- Place the downloaded `framingham.csv` file in the root directory of this project.

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

Your web browser should automatically open with the interactive dashboard.

---

## Technologies Used

- **Python**
- **Streamlit** (for the web dashboard)
- **PyTorch** (for the VAE model)
- **Scikit-learn** (for classifiers, preprocessing, and KDE)
- **Pandas** (for data manipulation)
- **Matplotlib** (for plotting)

---

This work was completed as part of a group project. The full group repository can be viewed [here](https://github.com/your-username/Hybrid-VAE-for-CHD-Risk).
