import streamlit as st
import pandas as pd
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import os
from utils import load_and_prep_data, get_augmentation_and_models

# --- STREAMLIT UI ---

st.set_page_config(layout="wide", page_title="CHD Risk Prediction Dashboard")

st.title("🩺 Tri-Brid VAE Augmentation Dashboard")
st.markdown("""
This dashboard demonstrates a solution to a common machine learning challenge: **predicting outcomes for under-represented populations.**

Here, we focus on predicting 10-year coronary heart disease (CHD) risk specifically for **younger patients (age < 50)** from the Framingham dataset. This group has very few positive CHD cases, making it difficult for a model to learn. Our Tri-Brid VAE method generates high-quality synthetic data to balance the dataset.

**Note:** While the final model isn't perfect, this project's goal is to show that this data augmentation technique can **significantly improve a model's ability to identify at-risk individuals** in a data-scarce environment.
""")

DATA_FILE = 'framingham.csv'
if not os.path.exists(DATA_FILE):
    st.error(f"Fatal Error: '{DATA_FILE}' not found. Please make sure the CSV file is in the same directory as the script.")
    st.stop()

X_train, X_test, y_train, y_test, well_behaved_cont, problematic_cont, cat_features, top_features = load_and_prep_data(DATA_FILE)


rf_baseline, rf_augmented, scaler, synthetic_df, X_train_minority = get_augmentation_and_models(X_train, y_train, well_behaved_cont, problematic_cont, cat_features)

# --- UI Layout ---
st.sidebar.header("Live Prediction Simulator")
st.sidebar.markdown("Input patient data to get a risk prediction from the **Augmented Model**.")


input_data = {}
for feature in top_features:
    if X_train[feature].nunique() <= 2: # Binary feature
        input_data[feature] = st.sidebar.selectbox(f'{feature}', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    else: 
        min_val = float(X_train[feature].min())
        max_val = float(X_train[feature].max())
        mean_val = float(X_train[feature].mean())
        input_data[feature] = st.sidebar.slider(f'{feature}', min_val, max_val, mean_val)

if st.sidebar.button("Predict 10-Year CHD Risk", type="primary"):
   
    input_df = pd.DataFrame([input_data])
    
    
    all_continuous_features = well_behaved_cont + problematic_cont
    input_df[all_continuous_features] = scaler.transform(input_df[all_continuous_features])
    
   
    prediction_proba = rf_augmented.predict_proba(input_df[top_features])[:, 1][0]
    prediction = rf_augmented.predict(input_df[top_features])[0]

    if prediction == 1:
        st.sidebar.error(f"**High Risk** of 10-Year CHD (Confidence: {prediction_proba:.1%})")
    else:
        st.sidebar.success(f"**Low Risk** of 10-Year CHD (Confidence: {1-prediction_proba:.1%})")

# Main content area
st.header("Pipeline Visualization and Results")

# Section 1: Initial Data Analysis
with st.expander("1. Initial Data Analysis: The Class Imbalance Problem", expanded=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("The dataset of patients under 50 is highly imbalanced.")
        imbalance_df = pd.DataFrame(y_train.value_counts()).rename(columns={'count': 'Count'})
        imbalance_df.index = ['No CHD (Class 0)', 'CHD (Class 1)']
        st.table(imbalance_df)
        st.markdown(f"The minority class (CHD) makes up only **{y_train.mean()*100:.1f}%** of the training data. This makes it very difficult for a standard model to learn its patterns.")
    
    with col2:
        fig, ax = plt.subplots()
        y_train.value_counts().plot(kind='bar', ax=ax, color=['#4A90E2', '#D0021B'])
        ax.set_title('Class Distribution in Training Data')
        ax.set_xlabel('TenYearCHD')
        ax.set_ylabel('Number of Patients')
        ax.set_xticklabels(['No CHD', 'CHD'], rotation=0)
        st.pyplot(fig)

# Section 2: Synthetic Data Generation
with st.expander("2. Tri-Brid Data Augmentation: Creating Synthetic Patients", expanded=True):
    st.markdown("We use our Tri-Brid method to generate realistic synthetic data for the minority class (patients with CHD) to balance the dataset.")
    
    plot_feature = st.selectbox("Compare Real vs. Synthetic Data Distribution for a Feature:", options=well_behaved_cont + problematic_cont)
    
    fig, ax = plt.subplots()
    X_train_minority[plot_feature].plot(kind='kde', ax=ax, label='Real Data', color='#4A90E2', linewidth=2)
    synthetic_df[plot_feature].plot(kind='kde', ax=ax, label='Synthetic Data', color='#34C759', linestyle='--', linewidth=2)
    ax.set_title(f'Distribution Comparison for "{plot_feature}"')
    ax.legend()
    st.pyplot(fig)
    st.markdown("The synthetic data's distribution closely mimics the real data, indicating that our VAE and KDE models have learned the underlying patterns effectively.")

# Section 3: Model Comparison
with st.expander("3. Model Performance Comparison", expanded=True):
    st.markdown("We train two models: a **Baseline** model on the original imbalanced data (with class weights) and an **Augmented** model on the balanced data. We evaluate them on the unseen test set.")
    
    # Calculate recall scores
    X_test_scaled = X_test.copy()
    all_continuous_features = well_behaved_cont + problematic_cont
    X_test_scaled[all_continuous_features] = scaler.transform(X_test[all_continuous_features])
    
    y_pred_baseline = rf_baseline.predict(X_test_scaled)
    y_pred_augmented = rf_augmented.predict(X_test_scaled)
    
    recall_baseline = recall_score(y_test, y_pred_baseline)
    recall_augmented = recall_score(y_test, y_pred_augmented)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline Model")
        st.metric(label="Minority Class Recall", value=f"{recall_baseline:.1%}")
        st.info("The baseline model struggles to identify positive CHD cases from the imbalanced data, achieving very low recall.")

    with col2:
        st.subheader("Augmented Model (with Tri-Brid Data)")
        st.metric(label="Minority Class Recall", value=f"{recall_augmented:.1%}", delta=f"{recall_augmented - recall_baseline:.1%} improvement")
        st.success("By training on the balanced dataset, the augmented model's ability to correctly identify at-risk patients (Recall) is dramatically improved.")
