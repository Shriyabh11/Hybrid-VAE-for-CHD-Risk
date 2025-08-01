import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
import torch


from model import BoostedVAE, train_boosted_vae, tri_brid_generate_data, device

@st.cache_data
def load_and_prep_data(file_path):
    """Loads data, performs cleaning, feature selection, and splitting."""
    df = pd.read_csv(file_path).dropna()
    df_imbalanced = df[df['age'] < 50].copy().reset_index(drop=True)
    
    X = df_imbalanced.drop('TenYearCHD', axis=1)
    y = df_imbalanced['TenYearCHD']
    
    selector = SelectKBest(f_classif, k=8)
    X_top_np = selector.fit_transform(X, y)
    top_features_mask = selector.get_support()
    top_features_names = X.columns[top_features_mask].tolist()
    X_top = pd.DataFrame(X_top_np, columns=top_features_names)
    
    problematic_features = [col for col in ['cigsPerDay'] if col in top_features_names]
    categorical_features = [col for col in top_features_names if X_top[col].nunique() <= 2]
    well_behaved_continuous = [col for col in top_features_names if col not in categorical_features and col not in problematic_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.3, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, well_behaved_continuous, problematic_features, categorical_features, top_features_names

@st.cache_data
def get_augmentation_and_models(_X_train, _y_train, well_behaved_cont, problematic_cont, cat_features):
    """Trains VAE, generates data, and trains both classifiers."""
    # VAE Training
    X_train_minority = _X_train[_y_train == 1]
    vae_input_data = QuantileTransformer(output_distribution='normal').fit_transform(X_train_minority[well_behaved_cont])
    vae_dataset = TensorDataset(torch.FloatTensor(vae_input_data))
    vae_loader = DataLoader(vae_dataset, batch_size=16, shuffle=True)
    
    vae = BoostedVAE(input_dim=len(well_behaved_cont), latent_dim=8).to(device)
    with st.spinner('Training VAE on minority class data... (this happens once)'):
        vae = train_boosted_vae(vae, vae_loader)

    # Tri-Brid Synthesis
    n_majority, n_minority = _y_train.value_counts()
    n_to_generate = n_majority - n_minority
    synthetic_df = tri_brid_generate_data(vae, X_train_minority, well_behaved_cont, problematic_cont, cat_features, n_to_generate, 8)
    
    # Create Augmented Dataset
    y_synthetic = pd.Series([1] * len(synthetic_df), name='TenYearCHD')
    X_train_aug = pd.concat([_X_train, synthetic_df], ignore_index=True)[_X_train.columns]
    y_train_aug = pd.concat([_y_train, y_synthetic], ignore_index=True)

    # Scaling
    all_continuous_features = well_behaved_cont + problematic_cont
    scaler = StandardScaler().fit(_X_train[all_continuous_features])
    
    X_train_scaled = _X_train.copy()
    X_train_scaled[all_continuous_features] = scaler.transform(_X_train[all_continuous_features])
    
    X_train_aug_scaled = X_train_aug.copy()
    X_train_aug_scaled[all_continuous_features] = scaler.transform(X_train_aug[all_continuous_features])

    # Classifier Training
    with st.spinner('Training classifiers... (this happens once)'):
        # Baseline
        rf_baseline = RandomForestClassifier(n_estimators=400, max_depth=15, class_weight={0: 1, 1: 25}, random_state=42)
        rf_baseline.fit(X_train_scaled, _y_train)
        
        # Augmented
        rf_augmented = RandomForestClassifier(n_estimators=400, max_depth=15, random_state=42)
        rf_augmented.fit(X_train_aug_scaled, y_train_aug)

    return rf_baseline, rf_augmented, scaler, synthetic_df, X_train_minority
