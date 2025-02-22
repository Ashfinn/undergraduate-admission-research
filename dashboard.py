import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# Load and cache data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Ashfinn/admission-feature-analysis/main/Undergraduate%20Admission%20Test%20Survey%20in%20Bangladesh.csv"
    df = pd.read_csv(url)
    return df

# Feature engineering function
def engineer_features(df):
    X = df.drop(columns=['University'])
    y = df['University']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X = pd.get_dummies(X, columns=['Residence'], drop_first=True)
    
    # Feature engineering
    X_fe = X.copy()
    X_fe['Average_GPA'] = (X_fe['SSC_GPA'] + X_fe['HSC_GPA']) / 2
    X_fe['GPA_Diff'] = X_fe['HSC_GPA'] - X_fe['SSC_GPA']
    X_fe['Study_Efficiency'] = X_fe['Duration_of_Study'] / X_fe['Average_GPA'].replace(0, 1e-6)
    X_fe['Social_Impact'] = X_fe['Social_Media_Engagement'] * X_fe['Average_GPA']
    X_fe['Family_Support'] = X_fe['Family_Economy'] + X_fe['Family_Education']
    X_fe['Study_Social_Ratio'] = X_fe['Duration_of_Study'] / X_fe['Social_Media_Engagement'].replace(0, 1e-6)
    X_fe = X_fe.drop(columns=['SSC_GPA', 'HSC_GPA', 'Social_Media_Engagement', 'External_Factors', 
                            'Politics', 'Duration_of_Study', 'Family_Education'])
    
    return X_fe, y

# Model training function
def train_models(X_train, X_test, y_train, y_test, scaler_type='standard'):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    # Scale features
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, 
                              min_samples_leaf=3, random_state=42)
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), 
                           n_estimators=200, learning_rate=0.5, random_state=42)
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, 
                                max_depth=6, learning_rate=0.1, random_state=42)
    
    # Fit models
    rf.fit(X_train_scaled, y_train)
    ada.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Predictions and metrics
    models = {'Random Forest': rf, 'AdaBoost': ada, 'XGBoost': xgb_model}
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
    
    return results

# Main app
def main():
    st.set_page_config(page_title="Admission Analysis Dashboard", layout="wide")
    
    # Title and description
    st.title("Undergraduate Admission Test Survey Analysis")
    st.write("Interactive dashboard analyzing factors affecting university admission in Bangladesh")
    
    # Load data
    df = load_data()
    
    # Sidebar for options
    st.sidebar.header("Options")
    section = st.sidebar.radio("Select Section", 
                             ["Data Exploration", "Feature Importance", "Model Comparison"])
    scaler_choice = st.sidebar.selectbox("Scaler Type", ["Standard", "Robust"])
    
    # Data preprocessing
    X_fe, y = engineer_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X_fe, y, test_size=0.2, 
                                                       random_state=42, stratify=y)
    
    # Sections
    if section == "Data Exploration":
        st.header("Data Exploration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"Shape: {df.shape}")
            st.write("Missing Values:")
            st.write(df.isnull().sum())
            st.write(f"Duplicates: {df.duplicated().sum()}")
        
        with col2:
            st.subheader("Target Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='University', data=df, ax=ax)
            ax.set_title("Admission Outcome Distribution")
            st.pyplot(fig)
        
        st.subheader("Basic Statistics")
        st.write(df.describe())
    
    elif section == "Feature Importance":
        st.header("Feature Importance")
        
        # Train RF for feature importance
        scaler = StandardScaler() if scaler_choice == "Standard" else RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        rf.fit(X_train_scaled, LabelEncoder().fit_transform(y_train))
        
        # Feature importance plot
        importance = pd.DataFrame({'Feature': X_fe.columns, 'Importance': rf.feature_importances_})
        importance = importance.sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance, ax=ax)
        ax.set_title(f"Feature Importance (Random Forest - {scaler_choice} Scaler)")
        st.pyplot(fig)
    
    elif section == "Model Comparison":
        st.header("Model Comparison")
        
        # Train models
        with st.spinner("Training models..."):
            results = train_models(X_train, X_test, y_train, y_test, 
                                scaler_type=scaler_choice.lower())
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Accuracy Comparison")
            accuracy_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[m]['accuracy'] for m in results]
            })
            fig, ax = plt.subplots()
            sns.barplot(x='Model', y='Accuracy', data=accuracy_df, ax=ax)
            ax.set_title(f"Model Accuracy ({scaler_choice} Scaler)")
            st.pyplot(fig)
        
        with col2:
            st.subheader("ROC-AUC Comparison")
            roc_df = pd.DataFrame({
                'Model': list(results.keys()),
                'ROC-AUC': [results[m]['roc_auc'] for m in results]
            })
            fig, ax = plt.subplots()
            sns.barplot(x='Model', y='ROC-AUC', data=roc_df, ax=ax)
            ax.set_title(f"Model ROC-AUC ({scaler_choice} Scaler)")
            st.pyplot(fig)
        
        st.subheader("Detailed Results")
        st.write(pd.DataFrame(results).T)

if __name__ == "__main__":
    main()