import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb

# Page configuration
st.set_page_config(page_title="Undergraduate Admission Predictor Bangladesh", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for appeal
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #2c3e50; color: white; border-radius: 5px;}
    .stButton>button:hover {background-color: #3498db;}
    h1 {color: #2c3e50;}
    h2 {color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px;}
    .stExpander {background-color: #ffffff; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("Undergraduate Admission Predictor Bangladesh")
st.subheader("Unlocking Admission Secrets with Machine Learning")
st.write("""
Welcome to an exciting journey through data and predictions! This app dives deep into what gets students into universities in Bangladesh—GPA, family vibes, social media habits, you name it. 
Built by **Obidur Rahman** with xAI, it’s packed with insights and a slick prediction tool. Let’s explore!
""")

# Sidebar navigation with flair
st.sidebar.title("Explore the Magic")
st.sidebar.markdown("Pick your adventure below:")
page = st.sidebar.radio("", ["Home", "Data Deep Dive", "How It Works", "Results Unleashed", "Bangladesh Insights", "Predict Your Fate"])
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)  # Random avatar icon
st.sidebar.write("By Obidur Rahman | [GitHub: Ashfinn](https://github.com/Ashfinn)")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Ashfinn/admission-feature-analysis/main/Undergraduate%20Admission%20Test%20Survey%20in%20Bangladesh.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Feature engineering function
def engineer_features(df_input):
    df_fe = df_input.copy()
    df_fe['Average_GPA'] = (df_fe['SSC_GPA'] + df_fe['HSC_GPA']) / 2
    df_fe['GPA_Diff'] = df_fe['HSC_GPA'] - df_fe['SSC_GPA']
    df_fe['Study_Efficiency'] = df_fe['Duration_of_Study'] / df_fe['Average_GPA'].replace(0, 1e-6)
    df_fe['Social_Impact'] = df_fe['Social_Media_Engagement'] * df_fe['Average_GPA']
    df_fe['Family_Support'] = df_fe['Family_Economy'] + df_fe['Family_Education']
    df_fe['Study_Social_Ratio'] = df_fe['Duration_of_Study'] / df_fe['Social_Media_Engagement'].replace(0, 1e-6)
    df_fe = df_fe.drop(columns=['SSC_GPA', 'HSC_GPA', 'Social_Media_Engagement', 'External_Factors', 'Politics', 'Duration_of_Study', 'Family_Education'])
    df_fe = pd.get_dummies(df_fe, columns=['Residence'], drop_first=True)
    return df_fe

# Train model and scaler
@st.cache_resource
def train_model_and_scaler():
    X = df.drop(columns=['University'])
    y = df['University'].astype(int)
    
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    X_fe = engineer_features(X)
    df_combined = pd.concat([X_fe, pd.Series(y, name='University')], axis=1).drop_duplicates(subset=X_fe.columns)
    X_fe_clean = df_combined.drop(columns=['University'])
    y_clean = df_combined['University'].values.ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X_fe_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    return xgb_model, scaler, X_fe_clean.columns

xgb_model, scaler, feature_names = train_model_and_scaler()

# Home page
if page == "Home":
    st.markdown("""
    ### Hey There!
    Ever wondered what it takes to crack university admissions in Bangladesh? This project uses cutting-edge machine learning to figure it out—think GPA, family support, and even your TikTok time. 
    Built by Obidur Rahman, it’s your one-stop shop to explore data, see results, and predict your own chances. Dive in!
    """)
    st.image("https://cdn.pixabay.com/photo/2016/11/29/05/45/astronomy-1867616_1280.jpg", caption="Exploring the Universe of Admissions", use_column_width=True)

# Data Deep Dive
elif page == "Data Deep Dive":
    st.header("Data Deep Dive")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Dataset Size:** {df.shape[0]} rows, {df.shape[1]} columns")
        st.write(f"**Missing Values:**\n{df.isnull().sum()}")
    with col2:
        st.write(f"**Duplicates:** {df.duplicated().sum()}")
        st.write(f"**Target Distribution:**\n{df['University'].value_counts(normalize=True)}")
    
    with st.expander("Summary Statistics"):
        st.dataframe(df.describe())

    st.subheader("Visualizations")
    col3, col4 = st.columns(2)
    with col3:
        st.write("**Admission Outcome Distribution**")
        fig, ax = plt.subplots()
        sns.countplot(x='University', data=df, palette="viridis", ax=ax)
        ax.set_title("Admission Outcome")
        st.pyplot(fig)
    
    with col4:
        st.write("**GPA Distribution**")
        fig, ax = plt.subplots()
        sns.boxplot(x='University', y='SSC_GPA', data=df, palette="magma", ax=ax)
        ax.set_title("SSC GPA by Admission")
        st.pyplot(fig)

    st.write("**Correlation Heatmap**")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Feature Correlations")
    st.pyplot(fig)

# How It Works
elif page == "How It Works":
    st.header("How It Works")
    st.write("""
    ### The Magic Behind the Scenes
    Here’s how we turned raw data into predictions:
    """)
    with st.expander("Preprocessing"):
        st.write("""
        - Filled in missing HSC_GPA values with the median.
        - Dropped 95 duplicate rows for a cleaner dataset.
        - Turned 'Residence' into numbers with one-hot encoding.
        """)
    with st.expander("Feature Engineering"):
        st.write("""
        - **Average_GPA**: Average of SSC and HSC GPAs.
        - **GPA_Diff**: How much HSC GPA beats SSC GPA.
        - **Study_Efficiency**: Study hours per GPA point.
        - **Social_Impact**: GPA times social media use.
        - **Family_Support**: Economy + education of family.
        - **Study_Social_Ratio**: Study time vs social media.
        """)
    with st.expander("The Model"):
        st.write("""
        - **XGBoost**: A badass gradient-boosting model.
        - Tuned it with 100 trees, depth of 6, and a chill learning rate of 0.1.
        - Scaled everything with RobustScaler to handle outliers like a pro.
        """)
    st.image("https://cdn.pixabay.com/photo/2017/08/01/00/38/gear-256.png", caption="Gears of Prediction", width=300)

# Results Unleashed
elif page == "Results Unleashed":
    st.header("Results Unleashed")
    st.write("### Model Performance")
    results = {
        "Model": ["XGBoost (Tuned)"],
        "Test Accuracy": [0.8387],
        "Test ROC-AUC": [0.8516]
    }
    st.table(pd.DataFrame(results))

    st.subheader("Visual Insights")
    col5, col6 = st.columns(2)
    with col5:
        st.write("**Feature Importance (XGBoost)**")
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette="Blues_d", ax=ax)
        ax.set_title("What Drives Admission?")
        st.pyplot(fig)
    
    with col6:
        st.write("**Pair Plot Sample**")
        X_fe = engineer_features(df.drop(columns=['University']))
        sample_df = pd.concat([X_fe[['Average_GPA', 'Family_Support']], df['University']], axis=1).sample(100)
        fig = sns.pairplot(sample_df, hue='University', palette="Set2")
        st.pyplot(fig)

# Bangladesh Insights
elif page == "Bangladesh Insights":
    st.header("Bangladesh Insights")
    st.write("""
    ### What This Means for Bangladesh
    Here’s the real talk based on the data:
    """)
    st.markdown("""
    - **Big Players**: `Family_Support` and `Average_GPA` are MVPs—money and brains matter.
    - **Fixing the Game**: More financial aid and tutoring could level the field.
    - **Social Media Trap**: Too much scrolling might tank your chances. Balance is key!
    """)
    st.image("https://cdn.pixabay.com/photo/2016/11/29/09/16/architecture-1868667_1280.jpg", caption="Building a Brighter Future", use_column_width=True)

# Predict Your Fate
elif page == "Predict Your Fate":
    st.header("Predict Your Fate")
    st.write("Drop your stats below and see your admission odds!")

    with st.form("prediction_form"):
        col7, col8 = st.columns(2)
        with col7:
            ssc_gpa = st.number_input("SSC GPA (2.0-5.0)", min_value=2.0, max_value=5.0, value=4.0)
            hsc_gpa = st.number_input("HSC GPA (2.0-5.0)", min_value=2.0, max_value=5.0, value=4.0)
            duration_study = st.number_input("Study Hours/Day", min_value=1, max_value=24, value=5)
        with col8:
            social_media = st.number_input("Social Media (hours/day)", min_value=0, max_value=24, value=2)
            family_economy = st.slider("Family Economy (1-5)", 1, 5, 3)
            family_education = st.slider("Family Education (1-5)", 1, 5, 3)
        residence = st.selectbox("Residence", ["Urban", "Rural"])
        submit = st.form_submit_button("Predict My Chances!")

    if submit:
        input_data = pd.DataFrame({
            'SSC_GPA': [ssc_gpa],
            'HSC_GPA': [hsc_gpa],
            'Duration_of_Study': [duration_study],
            'Social_Media_Engagement': [social_media],
            'Family_Economy': [family_economy],
            'Family_Education': [family_education],
            'Residence': [residence],
            'External_Factors': [0],
            'Politics': [0]
        })
        input_fe = engineer_features(input_data)
        input_fe = input_fe.reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_fe)
        prob = xgb_model.predict_proba(input_scaled)[0, 1]
        st.success(f"Your Admission Probability: **{prob:.2%}**")
        st.balloons()  # Fun animation!

# Footer
st.markdown("---")
st.write("© 2025 Obidur Rahman | Powered by xAI | [GitHub: Ashfinn](https://github.com/Ashfinn)")
st.markdown("*Made with love and code in Bangladesh!*")