# Undergraduate Admission Analysis

## Overview
This Jupyter Notebook analyzes undergraduate admissions data to explore factors influencing admission decisions. The analysis includes data preprocessing, exploratory data analysis (EDA), and predictive modeling.

## Contents
- **Data Preprocessing:** Handling missing values, feature engineering, and standardization.
- **Exploratory Data Analysis (EDA):** Visualizing trends and relationships in the dataset.
- **Predictive Modeling:** Applying machine learning techniques to predict admission outcomes.
- **Results and Insights:** Key findings from the data analysis and model performance.

## Requirements
To run the notebook, install the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn autoviz xgboost
```

## Usage
1. Open the notebook in Jupyter:
   ```bash
   jupyter notebook undergraduate_admission.ipynb
   ```
2. Execute the cells sequentially to follow the analysis.

## Data Source
The dataset includes various factors affecting undergraduate admissions, such as GPA, test scores, and other metrics.

## Predictive Modeling
The following steps were taken to develop and evaluate predictive models:
1. **Feature Selection & Engineering:** Selected key features such as GPA, test scores, and extracurricular involvement.
2. **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets.
3. **Model Training:** Various machine learning models were tested, including:
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - XGBoost
4. **Hyperparameter Tuning:** Grid search and cross-validation were used to optimize model performance.
5. **Evaluation Metrics:** Performance was assessed using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC score

## Results and Insights
- **Data Preprocessing:** The dataset underwent cleaning, feature engineering, and transformation for better model performance.
- **EDA Findings:** GPA and standardized test scores were strongly correlated with admission chances.
- **Model Performance:**
   - Logistic Regression: Accuracy ~ 82%
   - Decision Tree: Accuracy ~ 78%
   - Random Forest: Accuracy ~ 85%
   - XGBoost: Accuracy ~ 88%
- **Key Observations:** Feature importance analysis highlighted the significance of test scores and GPA in predicting admission likelihood. XGBoost performed the best in terms of accuracy and overall performance.

## Notes
- Ensure that all dependencies are installed before running the notebook.
- Modify parameters or preprocessing steps as needed for custom analyses.

## License
This project is for educational and research purposes only. Feel free to modify and use it for learning.

