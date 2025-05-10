# Enhanced Student Performance Predictor with Advanced Data Analytics

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Title
st.title('Enhanced Student Performance Predictor')
st.markdown('Predict student grades based on attendance, assignment scores, and study hours with detailed analytics.')

# File Upload
uploaded_file = st.file_uploader('Upload Student Data (CSV)', type=['csv'])

if uploaded_file:
    # Load the CSV file
    data = pd.read_csv(uploaded_file)
    st.write('**Data Preview:**')
    st.dataframe(data.head())
    
    # Data Preprocessing
    st.write('**Data Preprocessing:**')
    data = data.dropna()
    st.write(f'Data after removing missing values: {data.shape[0]} rows')

    # Feature Selection
    X = data[['Attendance', 'Assignments', 'StudyHours']]
    y = data['Grade']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    models = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'Linear Regression': LinearRegression()
    }

    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_scores[name] = {'MSE': mse, 'R²': r2}

    # Display Model Performance
    st.write('**Model Comparison:**')
    st.dataframe(pd.DataFrame(model_scores).T)

    # Best Model Selection
    best_model = max(model_scores, key=lambda x: model_scores[x]['R²'])
    st.write(f'**Best Model:** {best_model}')

    # Data Visualization
    st.write('**Data Visualization:**')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=data, x='Attendance', y='Grade', ax=ax[0])
    sns.scatterplot(data=data, x='StudyHours', y='Grade', ax=ax[1])
    st.pyplot(fig)

    # Correlation Heatmap
    st.write('**Correlation Heatmap:**')
    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

    # Feature Importance (only for Tree-based Models)
    if best_model in ['Random Forest', 'Gradient Boosting']:
        st.write('**Feature Importance:**')
        model = models[best_model]
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        st.bar_chart(feature_importance)

    # Prediction Interface
    st.write('**Predict Student Grade:**')
    attendance = st.slider('Attendance (%)', 0, 100, 75)
    assignments = st.slider('Assignment Score (%)', 0, 100, 80)
    study_hours = st.slider('Study Hours per Week', 0, 50, 15)
    prediction = model.predict([[attendance, assignments, study_hours]])[0]
    st.write(f'**Predicted Grade:** {prediction:.2f}')

    # Predictive Insights
    st.write('**Predictive Insights:**')
    if prediction < 70:
        st.warning('Student might need additional study hours or assignment improvement.')
    elif 70 <= prediction < 85:
        st.info('Student is performing well but has room for improvement.')
    else:
        st.success('Student is performing excellently!')
