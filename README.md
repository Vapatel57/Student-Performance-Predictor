# Student Performance Predictor

This project uses machine learning models to predict student grades based on three key features: **attendance**, **assignment scores**, and **study hours**. The app provides interactive predictions and includes advanced data analytics features like **correlation heatmaps** and **feature importance**.

## Features
- **Prediction Model**: Predict student grades based on attendance, assignments, and study hours.
- **Model Comparison**: Compare the performance of multiple models including **Random Forest**, **Gradient Boosting**, and **Linear Regression**.
- **Data Analytics**:
  - **Correlation Heatmap**: Visualize relationships between different features (attendance, assignments, study hours, grades).
  - **Feature Importance**: Identify which features have the most impact on student grades using tree-based models.
- **Interactive Interface**: Allows users to input their own values for attendance, assignment scores, and study hours to get a predicted grade.
- **Predictive Insights**: Provides actionable insights based on the predicted grade (e.g., study improvement tips).

---

## Requirements
To run this project, you will need the following Python libraries:
- **streamlit**
- **pandas**
- **numpy**
- **scikit-learn**
- **matplotlib**
- **seaborn**
  
---

You can install these dependencies using the following command:
```bash
pip install -r requirements.txt
```
---
How to Run the Project

1.Clone the repository
   ```bash
   git clone https://github.com/yourusername/Student-Performance-Predictor.git
   cd Student-Performance-Predictor
   ```
2.Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
2. Run the app
  ```bash
  streamlit run app.py
  ```

---
