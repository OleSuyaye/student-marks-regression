# Student Marks Regression Project

This project applies supervised machine learning regression techniques to predict student marks based on the number of courses a student takes and the amount of time spent studying.

The project covers the complete regression workflow, including data preparation, model training, evaluation, hyperparameter tuning, cross-validation, pipelines, model saving, and deployment using Streamlit.

## Project Objective

The goal is to predict student marks using:

- `number_courses`
- `time_study`

Target:

- `Marks`

## Dataset

Dataset source:

https://www.kaggle.com/datasets/yasserh/student-marks-dataset/data

The dataset contains 100 observations and three columns.

| Column | Description |
|---|---|
| `number_courses` | Number of courses taken |
| `time_study` | Time spent studying |
| `Marks` | Student marks |

## Models Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors Regressor

## Model Comparison

| Model | Train R² | Test R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| Linear Regression | 0.9344 | 0.9460 | 3.0793 | 3.7684 |
| Tuned Decision Tree | 1.0000 | 0.9857 | 1.6166 | 1.9378 |
| Tuned Random Forest | 0.9973 | 0.9927 | 1.1954 | 1.3877 |
| Tuned KNN | 1.0000 | 0.9922 | 1.1212 | 1.4309 |

`Note:` I decided to go with KNN for the final deployment because, we used a pipeline for scaling, which was the only preprocessing done for this data. All other models required no preprocessing because the data was clean, however, KNN requires scaling.


## KNN Hyperparameter Tuning

```python
param_grid = {
    "knn__n_neighbors": [2, 3, 4, 5, 6, 7],
    "knn__weights": ["uniform", "distance"]
}
```

Best parameters:

```python
{
    "knn__n_neighbors": 5,
    "knn__weights": "distance"
}
```

Best average CV R²:

```text
0.9712
```

## Pipeline

```python
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor())
])
```

The pipeline ensures that feature scaling happens correctly inside each cross-validation fold and helps prevent data leakage.

## Final KNN Performance

```text
Training R²: 1.0000
Test R²:     0.9922
MAE:         1.1212
MSE:         2.0475
RMSE:        1.4309
```

## Saving the Model

```python
import joblib

joblib.dump(best_knn, "../models/best_knn.pkl")
```

Load later with:

```python
model = joblib.load("models/best_knn.pkl")
```

## Example Prediction

```python
import joblib
import pandas as pd

model = joblib.load("models/best_knn.pkl")

new_student = pd.DataFrame({
    "number_courses": [6],
    "time_study": [7.0]
})

prediction = model.predict(new_student)

print(f"Predicted Marks: {prediction[0]:.2f}")
```

## Streamlit App

The deployed app allows users to enter:

- Number of courses
- Time studied

and receive a predicted mark.

Live app:

https://olesuyayestudentmarksknnregression.streamlit.app

## Project Structure

```text
supervised_regression/
│
├── data/
│   └── Student_Marks.csv
├── models/
│   └── best_knn.pkl
├── notebooks/
│   └── student_marks.ipynb
├── app.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Running Locally

```bash
git clone https://github.com/OleSuyaye/student-marks-regression.git
cd student-marks-regression
pip install -r requirements.txt
streamlit run app.py
```

## Requirements

```text
streamlit==1.51.0
pandas==2.3.3
scikit-learn==1.7.2
joblib==1.5.2
```

## Key Concepts Practiced

- Supervised machine learning
- Regression
- Train/test splitting
- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- KNN Regression
- Feature scaling
- StandardScaler
- MAE, MSE, RMSE, R²
- Cross-validation
- GridSearchCV
- Pipelines
- Data leakage prevention
- Model saving with Joblib
- Streamlit deployment
- Git and GitHub

## Author

**Kevin Ngang'a**

GitHub: https://github.com/OleSuyaye