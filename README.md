<div align="center">

# 📚 Student Academic Performance Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-lightgrey?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.0-blue?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)](https://www.docker.com/)

</div>

An AI/ML web application that predicts a student's academic performance grade (Poor / Average / Good / Excellent) based on study habits, attendance, and academic history.

---

## 📸 Interface Preview

![Student Performance Interface](https://raw.githubusercontent.com/Tayyabah-Rehman/Student-Performance-Predictor/main/data/Student%20Performance%20Interface.PNG)

*Screenshot of the Student Performance Predictor showing input features and grade prediction output.*

---

## Project Structure

```
student_performance/
├── app.py                    # Flask REST API
├── generate_and_train.py     # Dataset generation + model training
├── requirements.txt
├── data/
│   └── student_performance.csv   # 800-record synthetic dataset
├── model/
│   ├── model.pkl             # Trained model (Logistic Regression)
│   ├── scaler.pkl            # StandardScaler
│   ├── label_encoder.pkl     # LabelEncoder
│   └── meta.json             # Metrics, feature importances, model comparison
└── templates/
    └── index.html            # Single-page frontend
```

---

## Features

| Feature | Range |
|---|---|
| Study Hours Per Day | 0.5 – 10 h |
| Attendance Percentage | 40 – 100 % |
| Assignments Completed | 0 – 10 |
| Previous Semester Marks | 30 – 100 |
| Class Participation | 1 – 5 (star rating) |
| Sleep Hours Per Night | 4 – 10 h |
| Extracurricular Activities | 0 – 3 |

**Target:** Performance Grade → `Poor`, `Average`, `Good`, `Excellent`

---

## Model Performance

| Model | CV Accuracy |
|---|---|
| Logistic Regression ✅ | ~80% |
| Random Forest | ~68% |
| Gradient Boosting | ~63% |

Best model selected automatically by cross-validation.  
Test accuracy: **~76%** | Weighted F1: **~75%**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset & train model (already done; skip if model/ exists)
python generate_and_train.py

# 3. Run the web app
python app.py

# 4. Open browser
# http://localhost:5000
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict` | Returns grade prediction + probabilities |
| GET | `/api/meta` | Model metrics, feature importances |
| GET | `/data/student_performance.csv` | Download dataset |

### POST `/api/predict` — example body
```json
{
  "study_hours_per_day": 5.0,
  "attendance_percentage": 85,
  "assignments_completed": 8,
  "previous_semester_marks": 72,
  "class_participation": 4,
  "sleep_hours": 7,
  "extracurricular_activities": 1
}
```

### Response
```json
{
  "prediction": "Good",
  "probabilities": { "Poor": 5.2, "Average": 18.1, "Good": 61.3, "Excellent": 15.4 },
  "model_used": "LogisticRegression"
}
```

---

## Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **ML:** scikit-learn (Logistic Regression, Random Forest, Gradient Boosting), joblib
- **Data:** pandas, numpy
- **Frontend:** Vanilla HTML/CSS/JS — no framework dependencies

---

*Day 1 Deliverable — Student Performance Prediction Task*
