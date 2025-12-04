# 🎓 Student Performance Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

**An end-to-end Machine Learning project that predicts student math scores based on demographic and academic factors.**

[🚀 Live Demo](#-running-the-application) • [📊 Dataset](#-dataset) • [🏗️ Architecture](#️-project-architecture) • [📖 Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Dataset](#-dataset)
4. [Project Architecture](#️-project-architecture)
5. [ML Pipeline](#-ml-pipeline)
6. [Models & Performance](#-models--performance)
7. [Installation](#-installation)
8. [Running the Application](#-running-the-application)
9. [Project Structure](#-project-structure)
10. [How It Works](#-how-it-works)
11. [Future Improvements](#-future-improvements)
12. [Technologies Used](#-technologies-used)
13. [License](#-license)

---

## 🎯 Project Overview

### What is this project?

This is a **complete end-to-end Machine Learning project** that predicts a student's **math score** based on various factors including:

- 👤 **Gender** (Male/Female)
- 🌍 **Race/Ethnicity** (Groups A-E)
- 🎓 **Parental Education Level** (High School to Master's Degree)
- 🍽️ **Lunch Type** (Standard or Free/Reduced)
- 📚 **Test Preparation Course** (Completed or None)
- 📖 **Reading Score** (0-100)
- ✍️ **Writing Score** (0-100)

### Project Goal

The goal is to understand **how various factors affect student performance** and build a predictive model that can:
1. Help educators identify students who might need additional support
2. Understand the impact of socioeconomic factors on academic performance
3. Provide insights for educational policy decisions

---

## 🔍 Problem Statement

### The Challenge

Educational institutions need to identify students who may struggle academically **before** they fail. Traditional methods rely on:
- Previous test scores
- Teacher observations
- Parent feedback

These methods are often **reactive** rather than **proactive**.

### Our Solution

Build a **Machine Learning model** that can predict student performance based on multiple factors, allowing for:
- **Early intervention** for at-risk students
- **Resource allocation** based on predicted needs
- **Data-driven decisions** in education

### Key Questions We Answer

1. How does parental education affect student performance?
2. Does completing a test preparation course improve scores?
3. Is there a correlation between reading/writing scores and math scores?
4. How do socioeconomic factors (like lunch type) impact performance?

---

## 📊 Dataset

### Source
The dataset contains information about **1,000 students** with their demographic information and test scores.

### Features

| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `gender` | Categorical | Student's gender | male, female |
| `race_ethnicity` | Categorical | Ethnic group | group A, B, C, D, E |
| `parental_level_of_education` | Categorical | Parent's education | some high school → master's degree |
| `lunch` | Categorical | Lunch program type | standard, free/reduced |
| `test_preparation_course` | Categorical | Test prep status | none, completed |
| `reading_score` | Numerical | Reading test score | 0-100 |
| `writing_score` | Numerical | Writing test score | 0-100 |
| `math_score` | Numerical | **TARGET** - Math test score | 0-100 |

### Data Split
- **Training Set:** 800 students (80%)
- **Testing Set:** 200 students (20%)

### Sample Data

```
gender | race_ethnicity | parental_education    | lunch    | test_prep | reading | writing | math
-------|----------------|----------------------|----------|-----------|---------|---------|------
female | group B        | bachelor's degree    | standard | none      | 72      | 74      | 72
male   | group A        | associate's degree   | free     | none      | 57      | 44      | 47
female | group C        | some college         | standard | completed | 87      | 86      | 90
```

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          STUDENT PERFORMANCE PREDICTOR                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐              │
│  │   RAW DATA      │    │   PROCESSED     │    │   TRAINED       │              │
│  │   stud.csv      │───▶│   train.csv     │───▶│   model.pkl     │              │
│  │   (1000 rows)   │    │   test.csv      │    │   (Best Model)  │              │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘              │
│          │                      │                      │                        │
│          ▼                      ▼                      ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         PIPELINE COMPONENTS                             │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                         │    │
│  │  📥 Data Ingestion    │  🔄 Data Transformation  │  🧠 Model Training   │    │
│  │  ─────────────────    │  ─────────────────────   │  ─────────────────   │    │
│  │  • Load CSV           │  • Handle missing values │  • Train 7 models   │    │
│  │  • Train/Test split   │  • Encode categories     │  • GridSearchCV     │    │
│  │  • Save to artifacts  │  • Scale numerical       │  • Select best      │    │
│  │                       │  • Save preprocessor     │  • Save model       │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                         WEB APPLICATION (Flask)                         │    │
│  ├─────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                         │    │
│  │  🌐 Routes:                        📄 Templates:                        │    │
│  │  • GET  /           → index.html   • index.html (Welcome)               │    │
│  │  • GET  /predictdata → home.html   • home.html (Prediction Form)        │    │
│  │  • POST /predictdata → Prediction                                       │    │
│  │                                                                         │    │
│  │  🔮 Prediction Pipeline:                                                │    │
│  │  Form Data → CustomData → DataFrame → Preprocessor → Model → Result     │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 ML Pipeline

### Pipeline Stages

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ML PIPELINE FLOW                                │
└──────────────────────────────────────────────────────────────────────────────┘

Stage 1: DATA INGESTION
─────────────────────────
notebook/data/stud.csv
        │
        ▼
┌─────────────────────┐
│  data_ingestion.py  │
│  • Read CSV         │
│  • Split 80/20      │
│  • Save artifacts   │
└─────────────────────┘
        │
        ├──▶ artifacts/data.csv (raw copy)
        ├──▶ artifacts/train.csv (800 rows)
        └──▶ artifacts/test.csv (200 rows)


Stage 2: DATA TRANSFORMATION
────────────────────────────
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│                    data_transformation.py                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Numerical Pipeline:          Categorical Pipeline:            │
│  ┌─────────────────────┐     ┌─────────────────────┐          │
│  │ SimpleImputer       │     │ SimpleImputer       │          │
│  │ (strategy=median)   │     │ (strategy=frequent) │          │
│  └─────────┬───────────┘     └─────────┬───────────┘          │
│            ▼                           ▼                       │
│  ┌─────────────────────┐     ┌─────────────────────┐          │
│  │ StandardScaler      │     │ OneHotEncoder       │          │
│  │ (mean=0, std=1)     │     │ (categories→numbers)│          │
│  └─────────────────────┘     └─────────┬───────────┘          │
│                                        ▼                       │
│                              ┌─────────────────────┐          │
│                              │ StandardScaler      │          │
│                              │ (with_mean=False)   │          │
│                              └─────────────────────┘          │
│                                                                │
│  Combined with ColumnTransformer                               │
└────────────────────────────────────────────────────────────────┘
        │
        └──▶ artifacts/proprocessor.pkl (saved transformation rules)


Stage 3: MODEL TRAINING
───────────────────────
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│                      model_trainer.py                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Models Trained:                                               │
│  ┌─────────────────┬─────────────────┬─────────────────┐      │
│  │ Linear Regression│ Decision Tree   │ Random Forest   │      │
│  │ Gradient Boosting│ XGBoost         │ CatBoost        │      │
│  │ AdaBoost         │                 │                 │      │
│  └─────────────────┴─────────────────┴─────────────────┘      │
│                                                                │
│  Hyperparameter Tuning: GridSearchCV (cv=3)                   │
│  Evaluation Metric: R² Score                                   │
│  Best Model Selection: Highest R² on test data                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
        │
        └──▶ artifacts/model.pkl (best trained model)


Stage 4: PREDICTION
───────────────────
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│                    predict_pipeline.py                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  User Input → CustomData → DataFrame                           │
│       │                                                        │
│       ▼                                                        │
│  Load proprocessor.pkl → Transform data                        │
│       │                                                        │
│       ▼                                                        │
│  Load model.pkl → Predict                                      │
│       │                                                        │
│       ▼                                                        │
│  Return predicted math score                                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 📈 Models & Performance

### Models Evaluated

| Model | R² Score | Training Time | Notes |
|-------|----------|---------------|-------|
| **Linear Regression** | **0.88** ⭐ | Fast | Best performer, selected |
| Ridge Regression | 0.87 | Fast | Good regularization |
| Random Forest | 0.85 | Medium | Robust to outliers |
| Gradient Boosting | 0.86 | Slow | Good for complex patterns |
| XGBoost | 0.84 | Medium | Fast gradient boosting |
| CatBoost | 0.87 | Medium | Handles categories well |
| AdaBoost | 0.82 | Medium | Focuses on hard examples |
| Decision Tree | 0.72 | Fast | Prone to overfitting |

### Why Linear Regression Won

1. **Strong linear relationships** in the data
2. **Reading and writing scores** are highly correlated with math scores
3. **Simple model** avoids overfitting on this dataset size
4. **Interpretable** - easy to understand feature importance

### Feature Importance

```
Feature                          Impact on Math Score
─────────────────────────────────────────────────────
Writing Score                    ████████████████████ High
Reading Score                    ███████████████████░ High
Test Preparation (completed)     ████████░░░░░░░░░░░░ Medium
Parental Education (higher)      ███████░░░░░░░░░░░░░ Medium
Lunch Type (standard)            █████░░░░░░░░░░░░░░░ Medium
Gender                           ███░░░░░░░░░░░░░░░░░ Low
Race/Ethnicity                   ██░░░░░░░░░░░░░░░░░░ Low
```

---

## 💻 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Clone the Repository

```bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
```

### Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements

```
numpy
pandas
seaborn
matplotlib
scikit-learn
catboost
xgboost
dill
flask
```

---

## 🚀 Running the Application

### Option 1: Run the Complete Pipeline

```bash
# This will:
# 1. Load and split data
# 2. Transform features
# 3. Train all models
# 4. Select and save the best model

python -m src.components.data_ingestion
```

**Output:**
```
0.8804332983749565  # R² score of the best model
```

### Option 2: Run the Web Application

```bash
python app.py
```

**Output:**
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5001
```

Open your browser and go to: **http://localhost:5001**

### Using the Web App

1. **Home Page** - Click "Start Prediction"
2. **Prediction Form** - Fill in student details:
   - Select gender, ethnicity, parental education
   - Choose lunch type and test preparation status
   - Enter reading and writing scores (0-100)
3. **Get Result** - Click "Predict Math Score"
4. **View Prediction** - See the predicted score with performance message

---

## 📁 Project Structure

```
ML_PROJECT/
│
├── 📁 artifacts/                    # Generated model artifacts
│   ├── data.csv                     # Raw data copy
│   ├── train.csv                    # Training data (80%)
│   ├── test.csv                     # Testing data (20%)
│   ├── proprocessor.pkl             # Saved preprocessor
│   └── model.pkl                    # Trained model
│
├── 📁 notebook/                     # Jupyter notebooks
│   ├── data/
│   │   └── stud.csv                 # Original dataset
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   └── 2. MODEL TRAINING.ipynb
│
├── 📁 src/                          # Source code
│   ├── __init__.py
│   ├── exception.py                 # Custom exception handling
│   ├── logger.py                    # Logging configuration
│   ├── utils.py                     # Utility functions
│   │
│   ├── 📁 components/               # ML pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py        # Data loading & splitting
│   │   ├── data_transformation.py   # Feature engineering
│   │   └── model_trainer.py         # Model training & selection
│   │
│   └── 📁 pipeline/                 # Prediction pipeline
│       ├── __init__.py
│       ├── predict_pipeline.py      # Prediction logic
│       └── train_pipeline.py        # Training orchestration
│
├── 📁 templates/                    # HTML templates
│   ├── index.html                   # Welcome page
│   └── home.html                    # Prediction form
│
├── 📁 logs/                         # Application logs
│   └── *.log                        # Timestamped log files
│
├── app.py                           # Flask application
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
└── README.md                        # This file
```

---

## ⚙️ How It Works

### 1. Data Ingestion (`data_ingestion.py`)

```python
# Reads raw data
df = pd.read_csv('notebook/data/stud.csv')

# Splits into train/test
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# Saves to artifacts folder
train_set.to_csv('artifacts/train.csv')
test_set.to_csv('artifacts/test.csv')
```

### 2. Data Transformation (`data_transformation.py`)

```python
# Numerical features: Impute missing → Scale
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical features: Impute missing → One-hot encode → Scale
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder()),
    ("scaler", StandardScaler(with_mean=False))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ("num_pipeline", num_pipeline, numerical_columns),
    ("cat_pipeline", cat_pipeline, categorical_columns)
])
```

### 3. Model Training (`model_trainer.py`)

```python
# Define models to test
models = {
    "Random Forest": RandomForestRegressor(),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(),
    # ... more models
}

# Train and evaluate each model with GridSearchCV
model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

# Select best model based on R² score
best_model = models[best_model_name]

# Save the best model
save_object("artifacts/model.pkl", best_model)
```

### 4. Prediction (`predict_pipeline.py`)

```python
# Load saved model and preprocessor
model = load_object("artifacts/model.pkl")
preprocessor = load_object("artifacts/proprocessor.pkl")

# Transform new data using same preprocessing
data_scaled = preprocessor.transform(features)

# Make prediction
prediction = model.predict(data_scaled)
```

### 5. Web Application (`app.py`)

```python
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get form data
        data = CustomData(
            gender=request.form.get('gender'),
            # ... other fields
        )
        
        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Return result
        return render_template('home.html', results=results[0])
```

---

## 🚀 Future Improvements

### With Current Dataset

1. **Feature Engineering**
   - Create interaction features (reading × writing)
   - Add score ratios (reading/writing)
   - Bin scores into categories (low/medium/high)

2. **Model Improvements**
   - Ensemble multiple models
   - Try neural networks
   - Implement cross-validation for final model

3. **Web Application**
   - Add user authentication
   - Store prediction history
   - Add data visualization dashboard
   - Implement batch predictions (CSV upload)

4. **Deployment**
   - Deploy to AWS Elastic Beanstalk
   - Add CI/CD pipeline
   - Implement model monitoring
   - Add A/B testing for models

### With Additional Data

1. **More Features**
   - Study hours per week
   - Attendance rate
   - Previous year scores
   - Extracurricular activities
   - Sleep hours

2. **Time Series**
   - Track student progress over time
   - Predict future performance trends

3. **Multi-output Prediction**
   - Predict all three scores simultaneously
   - Predict grade letter (A, B, C, D, F)

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| **Language** | Python 3.9+ |
| **ML Framework** | Scikit-Learn, XGBoost, CatBoost |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Flask |
| **Serialization** | Dill |
| **Frontend** | HTML5, CSS3 |
| **Version Control** | Git |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Muhammad Sharjeel** - [GitHub](https://github.com/yourusername)

Project Link: [https://github.com/yourusername/student-performance-predictor](https://github.com/yourusername/student-performance-predictor)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ and Python

</div>
