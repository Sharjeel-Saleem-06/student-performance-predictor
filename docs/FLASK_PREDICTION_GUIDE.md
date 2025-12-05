# 🌐 Complete Guide: Flask Web App & Prediction Pipeline

## 📚 Table of Contents
1. [Overview: What is a Web App?](#-overview-what-is-a-web-app)
2. [Understanding Flask](#-understanding-flask)
3. [The app.py File (Complete Breakdown)](#-the-apppy-file-complete-breakdown)
4. [The predict_pipeline.py File](#-the-predict_pipelinepy-file)
5. [The HTML Templates](#-the-html-templates)
6. [How Everything Works Together](#-how-everything-works-together)
7. [The Complete Request-Response Flow](#-the-complete-request-response-flow)
8. [Testing the App](#-testing-the-app)

---

## 🌟 Overview: What is a Web App?

### What We Built
We built a **web application** that:
1. Shows a form where users enter student information
2. Takes that information and predicts the math score
3. Shows the prediction result

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WEB APPLICATION FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  👤 USER (Browser)                                                          │
│       │                                                                     │
│       │ 1. Opens http://localhost:5001/                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        FLASK SERVER (app.py)                        │    │
│  │                                                                     │    │
│  │  Route: /                                                           │    │
│  │  └── Returns: index.html (Welcome page)                             │    │
│  │                                                                     │    │
│  │  Route: /predictdata (GET)                                          │    │
│  │  └── Returns: home.html (Empty form)                                │    │
│  │                                                                     │    │
│  │  Route: /predictdata (POST)                                         │    │
│  │  └── Receives: Form data                                            │    │
│  │  └── Calls: predict_pipeline.py                                     │    │
│  │  └── Returns: home.html (With prediction result)                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                     │
│       │ 2. Prediction request                                               │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     PREDICT PIPELINE                                │    │
│  │                                                                     │    │
│  │  1. CustomData → Creates DataFrame from form data                   │    │
│  │  2. PredictPipeline → Loads model & preprocessor                    │    │
│  │  3. Transform → Applies same preprocessing as training              │    │
│  │  4. Predict → Gets prediction from model                            │    │
│  │  5. Return → Sends result back to Flask                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                     │
│       │ 3. Result: 71.68                                                    │
│       ▼                                                                     │
│  👤 USER sees: "THE prediction is 71.68"                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🍶 Understanding Flask

### What is Flask?

**Flask** is a Python web framework - it helps you build websites and web applications.

**Analogy:**
- **Python** = The language you speak
- **Flask** = The waiter that takes orders and delivers food
- **HTML** = The menu that customers see
- **Your ML Model** = The chef that cooks the food

### Key Flask Concepts

#### 1. Routes (URLs)

```python
@app.route('/')
def index():
    return "Hello!"
```

**What is a Route?**
- A route is a URL pattern that your app responds to
- `'/'` means the home page (e.g., `http://localhost:5001/`)
- `'/predictdata'` means `http://localhost:5001/predictdata`

```
URL                              Route
───────────────────────────────────────────────
http://localhost:5001/           '/'
http://localhost:5001/predictdata '/predictdata'
http://localhost:5001/about      '/about'
```

---

#### 2. HTTP Methods (GET vs POST)

```python
@app.route('/predictdata', methods=['GET', 'POST'])
```

**What are HTTP Methods?**

| Method | Purpose | Example |
|--------|---------|---------|
| **GET** | Request data (view a page) | Opening the form |
| **POST** | Send data (submit a form) | Submitting the prediction form |

```
GET Request:
User types URL → Browser asks server "Give me this page" → Server sends page

POST Request:
User fills form → Clicks Submit → Browser sends data to server → Server processes → Sends response
```

---

#### 3. Templates (HTML Files)

```python
return render_template('home.html', results=71.68)
```

**What is render_template?**
- Flask looks for HTML files in the `templates/` folder
- It fills in variables (like `results`) into the HTML
- Returns the complete HTML to the browser

```
templates/
├── index.html    ← Welcome page
└── home.html     ← Prediction form
```

---

#### 4. Request Object

```python
from flask import request

gender = request.form.get('gender')  # Gets form data
```

**What is the request object?**
- Contains all data sent by the user
- `request.form` - Data from HTML form
- `request.method` - 'GET' or 'POST'

---

## 📄 The app.py File (Complete Breakdown)

### The Complete Code

```python
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
```

---

### Line-by-Line Explanation

#### Imports (Lines 1-6)

```python
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
```

| Import | Purpose |
|--------|---------|
| `Flask` | The main Flask class to create the app |
| `request` | Access data sent by the user |
| `render_template` | Load and display HTML files |
| `numpy`, `pandas` | Data manipulation (used by pipeline) |
| `StandardScaler` | (Not actually used here, could be removed) |
| `CustomData` | Our class to create DataFrames from form data |
| `PredictPipeline` | Our class to make predictions |

---

#### Creating the Flask App (Lines 8-10)

```python
application = Flask(__name__)
app = application
```

**What is `Flask(__name__)`?**
```python
Flask(__name__)
      │
      └── __name__ tells Flask where to find templates, static files, etc.
          In this case, __name__ = "app" (the filename without .py)
```

**Why two variables?**
```python
application = Flask(__name__)  # AWS Elastic Beanstalk expects 'application'
app = application              # We use 'app' for convenience
```

---

#### Route 1: Home Page (Lines 14-16)

```python
@app.route('/')
def index():
    return render_template('index.html')
```

**What happens:**
1. User visits `http://localhost:5001/`
2. Flask calls the `index()` function
3. Flask loads `templates/index.html`
4. Flask sends the HTML to the user's browser

```
User: "Give me /"
Flask: "Here's index.html"
Browser: Shows the welcome page
```

---

#### Route 2: Prediction Page (Lines 18-41)

```python
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
```

**What does `methods=['GET', 'POST']` mean?**
- This route accepts BOTH GET and POST requests
- GET = Show the empty form
- POST = Process the submitted form

---

##### GET Request (Show Form)

```python
if request.method == 'GET':
    return render_template('home.html')
```

**What happens:**
1. User visits `http://localhost:5001/predictdata`
2. `request.method` is `'GET'`
3. Flask returns the empty form (home.html)

---

##### POST Request (Process Form)

```python
else:
    data = CustomData(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        reading_score=float(request.form.get('writing_score')),
        writing_score=float(request.form.get('reading_score'))
    )
```

**What is `request.form.get('gender')`?**
```
HTML Form:
<select name="gender">        ← name="gender"
    <option value="male">Male</option>
    <option value="female">Female</option>
</select>

When user selects "Female":
request.form.get('gender') → "female"
```

**The form field names must match:**
```
HTML: name="gender"           → request.form.get('gender')
HTML: name="ethnicity"        → request.form.get('ethnicity')
HTML: name="parental_level_of_education" → request.form.get('parental_level_of_education')
```

---

```python
    pred_df = data.get_data_as_data_frame()
```

**What this creates:**
```
pred_df:
┌────────┬───────────────┬─────────────────────────────┬──────────┬────────────────────────┬───────────────┬───────────────┐
│ gender │ race_ethnicity│ parental_level_of_education │ lunch    │ test_preparation_course│ reading_score │ writing_score │
├────────┼───────────────┼─────────────────────────────┼──────────┼────────────────────────┼───────────────┼───────────────┤
│ female │ group B       │ bachelor's degree           │ standard │ completed              │ 85            │ 82            │
└────────┴───────────────┴─────────────────────────────┴──────────┴────────────────────────┴───────────────┴───────────────┘
```

---

```python
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('home.html', results=results[0])
```

**What happens:**
1. Create a `PredictPipeline` object
2. Call `predict()` with the DataFrame
3. Get the prediction (e.g., `[71.68]`)
4. Return home.html with `results=71.68`

---

#### Running the Server (Lines 44-45)

```python
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `host` | `"0.0.0.0"` | Accept connections from any IP |
| `port` | `5001` | Run on port 5001 |
| `debug` | `True` | Auto-reload on code changes, show errors |

**What is `if __name__ == "__main__"`?**
- Only runs when you execute `python app.py` directly
- Does NOT run when imported by another file

---

## 🔮 The predict_pipeline.py File

### The Complete Code

```python
import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'proprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
```

---

### Class 1: PredictPipeline

#### Purpose
Loads the trained model and preprocessor, then makes predictions.

#### The predict() Method

```python
def predict(self, features):
    try:
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join('artifacts', 'proprocessor.pkl')
```

**What are these paths?**
```
artifacts/
├── model.pkl         ← The trained ML model (Linear Regression)
└── proprocessor.pkl  ← The preprocessing pipeline (OneHotEncoder, Scaler, etc.)
```

---

```python
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)
```

**What is load_object?**
- Function from `utils.py`
- Uses `dill` library to load saved Python objects
- Returns the actual model/preprocessor objects

```python
# What load_object does internally:
with open(file_path, "rb") as f:
    return dill.load(f)
```

---

```python
        data_scaled = preprocessor.transform(features)
```

**Why transform?**
- The model was trained on TRANSFORMED data
- New data must be transformed THE SAME WAY
- The preprocessor knows how (it was saved during training)

```
Input (features):
┌────────┬───────────────┬───────────────┬───────────────┐
│ gender │ race_ethnicity│ reading_score │ writing_score │
├────────┼───────────────┼───────────────┼───────────────┤
│ female │ group B       │ 85            │ 82            │
└────────┴───────────────┴───────────────┴───────────────┘

After preprocessor.transform():
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ 0.45  │ 1.0   │ 0.0   │ 0.0   │ 1.0   │ 0.0   │ 0.72  │ 0.68  │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
  (scaled numbers, one-hot encoded categories)
```

---

```python
        preds = model.predict(data_scaled)
        return preds
```

**What happens:**
1. Model receives the transformed data
2. Model calculates the prediction
3. Returns array like `[71.68]`

---

### Class 2: CustomData

#### Purpose
Converts form data into a pandas DataFrame that the model can use.

#### The __init__ Method

```python
def __init__(self,
    gender: str,
    race_ethnicity: str,
    parental_level_of_education,
    lunch: str,
    test_preparation_course: str,
    reading_score: int,
    writing_score: int):

    self.gender = gender
    self.race_ethnicity = race_ethnicity
    # ... etc
```

**What is this?**
- Stores all the form values as instance variables
- `str` and `int` are type hints (documentation)

---

#### The get_data_as_data_frame Method

```python
def get_data_as_data_frame(self):
    custom_data_input_dict = {
        "gender": [self.gender],
        "race_ethnicity": [self.race_ethnicity],
        "parental_level_of_education": [self.parental_level_of_education],
        "lunch": [self.lunch],
        "test_preparation_course": [self.test_preparation_course],
        "reading_score": [self.reading_score],
        "writing_score": [self.writing_score],
    }

    return pd.DataFrame(custom_data_input_dict)
```

**Why a dictionary with lists?**
```python
# DataFrame needs lists for columns:
{"gender": ["female"]}  # Creates 1 row with "female" in gender column

# Without list, it would fail:
{"gender": "female"}    # ERROR!
```

**What it creates:**
```
pd.DataFrame(custom_data_input_dict):

       gender  race_ethnicity  parental_level_of_education  lunch  ...
0      female  group B         bachelor's degree            standard ...
```

---

## 📝 The HTML Templates

### Template 1: index.html (Welcome Page)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Student Performance Predictor</title>
    <style>
        /* CSS styling */
    </style>
</head>
<body>
    <div class="container">
        <div class="emoji">📚</div>
        <h1>Student Performance Predictor</h1>
        <p>
            Predict a student's math score based on their background, 
            reading score, and writing score using Machine Learning!
        </p>
        <a href="/predictdata" class="btn">Start Prediction →</a>
    </div>
</body>
</html>
```

**Key parts:**
- `<a href="/predictdata">` - Link to the prediction form
- When clicked, browser sends GET request to `/predictdata`

---

### Template 2: home.html (Prediction Form)

#### The Form Tag

```html
<form action="{{ url_for('predict_datapoint')}}" method="post">
```

| Attribute | Value | Meaning |
|-----------|-------|---------|
| `action` | `{{ url_for('predict_datapoint') }}` | Where to send the form data |
| `method` | `post` | Send as POST request |

**What is `{{ url_for('predict_datapoint') }}`?**
- Jinja2 template syntax (Flask's template engine)
- `{{ }}` = Insert Python value here
- `url_for('predict_datapoint')` = Get URL for the function named `predict_datapoint`
- Result: `/predictdata`

---

#### Form Fields

```html
<select name="gender" required>
    <option value="">Select your Gender</option>
    <option value="male">Male</option>
    <option value="female">Female</option>
</select>
```

**Important attributes:**
- `name="gender"` - This becomes `request.form.get('gender')` in Python
- `value="male"` - The value sent when this option is selected
- `required` - Form won't submit without a selection

---

```html
<input type="number" name="reading_score" min='0' max='100' />
```

**Important attributes:**
- `type="number"` - Only allows numbers
- `name="reading_score"` - Becomes `request.form.get('reading_score')`
- `min='0' max='100'` - Validates the range

---

#### Displaying the Result

```html
<h2>
   THE prediction is {{results}}
</h2>
```

**What is `{{results}}`?**
- Jinja2 template syntax
- Flask replaces this with the actual value
- From: `render_template('home.html', results=71.68)`
- Result: `THE prediction is 71.68`

---

## 🔄 How Everything Works Together

### The Complete Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE REQUEST-RESPONSE FLOW                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  STEP 1: User Opens Home Page                                                  │
│  ──────────────────────────────                                                │
│  Browser → GET http://localhost:5001/                                          │
│         → Flask calls index()                                                  │
│         → Returns index.html                                                   │
│         → Browser shows welcome page                                           │
│                                                                                │
│  STEP 2: User Clicks "Start Prediction"                                        │
│  ──────────────────────────────────────                                        │
│  Browser → GET http://localhost:5001/predictdata                               │
│         → Flask calls predict_datapoint()                                      │
│         → request.method == 'GET'                                              │
│         → Returns home.html (empty form)                                       │
│         → Browser shows form                                                   │
│                                                                                │
│  STEP 3: User Fills Form and Clicks Submit                                     │
│  ─────────────────────────────────────────                                     │
│  Browser → POST http://localhost:5001/predictdata                              │
│         → With form data: gender=female, ethnicity=group B, ...                │
│         → Flask calls predict_datapoint()                                      │
│         → request.method == 'POST'                                             │
│                                                                                │
│  STEP 4: Flask Processes the Form                                              │
│  ────────────────────────────────                                              │
│  Flask → request.form.get('gender') → "female"                                 │
│       → request.form.get('ethnicity') → "group B"                              │
│       → ... (all form fields)                                                  │
│       → Creates CustomData object                                              │
│       → Calls data.get_data_as_data_frame()                                    │
│       → Gets DataFrame                                                         │
│                                                                                │
│  STEP 5: Prediction Pipeline                                                   │
│  ───────────────────────────                                                   │
│  Pipeline → load_object("artifacts/model.pkl")                                 │
│          → load_object("artifacts/proprocessor.pkl")                           │
│          → preprocessor.transform(features)                                    │
│          → model.predict(data_scaled)                                          │
│          → Returns [71.68]                                                     │
│                                                                                │
│  STEP 6: Return Result                                                         │
│  ─────────────────────                                                         │
│  Flask → render_template('home.html', results=71.68)                           │
│       → Jinja2 replaces {{results}} with 71.68                                 │
│       → Returns HTML to browser                                                │
│       → Browser shows: "THE prediction is 71.68"                               │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing the App

### Running the App

```bash
cd /Users/muhammadsharjeel/Documents/ML_PROJECT
python app.py
```

**Output:**
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5001
```

### Testing with Browser

1. Open `http://localhost:5001/` - See welcome page
2. Click "Start Prediction" - See form
3. Fill in the form and submit - See prediction

### Testing with curl

```bash
curl -X POST http://127.0.0.1:5001/predictdata \
  -d "gender=female" \
  -d "ethnicity=group B" \
  -d "parental_level_of_education=bachelor's degree" \
  -d "lunch=standard" \
  -d "test_preparation_course=completed" \
  -d "writing_score=85" \
  -d "reading_score=82"
```

**Result:**
```
THE prediction is 71.68204773399644
```

---

## 📊 File Structure Summary

```
ML_PROJECT/
├── app.py                    # Flask application (the server)
├── templates/                # HTML templates
│   ├── index.html            # Welcome page
│   └── home.html             # Prediction form
├── src/
│   ├── pipeline/
│   │   └── predict_pipeline.py  # Prediction logic
│   └── utils.py              # Helper functions (load_object)
└── artifacts/
    ├── model.pkl             # Trained ML model
    └── proprocessor.pkl      # Preprocessing pipeline
```

---

## 🎯 Key Concepts Summary

| Concept | What It Is |
|---------|------------|
| **Flask** | Python web framework |
| **Route** | URL pattern that triggers a function |
| **GET** | Request to view a page |
| **POST** | Request to submit data |
| **render_template** | Load HTML and fill in variables |
| **request.form** | Access form data submitted by user |
| **Jinja2** | Template engine ({{ variable }}) |
| **CustomData** | Class to convert form data to DataFrame |
| **PredictPipeline** | Class to load model and make predictions |
| **load_object** | Function to load .pkl files |

---

## 🚀 What Happens When You Make a Prediction

```
1. User fills form:
   ├── Gender: Female
   ├── Ethnicity: Group B
   ├── Parent Education: Bachelor's degree
   ├── Lunch: Standard
   ├── Test Prep: Completed
   ├── Reading Score: 85
   └── Writing Score: 82

2. Form data sent to Flask:
   POST /predictdata
   gender=female&ethnicity=group B&...

3. Flask creates CustomData:
   CustomData(gender="female", ...)

4. CustomData creates DataFrame:
   ┌────────┬───────────────┬─────────────────────────────┐
   │ gender │ race_ethnicity│ parental_level_of_education │...
   ├────────┼───────────────┼─────────────────────────────┤
   │ female │ group B       │ bachelor's degree           │...
   └────────┴───────────────┴─────────────────────────────┘

5. PredictPipeline loads model:
   model = LinearRegression (from model.pkl)
   preprocessor = ColumnTransformer (from proprocessor.pkl)

6. Transform data:
   [0.45, 1.0, 0.0, 0.0, 1.0, 0.0, 0.72, 0.68, ...]

7. Make prediction:
   model.predict([0.45, 1.0, ...]) → [71.68]

8. Return to user:
   "THE prediction is 71.68"
```

---

**Congratulations!** 🎉 You now understand how Flask, the prediction pipeline, and HTML templates work together to create a complete ML web application!

