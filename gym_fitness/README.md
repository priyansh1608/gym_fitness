# Gym Fitness App

AI based fitness recommendation web application.

## Technologies

- Python
- Flask
- SQLite
- SQL
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- MediaPipe
- HTML
- CSS
- JavaScript

## Features

### User Profile

User enters:

- Name
- Age
- Gender
- Height
- Weight
- Fitness goal
- Activity level
- Food preference

### BMI

The application calculates BMI automatically.

### Machine Learning

Random Forest is used to classify the recommended workout category.

### Body Scanner

Browser webcam access is used as the foundation for pose/body landmark analysis.

### Diet Recommendation

The system provides educational diet suggestions based on:

- Goal
- Food preference
- Weight

### Workout Recommendation

The system provides workout recommendations based on:

- Goal
- Activity level
- BMI

## Run

Create virtual environment:

python -m venv venv

Activate:

Windows:

venv\Scripts\activate

Install:

pip install -r requirements.txt

Train model:

python train_model.py

Run:

python app.py

Open:

http://127.0.0.1:5000

## Disclaimer

This application is an educational software project.
It does not diagnose medical conditions and should not be
used as a replacement for professional medical, nutrition,
or fitness advice.