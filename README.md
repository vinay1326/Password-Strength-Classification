# Password Strength Classification

## Overview
This project classifies passwords into different strength levels using TF-IDF vectorization and multinomial logistic regression. It involves extracting features from passwords and training a machine learning model to predict password strength.

## Steps Involved

### 1. Importing Required Libraries
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `sqlite3` to connect to the database
- `sklearn` for machine learning models

### 2. Loading Data
- The dataset is stored in an SQLite database (`password_Data.sqlite`).
- Data is retrieved using SQL queries (`SELECT * FROM Users`).

### 3. Exploratory Data Analysis (EDA)
- Checking for missing values, duplicates, and overall structure of the dataset.
- Identifying different password types (numeric, alphabetic, alphanumeric, uppercase, etc.).

### 4. Feature Engineering
- Extracting features from passwords:
  - `length`: Total length of the password.
  - `lower_case_frequency`: Ratio of lowercase letters.
  - `upper_case_frequency`: Ratio of uppercase letters.
  - `digit_case_frequency`: Ratio of digits.
  - `special_char_freq`: Ratio of special characters.

### 5. TF-IDF Transformation
- Converts passwords into numerical vectors using `TfidfVectorizer` (character-level analysis).
- Generates a feature matrix with TF-IDF values.

### 6. Data Preparation
- Combines TF-IDF features with engineered features.
- Splits data into training and testing sets using `train_test_split`.

### 7. Model Training
- Implements multinomial logistic regression using `LogisticRegression(multi_class='multinomial')`.
- Trains the model with `X_train` and `y_train`.

### 8. Model Evaluation
- Predicts password strength on `X_test`.
- Evaluates model performance using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report

### 9. Password Prediction Function
- Takes user input and converts it into the same feature format.
- Uses the trained model to classify password strength into:
  - Weak
  - Normal
  - Strong

## How to Use
1. Ensure the SQLite database file is available.
2. Run the notebook step-by-step.
3. Train the model and use `predict()` function to test password strength.

## Dependencies
- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn
- SQLite3

## Results
- The model achieves a reasonable accuracy score.
- Classification report provides precision, recall, and F1-score for each password strength category.

## Future Improvements
- Experimenting with other machine learning models (e.g., SVM, Random Forest).
- Expanding dataset to improve model generalization.
- Using deep learning models for better accuracy.

