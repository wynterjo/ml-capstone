import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Dummy Data: Hours Spent Studying vs. Exam Score
###  This simulates the 'Data Wrangling' part of my Capstone ###
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]) # Hours
y = np.array([45, 50, 60, 65, 75, 80, 90, 95])         # Scores

# 2. Initialize and Train the Model
model = LinearRegression()
model.fit(X, y)

# 3. Make a Prediction
# Let's predict the score for someone who studied 10 hours
hours_to_predict = [[10]]
predicted_score = model.predict(hours_to_predict)

print(f"Prediction: A student who studies 10 hours is predicted to score: {predicted_score[0]:.2f}%")
print(f"Model Accuracy (R^2): {model.score(X, y):.4f}")