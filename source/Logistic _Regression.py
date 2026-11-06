import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("logistic_data.csv")

# ==============================
# Data Visualization (Optional)
# ==============================
# df.plot(kind="scatter", x="Hours_Studied", y="Pass")
# df.plot(kind="scatter", x="Attendance", y="Pass")
# plt.show()

# ==============================
# Feature Selection
# ==============================
X = df[['Hours_Studied', 'Attendance']]   # Independent Variables
y = df['Pass']                            # Target Variable

# ==============================
# Split Dataset
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42
)

# ==============================
# Train Logistic Regression Model
# ==============================
model = LogisticRegression()
model.fit(X_train, y_train)

# ==============================
# Make Predictions
# ==============================
y_pred = model.predict(X_test)

# ==============================
# Evaluate Model
# ==============================
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# Predict New Student
# ==============================
hours_studied = 6
attendance = 85

prediction = model.predict([[hours_studied, attendance]])

print("\nPrediction:")

if prediction[0] == 1:
    print("Student is likely to PASS.")
else:
    print("Student is likely to FAIL.")