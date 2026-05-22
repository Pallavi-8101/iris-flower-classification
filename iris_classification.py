# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load dataset
df = pd.read_csv("D:\Oasis Internship 2026\Iris Flower Classification\Iris.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())


# Dataset information
print("\nDataset Info:")
print(df.info())


# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Remove Id column
df = df.drop("Id", axis=1)


# Data visualization
sns.pairplot(df, hue="Species")
plt.show()


# Features (Input)
X = df.drop("Species", axis=1)

# Target (Output)
y = df["Species"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create Random Forest model
model = RandomForestClassifier(random_state=42)


# Train model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Predict new flower
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\nPredicted Species:")
print(prediction[0])