import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
df = pd.read_csv('heart.csv')

# 2. Display the first five records
print("--- First 5 Records ---")
print(df.head())

# 3. Identify Numerical Features and Target Variable
# Standard Kaggle dataset target column is 'target'
target_var = 'target'
num_features = [col for col in df.columns if col != target_var]

print(f"\nTarget Variable: {target_var}")
print(f"Numerical Features ({len(num_features)}): {num_features}")

# 4. Check for missing values
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# Clean missing values if any exist
df = df.dropna()

# 5. Split dataset into 80% training and 20% testing
X = df[num_features]
y = df[target_var]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# --- TASK 2: Model Development ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Evaluation ---")
print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")

# Save trained model
joblib.dump(model, 'model.pkl')
print("\nModel saved successfully as 'model.pkl'.")