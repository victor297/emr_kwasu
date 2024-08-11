import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

# Load the synthetic data
df = pd.read_csv('synthetic_emr_data.csv')

# Define conditions and symptoms
conditions_symptoms = {
    'Hypertension': ['headache', 'dizziness', 'shortness of breath', 'nosebleeds'],
    'Diabetes': ['increased thirst', 'frequent urination', 'fatigue', 'blurred vision'],
    'Asthma': ['wheezing', 'shortness of breath', 'chest tightness', 'coughing'],
    'COPD': ['chronic cough', 'wheezing', 'shortness of breath', 'chest tightness'],
    'Heart Disease': ['chest pain', 'shortness of breath', 'fatigue', 'irregular heartbeat'],
    'None': ['healthy', 'no symptoms'],
    'Malaria': ['fever', 'chills', 'headache', 'muscle pain'],
    'Typhoid': ['fever', 'weakness', 'abdominal pain', 'rash'],
    'Ulcer': ['stomach pain', 'bloating', 'nausea', 'vomiting'],
    'HIV': ['fever', 'night sweats', 'weight loss', 'swollen lymph nodes']
}

# Define target variable: 1 for serious condition, 0 for non-serious condition
serious_conditions = ['Hypertension', 'Diabetes', 'Asthma', 'COPD', 'Heart Disease', 'Typhoid', 'Ulcer', 'HIV']
df['outcome'] = df['medical_condition'].apply(lambda x: 1 if x in serious_conditions else 0)

# Preprocess text data: TF-IDF vectorization of clinical notes
tfidf = TfidfVectorizer(max_features=100)
X_tfidf = tfidf.fit_transform(df['clinical_notes']).toarray()

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'ethnicity']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Combine all features
X = pd.concat([df[['age', 'gender', 'ethnicity']], pd.DataFrame(X_tfidf)], axis=1)
y = df['outcome']

# Train/test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Save the model, encoders, and TF-IDF vectorizer
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

# Evaluate the model
y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot other metrics
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(10, 5))
plt.bar(names, values)
plt.title('Model Evaluation Metrics')
plt.ylim(0, 1)
plt.savefig('evaluation_metrics.png')
