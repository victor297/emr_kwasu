import streamlit as st
import pandas as pd
import joblib

# Load the model, encoders, and TF-IDF vectorizer
xgb_model = joblib.load('xgb_model.pkl')
tfidf = joblib.load('tfidf.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

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

# Function to predict condition based on symptoms
def predict_condition(age, gender, ethnicity, symptoms):
    if 'no symptoms' in symptoms or 'healthy' in symptoms:
        return 'Non-Serious', ['None']

    symptoms_str = ', '.join(symptoms)
    clinical_notes = f"Patient presents with symptoms of {symptoms_str}."
    X_tfidf = tfidf.transform([clinical_notes]).toarray()
    
    gender_encoded = label_encoders['gender'].transform([gender])[0]
    ethnicity_encoded = label_encoders['ethnicity'].transform([ethnicity])[0]
    
    X_new = pd.DataFrame([[age, gender_encoded, ethnicity_encoded] + list(X_tfidf[0])], columns=feature_names)
    
    prediction = xgb_model.predict(X_new)[0]
    condition = 'Serious' if prediction == 1 else 'Non-Serious'
    
    # Determine clinical findings
    clinical_findings = []
    for cond, sym in conditions_symptoms.items():
        if all(symptom in sym for symptom in symptoms):
            clinical_findings.append(cond)
    
    return condition, clinical_findings

# Streamlit UI

st.header('Input Data')

age = st.slider('Age', 3, 90, 30)
gender = st.selectbox('Gender', ['Male', 'Female'])
ethnicity = st.selectbox('Ethnicity', ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'])
symptoms = st.multiselect('Symptoms', ['headache', 'dizziness', 'shortness of breath', 'nosebleeds', 'increased thirst', 'frequent urination',
'fatigue', 'blurred vision', 'wheezing', 'chest tightness', 'coughing', 'chronic cough', 'chest pain',
'irregular heartbeat', 'fever', 'chills', 'muscle pain', 'weakness', 'abdominal pain', 'rash',
'stomach pain', 'bloating', 'nausea', 'vomiting', 'night sweats', 'weight loss', 'swollen lymph nodes',
'healthy', 'no symptoms'])

if st.button('Predict Condition'):
    if not symptoms:
        st.error('Please select at least one symptom.')
    else:
        condition, clinical_findings = predict_condition(age, gender, ethnicity, symptoms)
        st.subheader('Predicted Condition')
        st.write(condition)
        
        st.subheader('Possible Medical Conditions Based on Symptoms')
        if clinical_findings:
            st.write(', '.join(clinical_findings))
        else:
            st.write('No specific medical conditions found for the given symptoms.')
