import pandas as pd
from faker import Faker
import random

# Initialize Faker
fake = Faker()

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

# Function to generate synthetic patient data
def generate_patient_record():
    condition = random.choice(list(conditions_symptoms.keys()))
    symptoms = random.choices(conditions_symptoms[condition], k=random.randint(1, len(conditions_symptoms[condition])))
    record = {
        'patient_id': fake.uuid4(),
        'name': fake.name(),
        'age': random.randint(18, 90),
        'gender': random.choice(['Male', 'Female']),
        'ethnicity': random.choice(['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other']),
        'medical_condition': condition,
        'symptoms': ', '.join(symptoms),
        'clinical_notes': generate_clinical_notes(condition, symptoms)
    }
    return record

# Function to generate clinical notes based on condition and symptoms
def generate_clinical_notes(condition, symptoms):
    symptoms_str = ', '.join(symptoms)
    notes = f"Patient presents with symptoms of {symptoms_str} associated with {condition}."
    return notes

# Generate a list of synthetic patient records
synthetic_data = [generate_patient_record() for _ in range(1000)]

# Convert the data to a pandas DataFrame
df = pd.DataFrame(synthetic_data)

# Save the DataFrame to a CSV file
df.to_csv('synthetic_emr_data.csv', index=False)

# Print a message to confirm the data has been saved
print("Synthetic EMR data has been saved to 'synthetic_emr_data.csv'")
