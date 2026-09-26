import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime

# ---------------------------
# INITIALIZATION
# ---------------------------

fake = Faker()
np.random.seed(42)

N = 10000

# ---------------------------
# DEMOGRAPHICS
# ---------------------------

patient_id = np.arange(1, N + 1)

age = np.random.randint(18, 90, N)

gender = np.random.choice(
    ["Male", "Female"],
    N,
    p=[0.52, 0.48]
)

height = np.random.normal(165, 10, N)
height = np.clip(height, 140, 200)

weight = np.random.normal(70, 15, N)
weight = np.clip(weight, 40, 150)

bmi = weight / ((height / 100) ** 2)

blood_group = np.random.choice(
    ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
    N
)

# ---------------------------
# LIFESTYLE
# ---------------------------

smoking = np.random.choice(
    ["Yes", "No"],
    N,
    p=[0.25, 0.75]
)

alcohol = np.random.choice(
    ["Yes", "No"],
    N,
    p=[0.35, 0.65]
)

physical_activity = np.random.choice(
    ["Low", "Moderate", "High"],
    N,
    p=[0.30, 0.50, 0.20]
)

# ---------------------------
# DIABETES
# ---------------------------

diabetes = []

for i in range(N):

    risk = 0.05

    if age[i] > 45:
        risk += 0.20

    if bmi[i] > 30:
        risk += 0.25

    if smoking[i] == "Yes":
        risk += 0.10

    diabetes.append(np.random.rand() < risk)

diabetes = np.array(diabetes)

# ---------------------------
# HYPERTENSION
# ---------------------------

hypertension = []

for i in range(N):

    risk = 0.08

    if age[i] > 50:
        risk += 0.25

    if bmi[i] > 28:
        risk += 0.20

    if smoking[i] == "Yes":
        risk += 0.10

    hypertension.append(np.random.rand() < risk)

hypertension = np.array(hypertension)

# ---------------------------
# HEART DISEASE
# ---------------------------

heart_disease = []

for i in range(N):

    risk = 0.03

    if hypertension[i]:
        risk += 0.25

    if diabetes[i]:
        risk += 0.15

    if smoking[i] == "Yes":
        risk += 0.20

    if age[i] > 60:
        risk += 0.20

    heart_disease.append(np.random.rand() < risk)

heart_disease = np.array(heart_disease)

# ---------------------------
# KIDNEY DISEASE
# ---------------------------

kidney_disease = []

for i in range(N):

    risk = 0.02

    if diabetes[i]:
        risk += 0.15

    if age[i] > 60:
        risk += 0.10

    kidney_disease.append(np.random.rand() < risk)

kidney_disease = np.array(kidney_disease)

# ---------------------------
# BLOOD PRESSURE
# ---------------------------

systolic_bp = []
diastolic_bp = []

for i in range(N):

    if hypertension[i]:

        systolic_bp.append(
            np.random.normal(150, 12)
        )

        diastolic_bp.append(
            np.random.normal(95, 8)
        )

    else:

        systolic_bp.append(
            np.random.normal(120, 10)
        )

        diastolic_bp.append(
            np.random.normal(80, 6)
        )

# ---------------------------
# HEART RATE
# ---------------------------

heart_rate = []

for i in range(N):

    if heart_disease[i]:
        heart_rate.append(
            np.random.normal(95, 15)
        )
    else:
        heart_rate.append(
            np.random.normal(75, 10)
        )

# ---------------------------
# OXYGEN SATURATION
# ---------------------------

spo2 = np.random.normal(97, 2, N)
spo2 = np.clip(spo2, 85, 100)

# ---------------------------
# BLOOD SUGAR
# ---------------------------

fasting_sugar = []

for i in range(N):

    if diabetes[i]:
        fasting_sugar.append(
            np.random.normal(180, 40)
        )
    else:
        fasting_sugar.append(
            np.random.normal(95, 12)
        )

# ---------------------------
# HBA1C
# ---------------------------

hba1c = []

for i in range(N):

    if diabetes[i]:
        hba1c.append(
            np.random.normal(8.5, 1)
        )
    else:
        hba1c.append(
            np.random.normal(5.2, 0.4)
        )

# ---------------------------
# HEMOGLOBIN
# ---------------------------

hemoglobin = np.random.normal(
    13.5,
    1.5,
    N
)

# ---------------------------
# WBC
# ---------------------------

wbc = np.random.normal(
    7500,
    1500,
    N
)

# ---------------------------
# PLATELETS
# ---------------------------

platelets = np.random.normal(
    250000,
    50000,
    N
)

# ---------------------------
# CHOLESTEROL
# ---------------------------

cholesterol = []

for i in range(N):

    if heart_disease[i]:
        cholesterol.append(
            np.random.normal(250, 30)
        )
    else:
        cholesterol.append(
            np.random.normal(180, 25)
        )

# ---------------------------
# HDL
# ---------------------------

hdl = np.random.normal(
    50,
    10,
    N
)

# ---------------------------
# LDL
# ---------------------------

ldl = np.random.normal(
    120,
    30,
    N
)

# ---------------------------
# TRIGLYCERIDES
# ---------------------------

triglycerides = np.random.normal(
    150,
    40,
    N
)

# ---------------------------
# CREATININE
# ---------------------------

creatinine = []

for i in range(N):

    if kidney_disease[i]:
        creatinine.append(
            np.random.normal(2.5, 0.8)
        )

    elif diabetes[i]:
        creatinine.append(
            np.random.normal(1.4, 0.3)
        )

    else:
        creatinine.append(
            np.random.normal(0.9, 0.2)
        )

# ---------------------------
# ECG
# ---------------------------

ecg_result = []

for i in range(N):

    if heart_disease[i]:

        ecg_result.append(
            np.random.choice(
                [
                    "Arrhythmia",
                    "Ischemia",
                    "Abnormal ECG"
                ]
            )
        )

    else:
        ecg_result.append("Normal")

# ---------------------------
# DIAGNOSIS
# ---------------------------

diagnosis = []

for i in range(N):

    diseases = []

    if diabetes[i]:
        diseases.append("Diabetes")

    if hypertension[i]:
        diseases.append("Hypertension")

    if heart_disease[i]:
        diseases.append("Heart Disease")

    if kidney_disease[i]:
        diseases.append("Kidney Disease")

    if len(diseases) == 0:
        diagnosis.append("Healthy")

    else:
        diagnosis.append(
            ", ".join(diseases)
        )

# ---------------------------
# VISIT DATE
# ---------------------------

start_date = datetime(2025, 1, 1)

visit_date = [
    fake.date_between(
        start_date='-1y',
        end_date='today'
    )
    for _ in range(N)
]

# ---------------------------
# DEPARTMENT
# ---------------------------

department = []

for d in diagnosis:

    if "Heart Disease" in d:
        department.append("Cardiology")

    elif "Diabetes" in d:
        department.append("Endocrinology")

    elif "Kidney Disease" in d:
        department.append("Nephrology")

    else:
        department.append("General Medicine")

# ---------------------------
# DATAFRAME
# ---------------------------

df = pd.DataFrame({

    "Patient_ID": patient_id,
    "Visit_Date": visit_date,

    "Age": age,
    "Gender": gender,

    "Height_cm": np.round(height, 1),
    "Weight_kg": np.round(weight, 1),
    "BMI": np.round(bmi, 2),

    "Blood_Group": blood_group,

    "Smoking": smoking,
    "Alcohol": alcohol,
    "Physical_Activity": physical_activity,

    "Diabetes": diabetes,
    "Hypertension": hypertension,
    "Heart_Disease": heart_disease,
    "Kidney_Disease": kidney_disease,

    "Systolic_BP": np.round(systolic_bp),
    "Diastolic_BP": np.round(diastolic_bp),

    "Heart_Rate": np.round(heart_rate),
    "SpO2": np.round(spo2, 1),

    "Hemoglobin": np.round(hemoglobin, 1),
    "WBC_Count": np.round(wbc),
    "Platelet_Count": np.round(platelets),

    "Fasting_Sugar": np.round(fasting_sugar),
    "HbA1c": np.round(hba1c, 1),

    "Creatinine": np.round(creatinine, 2),

    "Total_Cholesterol": np.round(cholesterol),
    "HDL": np.round(hdl),
    "LDL": np.round(ldl),
    "Triglycerides": np.round(triglycerides),

    "ECG_Result": ecg_result,

    "Diagnosis": diagnosis,
    "Department": department
})

# ---------------------------
# SAVE DATASET
# ---------------------------

df.to_csv(
    "synthetic_healthcare_dataset.csv",
    index=False
)

print(df.head())

print("\nDataset Shape:", df.shape)

print(
    "\nCSV saved as "
    "'synthetic_healthcare_dataset.csv'"
)