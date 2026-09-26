import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("synthetic_healthcare_dataset.csv")

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)

print("Shape:", df.shape)

# ==========================================
# CHECK MISSING VALUES
# ==========================================

print("\nMissing Values")

missing_values = df.isnull().sum()

print(missing_values)

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

numeric_cols = df.select_dtypes(
    include=np.number
).columns

for col in numeric_cols:

    df[col].fillna(
        df[col].median(),
        inplace=True
    )

categorical_cols = df.select_dtypes(
    exclude=np.number
).columns

for col in categorical_cols:

    df[col].fillna(
        df[col].mode()[0],
        inplace=True
    )

print("\nMissing values handled")

# ==========================================
# REMOVE DUPLICATES
# ==========================================

duplicates_before = df.duplicated().sum()

print("\nDuplicate Records:", duplicates_before)

df.drop_duplicates(
    inplace=True
)

duplicates_after = df.duplicated().sum()

print("Duplicates After Removal:",
      duplicates_after)

# ==========================================
# OUTLIER DETECTION
# IQR METHOD
# ==========================================

print("\nChecking Outliers")

numeric_cols = [
    "Age",
    "BMI",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "Hemoglobin",
    "Fasting_Sugar",
    "HbA1c",
    "Creatinine",
    "Total_Cholesterol"
]

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(
        f"{col}: {len(outliers)} outliers"
    )

# ==========================================
# OPTIONAL OUTLIER REMOVAL
# ==========================================

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[
        (df[col] >= lower) &
        (df[col] <= upper)
    ]

print("\nShape After Outlier Removal:")
print(df.shape)

# ==========================================
# SAVE CLEANED DATA
# ==========================================

df.to_csv(
    "healthcare_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved")

# ==========================================
# MIN MAX NORMALIZATION
# ==========================================

normalize_cols = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart_Rate",
    "SpO2",
    "Hemoglobin",
    "WBC_Count",
    "Platelet_Count",
    "Fasting_Sugar",
    "HbA1c",
    "Creatinine",
    "Total_Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides"
]

scaler = MinMaxScaler()

df[normalize_cols] = scaler.fit_transform(
    df[normalize_cols]
)

print("\nNormalization Completed")

# ==========================================
# VERIFY NORMALIZATION
# ==========================================

print("\nNormalized Data Sample")

print(
    df[normalize_cols]
    .head()
)

# ==========================================
# SAVE NORMALIZED DATA
# ==========================================

df.to_csv(
    "healthcare_normalized.csv",
    index=False
)

print("\nNormalized dataset saved")

print("\nFinal Shape:", df.shape)

print("\nProcessing Completed Successfully")