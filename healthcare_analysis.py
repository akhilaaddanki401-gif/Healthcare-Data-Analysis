import pandas as pd

import os


csv_path = os.path.join(os.path.dirname(__file__), "healthcare_dataset.csv")
df = pd.read_csv(csv_path)

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nUnique Values in Each Column:")
print(df.nunique())

print("\nData Types of Each Column:")
print(df.dtypes)

print("\nDuplicate Records:")
print(df[df.duplicated()].head(10))

print("\nMissing Values Percentage:")
print((df.isnull().sum() / len(df)) * 100)

print("\nGender Values:")
print(df["Gender"].unique())
print("\nMedical Condition Values:")
print(df["Medical Condition"].unique())
print("\nGender Distribution:")
print(df["Gender"].value_counts())

print("\nMedical Condition Distribution:")
print(df["Medical Condition"].value_counts())
print("\nTest Results Distribution:")
print(df["Test Results"].value_counts())
# STEP 11 - Data Cleaning

print("\n--- DATA CLEANING ---")

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Remove duplicate rows

duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows Before Cleaning:", duplicate_count)

df = df.drop_duplicates()

print("Duplicate Rows After Cleaning:", df.duplicated().sum())
# Separate numerical and categorical columns

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = df.select_dtypes(include=["object"]).columns

print("\nNumerical Columns:")
print(numerical_columns.tolist())

print("\nCategorical Columns:")
print(categorical_columns.tolist())
# Fill missing values

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())
print("\nDataset Shape After Cleaning:")
print(df.shape)

print("\nFirst 5 Rows After Cleaning:")
print(df.head())

# STEP 12 - Date Processing

print("\n--- DATE PROCESSING ---")

# Convert date columns to datetime format
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])

print("\nData Types After Date Conversion:")
print(df[["Date of Admission", "Discharge Date"]].dtypes)
# STEP 13 - Calculate Length of Stay

df["Length of Stay"] = (
    df["Discharge Date"] - df["Date of Admission"]
).dt.days

print("\nLength of Stay:")
print(df["Length of Stay"].head())

print("\nLength of Stay Statistics:")
print(df["Length of Stay"].describe())
# STEP 14 - Numerical Data Analysis

print("\n--- NUMERICAL DATA ANALYSIS ---")

print("\nNumerical Columns:")
print(numerical_columns.tolist())

print("\nNumerical Data Summary:")
print(df[numerical_columns].describe())

# STEP 15 - Detect Invalid Billing Amounts

print("\n--- INVALID BILLING AMOUNTS ---")

negative_billing = df[df["Billing Amount"] < 0]

print("\nNumber of Negative Billing Amounts:")
print(len(negative_billing))

print("\nNegative Billing Records:")
print(negative_billing[["Billing Amount", "Medical Condition"]].head(10))
# STEP 16 - Remove Invalid Billing Amounts

print("\n--- REMOVING INVALID BILLING AMOUNTS ---")

rows_before = len(df)

# Remove records where Billing Amount is negative
df = df[df["Billing Amount"] >= 0]

rows_after = len(df)

print("\nRows Before Removing Invalid Values:", rows_before)
print("Rows After Removing Invalid Values:", rows_after)
print("Invalid Rows Removed:", rows_before - rows_after)

print("\nMinimum Billing Amount After Cleaning:")
print(df["Billing Amount"].min())
# STEP 17 - Verify Billing Amount Cleaning

print("\n--- BILLING AMOUNT VERIFICATION ---")

print("Negative Billing Amounts Remaining:")
print((df["Billing Amount"] < 0).sum())

print("\nBilling Amount Summary After Cleaning:")
print(df["Billing Amount"].describe())
# STEP 18 - Categorical Data Analysis

print("\n--- CATEGORICAL COLUMNS ---")

categorical_columns = df.select_dtypes(include=["object"]).columns

print("\nCategorical Columns:")
print(categorical_columns.tolist())

print("\nNumber of Categorical Columns:")
print(len(categorical_columns))

# STEP 18B - Check Unique Values of Categorical Columns

print("\n--- UNIQUE VALUES OF CATEGORICAL COLUMNS ---")

for column in categorical_columns:
    print(f"\n{column}: {df[column].nunique()} unique values")
    
    # STEP 18C - Remove High-Cardinality Columns

print("\n--- REMOVING HIGH-CARDINALITY COLUMNS ---")

columns_to_remove = ["Name", "Doctor", "Hospital"]

df = df.drop(columns=columns_to_remove)

print("\nRemoved Columns:")
print(columns_to_remove)

print("\nRemaining Columns:")
print(df.columns.tolist())

print("\nDataset Shape After Removing High-Cardinality Columns:")
print(df.shape)

# STEP 18D - One-Hot Encoding

print("\n--- ONE-HOT ENCODING ---")

categorical_to_encode = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results"
]

df = pd.get_dummies(
    df,
    columns=categorical_to_encode,
    drop_first=True,
    dtype=int
)

print("\nDataset Shape After Encoding:")
print(df.shape)

print("\nColumns After Encoding:")
print(df.columns.tolist())

# STEP 19 - Check Data Types After Encoding

print("\n--- DATA TYPES AFTER ENCODING ---")

print(df.dtypes)
# STEP 20 - Final Missing Values Check

print("\n--- FINAL MISSING VALUES CHECK ---")

print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())
# STEP 21 - Final Duplicate Check

print("\n--- FINAL DUPLICATE CHECK ---")

print("Duplicate Rows Remaining:")
print(df.duplicated().sum())
# STEP 22 - Create Date-Based Features

print("\n--- DATE-BASED FEATURES ---")

# Admission year
df["Admission Year"] = df["Date of Admission"].dt.year

# Admission month
df["Admission Month"] = df["Date of Admission"].dt.month

# Admission day of week
df["Admission Day"] = df["Date of Admission"].dt.day_name()

print("\nNew Date Features:")
print(df[[
    "Date of Admission",
    "Admission Year",
    "Admission Month",
    "Admission Day"
]].head())

print("\nAdmission Year Distribution:")
print(df["Admission Year"].value_counts().sort_index())

print("\nAdmission Month Distribution:")
print(df["Admission Month"].value_counts().sort_index())

# STEP 23 - Numerical EDA

print("\n--- NUMERICAL EDA ---")

# Recalculate numerical columns after all transformations
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nNumerical Columns:")
print(numerical_columns.tolist())

print("\nNumerical Statistics:")
print(df[numerical_columns].describe())
# STEP 24 - Medical Condition EDA

print("\n--- MEDICAL CONDITION ANALYSIS ---")

condition_counts = df[
    [
        "Medical Condition_Asthma",
        "Medical Condition_Cancer",
        "Medical Condition_Diabetes",
        "Medical Condition_Hypertension",
        "Medical Condition_Obesity"
    ]
].sum()

# Arthritis is the reference category because drop_first=True was used
arthritis_count = len(df) - condition_counts.sum()

condition_counts["Medical Condition_Arthritis"] = arthritis_count

print("\nPatient Count by Medical Condition:")
print(condition_counts.sort_values(ascending=False))
# STEP 25 - Gender vs Medical Condition Analysis

print("\n--- GENDER VS MEDICAL CONDITION ---")

# Convert encoded gender back to readable labels
df["Gender_Label"] = df["Gender_Male"].map({
    0: "Female",
    1: "Male"
})

medical_conditions = [
    "Asthma",
    "Cancer",
    "Diabetes",
    "Hypertension",
    "Obesity",
    "Arthritis"
]

# Count Male and Female patients for each condition
for condition in medical_conditions:

    if condition == "Arthritis":
        condition_data = df[
            (df["Medical Condition_Asthma"] == 0) &
            (df["Medical Condition_Cancer"] == 0) &
            (df["Medical Condition_Diabetes"] == 0) &
            (df["Medical Condition_Hypertension"] == 0) &
            (df["Medical Condition_Obesity"] == 0)
        ]
    else:
        condition_data = df[
            df[f"Medical Condition_{condition}"] == 1
        ]

    print(f"\n{condition}:")
    print(condition_data["Gender_Label"].value_counts())
    # STEP 25 - Gender vs Medical Condition Analysis

print("\n--- GENDER VS MEDICAL CONDITION ---")

df["Gender_Label"] = df["Gender_Male"].map({
    0: "Female",
    1: "Male"
})

medical_conditions = [
    "Asthma",
    "Cancer",
    "Diabetes",
    "Hypertension",
    "Obesity",
    "Arthritis"
]

for condition in medical_conditions:

    if condition == "Arthritis":
        condition_data = df[
            (df["Medical Condition_Asthma"] == 0) &
            (df["Medical Condition_Cancer"] == 0) &
            (df["Medical Condition_Diabetes"] == 0) &
            (df["Medical Condition_Hypertension"] == 0) &
            (df["Medical Condition_Obesity"] == 0)
        ]
    else:
        condition_data = df[
            df[f"Medical Condition_{condition}"] == 1
        ]

    print(f"\n{condition}:")
    print(condition_data["Gender_Label"].value_counts())
    # STEP 26 - Age Group Analysis

print("\n--- AGE GROUP ANALYSIS ---")

# Create age groups
df["Age Group"] = pd.cut(
    df["Age"],
    bins=[0, 18, 30, 45, 60, 75, 100],
    labels=[
        "0-18",
        "19-30",
        "31-45",
        "46-60",
        "61-75",
        "76+"
    ]
)

print("\nPatient Count by Age Group:")
print(df["Age Group"].value_counts().sort_index())

print("\nAverage Billing Amount by Age Group:")
print(
    df.groupby(
        "Age Group",
        observed=True
    )["Billing Amount"].mean()
)
# STEP 27 - Admission Type Analysis

print("\n--- ADMISSION TYPE ANALYSIS ---")

# Admission Type is encoded:
# Elective = reference category
# Emergency and Urgent have separate columns

admission_counts = {
    "Elective": len(df) - (
        df["Admission Type_Emergency"].sum()
        + df["Admission Type_Urgent"].sum()
    ),
    "Emergency": df["Admission Type_Emergency"].sum(),
    "Urgent": df["Admission Type_Urgent"].sum()
}

print("\nPatient Count by Admission Type:")

for admission_type, count in admission_counts.items():
    print(f"{admission_type}: {count}")

print("\nAverage Billing Amount by Admission Type:")

for admission_type in admission_counts:

    if admission_type == "Elective":
        data = df[
            (df["Admission Type_Emergency"] == 0) &
            (df["Admission Type_Urgent"] == 0)
        ]

    elif admission_type == "Emergency":
        data = df[df["Admission Type_Emergency"] == 1]

    else:
        data = df[df["Admission Type_Urgent"] == 1]

    print(
        f"{admission_type}: "
        f"{data['Billing Amount'].mean():.2f}"
    )
    # STEP 28 - Billing Amount Analysis

print("\n--- BILLING AMOUNT ANALYSIS ---")

print("\nBilling Amount Statistics:")
print(df["Billing Amount"].describe())

print("\nAverage Billing Amount:")
print(df["Billing Amount"].mean())

print("\nMedian Billing Amount:")
print(df["Billing Amount"].median())

print("\nMinimum Billing Amount:")
print(df["Billing Amount"].min())

print("\nMaximum Billing Amount:")
print(df["Billing Amount"].max())
# STEP 29 - Length of Stay Analysis

print("\n--- LENGTH OF STAY ANALYSIS ---")

print("\nLength of Stay Statistics:")
print(df["Length of Stay"].describe())

print("\nAverage Length of Stay:")
print(df["Length of Stay"].mean())

print("\nMedian Length of Stay:")
print(df["Length of Stay"].median())

print("\nMinimum Length of Stay:")
print(df["Length of Stay"].min())

print("\nMaximum Length of Stay:")
print(df["Length of Stay"].max())

print("\nLength of Stay Distribution:")
print(df["Length of Stay"].value_counts().sort_index().head(20))
# STEP 30 - Medical Condition vs Billing Amount

print("\n--- MEDICAL CONDITION VS BILLING AMOUNT ---")

medical_conditions = [
    "Asthma",
    "Cancer",
    "Diabetes",
    "Hypertension",
    "Obesity",
    "Arthritis"
]

for condition in medical_conditions:

    if condition == "Arthritis":
        condition_data = df[
            (df["Medical Condition_Asthma"] == 0) &
            (df["Medical Condition_Cancer"] == 0) &
            (df["Medical Condition_Diabetes"] == 0) &
            (df["Medical Condition_Hypertension"] == 0) &
            (df["Medical Condition_Obesity"] == 0)
        ]
    else:
        condition_data = df[
            df[f"Medical Condition_{condition}"] == 1
        ]

    print(f"\n{condition}:")
    print("Average Billing Amount:",
          round(condition_data["Billing Amount"].mean(), 2))

    print("Median Billing Amount:",
          round(condition_data["Billing Amount"].median(), 2))

    print("Minimum Billing Amount:",
          round(condition_data["Billing Amount"].min(), 2))

    print("Maximum Billing Amount:",
          round(condition_data["Billing Amount"].max(), 2))
    # STEP 31 - Medical Condition vs Length of Stay

print("\n--- MEDICAL CONDITION VS LENGTH OF STAY ---")

medical_conditions = [
    "Asthma",
    "Cancer",
    "Diabetes",
    "Hypertension",
    "Obesity",
    "Arthritis"
]

for condition in medical_conditions:

    if condition == "Arthritis":
        condition_data = df[
            (df["Medical Condition_Asthma"] == 0) &
            (df["Medical Condition_Cancer"] == 0) &
            (df["Medical Condition_Diabetes"] == 0) &
            (df["Medical Condition_Hypertension"] == 0) &
            (df["Medical Condition_Obesity"] == 0)
        ]
    else:
        condition_data = df[
            df[f"Medical Condition_{condition}"] == 1
        ]

    print(f"\n{condition}:")
    print(
        "Average Length of Stay:",
        round(condition_data["Length of Stay"].mean(), 2)
    )
    print(
        "Median Length of Stay:",
        round(condition_data["Length of Stay"].median(), 2)
    )
    print(
        "Minimum Length of Stay:",
        condition_data["Length of Stay"].min()
    )
    print(
        "Maximum Length of Stay:",
        condition_data["Length of Stay"].max()
    )
    # STEP 32 - Insurance Provider Analysis

print("\n--- INSURANCE PROVIDER ANALYSIS ---")

insurance_columns = [
    "Insurance Provider_Blue Cross",
    "Insurance Provider_Cigna",
    "Insurance Provider_Medicare",
    "Insurance Provider_UnitedHealthcare"
]

# A provider was dropped during one-hot encoding,
# so calculate its count from the remaining records.
provider_counts = {}

for column in insurance_columns:
    provider_name = column.replace("Insurance Provider_", "")
    provider_counts[provider_name] = df[column].sum()

provider_counts["Aetna"] = len(df) - sum(provider_counts.values())

print("\nPatient Count by Insurance Provider:")

for provider, count in provider_counts.items():
    print(f"{provider}: {int(count)}")

print("\nAverage Billing Amount by Insurance Provider:")

for provider in provider_counts:

    if provider == "Aetna":
        provider_data = df[
            (df["Insurance Provider_Blue Cross"] == 0) &
            (df["Insurance Provider_Cigna"] == 0) &
            (df["Insurance Provider_Medicare"] == 0) &
            (df["Insurance Provider_UnitedHealthcare"] == 0)
        ]
    else:
        provider_data = df[
            df[f"Insurance Provider_{provider}"] == 1
        ]

    print(
        f"{provider}: "
        f"{provider_data['Billing Amount'].mean():.2f}"
    )
    # STEP 33 - Medication Analysis

print("\n--- MEDICATION ANALYSIS ---")

medication_columns = [
    "Medication_Ibuprofen",
    "Medication_Lipitor",
    "Medication_Paracetamol",
    "Medication_Penicillin"
]

medication_counts = {}

for column in medication_columns:
    medication_name = column.replace("Medication_", "")
    medication_counts[medication_name] = df[column].sum()

# Aspirin is the reference category because drop_first=True was used
medication_counts["Aspirin"] = len(df) - sum(medication_counts.values())

print("\nPatient Count by Medication:")

for medication, count in medication_counts.items():
    print(f"{medication}: {int(count)}")

print("\nAverage Billing Amount by Medication:")

for medication in medication_counts:

    if medication == "Aspirin":
        medication_data = df[
            (df["Medication_Ibuprofen"] == 0) &
            (df["Medication_Lipitor"] == 0) &
            (df["Medication_Paracetamol"] == 0) &
            (df["Medication_Penicillin"] == 0)
        ]
    else:
        medication_data = df[
            df[f"Medication_{medication}"] == 1
        ]

    print(
        f"{medication}: "
        f"{medication_data['Billing Amount'].mean():.2f}"
    )
    # STEP 34 - Test Results Analysis

print("\n--- TEST RESULTS ANALYSIS ---")

test_result_columns = [
    "Test Results_Inconclusive",
    "Test Results_Normal"
]

test_result_counts = {}

for column in test_result_columns:
    result_name = column.replace("Test Results_", "")
    test_result_counts[result_name] = df[column].sum()

# Abnormal is the reference category
test_result_counts["Abnormal"] = (
    len(df) - sum(test_result_counts.values())
)

print("\nPatient Count by Test Result:")

for result, count in test_result_counts.items():
    print(f"{result}: {int(count)}")

print("\nAverage Billing Amount by Test Result:")

for result in test_result_counts:

    if result == "Abnormal":
        result_data = df[
            (df["Test Results_Inconclusive"] == 0) &
            (df["Test Results_Normal"] == 0)
        ]
    else:
        result_data = df[
            df[f"Test Results_{result}"] == 1
        ]

    print(
        f"{result}: "
        f"{result_data['Billing Amount'].mean():.2f}"
    )
    # STEP 35 - Correlation Analysis

print("\n--- CORRELATION ANALYSIS ---")

# Select numerical columns
correlation_columns = [
    "Age",
    "Billing Amount",
    "Room Number",
    "Length of Stay"
]

correlation_matrix = df[correlation_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

print("\nCorrelation with Billing Amount:")

print(
    correlation_matrix["Billing Amount"]
    .sort_values(ascending=False)
)
# STEP 36 - Data Visualization

import matplotlib.pyplot as plt

print("\n--- DATA VISUALIZATION ---")

# 36.1 Medical Condition Distribution
medical_conditions = [
    "Asthma",
    "Cancer",
    "Diabetes",
    "Hypertension",
    "Obesity",
    "Arthritis"
]

condition_counts = []

for condition in medical_conditions:

    if condition == "Arthritis":
        count = len(df) - (
            df["Medical Condition_Asthma"].sum()
            + df["Medical Condition_Cancer"].sum()
            + df["Medical Condition_Diabetes"].sum()
            + df["Medical Condition_Hypertension"].sum()
            + df["Medical Condition_Obesity"].sum()
        )
    else:
        count = df[f"Medical Condition_{condition}"].sum()

    condition_counts.append(count)

plt.figure(figsize=(8, 5))
plt.bar(medical_conditions, condition_counts)
plt.title("Patients by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Number of Patients")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 36.2 Age Group Distribution

plt.figure(figsize=(8, 5))
df["Age Group"].value_counts().sort_index().plot(kind="bar")
plt.title("Patients by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# 36.3 Billing Amount Distribution

plt.figure(figsize=(8, 5))
plt.hist(df["Billing Amount"], bins=30)
plt.title("Billing Amount Distribution")
plt.xlabel("Billing Amount")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()


# 36.4 Length of Stay Distribution

plt.figure(figsize=(8, 5))
plt.hist(df["Length of Stay"], bins=20)
plt.title("Length of Stay Distribution")
plt.xlabel("Length of Stay (Days)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()
# STEP 37 - Query-Based Experiments

print("\n--- QUERY BASED EXPERIMENTS ---")


# Experiment 1 - Patients with high billing amount
print("\nExperiment 1: High Billing Amount")

high_billing = df[df["Billing Amount"] > 40000]

print("Number of Patients:")
print(len(high_billing))

print("\nSample Records:")
print(
    high_billing[
        ["Age", "Billing Amount", "Room Number", "Length of Stay"]
    ].head(10)
)


# Experiment 2 - Long hospital stay
print("\nExperiment 2: Long Hospital Stay")

long_stay = df[df["Length of Stay"] > 10]

print("Number of Patients:")
print(len(long_stay))

print("\nSample Records:")
print(
    long_stay[
        ["Age", "Billing Amount", "Length of Stay"]
    ].head(10)
)


# Experiment 3 - Older patients with high billing
print("\nExperiment 3: Older Patients with High Billing")

older_high_billing = df[
    (df["Age"] >= 60) &
    (df["Billing Amount"] > 40000)
]

print("Number of Patients:")
print(len(older_high_billing))

print("\nSample Records:")
print(
    older_high_billing[
        ["Age", "Billing Amount", "Length of Stay"]
    ].head(10)
)


# Experiment 4 - Long stay with high billing
print("\nExperiment 4: Long Stay + High Billing")

long_stay_high_billing = df[
    (df["Length of Stay"] > 10) &
    (df["Billing Amount"] > 40000)
]

print("Number of Patients:")
print(len(long_stay_high_billing))

print("\nSample Records:")
print(
    long_stay_high_billing[
        ["Age", "Billing Amount", "Length of Stay"]
    ].head(10)
)
# STEP 38 - Save Final Cleaned Dataset

print("\n--- SAVING FINAL CLEANED DATASET ---")

#output_file = "healthcare_cleaned.csv"

#df.to_csv(output_file, index=False)

#print("\nFinal cleaned dataset saved successfully!")
#print("File Name:", output_file)
#print("Final Dataset Shape:", df.shape)
import os

output_file = os.path.join(
    os.path.dirname(__file__),
    "healthcare_cleaned.csv"
)

df.to_csv(output_file, index=False)

print("\nFinal cleaned dataset saved successfully!")
print("File:", output_file)
print("Final Dataset Shape:", df.shape)