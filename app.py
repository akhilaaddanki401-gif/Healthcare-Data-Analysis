
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Healthcare Data Analysis",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# LOAD CLEANED DATASET
# ============================================================

csv_path = os.path.join(
    os.path.dirname(__file__),
    "healthcare_cleaned.csv"
)

df = pd.read_csv(csv_path)


# Convert date columns if available
if "Date of Admission" in df.columns:
    df["Date of Admission"] = pd.to_datetime(
        df["Date of Admission"],
        errors="coerce"
    )

if "Discharge Date" in df.columns:
    df["Discharge Date"] = pd.to_datetime(
        df["Discharge Date"],
        errors="coerce"
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏥 Healthcare Analysis")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Dashboard",
        "📊 Dataset Overview",
        "🧹 Data Cleaning",
        "📈 EDA",
        "🧪 Experiments",
        "⬇️ Download Dataset"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🏥 Healthcare Data Analysis Dashboard")

    st.markdown(
        """
        ### Welcome!

        This dashboard presents the analysis of a healthcare dataset
        after performing data cleaning, preprocessing and exploratory
        data analysis.

        Use the **sidebar** to explore different sections.
        """
    )

    st.markdown("---")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Total Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Duplicate Rows",
            df.duplicated().sum()
        )

    with col4:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.markdown("---")

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    st.markdown("---")

    st.subheader("🔍 First 10 Records")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("📌 Column Names")

    st.write(
        list(df.columns)
    )

    st.subheader("📐 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ============================================================
# DATA CLEANING
# ============================================================

elif page == "🧹 Data Cleaning":

    st.title("🧹 Data Cleaning & Preprocessing")

    st.markdown(
        """
        The original healthcare dataset was processed using the following
        data cleaning techniques:
        """
    )

    cleaning_steps = [
        "Duplicate records were identified and removed.",
        "Missing numerical values were handled using the median.",
        "Missing categorical values were handled using the mode.",
        "Date columns were converted into datetime format.",
        "Length of Stay was calculated using admission and discharge dates.",
        "Records with negative Billing Amount were removed.",
        "High-cardinality columns such as Name, Doctor and Hospital were removed.",
        "Categorical variables were converted using One-Hot Encoding.",
        "Additional date features such as Year, Month and Day were created."
    ]

    for i, step in enumerate(cleaning_steps, start=1):
        st.write(f"**{i}.** {step}")

    st.markdown("---")

    st.subheader("Missing Values After Cleaning")

    missing_values = df.isnull().sum()

    missing_values = missing_values[
        missing_values > 0
    ]

    if missing_values.empty:
        st.success("✅ No missing values found!")
    else:
        st.dataframe(
            missing_values,
            use_container_width=True
        )

    st.subheader("Duplicate Records After Cleaning")

    st.write(
        f"Duplicate rows: **{df.duplicated().sum()}**"
    )


# ============================================================
# EDA
# ============================================================

elif page == "📈 EDA":

    st.title("📈 Exploratory Data Analysis")

    # --------------------------------------------------------
    # Medical Condition
    # --------------------------------------------------------

    st.subheader("🏥 Medical Condition Distribution")

    condition_columns = [
        col for col in df.columns
        if col.startswith("Medical Condition_")
    ]

    if condition_columns:

        condition_counts = {}

        for col in condition_columns:
            condition_name = col.replace(
                "Medical Condition_",
                ""
            )

            condition_counts[condition_name] = int(
                df[col].sum()
            )

        condition_series = pd.Series(
            condition_counts
        )

        st.bar_chart(condition_series)

    else:
        st.info(
            "Medical Condition encoded columns were not found."
        )

    # --------------------------------------------------------
    # Billing Amount
    # --------------------------------------------------------

    st.subheader("💰 Billing Amount Distribution")

    if "Billing Amount" in df.columns:

        fig, ax = plt.subplots()

        ax.hist(
            df["Billing Amount"].dropna(),
            bins=30
        )

        ax.set_xlabel("Billing Amount")
        ax.set_ylabel("Number of Patients")
        ax.set_title("Billing Amount Distribution")

        st.pyplot(fig)

    # --------------------------------------------------------
    # Length of Stay
    # --------------------------------------------------------

    st.subheader("🛏️ Length of Stay Distribution")

    if "Length of Stay" in df.columns:

        fig, ax = plt.subplots()

        ax.hist(
            df["Length of Stay"].dropna(),
            bins=20
        )

        ax.set_xlabel("Length of Stay")
        ax.set_ylabel("Number of Patients")
        ax.set_title("Length of Stay Distribution")

        st.pyplot(fig)

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

    st.subheader("👥 Age Distribution")

    if "Age" in df.columns:

        fig, ax = plt.subplots()

        ax.hist(
            df["Age"].dropna(),
            bins=20
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Patients")
        ax.set_title("Age Distribution")

        st.pyplot(fig)


# ============================================================
# EXPERIMENTS
# ============================================================

elif page == "🧪 Experiments":

    st.title("🧪 Healthcare Query Experiments")

    st.markdown(
        """
        Select an experiment below to perform a specific query
        on the cleaned healthcare dataset.
        """
    )

    experiment = st.selectbox(
        "Select Experiment",
        [
            "High Billing Amount",
            "Long Hospital Stay",
            "Older Patients with High Billing",
            "Long Stay with High Billing"
        ]
    )

    st.markdown("---")


    # --------------------------------------------------------
    # EXPERIMENT 1
    # --------------------------------------------------------

    if experiment == "High Billing Amount":

        st.subheader("💰 Patients with Billing Amount > 40,000")

        if "Billing Amount" in df.columns:

            result = df[
                df["Billing Amount"] > 40000
            ]

            st.metric(
                "Matching Records",
                result.shape[0]
            )

            st.dataframe(
                result,
                use_container_width=True
            )

        else:
            st.error(
                "Billing Amount column not found."
            )


    # --------------------------------------------------------
    # EXPERIMENT 2
    # --------------------------------------------------------

    elif experiment == "Long Hospital Stay":

        st.subheader("🛏️ Patients with Length of Stay > 10 Days")

        if "Length of Stay" in df.columns:

            result = df[
                df["Length of Stay"] > 10
            ]

            st.metric(
                "Matching Records",
                result.shape[0]
            )

            st.dataframe(
                result,
                use_container_width=True
            )

        else:
            st.error(
                "Length of Stay column not found."
            )


    # --------------------------------------------------------
    # EXPERIMENT 3
    # --------------------------------------------------------

    elif experiment == "Older Patients with High Billing":

        st.subheader(
            "👴 Patients with Age ≥ 60 and Billing Amount > 40,000"
        )

        if (
            "Age" in df.columns
            and "Billing Amount" in df.columns
        ):

            result = df[
                (df["Age"] >= 60)
                &
                (df["Billing Amount"] > 40000)
            ]

            st.metric(
                "Matching Records",
                result.shape[0]
            )

            st.dataframe(
                result,
                use_container_width=True
            )

        else:
            st.error(
                "Required columns not found."
            )


    # --------------------------------------------------------
    # EXPERIMENT 4
    # --------------------------------------------------------

    elif experiment == "Long Stay with High Billing":

        st.subheader(
            "🏥 Patients with Length of Stay > 10 and Billing Amount > 40,000"
        )

        if (
            "Length of Stay" in df.columns
            and "Billing Amount" in df.columns
        ):

            result = df[
                (df["Length of Stay"] > 10)
                &
                (df["Billing Amount"] > 40000)
            ]

            st.metric(
                "Matching Records",
                result.shape[0]
            )

            st.dataframe(
                result,
                use_container_width=True
            )

        else:
            st.error(
                "Required columns not found."
            )


# ============================================================
# DOWNLOAD DATASET
# ============================================================

elif page == "⬇️ Download Dataset":

    st.title("⬇️ Download Cleaned Dataset")

    st.write(
        "Download the final cleaned and preprocessed healthcare dataset."
    )

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download healthcare_cleaned.csv",
        data=csv_data,
        file_name="healthcare_cleaned.csv",
        mime="text/csv"
    )

    st.success(
        "Your cleaned dataset is ready for download!"
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.markdown("---")

st.sidebar.info(
    "Healthcare Data Wrangling & Preprocessing Project"
)

