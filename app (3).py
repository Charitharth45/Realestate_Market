
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction Dashboard")
st.write(
    "Explore housing data, compare model performance, "
    "and estimate house prices using machine learning."
)

# -----------------------------
# 2. Load dataset and model
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("House_Price_India_cleaned.csv")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    return data

@st.cache_resource
def load_model():
    return joblib.load("best_house_price_model.pkl")

try:
    df = load_data()
    model = load_model()
except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.stop()

# -----------------------------
# 3. Prepare dashboard data
# -----------------------------
df["Sale Year"] = df["Date"].dt.year
df["Sale Month"] = df["Date"].dt.month

# -----------------------------
# 4. Sidebar navigation
# -----------------------------
page = st.sidebar.radio(
    "Dashboard Navigation",
    [
        "Overview",
        "Data Visualization",
        "Model Performance",
        "Price Prediction"
    ]
)

# -----------------------------
# 5. Overview page
# -----------------------------
if page == "Overview":

    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Houses", f"{len(df):,}")
    col2.metric("Average Price", f"{df['Price'].mean():,.0f}")
    col3.metric("Minimum Price", f"{df['Price'].min():,.0f}")
    col4.metric("Maximum Price", f"{df['Price'].max():,.0f}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Information")
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
    st.write("Missing values:", int(df.isna().sum().sum()))

# -----------------------------
# 6. Data visualization page
# -----------------------------
elif page == "Data Visualization":

    st.header("House Price Data Visualization")

    st.subheader("House Price Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["Price"].dropna(), bins=40, kde=True, ax=ax)
    ax.set_xlabel("House Price")
    ax.set_title("Distribution of House Prices")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Price vs Living Area")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df,
        x="living area",
        y="Price",
        alpha=0.4,
        ax=ax
    )
    ax.set_title("House Price vs Living Area")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Price by House Grade")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x="grade of the house",
        y="Price",
        ax=ax
    )
    ax.set_title("House Price by Grade")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        ax=ax
    )
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# 7. Model performance page
# -----------------------------
elif page == "Model Performance":

    st.header("Machine Learning Model Performance")

    results_path = Path("results.csv")

    if results_path.exists():

        results_df = pd.read_csv(results_path)

        st.dataframe(results_df, use_container_width=True)

        if "RMSE" in results_df.columns:
            st.subheader("Model Comparison - RMSE")

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=results_df.sort_values("RMSE"),
                x="Model",
                y="RMSE",
                ax=ax
            )
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)
            plt.close(fig)

        if "R2 Score" in results_df.columns:
            st.subheader("Model Comparison - R² Score")

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=results_df.sort_values(
                    "R2 Score",
                    ascending=False
                ),
                x="Model",
                y="R2 Score",
                ax=ax
            )
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info(
            "No results.csv found. Save your model comparison "
            "DataFrame as results.csv to display model metrics."
        )

    st.caption(
        "The prediction model loaded by this dashboard is "
        "best_house_price_model.pkl."
    )

# -----------------------------
# 8. House price prediction page
# -----------------------------
elif page == "Price Prediction":

    st.header("Predict House Price")

    # Recreate the exact feature columns used during training
    X_reference = df.drop(
        columns=["id", "Date", "Price"],
        errors="ignore"
    ).copy()

    X_reference["Sale_Year"] = df["Date"].dt.year
    X_reference["Sale_Month"] = df["Date"].dt.month

    if "Postal Code" in X_reference.columns:
        X_reference["Postal Code"] = (
            X_reference["Postal Code"].astype("string")
        )

    st.write("Enter the house details below:")

    user_input = {}

    with st.form("prediction_form"):

        for column in X_reference.columns:

            series = X_reference[column]

            if (
                pd.api.types.is_object_dtype(series)
                or isinstance(series.dtype, pd.StringDtype)
                or isinstance(series.dtype, pd.CategoricalDtype)
            ):
                choices = sorted(
                    series.dropna().astype(str).unique().tolist()
                )

                if choices:
                    user_input[column] = st.selectbox(
                        column,
                        choices
                    )
                else:
                    user_input[column] = st.text_input(column, "")

            else:
                median_value = series.median()

                if pd.isna(median_value):
                    median_value = 0.0

                user_input[column] = st.number_input(
                    column,
                    value=float(median_value),
                    step=1.0
                )

        submitted = st.form_submit_button("Predict House Price")

    if submitted:

        input_df = pd.DataFrame(
            [user_input],
            columns=X_reference.columns
        )

        if "Postal Code" in input_df.columns:
            input_df["Postal Code"] = (
                input_df["Postal Code"].astype("string")
            )

        try:
            prediction = model.predict(input_df)[0]

            st.success("Prediction completed!")

            st.metric(
                "Estimated House Price",
                f"{prediction:,.2f}"
            )

            st.caption(
                "This is a machine learning estimate, not a "
                "guaranteed market valuation."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
