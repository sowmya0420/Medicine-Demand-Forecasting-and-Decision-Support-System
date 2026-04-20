import streamlit as st
import pandas as pd

# Import modules
from modules.obj3 import run_obj3_classification
from modules.obj1 import run_obj1_forecasting, get_forecast_result
from modules.obj2 import run_obj2_alerts
from modules.obj4 import run_obj4_expiry_optimization
# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Medicine Demand Forecasting System",
    layout="wide"
)

# -------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------
st.sidebar.title("Navigation")

module = st.sidebar.radio(
    "Go to:",
    [
        "Dashboard",
        "Module 1 - Medicine Movement Classification (Fuzzy)",
        "Module 2 - Demand Forecasting",
        "Module 3 - Intelligent Alert System",
        "Module 4 - Expiry Optimization (Fuzzy)"
    ]
)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/preprocessed_pharmacy.csv")
    return df

df = load_data()

# -------------------------------------------------
# Dashboard
# -------------------------------------------------
if module == "Dashboard":
    st.title("Medicine Demand Forecasting & Decision Support System")

    st.markdown("""
    This system helps in pharmacy inventory management by analyzing medicine sales,
    classifying medicine movement, forecasting future demand, generating alerts,
    and reducing expiry risk using fuzzy logic and soft computing techniques.
    """)

    st.markdown("---")

    st.subheader("System Workflow")

    st.markdown("""
    **Flow of the System:**

    Sales Data → Movement Classification → Demand Forecasting → Alert Generation → Expiry Optimization
    """)

    st.markdown("---")

    st.subheader("System Modules")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### Module 1")
        st.write("Medicine Movement Classification")
        st.write("Fuzzy Logic Based")

    with col2:
        st.markdown("### Module 2")
        st.write("Demand Forecasting")
        st.write("Soft Computing")

    with col3:
        st.markdown("### Module 3")
        st.write("Intelligent Alert System")
        st.write("Decision Support")

    with col4:
        st.markdown("### Module 4")
        st.write("Expiry Optimization")
        st.write("Fuzzy Logic Based")

    st.markdown("---")

    st.subheader("Dataset Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", len(df))
    c2.metric("Total Medicines", df['product_id'].nunique())
    c3.metric("Total Categories", df['category'].nunique())

    st.success("Select a module from the sidebar.")

# -------------------------------------------------
# Module 1
# -------------------------------------------------
elif module == "Module 1 - Medicine Movement Classification (Fuzzy)":
    run_obj3_classification(df)

# -------------------------------------------------
# Module 2
# -------------------------------------------------
elif module == "Module 2 - Demand Forecasting":
    run_obj1_forecasting(df)

# -------------------------------------------------
# Module 3
# -------------------------------------------------
elif module == "Module 3 - Intelligent Alert System":
    forecast_result = get_forecast_result(df)
    run_obj2_alerts(df, forecast_result)

# -------------------------------------------------
# Module 4
# -------------------------------------------------
elif module == "Module 4 - Expiry Optimization (Fuzzy)":
    run_obj4_expiry_optimization(df)

