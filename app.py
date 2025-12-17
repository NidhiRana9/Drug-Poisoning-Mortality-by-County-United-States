import streamlit as st
import pandas as pd
import joblib

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="Drug Poisoning Death Rate Predictor",
    page_icon="💊",
    layout="wide"
)

# =============================
# Load model & encoders
# =============================
model = joblib.load(r"C:\Users\Muskaan R\Documents\drug_poisoning_streamlit\rf_model.pkl")
le_state = joblib.load(r"C:\Users\Muskaan R\Documents\drug_poisoning_streamlit\le_state.pkl")
le_urban = joblib.load(r"C:\Users\Muskaan R\Documents\drug_poisoning_streamlit\le_urban.pkl")

df = pd.read_csv(r"C:\Users\Muskaan R\Documents\drug_poisoning_streamlit\processed_data.csv")

# =============================
# App Header
# =============================
st.title("💊 Drug Poisoning Death Rate Prediction")
st.markdown(
    """
    This application predicts **drug poisoning death rates per 100,000 population**
    using a **Random Forest regression model** trained on historical public health data.
    """
)

st.divider()

# =============================
# Sidebar Inputs
# =============================
st.sidebar.header("🧮 Input Features")

year = st.sidebar.selectbox(
    "📅 Select Year",
    sorted(df["year"].unique())
)

state = st.sidebar.selectbox(
    "📍 Select State",
    sorted(df["state"].unique())
)

urban = st.sidebar.selectbox(
    "🏙️ Urban / Rural Category",
    sorted(df["urban_rural_category"].unique())
)

# =============================
# Encode inputs
# =============================
state_encoded = le_state.transform([state])[0]
urban_encoded = le_urban.transform([urban])[0]

input_df = pd.DataFrame({
    "year": [year],
    "state": [state_encoded],
    "urban_rural_category": [urban_encoded]
})

# =============================
# Prediction
# =============================
prediction = model.predict(input_df)[0]

# =============================
# Display Prediction
# =============================
st.subheader("📊 Prediction Result")

col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        label="Predicted Death Rate",
        value=f"{prediction:.2f}",
        help="Deaths per 100,000 population"
    )

with col2:
    if prediction < 10:
        st.success("🟢 **Low risk level** compared to national averages.")
    elif prediction < 20:
        st.warning("🟡 **Moderate risk level** — monitoring recommended.")
    else:
        st.error("🔴 **High risk level** — public health intervention advised.")

st.divider()

# =============================
# Historical Trend
# =============================
st.subheader("📈 Historical Trend (Selected State & Category)")

trend_df = df[
    (df["state"] == state) &
    (df["urban_rural_category"] == urban)
].sort_values("year")

if not trend_df.empty:
    st.line_chart(
        trend_df.set_index("year")["model_based_death_rate"]
    )
else:
    st.info("No historical data available for the selected combination.")

st.divider()

# =============================
# Dataset Preview
# =============================
with st.expander("🔍 Show Sample of Dataset"):
    st.dataframe(df.head(10))
