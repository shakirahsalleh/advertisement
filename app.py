import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "xgb_model_advertisement.pkl.pkl"  # change to your pkl filename

st.set_page_config(page_title="Sales Predictor", page_icon="📈")


@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


st.title("📈 Sales Prediction")
st.write("Enter your advertising budget for each channel to get a prediction.")

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH.name}. Put it in the same folder as app.py.")
    st.stop()

col1, col2, col3 = st.columns(3)
tv = col1.number_input("TV", min_value=0.0, value=100.0, step=1.0)
radio = col2.number_input("Radio", min_value=0.0, value=20.0, step=1.0)
newspaper = col3.number_input("Newspaper", min_value=0.0, value=10.0, step=1.0)

if st.button("Predict", type="primary"):
    # Column names and order must match what the model was trained on
    X = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"])
    try:
        prediction = model.predict(X)[0]
        st.success("Prediction complete")
        st.metric("Predicted result", f"{prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
