from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from rag.agent import answer_chat

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "simple_house_properties_clean.csv")
MODEL_PATH = PROJECT_ROOT / "models" / "best_regression_model.joblib"
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(page_title="House AI", page_icon="🏠", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stChatMessage"],
    [data-testid="stChatMessageContent"] {
        width: 100%;
        max-width: none;
    }

    [data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: 24rem;
        right: 3rem;
        width: auto;
        z-index: 1000;
        padding: 0.75rem 0 1rem;
        background: transparent;
        border: none;
        box-shadow: none;
    }

    [data-testid="stAppViewContainer"]:has(
        [data-testid="stSidebar"][aria-expanded="false"]
    ) [data-testid="stChatInput"] {
        left: 3rem;
    }

    [data-testid="stMainBlockContainer"],
    .main .block-container {
        max-width: none;
        padding-bottom: 7rem;
    }

    @media (max-width: 768px) {
        [data-testid="stChatInput"] {
            left: 1rem;
            right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_properties():
    return pd.read_csv(DATA_PATH)

@st.cache_resource(show_spinner=False)
def load_regression_model():
    return joblib.load(MODEL_PATH)

def price_segment(price_lakh):
    if price_lakh < 50:
        return "Affordable"
    if price_lakh <= 150:
        return "Mid-range"
    return "Premium"

def predict_price(model, input_data):
    categorical_columns = [
        "city",
        "status",
        "property_type",
        "transaction",
        "furnishing",
        "ownership",
    ]

    # One hot encoding
    processed_input = pd.get_dummies(input_data, columns=categorical_columns)
    if hasattr(model, "feature_names_in_"):
        processed_input = processed_input.reindex(
            columns=model.feature_names_in_,
            fill_value=0,
        )

    log_price = model.predict(processed_input)[0]
    price_lakh = float(np.expm1(log_price))
    return round(price_lakh, 2), price_segment(price_lakh)

properties = load_properties()
regression_model = load_regression_model()

st.title("House AI")
st.caption("Simple Streamlit app for the saved regression model.")

with st.sidebar:
    st.subheader("House AI")
    st.caption("Price prediction and assistant.")

tab_predict, tab_chat = st.tabs(["Price Prediction", "Assistant"])

with tab_predict:
    st.subheader("Predict Price With Best Trained Model")
    cities = sorted(properties["city"].dropna().unique())

    with st.form("prediction_form"):
        col1, col2, col3, col4 = st.columns(4)
        city = col1.selectbox(
            "City",
            cities,
            index=cities.index("Pune") if "Pune" in cities else 0,
        )

        area_sqft = col2.number_input(
            "Area sqft",
            min_value=100.0,
            max_value=20000.0,
            value=1000.0,
            step=50.0,
        )

        bhk = col3.number_input(
            "BHK",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=1.0,
        )

        bathrooms = col4.number_input(
            "Bathrooms",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=1.0,
        )

        col5, col6, col7, col8 = st.columns(4)
        balconies = col5.number_input(
            "Balconies",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=1.0,
        )

        parking_count = col6.number_input(
            "Parking",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=1.0,
        )

        current_floor = col7.number_input(
            "Current floor",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=1.0,
        )

        total_floors = col8.number_input(
            "Total floors",
            min_value=0.0,
            max_value=100.0,
            value=12.0,
            step=1.0,
        )

        col9, col10, col11 = st.columns(3)
        furnishing = col9.selectbox(
            "Furnishing",
            sorted(properties["furnishing"].dropna().unique()),
        )

        transaction = col10.selectbox(
            "Transaction",
            sorted(properties["transaction"].dropna().unique()),
        )

        ownership = col11.selectbox(
            "Ownership",
            sorted(properties["ownership"].dropna().unique()),
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "area_sqft": area_sqft,
                    "bhk": bhk,
                    "bathrooms": bathrooms,
                    "balconies": balconies,
                    "parking_count": parking_count,
                    "current_floor": current_floor,
                    "total_floors": total_floors,
                    "city": city,
                    "status": "Unknown",
                    "property_type": "Flat",
                    "transaction": transaction,
                    "furnishing": furnishing,
                    "ownership": ownership,
                }
            ]
        )

        price_lakh, segment = predict_price(regression_model, input_data)
        c1, c2 = st.columns(2)
        c1.metric("Predicted price", f"INR {price_lakh:,.2f} lakh")
        c2.metric("Price segment", segment)

with tab_chat:
    st.subheader("House AI Assistant")

    if st.button("Clear Chat"):
        st.session_state.pop("chat_messages", None)
        st.rerun()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant", "content": (
                    "Ask me about estimated prices, market summaries, or good deals. "
                    "Example: 2 BHK in 80 lakhs in Pune. Is it good deal?"), }]

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about price, city, BHK, budget, or good deals")
    if prompt:
        st.session_state.chat_messages.append(
            {"role": "user", "content": prompt}
        )
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Groq agent is choosing and running tools..."):
                answer = answer_chat(st.session_state.chat_messages)
            st.markdown(answer)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )
