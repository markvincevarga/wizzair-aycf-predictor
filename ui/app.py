import streamlit as st
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui.utils import load_data
from ui.views import render_map_view, render_calendar_view

st.set_page_config(
    page_title="WizzAir AYCF Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ WizzAir All You Can Fly Predictor")

# Load data once
with st.spinner("Loading recent flight data..."):
    data = load_data()

# Sidebar
st.sidebar.title("Navigation")
view_option = st.sidebar.radio("Go to", ["Where to? (Map)", "When? (Calendar)"])

if view_option == "Where to? (Map)":
    render_map_view(data)
elif view_option == "When? (Calendar)":
    render_calendar_view(data)

st.sidebar.markdown("---")
st.sidebar.info(
    "Data loaded: Past 30 days history + Next 30 days predictions.\n"
    "Predictions are based on XGBoost model trained on historical availability."
)


