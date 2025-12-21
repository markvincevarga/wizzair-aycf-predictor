import streamlit as st
import pandas as pd
from datetime import date, timedelta
from storage.database import DatabaseWrapper
from storage.predictions import Predictions
from storage.availabilities import Availabilities

DB_NAME = "wizz-aycf"

@st.cache_resource
def get_db():
    return DatabaseWrapper(database_name=DB_NAME)

@st.cache_data(ttl=3600)
def load_data():
    db = get_db()
    
    today = date.today()
    start_date = today - timedelta(days=30)
    end_date = today + timedelta(days=30)
    
    preds_repo = Predictions(db)
    avail_repo = Availabilities(db)
    
    # Fetch data
    df_preds = preds_repo.get_recent_predictions(start_date, end_date)
    df_avail = avail_repo.get_recent_availabilities(start_date, end_date)
    
    # Ensure dates are dates (not timestamps or datetime) for easier filtering
    if not df_preds.empty:
        df_preds['availability_start'] = df_preds['availability_start'].dt.date
    
    if not df_avail.empty:
        df_avail['availability_start'] = df_avail['availability_start'].dt.date
        
    return {
        "predictions": df_preds,
        "history": df_avail
    }

