# constants.py
import streamlit as st

# Obtener API Key desde los secretos de Streamlit
API_KEY = st.secrets["API_KEY"]
BASE_URL = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}/"
