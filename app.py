import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Analytics")
st.write("This Streamlit deployment acts as the live demo for our analytics project.")
if os.path.exists("churn.csv"):
    df = pd.read_csv("churn.csv")
    st.write("Dataset Sample:")
    st.dataframe(df.head(15))
st.write("Review the full pipeline inside churn.ipynb.")
