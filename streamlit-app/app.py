import streamlit as st
import pandas as pd
import kagglehub
df=pd.read_csv("startup_funding.csv")
df["Investors name"]=df["Investors Name"].fillna("Undisclosed")
st.sidebar.title("Startup Fund Analysis")
option=st.sidebar.selectbox("Select an option",["Overall Analysis","Startup Analysis","Investor Analysis"])
if option=="Overall Analysis":
    st.title("Overall Analysis")
elif option=="Startup Analysis":
    st.title("Startup Analysis")
    st.sidebar.selectbox("Select a startup",sorted(df["Startup Name"].unique().tolist()))
    button=st.sidebar.button("Find startup details")
else:
    st.title("Investor Analysis")
    st.sidebar.selectbox("Select an investor",sorted(df["Investors name"].unique().tolist()))
    button2=st.sidebar.button("Find investor details")