import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
st.set_page_config(layout="wide")
df=pd.read_csv("india.csv")
states=df["State"].unique()
states.insert(0,"Overall India")
columns=list(df.columns[5:])
st.sidebar.title("India data visualization")
selected_state=st.sidebar.selectbox("Select a state",states)
primary=st.sidebar.selectbox("Select primary",columns)
secondary=st.sidebar.selectbox("Select secondary",columns)
plot=st.sidebar.button("plot graph")
if plot:
    if selected_state=="Overall India":
        fig=px.scatter_mapbox(df,lat="Latitude",lon="Longitude",size=primary,size_max=35,zoom=4,color=secondary,hover_name="State")
        st.plotly_chart(fig,use_container_width=True)
    else:
        df=df[df["State"]==selected_state]
        fig=px.scatter_mapbox(df,lat="Latitude",lon="Longitude",size=primary,size_max=35,zoom=6,color=secondary,hover_name="District")
        st.plotly_chart(fig,use_container_width=True)