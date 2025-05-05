import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("startup_cleaned.csv")
st.sidebar.title("Startup Fund Analysis")
option=st.sidebar.selectbox("Select an option",["Overall Analysis","Startup Analysis","Investor Analysis"])
df.dropna(subset=["date","startup","vertical","city","round","amount"],inplace=True)
df["date"]=pd.to_datetime(df["date"])
df["month"]=df["date"].dt.month
df["year"]=df["date"].dt.year
def load_investor_details(investor):
    st.title(investor)
    st.subheader("Recent Investments")
    st.dataframe(df[df["investor"].str.contains(investor)][["date","startup","vertical","city","round","amount"]].head(5))
    st.subheader("Biggest Investment")
    st.dataframe(df[df["investor"].str.contains(investor)].groupby("startup")["amount"].sum().sort_values(ascending=False).head(1))
    startup=df[df["investor"].str.contains(investor)].groupby("vertical")["amount"].sum()
    city=df[df["investor"].str.contains(investor)].groupby("city")["amount"].sum()
    stage=df[df["investor"].str.contains(investor)].groupby("round")["amount"].sum()
    col1,col2,col3=st.columns(3)
    with col1:
        st.subheader("Sector Investments")
        fig, ax = plt.subplots()
        ax.pie(startup,labels=startup.index,autopct="%1.1f%%")
        st.pyplot(fig)
    with col2:
        st.subheader("City Investments")
        fig1, ax1 = plt.subplots()
        ax1.pie(city, labels=city.index, autopct="%1.1f%%")
        st.pyplot(fig1)
    with col3:
        st.subheader("Round Investments")
        fig2, ax2 = plt.subplots()
        ax2.pie(stage, labels=stage.index, autopct="%1.1f%%")
        st.pyplot(fig2)
    df["year"]=df["date"].dt.year
    year=df[df["investor"].str.contains(investor)].groupby("year")["amount"].sum()
    st.subheader("YoY Investments")
    fig3, ax3 = plt.subplots()
    ax3.plot(year.index,year.values)
    st.pyplot(fig3)
def load_overall_analysis():
    col1,col2,col3=st.columns(3)
    with col1:
        total = df["amount"].sum()
        st.metric("Total", str(round(total)) + " Cr")
    with col2:
        maximum = df["amount"].max()
        st.metric("Max", str(round(maximum))+" Cr")
    with col3:
        average = df["amount"].mean()
        st.metric("Average", str(round(average))+" Cr")
    option=st.selectbox("Select an option",["total","count"])
    if option=="count":
      temp_df=df.groupby(["year","month"])["startup"].count().reset_index()
      temp_df["x_axis"]=temp_df["month"].astype("str")+"-"+temp_df["year"].astype("str")
      st.subheader("MoM Investments")
      fig4, ax4 = plt.subplots()
      ax4.plot(temp_df["x_axis"],temp_df["startup"])
      st.pyplot(fig4)
    else:
        temp_df = df.groupby(["year", "month"])["amount"].sum().reset_index()
        temp_df["x_axis"] = temp_df["month"].astype("str") + "-" + temp_df["year"].astype("str")
        st.subheader("MoM Investments")
        fig5, ax5 = plt.subplots()
        ax5.plot(temp_df["x_axis"], temp_df["amount"])
        st.pyplot(fig5)
if option=="Overall Analysis":
    st.title("Overall Analysis")
    load_overall_analysis()
elif option=="Startup Analysis":
    st.title("Startup Analysis")
    st.sidebar.selectbox("Select a startup",sorted(df["startup"].unique().tolist()))
    button=st.sidebar.button("Find startup details")
else:
    investor=st.sidebar.selectbox("Select an investor",list(set(df["investor"].str.split(",").sum())))
    button2=st.sidebar.button("Find investor details")
    if button2:
        load_investor_details(investor)