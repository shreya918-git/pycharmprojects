import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import preprocessor
import helper

st.sidebar.title("Whatsapp chat analyzer")

uploaded_file=st.sidebar.file_uploader("Select a file")
if uploaded_file is not None:
    bytes=uploaded_file.getvalue()
    data=bytes.decode("utf-8")
    df=preprocessor.process(data)
    user_list=df["user"].unique().tolist()
    user_list.remove("group notification")
    user_list.sort()
    user_list.insert(0,"Overall")
    selected_user=st.sidebar.selectbox("Show analysis wrt", user_list)
    button=st.sidebar.button("Show analysis")
    if button:
        col1,col2,col3,col4=st.columns(4)
        with col1:
            st.subheader("Total Messages")
            total_messages=helper.fetch_stats(selected_user,df)[0]
            st.html("<h2>{}</h2>".format(total_messages))
        with col2:
            st.subheader("Total Words")
            total_words=helper.fetch_stats(selected_user,df)[1]
            st.html("<h2>{}</h2>".format(total_words))
        with col3:
            st.subheader("Total Media")
            media=helper.fetch_stats(selected_user,df)[2]
            st.html("<h2>{}</h2>".format(media))
        with col4:
            st.subheader("Total Links")
            links=helper.fetch_stats(selected_user,df)[3]
            st.html("<h2>{}</h2>".format(links))
        col1,col2=st.columns(2)
        with col1:
            st.subheader("Most busy months")
            temp_df=helper.timeline(selected_user,df)
            fig,ax=plt.subplots()
            ax.plot(temp_df["time"],temp_df["messages"])
            plt.xticks(rotation="vertical")
            st.pyplot(fig)
        with col2:
            st.subheader("Most busy days")
            temp_df=helper.most_busy_date(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(temp_df["only_date"],temp_df["messages"])
            plt.xticks(rotation="vertical")
            st.pyplot(fig)
        if selected_user=="Overall":
          col1,col2=st.columns(2)
          with col1:
            st.subheader("Most active users")
            temp_df=df["user"].value_counts().head(5)
            fig,ax=plt.subplots()
            ax.bar(temp_df.index,temp_df.values)
            st.pyplot(fig)
          with col2:
            temp_df=helper.most_busy_users(df)
            st.dataframe(temp_df)
        st.subheader("Wordcloud")
        fig,ax=plt.subplots()
        df_wc=helper.create_word_cloud(selected_user,df)
        ax.imshow(df_wc)
        st.pyplot(fig)
        st.subheader("Most used words")
        temp_df=helper.most_used_words(df)
        fig,ax=plt.subplots()
        ax.barh(temp_df[0],temp_df[1])
        plt.xticks(rotation="vertical")
        st.pyplot(fig)
        st.subheader("Most used emojis")
        col1,col2=st.columns(2)
        with col1:
            temp_df=helper.emoji_analysis(selected_user,df)
            st.dataframe(temp_df)
        with col2:
            temp_df=helper.emoji_analysis(selected_user,df)
            fig,ax=plt.subplots()
            ax.pie(temp_df[1].head(5),labels=temp_df[0].head(5),autopct="%0.1f%%")
            st.pyplot(fig)
        st.subheader("Heatmap")
        temp_df=helper.heatmap(selected_user,df)
        fig,ax=plt.subplots()
        sns.heatmap(temp_df)
        st.pyplot(fig)


