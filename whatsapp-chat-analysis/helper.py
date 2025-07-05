import numpy as np
import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji

def fetch_stats(selected_user,df):
    if selected_user != "Overall":
        df=df[df["user"]==selected_user]
    total_messages=df.shape[0]
    words=[]
    for i in df["messages"].str.split(" "):
        words.extend(i)
    temp_df=df[df["messages"].str.contains("Media")]
    media=temp_df.shape[0]
    extractor=URLExtract()
    extractor.update()
    links=[]
    for i in df["messages"]:
        links.extend(extractor.find_urls(i))
    return total_messages,len(words),media,len(links)

def most_busy_users(df):
    temp_df=round((df["user"].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={"count":"percent"})
    return temp_df

def create_word_cloud(selected_user,df):
    if selected_user != "Overall":
        df=df[df["user"]==selected_user]
    df = df[df["user"] != "group notification"]
    df = df[~df["messages"].str.contains("Media")]
    f = open("stop_hinglish.txt", "r")
    stop_words = f.read().split("\n")
    def remove_stop_words(message):
        word_list=[]
        for i in message.lower().split(" "):
            for j in i:
                if j not in stop_words:
                    word_list.append(j)
        return "".join(word_list)
    wc=WordCloud(width=500,height=500,min_font_size=10,background_color="black")
    df["messages"].apply(remove_stop_words)
    df_wc=wc.generate(df["messages"].str.cat(sep=" "))
    return df_wc

def most_used_words(df):
    temp_df=df[df["user"] != "group notification"]
    temp_df=temp_df[~temp_df["messages"].str.contains("Media")]
    f=open("stop_hinglish.txt","r")
    stop_words=f.read().split("\n")
    word_list=[]
    for message in temp_df["messages"]:
        words=message.split(" ")
        for i in words:
            if i not in stop_words:
                word_list.append(i)
    temp_df=pd.DataFrame(Counter(word_list).most_common(20))
    return temp_df

def emoji_analysis(selected_user,df):
    if selected_user != "Overall":
        df=df[df["user"]==selected_user]
    emojis=[]
    for message in df["messages"]:
        emojis.extend([i for i in message if emoji.UNICODE_EMOJI])
    temp_df=pd.DataFrame(Counter(emojis).most_common(len(emojis)))
    return temp_df

def timeline(selected_user,df):
    if selected_user != "Overall":
        df=df[df["user"]==selected_user]
    temp_df = df.groupby(["year", "month"])["messages"].count().reset_index()
    time = []
    for i in range(temp_df.shape[0]):
        time.append(str(temp_df["month"][i]) + "-" + str(temp_df["year"][i]))
    temp_df["time"] = time
    return temp_df

def most_busy_date(selected_user,df):
    if selected_user != "Overall":
        df=df[df["user"]==selected_user]
    df["only_date"] = df["date"].dt.date
    temp_df = df.groupby("only_date")["messages"].count().reset_index()
    return temp_df

def heatmap(selected_user,df):
    if selected_user != "Overall":
        df=df[df["user"]==selected_user]

    def period(hour):
        if hour == 12:
            return str(hour) + ":" + "01"
        elif hour == 23:
            return str(hour) + ":" + "00"
        elif hour==00:
            return str(hour)+":"+"01"
        else:
            return str(hour) + ":" + str(hour + 1)

    df["period"] = df["hour"].apply(period)
    temp_df = df.pivot_table(index="day", columns="period", values="messages", aggfunc="count").fillna(0)
    return temp_df

