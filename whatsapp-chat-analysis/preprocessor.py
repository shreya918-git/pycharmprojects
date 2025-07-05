import numpy as np
import pandas as pd
import re
def process(data):
    pattern = "\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}\s?[ap]m\s-"
    messages = re.split(pattern, data)[1:]
    message_dates = re.findall(pattern, data)
    df = pd.DataFrame({"message": messages, "message_date": message_dates})
    df["message_date"] = df["message_date"].str.replace(" -", "", regex=False)
    df["message_date"] = df["message_date"].str.replace("\u202f", "", regex=False)
    df["message_date"] = pd.to_datetime(df["message_date"], format="%d/%m/%Y, %I:%M%p")
    df.rename(columns={"message_date": "date"}, inplace=True)
    user = []
    messages = []
    for message in df["message"]:
        text = re.split("^([A-Za-z\s]+):", message)
        if text[1:]:
            user.append(text[1])
            messages.append(text[2])
        else:
            user.append("group notification")
            messages.append(text[0])
    df["user"] = user
    df["messages"] = messages
    df.drop(columns=["message"], inplace=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["minutes"] = df["date"].dt.minute

    return df