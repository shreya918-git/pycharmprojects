import numpy as np
import pandas as pd
import streamlit as st
import sklearn
import xgboost
import pickle
import os

st.title("Laptop Price Predictor")

df=pickle.load(open("df.pkl","rb"))
pipe=pickle.load(open("pipe.pkl","rb"))
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor

# Access the stacking regressor
stacking = pipe.named_steps['step2']

# Remove 'gpu_id' from each estimator if it exists
for name, est in stacking.estimators:
    if isinstance(est, XGBRegressor) and hasattr(est, "gpu_id"):
        delattr(est, "gpu_id")

# Also from final estimator if needed
if isinstance(stacking.final_estimator_, XGBRegressor) and hasattr(stacking.final_estimator_, "gpu_id"):
    delattr(stacking.final_estimator_, "gpu_id")


company=st.selectbox("Select a company",df["Company"].unique())
type=st.selectbox("Select Type",df['TypeName'].unique())
ram=st.selectbox("Select RAM(in GB)",[2,4,6,8,10,12,16,24,32,64])
weight=st.number_input(label="weight")
touchscreen=st.selectbox("Select Touchscreen",[True,False])
ips=st.selectbox("Select IPS",[True,False])
screen_size=st.slider("Select screen size",10.0,18.0,13.0)
resolution=st.selectbox("Select Resolution",['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu=st.selectbox("Select CPU",df["Cpu_brand"].unique())
hdd=st.selectbox("Select HDD(in GB)",[0,128,256,512,1024,2048])
sdd=st.selectbox("Select SDD(in GB)",[0,8,128,256,512,1024])
gpu=st.selectbox("Select GPU",df["Gpu_brand"].unique())
os=st.selectbox("Select OS",df["os"].unique())
predict_price=st.button("Predict Price")
if predict_price:
    ppi=(int(resolution.strip("x")[0])**2+int(resolution.strip("x")[1])**2)**0.5/int(screen_size)
    if touchscreen=="True":
        touchscreen=1
    else:
        touchscreen=0
    if ips=="True":
        ips=1
    else:
        ips=0
    query=np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,sdd,gpu,os])
    query=query.reshape(1,12)

    import pandas as pd
    import numpy as np
    import unicodedata

    # If query was created from np.array, wrap it properly
    query = pd.DataFrame([[
        company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,sdd,gpu,os
    ]], columns=[
        'Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'ppi',
        'Cpu_brand', 'HDD', 'SSD', 'Gpu_brand', 'os'
    ])

    # Remove any Unicode characters from string columns
    for col in query.select_dtypes(include='object').columns:
        query[col] = query[col].apply(
            lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode('utf-8')
        )

    # Predict
    pred = pipe.predict(query)
    price = np.exp(pred[0])

    st.success(f"Predicted Price: ₹{price:,.0f}")