import streamlit as st
import time
st.title("Startup Dashboard")
st.header("I am learning streamlit")
st.subheader("streamlit")
st.markdown("""
### my tech stack is-
- ml
- python
- numpy 
- pandas
""")
st.write("this is a text")
st.code("""
def square(x):
  return x**2
square(4)
""")
st.metric("Revenue","3L","3%")
st.latex("x^2+y^2+z=0")
col1,col2=st.columns(2)
with col1:
    st.image("img.png")
with col2:
    st.image("img.png")
st.text_input("Enter Email")
st.number_input("Enter age")
st.date_input("Enter date")
st.success("login successful")
st.error("login failed")
st.info("this is a info")
st.warning("this is a warning")
st.sidebar.title("Sidebar title")
bar=st.progress(0)
for i in range(1,101):
    time.sleep(1)
    bar.progress(i)
name=st.text_input("Enter your name")
email=st.text_input("Enter your email")
button=st.button("Login")
if button:
 if name=="Shreya" and email=="shreyasharma918nds@gmail.com":
    st.balloons()
    st.success("login successful")
 else:
    st.error("login failed")
file=st.file_uploader("Upload a csv file")
if file is not None:
    df=pd.read_csv(file)
    st.dataframe(df.describe())