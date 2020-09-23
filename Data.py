import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from PIL import Image
import plotly.express as px


img = Image.open("8.jpg")
st.image(img)
st.title("Insurance Policy") 

df = pd.read_csv("ins.csv")
all_col = df.columns.to_list()

if st.sidebar.checkbox("View Sample Data"):
    st.write("Sample Data:")
    st.dataframe(df.head())
if st.sidebar.checkbox("Describe Data"):
    st.write("Data Description:",df.describe())  
    
if st.checkbox("Line Chart"):
    st.line_chart(df)
        
if st.checkbox("Bar Chart"):
    bar = st.multiselect("Select Features ",all_col,key = 'a')
    new_df = df[bar]
    st.bar_chart(df[bar])
    
if st.checkbox("Area_chart"):        
    area = st.multiselect("Select Features",all_col)
    new_dff = df[area]
    st.area_chart(df[area])    
    
if st.checkbox("Histogram"):
    fig = px.histogram(df)
    fig.show()  
 

if st.checkbox("Model <ignore warnings>"):
    X = df[['months_as_customer','policy_annual_premium','bodily_injuries']]
    y = df['total_claim_amount']


#creating model
    clf = LinearRegression()
    clf.fit(X,y)
#st.sidebar.button("Get Rank")
#making prediction

    u1 = float(st.text_input("Enter months_as_customer:"))
    u2 = float(st.text_input("Enter policy_annual_premium"))
    u3 = float(st.text_input("Enter bodily_injuries"))

    if st.button("Calculate"):
        predict = clf.predict([(u1,u2,u3)])
        predict = int(predict)
        st.write("###","Total Claim Amount will be:",predict)

 
    
 
