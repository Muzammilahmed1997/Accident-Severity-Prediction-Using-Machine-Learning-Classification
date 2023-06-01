import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64


values =  pd.read_csv("value.csv")

title1 = st.title("Accident Severity Prediction")

values['Speed_limit']  =  st.number_input("Speed Limit")

values['Number_of_Vehicles'] = st.number_input("Number of Vehicles")

values['Number_of_Casualties'] = st.number_input("Number of Casualities")

values['1st_Road_Class'] = st.number_input("Number of Road")

values['Day_of_Week'] = st.number_input("Day")

values['Urban_or_Rural_Area'] = st.number_input("Urban or Rural")


#loading the trained model
pickle_in = open('pred2.sav', 'rb') 
classifier = pickle.load(pickle_in)
pred = classifier.predict(values)



if st.button("Predict"): 
        st.write(pred) 



