# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:49:22 2024

@author: jjin
"""

import streamlit as st
import pandas as pd
import os


# Import profiling capability
from typing_extensions import Annotated
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

# ML stuff
from pycaret.regression import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2024/02/shutterstock_1166533285-Converted-02.png");
    st.title("AutoStreamML");
    choise = st.radio("Navigation",["Upload","Profiling","ML","Download"]);
    st.info("This application allows you to build an automated ML pipeline, Pandas Profiling and PyCaret.")

st.write("Hello, Jason.Jin")

sourceName = "sourcedata.csv";

if os.path.exists(sourceName):
    df = pd.read_csv(sourceName,index_col=None)

if choise == "Upload":
    st.title("Upload your data for modelling!")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv("sourcedata.csv",index=None)
        st.dataframe(df)


if choise == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)


if choise == "ML": 
    st.title("Machine learning go")
    target = st.selectbox("select your target", df.columns)
    setup(df, target = target, verbose = False)
    setup_df = pull()
    st.info("this is the ML experiment settings")
    st.dataframe(setup_df)
    best_model = compare_models()
    compare_df = pull()
    st.info("This is the ML Model")
    st.dataframe(compare_df)
    best_model
    save_model(best_model,'best_model')

if choise == "Download":
    st.title("Download trained model")
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pk1")
    

st.markdown(
    r"""
    <style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {
            visibility: hidden;
        }
    .stAppToolbar{
            visibility: hidden;
        }
        
    </style>
    """, unsafe_allow_html=True    
)


