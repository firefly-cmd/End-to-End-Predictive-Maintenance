import streamlit as st
import pandas as pd


# Introduction tab function of exploratory data analysis
def introductory_analysis(df_eda):
    st.title("Introduction")
    st.markdown(
        """
                AI4I 2020 Predictive Maintenance Dataset Data Set is a synthetic dataset about single
                machine that represents the real life characteristics. It has 10k data measurements
                with no missing values. Since the real datasets about the machinary used in the production
                are hard to publish, it is created synthetically.
                
                The consists of 10000 datapoints with 14 features: \n
                UDI: Unique identifier corresponds to the time column with an increasing range \n
                Product ID: Serial number of the created products. Consists a letter L, M or H for low, 
                medium or high product qualities. \n
                Type: Quality type of the product \n
                Air temperature [K]: Air temperature measurements \n
                Process temperature [K]: Process temperature measurements \n
                Rotational speed [rpm]: Rotational speed of the machine \n
                Torque [Nm]: Torque measurements of the machine \n
                Tool wear [min]: Tool wear level of the used tool in the process \n
                Machine failure: Indicator of machine has failed in that particular point or not \n
                
                The machine can fail for 5 different reasons:
                Tool wear failure (TWF): Used tool will wear and create a failure after some usage \n
                Heat dissipation failure (HDF): Heat dissipation causes a failure \n
                Power failure (PWF): Machine failures due to power levels \n
                Overstrain failure (OSF):Machine failures due to overstrain of the tool used \n
                Random failure (RNF): Process fails randomly
                
    """
    )

    st.markdown("## Dataset Overview")
    st.dataframe(df_eda)
