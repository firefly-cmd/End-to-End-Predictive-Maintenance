import streamlit as st
import pandas as pd

# Import the tab functions
from pages.eda_tabs.variable_distribution import variable_distribution_analysis
from pages.eda_tabs.pairwise_relations import pairwise_relations_analysis
from pages.eda_tabs.failure_analysis import failure_analysis
from pages.eda_tabs.introduction import introductory_analysis


@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv("data/ai4i2020.csv")


def eda():
    # Load the data
    df_eda = get_data()

    # Create time series plots
    # Sort the data according to UID
    df_eda = df_eda.sort_values(by=["UDI"])

    # Split the page into 4 tabs
    (
        introduction_tab,
        variable_distribution_tab,
        pairwise_relations_tab,
        failure_analysis_tab,
    ) = st.tabs(
        [
            "Introduction",
            "Variable Distribution",
            "Pairwise Relations",
            "Failure Analysis",
        ]
    )
    with introduction_tab:
        introductory_analysis(df_eda)

    # Create variable distribution tab
    with variable_distribution_tab:
        variable_distribution_analysis(df_eda)

    # Create pairwise distribution tab
    with pairwise_relations_tab:
        pairwise_relations_analysis(df_eda)

    # Create failure analysis tab
    with failure_analysis_tab:
        failure_analysis(df_eda)


eda()
