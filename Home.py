import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_extras.switch_page_button import switch_page
import base64


# Load the image as base64
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# Cache the data in order to not load over and over again
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv("data/ai4i2020.csv")


# Creates a line chart using the x and y value given
def create_line_chart(x_values, y_values, title, xlabel, ylabel):
    plot = px.line(x=x_values, y=y_values)
    plot.update_layout(title_text=title, xaxis_title=xlabel, yaxis_title=ylabel)

    st.plotly_chart(plot, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Real-Time Predictive Maintenance Dashboard",
        page_icon="✅",
        layout="wide",
    )

    # Retreive the data and cache it
    df = get_data()

    st.title("PREDICTIVE SENTINEL: GUARDING MACHINES OF TOMORROW")
    st.header(
        "Harnessing the Power of Machine Learning to Predict and Prevent Machine Failures"
    )

    st.markdown(
        """
    In the intricate realm of industry, unexpected machine downtimes can spell more than just inconvenience; 
    they often translate to substantial costs, halt in operations, and even looming safety issues. "Predictive Sentinel" 
    is my attempt to delve into this challenge using the AI4I 2020 Predictive Maintenance Dataset. Through this 
    portfolio case project, I'm exploring how data from this specific dataset can be analyzed to predict potential 
    machine issues. By leveraging the insights contained within the dataset and applying advanced machine learning 
    techniques, I aim to showcase a prototype that industries might use to spot and pre-empt potential hiccups. As you 
    navigate through, you'll see the foundation I've built and the insights uncovered in this ongoing journey.
    """
    )

    # Insert the introduction image
    st.image("introduction_factory.jpg", use_column_width=True)

    # Place the navigator buttons
    st.markdown("## Start Exploring")

    eda_page_switch_button = st.button(
        "DIVE DEEP INTO THE DATA AND DISCOVER PATTERNS",
        use_container_width=True,
        type="primary",
    )

    classification_page_switch_button = st.button(
        "LEARN ABOUT MACHINE LEARNING MODELS AND EXPERIMENTS",
        use_container_width=True,
        type="primary",
    )

    if eda_page_switch_button:
        switch_page("Exploratory_Data_Analysis")

    if classification_page_switch_button:
        switch_page("Classification")

    # Create the footer
    github_url = "https://github.com/firefly-cmd/End-to-End-Predictive-Maintenance"
    linkedin_url = "https://www.linkedin.com/in/fatih-kır"

    github_icon_base64 = get_image_base64("github_icon.png")
    linkedin_icon_base64 = get_image_base64("linkedin_icon.png")

    footer = f"""
    <style>
        .footer {{
            text-align: center;
            width: 100%;
            margin-top: 30px;
        }}
        .link {{
            padding: 5px;
            display: inline-block;
        }}
        .footer-text {{
            font-size: 16px;
            color: #888;
            padding: 10px 0;
        }}
    </style>
    <hr style="border-top: 1px solid #888; margin-bottom: 20px;">
    <div class="footer">
        <div class="footer-text">
            Hello! I'm Fatih KIR, a Data Scientist / Machine Learning Engineer specializing in time series sensory data analysis.
            Every day, I delve deep into the intricacies of time-stamped data, extracting meaningful patterns and insights. If you
            share a passion for transforming raw data into actionable intelligence, or just want to connect, find me on the platforms below!
        </div>
        <a class="link" href="{linkedin_url}" target="_blank" rel="noopener noreferrer"><img src="data:image/png;base64, {linkedin_icon_base64}" width=50></a>
        <a class="link" href="{github_url}" target="_blank" rel="noopener noreferrer"><img src="data:image/png;base64, {github_icon_base64}" width=50></a>
    </div>
    """

    st.markdown(footer, unsafe_allow_html=True)
