import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px


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

    st.title("AI4I PREDICTIVE MAINTENANCE APP")

    # Create a random starting point
    starting_index = np.random.randint(0, 9950)
    data_len = 10000

    ## Initiate the x and y values for the plots
    # Air temperature
    air_temperature_y = (
        df["Air temperature [K]"]
        .iloc[starting_index : int(starting_index + 50)]
        .tolist()
    )
    data_x = [i for i in range(50)]

    # Process temperature
    process_temperature_y = (
        df["Process temperature [K]"]
        .iloc[starting_index : int(starting_index + 50)]
        .tolist()
    )

    # Rotational speed
    rotational_speed_y = (
        df["Rotational speed [rpm]"]
        .iloc[starting_index : int(starting_index + 50)]
        .tolist()
    )

    # Torque
    torque_y = (
        df["Torque [Nm]"].iloc[starting_index : int(starting_index + 50)].tolist()
    )

    # Tool wear
    tool_wear_y = (
        df["Tool wear [min]"].iloc[starting_index : int(starting_index + 50)].tolist()
    )

    # TODO Create a single container for these and draw them in the same graph
    # Creating a single-element container for data elements
    placeholder_air_temperature = st.empty()
    placeholder_process_temperature = st.empty()
    placeholder_rotational_speed = st.empty()
    placeholder_torque = st.empty()
    placeholder_tool_wear = st.empty()

    # This app will be live for 300 seconds #TODO Make it live continuesly
    for seconds in range(300):
        # Create a live air temperature chart
        with placeholder_air_temperature.container():
            create_line_chart(
                x_values=data_x,
                y_values=air_temperature_y,
                title="Air Temperature",
                xlabel="Time",
                ylabel="Air Temperature [K]",
            )

        # Create a live process temperature chart
        with placeholder_process_temperature.container():
            create_line_chart(
                x_values=data_x,
                y_values=process_temperature_y,
                title="Process Temperature",
                xlabel="Time",
                ylabel="Process temperature [K]",
            )

        # Create a live rotational speed chart
        with placeholder_rotational_speed.container():
            create_line_chart(
                x_values=data_x,
                y_values=rotational_speed_y,
                title="Rotational Speed",
                xlabel="Time",
                ylabel="Rotational speed [rpm]",
            )

        # Create a live torque chart
        with placeholder_torque.container():
            create_line_chart(
                x_values=data_x,
                y_values=torque_y,
                title="Torque",
                xlabel="Time",
                ylabel="Torque [Nm]",
            )

        # Create a live tool wear chart
        with placeholder_torque.container():
            create_line_chart(
                x_values=data_x,
                y_values=tool_wear_y,
                title="Tool wear",
                xlabel="Time",
                ylabel="Tool wear [min]",
            )

        # Update the lists for all the data
        air_temperature_y.append(df["Air temperature [K]"].iloc[starting_index + 51])
        process_temperature_y.append(
            df["Process temperature [K]"].iloc[starting_index + 51]
        )
        rotational_speed_y.append(
            df["Rotational speed [rpm]"].iloc[starting_index + 51]
        )
        torque_y.append(df["Torque [Nm]"].iloc[starting_index + 51])
        tool_wear_y.append(df["Tool wear [min]"].iloc[starting_index + 51])

        # Add an element at the end
        air_temperature_y.pop(0)
        process_temperature_y.pop(0)
        rotational_speed_y.pop(0)
        torque_y.pop(0)
        tool_wear_y.pop(0)

        # Update the starting index
        starting_index += 1

        # Circle back to the beginning after reaching the last element of the data
        if starting_index > 9999:
            starting_index = 0

        # Update the x axis values
        data_x.append(data_x[-1] + 1)
        data_x.pop(0)

        time.sleep(1)
