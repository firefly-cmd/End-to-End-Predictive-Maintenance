import streamlit as st
import plotly.graph_objects as go
import numpy as np
import scipy.stats as stats


# Function to plot time series with moving average
def plot_time_series(data, column, window=50, timestamp_column="UDI", ma_color="red"):
    # Plot time series
    fig = go.Figure(
        data=go.Scatter(
            x=data[timestamp_column], y=data[column], mode="lines", name=column
        )
    )

    # Calculate moving average
    data["Moving Average"] = data[column].rolling(window=window).mean()

    # Add moving average to the plot
    fig.add_trace(
        go.Scatter(
            x=data[timestamp_column],
            y=data["Moving Average"],
            mode="lines",
            line=dict(color=ma_color),
            name=f"{window} point of period Moving Average",
        )
    )

    fig.update_layout(
        title_text="Time Series of " + column + " with Moving Average",
        xaxis_title="Time",
        yaxis_title=column,
    )

    fig.update_layout(
        legend=dict(
            x=1,
            y=1,
            xanchor="right",
            yanchor="top",
            orientation="h",  # this will make the legend horizontal
            bgcolor="rgba(255, 255, 255, 0.5)",  # semi-transparent white background
            traceorder="normal",
            font=dict(family="sans-serif", size=12, color="black"),
        )
    )

    st.plotly_chart(fig, use_container_width=True)


# Plot the variable distribution with addition information about the distribution such as
# mean, std, skewness and kurtosis to analyze it further
def plot_distribution(data, feature):
    # Calculate statistics
    mu = np.mean(data[feature])
    sigma = np.std(data[feature])
    skewness = stats.skew(data[feature])
    kurtosis = stats.kurtosis(data[feature])

    # Create histogram
    hist = go.Histogram(x=data[feature], opacity=0.75, name="Histogram", nbinsx=50)

    # Create layout
    layout = go.Layout(
        title=f"{feature} Distribution<br>"
        f"<sup>mean = {mu:.2f}, standard deviation = {sigma:.2f}, skewness = {skewness:.2f}, kurtosis = {kurtosis:.2f}</sup>",
        xaxis=dict(title=feature),
        yaxis=dict(title="Density"),
        showlegend=True,
    )

    # Add traces to figure and plot
    fig = go.Figure(data=[hist], layout=layout)
    st.plotly_chart(fig, use_container_width=True)


# Varible distribution tab function
def variable_distribution_analysis(df_eda):
    st.title("Variable Distribution Analysis")
    st.markdown(
        f"""
        This section provides a comprehensive analysis of the distribution of individual variables in the dataset. The variables represent different attributes related to the performance and condition of machines.
        Understanding the distribution of variables is an essential part of data analysis. It provides insights into patterns, potential anomalies, the range of each variable, and can aid in the prediction of future data.
        The following analysis will cover the distribution of variables such as air temperature, process temperature, rotational speed, and torque, visualized using histograms and density plots.
        The aim is to provide a detailed understanding of the variables in order to facilitate more accurate predictive modeling.
    """
    )
    st.markdown(
        f"""
            ## Air temperature
            
    """
    )

    # Plot the air temperature with its moving average value
    air_temperature_column = "Air temperature [K]"
    plot_time_series(df_eda, air_temperature_column)
    plot_distribution(df_eda, air_temperature_column)

    st.markdown(
        f"""
                Air temperature fluctuates between 295 K and 304 K most of the time. It alternates between increasing and decreasing trends after 
                some data points which may indicate some seasonality, however, we may not able catch it with 10k data points. Also the residuals are not high 
                so it follows a somewhat stable line comparing with the moving averages. The measurements mostly around 300 K and there may be increasing chance of
                machine failure especially heat failure when temperature drops too much or increases too much. The kurtosis value of of the distribution is low, also the skewness value
                of the distribution is very low which means that the distribution is close to the normal distribution and there is not much outliers. Most of the outliers
                present themself in the lower tail of the distribution which means that air temperature drops are more common than the very high temperatures
    """
    )

    st.markdown(
        f"""
            ## Process temperature
            
    """
    )

    # Plot the process temperature with its moving average value
    process_temperature_column = "Process temperature [K]"
    plot_time_series(df_eda, process_temperature_column)
    plot_distribution(df_eda, process_temperature_column)

    st.markdown(
        f"""
                
                Process temperature follows a similar path with the air temperature. It is higher compared to the air temperature. The similarities between air temperature
                and process temperature may indicate that the process temperature might be affected by the air temperature. This means that, the difference between process
                temperature and the air temperature may provide some insights about the machine failures. Also there is a sharp decrease in the process temperature around 1k
                data points which could increase the possibility of machine failures around these points.
                
    """
    )

    st.markdown(
        f"""
            ## Rotational Speed
            
    """
    )

    # Plot the rotational speed with its moving average value
    rotational_speed_column = "Rotational speed [rpm]"
    plot_time_series(df_eda, rotational_speed_column)
    plot_distribution(df_eda, rotational_speed_column)

    st.markdown(
        f"""
                
                General trend of the rotational speed is constant around 1600 rpm considering the moving average line. However, there are too much fluctuations
                and some measurements reaches much more than 2500 rpm. These instant increases in the rotational speed may be an indication of machine failure and needs
                further investigation
            
    
    """
    )

    # Plot the torque with its moving average value
    torque_column = "Torque [Nm]"
    plot_time_series(df_eda, torque_column)
    plot_distribution(df_eda, torque_column)

    st.markdown(
        f"""
                
                Torque follows a constant trend with fluctuations around 20 to 60 Nm. In spite of that, there are some measurements which are too low or too high compared to 
                the general trend. This may indicate machine failures.
                
    """
    )

    # Plot the tool wear with its moving average
    tool_wear_column = "Tool wear [min]"
    plot_time_series(df_eda, tool_wear_column)
    plot_distribution(df_eda, tool_wear_column)

    st.markdown(
        f"""
                Tool wear exibits a very different graph than the other features. It increases up to some point and resets again. The reset points may indicate a machine failure.
                This may mean that there could be strong correlation between the tool wear and the machine failure which will be investigated further in the eda process.
                
    """
    )

    st.markdown("""## Summary""")
    st.markdown(
        """
                The air temperature, mostly hovering around 300K, fluctuates between 295K and 304K, indicating potential seasonality, and suggesting that machine failure 
                risk could increase if temperatures vary too far from this range. The distribution of temperature measurements is fairly normal with few outliers, which 
                are more likely to occur at lower temperatures. The process temperature tends to follow a similar pattern to air temperature, albeit generally higher. 
                A strong correlation between the two could suggest that air temperature impacts the process temperature, and variations between these two measurements 
                might offer insights into machine failures. A sharp decrease in process temperature at around 1k data points could indicate an increased risk of 
                machine failure. The machine's rotational speed, though mostly constant around 1600 rpm, does experience fluctuations, with some measurements 
                exceeding 2500 rpm; these sudden increases might signify potential machine failures and warrant further investigation. Torque remains relatively 
                constant, fluctuating between 20 to 60 Nm, but outlier measurements could also point towards machine failures. A distinct trend in tool wear 
                involves it increasing to a point before resetting, with these reset points potentially marking machine failures. This possible correlation 
                between tool wear and machine failure will be further investigated during the exploratory data analysis (EDA) process.
                
    """
    )
