import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# Plots the number of failures per 500 data points as a histogram
def plot_failure_over_time(df, time_column, failure_column):
    """
    This function plots the failures of a machine over time using Plotly and displays it in a Streamlit app.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data
    time_column (str): The name of the column containing the time data
    failure_column (str): The name of the column containing the failure data

    Returns:
    None
    """
    # Filter the dataframe to only include rows where a failure occurred
    failure_df = df[df[failure_column] == 1]

    # Create a Plotly Figure
    fig = go.Figure()

    # Add a histogram trace for the failures over time
    fig.add_trace(
        go.Histogram(
            x=failure_df[time_column],
            xbins=dict(
                start=df[time_column].min(),  # first edge of first bin
                end=df[time_column].max(),  # last edge of last bin
                size=500,  # size of bins
            ),
            marker_color="red",
            name="Failures",
        )
    )

    # Update the layout
    fig.update_layout(
        title="Failures Over Time",
        xaxis_title="Time Ticks",
        yaxis_title="Number of Failures",
        autosize=False,
        width=500,
        height=500,
        xaxis_range=[df[time_column].min(), df[time_column].max()],
    )

    # Display the figure in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)


# Plot the feature column distributions by failure
def plot_feature_boxplot(data, feature, target):
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("No Failure", "Failure"), shared_yaxes=True
    )

    # For the cases where target variable is 0 (No Failure)
    fig.add_trace(
        go.Box(
            y=data[data[target] == 0][feature],
            name="No Failure",
            boxpoints="outliers",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=1,
        col=1,
    )

    # For the cases where target variable is 1 (Failure)
    fig.add_trace(
        go.Box(
            y=data[data[target] == 1][feature],
            name="Failure",
            boxpoints="outliers",
            jitter=0.3,
            pointpos=-1.8,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=600, width=800, title_text=feature, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)


# Plot how much time is passed until the next recovery
def plot_time_to_failure(df, time_ticks_col, machine_failure_col):
    # Create a copy of the dataframe
    df_copy = df.copy()

    # Create a new column that indicates the time of the last failure
    df_copy["last_failure"] = df_copy.loc[
        df_copy[machine_failure_col] == 1, time_ticks_col
    ]

    # Forward fill the last_failure column
    df_copy["last_failure"] = df_copy["last_failure"].ffill()

    # Compute the time since the last failure
    df_copy["time_since_last_failure"] = (
        df_copy[time_ticks_col] - df_copy["last_failure"]
    )

    # Create a Plotly figure
    fig = go.Figure()

    # Add a scatter trace for the time to failure
    fig.add_trace(
        go.Scatter(
            x=df_copy[time_ticks_col],
            y=df_copy["time_since_last_failure"],
            mode="lines",
            name="Time to Failure",
        )
    )

    # Update the layout
    fig.update_layout(
        title="Time to Failure",
        xaxis_title="Time",
        yaxis_title="Time since last failure",
    )

    st.plotly_chart(fig, use_container_width=True)


# Plot the the how much time left to failure graph based on the feature levels
def plot_feature_vs_time_to_failure(df, feature, time_column, failure_column):
    """
    This function creates a box plot of time to failure for different levels of a selected feature.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the data
    feature (str): The name of the feature to bin and compare against time to failure
    time_column (str): The name of the column containing the time data
    failure_column (str): The name of the column containing the failure data

    Returns:
    None
    """
    # Create a copy of the dataframe to avoid modifying the original one
    df_copy = df.copy()

    # Create the bins for the selected feature
    df_copy[feature + "_bin"] = pd.qcut(
        df_copy[feature], q=3, labels=["low", "medium", "high"]
    )

    # Calculate the time to failure for each data point
    failure_times = df_copy.loc[df_copy[failure_column] == 1, time_column]
    df_copy["time_to_failure"] = (
        failure_times.reindex(df_copy.index, method="bfill") - df_copy[time_column]
    )

    # Drop rows where time_to_failure could not be calculated
    df_copy = df_copy.dropna(subset=["time_to_failure"])

    # Create separate dataframes for each bin
    low_df = df_copy[df_copy[feature + "_bin"] == "low"]["time_to_failure"]
    medium_df = df_copy[df_copy[feature + "_bin"] == "medium"]["time_to_failure"]
    high_df = df_copy[df_copy[feature + "_bin"] == "high"]["time_to_failure"]

    # Create a Plotly Figure
    fig = go.Figure()

    # Add a box trace for each bin
    fig.add_trace(
        go.Box(y=low_df, name="Low " + feature, boxmean="sd", jitter=0.3, pointpos=-1.8)
    )
    fig.add_trace(
        go.Box(
            y=medium_df,
            name="Medium " + feature,
            boxmean="sd",
            jitter=0.3,
            pointpos=-1.8,
        )
    )
    fig.add_trace(
        go.Box(
            y=high_df, name="High " + feature, boxmean="sd", jitter=0.3, pointpos=-1.8
        )
    )

    # Update the layout
    fig.update_layout(
        title="Time to Failure vs " + feature,
        xaxis_title=feature,
        yaxis_title="Time to Failure",
        autosize=False,
        width=500,
        height=500,
    )

    # Display the figure in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)


# Calculate the feature importances based on the target column
@st.cache_data
def calculate_feature_importance(data, target_column):
    # Define the features and the target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Initialize the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model.feature_importances_


# Calculate and display the feature importances of the cariables based on target column
def plot_feature_importance(data, target_column):
    # Calculate feature importances
    feature_importances = calculate_feature_importance(data, target_column)

    # Create a DataFrame for visualization
    importance_df = pd.DataFrame(
        {
            "Feature": data.drop(target_column, axis=1).columns,
            "Importance": feature_importances,
        }
    )

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Plot feature importances
    fig = go.Figure([go.Bar(x=importance_df["Feature"], y=importance_df["Importance"])])
    fig.update_layout(
        title_text="Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance",
    )
    st.plotly_chart(fig, use_container_width=True)


# Failure analysis tab function for the exploratory data analysis
def failure_analysis(df_eda):
    st.title("Failure Analysis")

    st.markdown(
        """
                    In the preceding sections, the exploration and pairwise relationship analysis have yielded substantial insights 
                    into the variables that significantly influence machine failure. The observations highlight certain operational 
                    conditions—specifically, low rotational speed, high torque, elevated tool wear, and moderate increases in air 
                    and process temperatures—that correlate with an increased propensity for machine failure.

                    This section, titled "Failure Analysis", will engage in a more profound investigation of these failure events. 
                    The primary objective is to comprehend the specific circumstances and chronological sequences that precipitate 
                    machine failure. This involves closely scrutinizing the progression of various operational parameters leading 
                    up to the point of failure, with the aim of identifying any consistent patterns or anomalies that could 
                    potentially serve as early warning indicators.

                    Furthermore, the analysis will endeavor to discern the relationship between different failure modes and the 
                    observed operational parameters and conditions. Such insights could prove invaluable in enhancing predictive 
                    maintenance strategies and in averting future instances of machine failure.
                    
        """
    )

    st.markdown("## Failure over time period")

    st.markdown("### Machine Failures")
    plot_failure_over_time(df_eda, "UDI", "Machine failure")

    st.markdown(
        """
                    When investigating the failure distribution over time, there is a clear trend that, between 
                    4k and 5k datapoints much higher machine failures are observed. Other times failure counts are
                    pretty much the same. We can analyze the patterns before and after this period to find out what 
                    is the reasons for this trend.
                    
        """
    )

    st.markdown("### Tool Wear Failures")
    plot_failure_over_time(df_eda, "UDI", "TWF")
    st.markdown(
        """
                    There are more or less a constant trend across the time for tool wear failure. Considering the variable
                    distribution analysis, tool wear does not exceed some point around 200 - 250 mins and it resets, the hypothesis
                    that increasing tool wear after somewhere around 200 mins causes the machine failures seems true. The tool wear increases
                    up to a point, machine fails, tool wear resets and this trend continues throughout the observation period.
                    
                    
        """
    )

    st.markdown("### Heat Dissipation Failures")
    plot_failure_over_time(df_eda, "UDI", "HDF")
    st.markdown(
        """
                    The sharp increase in overall failures completely overlaps with the machine failures caused by heat dissipation. This 
                    perfectly matches with the air temperature rise observed in the variable distribution analysis. This supports the hypothesis 
                    of increasing air temperature increases the machine failures which is also consistent with the pairwise plot analysis with
                    the air temperature and other variables.
                    
                
        """
    )

    st.markdown("### Power Failures")
    plot_failure_over_time(df_eda, "UDI", "PWF")
    st.markdown(
        """
                    There are power failures throughout the observations with some fluctuations, considering the machine operates
                    with constant power most of the time, deviations from it may cause power failures across observations. Also,
                    the fluctuations may be caused by the failed recovery process since in some parts of the observations the power
                    failures are repeated and in other parts machine worked smoothly.
                    
                    
                    
                    
        """
    )

    st.markdown("### Over Strain Failures")
    plot_failure_over_time(df_eda, "UDI", "OSF")
    st.markdown(
        """
                    Over strain failures somewhat consistent across the time as well. It has similar structure power and tool wear failures.
                    These failures may be caused by features with similar characteristics
                    
        """
    )

    st.markdown("## Feature distribution by failure or not")
    st.markdown(
        """In this section, we will explore how the distributions of different features in our dataset vary based on 
                    whether a machine failure has occurred or not. This is an important step in understanding the underlying 
                    mechanisms of failure in our machines and identifying potential predictors of such failures.
                    For each feature, we will compare the distribution of values under normal operation (no failure) to 
                    the distribution of values when a failure has occurred. To do this, we will use box plots, a type of 
                    graphical representation that succinctly illustrates the central tendency, dispersion, and skewness 
                    of a dataset, while also highlighting any potential outliers.
                    By comparing these distributions side-by-side, we can gain valuable insights into how changes in a 
                    given feature might correlate with the occurrence of machine failures. This could, in turn, guide 
                    us towards a more effective predictive maintenance strategy and help us to prevent future failures. """
    )

    st.markdown("### Air Temperature")
    plot_feature_boxplot(df_eda, "Air temperature [K]", "Machine failure")

    st.markdown(
        """
                    There is a clear skew for the air temperature when there is a machine failure.
                    This means that there is an increasing probability when air temperatures becomes
                    higher than the optimal value
                
        """
    )

    st.markdown("### Process temperature")
    plot_feature_boxplot(df_eda, "Process temperature [K]", "Machine failure")

    st.markdown(
        """
                    
                    There are no skewness in the process temperature when the failures occurs. The process temperature may
                    not be a key indicator or the machine failures by itself
                    
        """
    )

    st.markdown("### Rotational speed")
    plot_feature_boxplot(df_eda, "Rotational speed [rpm]", "Machine failure")

    st.markdown(
        """
                    The machine does not seem to fail by high rotational speeds since there are too many points
                    that are high measurements but does not seem to cause machine failure. However, the shift in the 
                    data distribution may indicate that very low rotational speeds may cause machine failure
                    
        """
    )

    st.markdown("### Torque")
    plot_feature_boxplot(df_eda, "Torque [Nm]", "Machine failure")

    st.markdown(
        """
                    It seems that very low and very high torque values may cause the machine failure.
                    
                    
                    """
    )

    st.markdown("### Tool wear")
    plot_feature_boxplot(df_eda, "Tool wear [min]", "Machine failure")

    st.markdown(
        """
                    There is a clear shift in the data distribution when there is a failure happens for the tool wear. This means that, the higher 
                    the tool wear, the higher the chances of machine failure
                    
                    """
    )

    st.markdown("## Time to failure")
    st.markdown(
        """
                    In this section, we dive deeper into the temporal dynamics of machine failures. Instead of just looking at 
                    when failures occur, we're interested in understanding how long it takes for a failure to happen. This is 
                    known as the "time to failure" and it can provide valuable insights into the lifespan and reliability of our machinery.

                    First, we present a comprehensive view of the time to failure for overall machine failures. This view allows 
                    us to understand the overall distribution of the time until a machine fails. By analyzing the characteristics 
                    of this distribution, we can answer questions like: How often do failures occur? Is there a common time frame 
                    within which machines tend to fail?

                    Then, we turn our attention to how different operating conditions or maintenance activities might influence 
                    the time to failure. For each selected feature, we divide the data into three groups: low, medium, and high, 
                    based on the values of the feature. We then compare the time to failure across these three groups. This analysis 
                    can help us to identify conditions that are associated with faster or slower times to failure.

                    These insights are valuable for predicting future failures and for developing strategies to prevent them. 
                    They can inform decisions about maintenance schedules, usage guidelines, and system design improvements. 
                    By understanding the time to failure, we can take proactive steps to extend the lifespan of our machinery, 
                    minimize downtime, and maintain operational efficiency.

                    In the following visualizations, you'll see the distribution of the time to failure for the overall machinery 
                    and for different levels of selected features. These plots provide a visual representation of the relationships 
                    we've described above.
                    
                    """
    )

    st.markdown("### Overall machine failure")
    plot_time_to_failure(df_eda, "UDI", "Machine failure")
    st.markdown(
        """
                    This machine can work properly around 100 time points in general. Around 4k and 5k time points, lifespan of the 
                    machine between failures is dropped too much. This is mostly due to heat dissipation failures as discussed in earlier
                    sections. There are some areas that machine worked properly for more than 150 time points but these points 
                    are in the minority and may be due to timely periodic checks of the machine. Moreover, the time to failure values
                    of the machine shows somewhat periodical trend with slight fluctuations. This may indicate that seasonal changes 
                    in some variables may be affecting the machine lifespan.
                    
                    
        """
    )

    st.markdown("### Air temperature levels vs time to fail")
    plot_feature_vs_time_to_failure(
        df_eda, "Air temperature [K]", "UDI", "Machine failure"
    )
    st.markdown(
        """
                    As can be seen in this graph, high air temperature values clearly indicates machine failure. The median value of time to
                    failure of high temperatures is 17 time points which is almost 50 percent lower than the low temperature time to failure
                    median level of 36.
                    
        """
    )

    st.markdown("### Process temperature levels vs time to fail")
    plot_feature_vs_time_to_failure(
        df_eda, "Process temperature [K]", "UDI", "Machine failure"
    )
    st.markdown(
        """
                    Process temperature levels does not seem to affect the machine lifespan as in air temperature case. However, lower levels 
                    of process temperature does seen to affect the lifespan more than the moderate air temperature levels. 
                    
        """
    )

    st.markdown("### Rotational speed levels vs time to fail")
    plot_feature_vs_time_to_failure(
        df_eda, "Rotational speed [rpm]", "UDI", "Machine failure"
    )
    st.markdown(
        """
                    Rotational speed levels does seem to impact much time to failure of the machine. All 3 levels represents the same
                    characteristics. This may due to fast fluctuations of rotational speed which can be viewed in the variable 
                    distribution section. The rotational speed at some data point may not be helpful to predict the remaining lifetime of
                    the machine before another failure.
                    
        """
    )

    st.markdown("### Torque vs time to fail")
    plot_feature_vs_time_to_failure(df_eda, "Torque [Nm]", "UDI", "Machine failure")
    st.markdown(
        """
                    Torque graph represents a similar characteristics with the rotational speed since in a constant power scenario as 
                    discussed in the earlier sections. When rotational speed levels are high, torque is low nd visa versa. So, torque 
                    levels are not a clear predictor for the remaining lifetime of the machine before failing.
                    
        """
    )

    st.markdown("### Tool wear vs time to fail")
    plot_feature_vs_time_to_failure(df_eda, "Tool wear [min]", "UDI", "Machine failure")
    st.markdown(
        """
                    Even if there are outlier points, median value of the high tool wear is 12 and the median time point value 
                    of the low tool wear is 52. This dramatic change in the time to failure values depending on the levels of tool wear
                    suggest that, it is the most important predictor of the remaining time left to failure of the device.
                    
        """
    )

    st.markdown("## Failure feature importances")
    st.markdown(
        """
                    In this section, we focus on identifying the most important features that contribute to machine failures. 
                    We make use of a powerful machine learning algorithm known as the Random Forest Classifier. The Random 
                    Forest Classifier is an ensemble learning method that operates by constructing multiple decision trees 
                    during training and outputting the class that is the mode of the classes output by individual trees. One of the 
                    advantages of using a Random Forest Classifier is that it provides a built-in method for determining feature importances.

                    Feature importance gives us a score for each feature of our data, the higher the score more important or relevant 
                    is the feature towards our output variable. Feature importance is an inbuilt class that comes with Tree Based Classifier, 
                    we will be using Random Forest Classifier for extracting the top features for our dataset. Understanding which features 
                    are most important can provide valuable insights that could be used for engineering features, optimizing the model, and 
                    even making decisions about the maintenance and operation of the machines.

                    We train the Random Forest Classifier on our dataset and extract the feature importances. The results are visualized as 
                    a bar plot, allowing us to easily compare the importance of different features. The length of the bar represents the 
                    importance of the feature: longer bars indicate features that are more important in predicting machine failure.

                    This analysis aims to provide insights into the key factors that influence machine failure. Understanding these factors 
                    is crucial for effective predictive maintenance. It can help us to identify potential issues before they result in failure, 
                    and to take appropriate preventative measures.
                    """
    )

    st.markdown("### Overall machine failure feature importances")

    # Create the feature importances plots for machine failure
    feature_importance_target = "Machine failure"
    feature_importance_variables = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    feature_importance_variables.append(feature_importance_target)

    plot_feature_importance(
        df_eda[feature_importance_variables], feature_importance_target
    )

    st.markdown(
        """
                    The total contribution of torque and rotational speed sem be more than 50 percent, which indicates that
                    disturbances in power levels of the machine increases the likelyhood of machine failure
                    more than other factors. Heat seems to be second most effective predictor of machine failure with the combination
                    of air and process temperature which is around 30 percent. This confirms the previous findings on other sections. 
                    Moreover, tool wear is a strong predictor of the remaining lifetime of the machine which is found in previous sections. however,
                    there may be high number of failures due to the tool wear so that, the effect of tool wear seems to be behind of 
                    torque and the rotational speed.
                    
        """
    )

    st.markdown("### Tool wear failure feature importances")

    # Create the feature importances plots for tool wear failure
    feature_importance_target = "TWF"
    feature_importance_variables = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    feature_importance_variables.append(feature_importance_target)

    plot_feature_importance(
        df_eda[feature_importance_variables], feature_importance_target
    )

    st.markdown(
        """
                    Tool wear failure is affected mostly by the tool wear of course, however, there is not much of a difference 
                    between other factors which is an interesting finding to consider in upcoming calculations
                    
        """
    )

    st.markdown("### Heat dissipation failure feature importances")

    # Create the feature importances plots for heat dissipation failure
    feature_importance_target = "HDF"
    feature_importance_variables = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    feature_importance_variables.append(feature_importance_target)

    plot_feature_importance(
        df_eda[feature_importance_variables], feature_importance_target
    )

    st.markdown(
        """
                    Heat dissipation failure is mostly affected by the air temperature, process temperature and rotational speed.
                    This is intuitive because there could be more heat loss when the motor rotational speed is high.
                    
        """
    )

    st.markdown("### Power failure feature importances")

    # Create the feature importances plots for power failure
    feature_importance_target = "PWF"
    feature_importance_variables = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    feature_importance_variables.append(feature_importance_target)

    plot_feature_importance(
        df_eda[feature_importance_variables], feature_importance_target
    )

    st.markdown(
        """
                    Power failure seems only be affected by the torque and the rotational speed and does not depend on other 
                    factors. This is intuitivee too because power is composed of rotaional speed times torque and changes in power
                    only affects the torque an rotational speed of the machine
                    
        """
    )

    st.markdown("### Over strain failure feature importances")

    # Create the feature importances plots for power failure
    feature_importance_target = "OSF"
    feature_importance_variables = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    feature_importance_variables.append(feature_importance_target)

    plot_feature_importance(
        df_eda[feature_importance_variables], feature_importance_target
    )

    st.markdown(
        """
                    Overstrain happens when the tool is weared and becames weak and then high amounts of force is applied to
                    the system which is high amounts of torque in this case. This feature importance values are very intuitive as well as 
                    power failure and heat dissipation
                    
        """
    )

    st.markdown("## Summary")

    st.markdown(
        """
                    The failure analysis revealed several key insights. There was a noticeable increase in machine failures between the 
                    4,000 to 5,000 datapoints, suggesting conditions during this period contributed to the failures. Tool wear showed 
                    a strong correlation with machine failure, with failure typically occurring around 200-250 minutes of tool use. 
                    Heat dissipation failures aligned perfectly with an increase in air temperature, supporting the idea that high air 
                    temperatures can lead to more machine failures.

                    Power failures occurred intermittently, suggesting that deviations from optimal power levels could lead to failure. 
                    Overstrain failures were somewhat consistent across the observed timeframe, potentially influenced by the same 
                    factors causing power and tool wear failures.

                    High air temperatures and tool wear significantly increased the likelihood of machine failure, whereas process 
                    temperature and rotational speed were less influential. However, very low rotational speeds and extreme torque 
                    values could potentially lead to a failure.

                    The combined effects of torque and rotational speed were the largest contributors to machine failure, followed 
                    by heat. Even though tool wear is a strong predictor of the remaining lifespan of the machine, it has less 
                    influence compared to torque and rotational speed. These insights can guide future predictive maintenance efforts.
                    
                    
        """
    )
