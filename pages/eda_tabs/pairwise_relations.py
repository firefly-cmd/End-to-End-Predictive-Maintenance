import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Function to create hexbin plots for two variables
def plot_hexbin_pairs(df, column1, column2):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Machine Failure = 0", "Machine Failure = 1"),
    )

    df_failure_0 = df[df["Machine failure"] == 0]
    df_failure_1 = df[df["Machine failure"] == 1]

    fig.add_trace(
        go.Histogram2d(
            x=df_failure_0[column1],
            y=df_failure_0[column2],
            nbinsx=20,
            nbinsy=20,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram2d(
            x=df_failure_1[column1],
            y=df_failure_1[column2],
            nbinsx=20,
            nbinsy=20,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text=column1, row=1, col=1)
    fig.update_yaxes(title_text=column2, row=1, col=1)
    fig.update_xaxes(title_text=column1, row=1, col=2)
    fig.update_yaxes(title_text=column2, row=1, col=2)

    fig.update_layout(title_text=f"Hexbin Plot of {column1} vs {column2}")
    st.plotly_chart(fig, use_container_width=True)


# Function to plot correlation heatmap
def plot_corr(df):
    corr = df.corr()
    fig = go.Figure(
        data=go.Heatmap(z=corr, x=corr.columns, y=corr.columns, hoverongaps=False)
    )
    fig.update_layout(title_text="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)


# Pairwise relation tab function for exploratory data analysis page
def pairwise_relations_analysis(df_eda):
    st.title("Pairwaise Relations Analysis")
    st.markdown(
        f"""
                In this part of our analysis, we focus on the pairwise relationships between the different features in our dataset. Understanding these relationships is crucial, 
                as they can provide important insights into the structure of our data and potential interactions between features. 
                
                We're using hexbin plots for this analysis, which are particularly useful for large datasets. Hexbin plots display the relationship between two numerical variables, 
                similar to a scatter plot, but instead of individual points, the plotting area is divided into hexagons. The color of each hexagon represents the number of observations 
                within it, which makes it easier to interpret the density of data points in different areas of the plot.
                
    """
    )

    st.markdown(f"## Correlation between variables")

    # Take a part of dataframe which has no categorial varibles in it to use it inside the correlation matrix
    df_non_categorical = df_eda[
        df_eda.columns[
            ~df_eda.columns.isin(
                [
                    "Product ID",
                    "Type",
                    "UDI",
                    "Moving Average",
                    "TWF",
                    "HDF",
                    "PWF",
                    "OSF",
                    "RNF",
                ]
            )
        ]
    ]
    # Plot the correlation heatmap
    plot_corr(df_non_categorical)

    st.markdown(
        f"""
            There is a strong negative relationship between torque and the rotational speed due to the nature of motors. In many systems,
            there is a trade off between speed and the torque. Moreover there is a clear positive correlation between process temperature
            and the air temperature which confirms the findings in the variable distribution part. They are related and may be affecting each other.
            
            Considering the correlation between machine failure and the torque which is around 0.2, increases in torque may be somewhat
            related to the machine failures. Also considering the fluctuations and high number of elevated values observed in the variable distribution
            analysis part, there is a high chance that elevated torque values is an indicator of machine failure.
            
            Moreover, the tool wear has the correlation factor of 0.1 which is the second highest among other features. Together with the shape of the 
            distribution which has very low number of observations after 200 mins of tool wear, after somewhere around 200 mins of tool wear, we may
            conclude that, there is an increasing chance of machine failure. 
            
            Temperature values does not seem to have much correlation with the machine failure comparing with the other factors. This may due to there are 
            less failures due to temperature than the other factors
                
                
    """
    )

    st.markdown(f"## Pairwise plots")
    st.markdown(
        f"""
                
    Pair plots, also known as scatterplot matrices, are a great method to visualize pairwise relationships in our dataset. 
    It constructs a matrix of scatter plots where each feature in the dataset is plotted against every other feature. 
    In other words, if we have n features in our dataset, a pair plot will have n*n scatter plots.

    In the context of this predictive maintenance dataset, pair plots can be particularly useful in visualizing the relationships 
    between different sensor readings or features. For example, we might want to understand if there is a relationship between 
    air temperature and rotational speed, or between torque and tool wear.

    However, with a large number of data points, traditional scatter plots can become less effective due to overplotting, where 
    the density of points makes it difficult to see patterns. This is where hexbin plots can be particularly useful. Hexbin plots 
    divide the plot into hexagonal bins and the color of each hexagon represents the number of points in that bin. This can make 
    it easier to see patterns, especially in areas of high density.

    In the next section, we will generate a series of pair plots using hexbin plots for 10 different pairs of features in our dataset. 
    By examining these plots, we will gain insights into the relationships between these features and how these relationships might 
    impact machine failure.
                
    """
    )
    st.markdown(f"### Air Temperature - Process temperature Relationship")
    plot_hexbin_pairs(df_eda, "Air temperature [K]", "Process temperature [K]")

    st.markdown(
        f"""
            There is a clear linear trend when it comes to air temperature and process temperature.
            Moreover, if we compare the highlighted parts in the subplots which machine failure is observed or not,
            there is not much clear difference between measurements when machine failure happens or not.
                """
    )

    st.markdown(f"### Air Temperature - Rotational Speed")
    plot_hexbin_pairs(df_eda, "Air temperature [K]", "Rotational speed [rpm]")

    st.markdown(
        f"""
            When it comes to relationship between air temperature and the rotational speed, there are clearly
        visible highlighted areas in the machine failure observed subplot. This may mean that, low levels of
        rotational speed combined with the slightly higher temperature levels may cause a 
        machine failure.
                
    """
    )

    st.markdown(f"### Air Temperature - Torque")
    plot_hexbin_pairs(df_eda, "Air temperature [K]", "Torque [Nm]")

    st.markdown(
        f"""
            The air temperature and the torque relationship is expected considering the air temperature and rotational
            speed. Since there is a clear trade off between rotational speed and the torque, when the rotational speed
            is low torque is high and combined with the slight increase in the air temperature may cause machine failure.
                
    """
    )

    st.markdown(f"### Air Temperature - Tool Wear")
    plot_hexbin_pairs(df_eda, "Air temperature [K]", "Tool wear [min]")

    st.markdown(
        f"""
            There are highlighted areas in the machine failure observation plot (in the right figure). However these
            highlighted areas indicates that there is an increasing chance of machine failure when tool wear is above
            around 200 independent of air temperature, and slighly less highlighted areas that air temperature is high 
            for every tool wear levels. It seems that these two variables are less related than the others and high tool
            wear levels and high process temperature levels indicate machine failure.
                
    """
    )

    st.markdown(f"### Process Temperature - Rotational Speed")
    plot_hexbin_pairs(df_eda, "Process temperature [K]", "Rotational speed [rpm]")

    st.markdown(
        f"""
        When it comes to the process temperature and rotational speed levels, the machine failures are highly focused on
        the low levels of rotational speed and moderate to high levels of process temperature. This result is very similar
        to air temperature and rotational speed case.
                
    """
    )

    st.markdown(f"### Process Temperature - Torque")
    plot_hexbin_pairs(df_eda, "Process temperature [K]", "Torque [Nm]")

    st.markdown(
        f"""
            Considering both the machine failure or not cases, there is a clear torque level cutoff which may indicate
            machine failures. We can say similar things about process temperature vs torque case with air temperature vs 
            torque case
                
                
    """
    )

    st.markdown(f"### Process Temperature - Tool wear")
    plot_hexbin_pairs(df_eda, "Process temperature [K]", "Tool wear [min]")

    st.markdown(
        f"""
        These results are similar with the air temperature vs tool wear case. Tool wear does not related with the 
        temperature values
                
    """
    )

    st.markdown(f"### Rotational Speed - Torque")
    plot_hexbin_pairs(df_eda, "Rotational speed [rpm]", "Torque [Nm]")

    st.markdown(
        f"""
            There is a reverse relationship with torque and the rotational speed. They have very high negative correlation
            factor and pair plot shows that this too. Considering this relationship, we can say that this machine mostly operates
            on constant power. The highlighted areas are showing that when the multiplicccation of torque and rotational speed is low so that power
            level of the machine is dropped, the machine fails.
                
    """
    )

    st.markdown(f"### Rotational Speed - Tool Wear")
    plot_hexbin_pairs(df_eda, "Rotational speed [rpm]", "Tool wear [min]")

    st.markdown(
        f"""
        Highlighted areas are too condensed in the machine failure plot. When rotational speed is low and there are high levels of tool wear
        especially higher than the 200, it is very probably that there could be a machine failure
                
    """
    )

    st.markdown(f"### Torque - Tool Wear")
    plot_hexbin_pairs(df_eda, "Torque [Nm]", "Tool wear [min]")

    st.markdown(
        f"""
        Since there is a reverse relation between rotational speed and the torque, this result of machine failures are expected.
        When there are low levels of rotational speed, there are high levels of torque and combined with the 
        high tool wear values, there is an increasing chance that there is a machine failure
                
    """
    )

    st.markdown("## Summary")
    st.markdown(
        """
    In summary, low rotational speed, high torque, and high tool wear are all significant factors that increase the likelihood of machine failure. 
    Additionally, slightly elevated air and process temperatures can also contribute to machine failure. On the other hand, the machine tends to 
    operate at constant power, and deviations from this pattern might signal an impending machine failure.
                
    """
    )
