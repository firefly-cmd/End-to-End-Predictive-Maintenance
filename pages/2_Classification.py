import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px

import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import plotly.figure_factory as ff
from streamlit_card import card
from streamlit_image_select import image_select
import base64
from pages.classification_tabs.introduction import classification_introduction


# Cache the data in order to not load over and over again
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv("data/ai4i2020.csv")


def load_and_plot_feature_importances(file_path: str):
    # Load the feature importances from the csv file
    feature_importances = pd.read_csv(file_path)

    # Plot the feature importances using Plotly
    fig = px.bar(
        feature_importances,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importances",
    )

    # Display the figure using Streamlit
    st.plotly_chart(fig, use_container_width=True)


def display_metrics(df_metrics):
    metric_container = st.container()

    with metric_container:
        col1, col2, col3 = st.columns(3)

        for column in df_metrics.columns:
            # st.metric(label=column, value="{:.2f}".format(df_metrics.loc[0, column]))

            with open("metric1.png", "rb") as f:
                data = f.read()
                encoded = base64.b64encode(data)
                data = "data:image/png;base64," + encoded.decode("utf-8")

            if column == "Precision" or column == "Cohen Kappa":
                with col1:
                    card(
                        text=column,
                        title="{:.2f}".format(df_metrics.loc[0, column]),
                        on_click=lambda *args: None,
                        image=data,
                    )
            elif column == "Recall" or column == "Balanced Accuracy":
                with col2:
                    card(
                        text=column,
                        title="{:.2f}".format(df_metrics.loc[0, column]),
                        on_click=lambda *args: None,
                        image=data,
                    )

            elif column == "Fbeta Score" or column == "Average Precision":
                with col3:
                    card(
                        text=column,
                        title="{:.2f}".format(df_metrics.loc[0, column]),
                        on_click=lambda *args: None,
                        image=data,
                    )


def load_and_display_metrics(file_path):
    df_metrics = pd.read_csv(file_path)
    display_metrics(df_metrics)


def load_and_plot_confusion_matrix(filename):
    cm = np.loadtxt(filename)

    x = ["Predicted Negative", "Predicted Positive"]
    y = ["Actual Negative", "Actual Positive"]

    # Create a heat map
    figure = ff.create_annotated_heatmap(
        z=cm,
        x=x,
        y=y,
        colorscale="Blues",
        showscale=True,
        annotation_text=cm.astype(int),
    )

    # Update layout
    figure.update_layout(
        height=500,
        width=500,
        title_text="<b>Confusion Matrix</b>",
        title_x=0.5,
        xaxis=dict(title="<b>Predicted Value</b>"),
        yaxis=dict(title="<b>Actual Value</b>"),
    )

    # Update xaxis and yaxis attributes
    figure.update_xaxes(side="top")

    st.plotly_chart(figure, use_container_width=True)


def main():
    st.title("Welcome to classification")

    st.markdown(
        """
    <style>
        div[data-baseweb="select"] > div {
            font-size: 20px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Create 5 tabs
    (
        introduction_tab,
        binary_classification_tab1,
        binary_classification_tab2,
        multivariate_tab1,
        multivariate_tab2,
    ) = st.tabs(
        [
            "Introduction",
            "Machine Failure Classification Models",
            "Machine Failure Classification Results",
            "Root Cause Analysis Models",
            "Root Cause Analysis Results",
        ]
    )

    with introduction_tab:
        classification_introduction()

    with binary_classification_tab1:
        st.title("SELECT A MACHINE LEARNING MODEL TO EXAMINE")

        selected_model = st.selectbox(
            label="xd",
            options=[
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "SVM",
                "XGBoost",
            ],
            label_visibility="hidden",
        )

        if selected_model == "Logistic Regression":  # Logistic regression
            st.markdown("# LOGISTIC REGRESSION")

            st.markdown(
                """
            ## INITIAL ANALYSIS WITH LOGISTIC REGRESSION

            Before we dive into the depths of feature engineering and model tuning, it is vital 
            to establish a baseline understanding of our dataset's predictive potential. At this 
            juncture, we will train a Logistic Regression model on our initial data, devoid of 
            any preprocessing or feature engineering.

            Logistic Regression is a robust and efficient algorithm for binary classification tasks, 
            making it an apt choice for our initial foray into understanding the predictive patterns 
            within our dataset. This model's simplicity and interpretability make it a good starting 
            point for our analysis.

            Why are we doing this? There are a few compelling reasons:

            1. **Establishing a Baseline:** This initial model will serve as a benchmark against 
            which we can compare the performance of our subsequent models. Any improvement in 
            prediction accuracy after preprocessing or feature engineering should exceed this baseline 
            performance.

            2. **Identifying Predictive Features:** Logistic Regression can provide some insight 
            into which features are most predictive of the outcome, even before any feature engineering. 
            This can guide our later efforts in feature selection and engineering.

            Remember, this is just the starting point. Our goal is to progressively enhance the 
            performance of our models by employing a range of data preprocessing, feature engineering, 
            and model tuning techniques. But first, let's see what our raw data can tell us!

            """
            )

            st.markdown("## INITIAL RESULTS")

            # Display the resulting model metrics
            initial_result_filepath = "ml/binary_classification/results/logistic_regression_experiment1/metrics.csv"

            load_and_display_metrics(file_path=initial_result_filepath)

            # Display the confusion matrix
            load_and_plot_confusion_matrix(
                filename="ml/binary_classification/results/logistic_regression_experiment1/confusion_matrix.txt"
            )

            # Load the feature importances and display them
            load_and_plot_feature_importances(
                file_path="ml/binary_classification/results/logistic_regression_experiment1/feature_importances.csv"
            )

            st.markdown(
                """

            ### 1. Features
            The selection of features encompasses a broad spectrum of potential indicators of machine failure. 
            This includes environmental conditions (Air temperature, Process temperature) and operational 
            parameters (Torque, Tool wear, Rotational speed).

            ### 2. Precision (0.13) and Recall (0.85)
            The trained model shows a high recall, meaning it correctly identifies actual failures most of the 
            time. However, the precision is low, indicating a high rate of false positives - situations where 
            the model predicts a machine failure when there isn't one. This could lead to unnecessary maintenance 
            activities or production stops, affecting operational efficiency. Improvements might be necessary 
            to strike a better balance between precision and recall, depending on the business context 
            and the associated costs of false positives and false negatives.

            ### 3. F-beta Score (0.40)
            Given the imbalance between precision and recall, the F-beta score of the model is relatively low. 
            This suggests that the model's ability to balance precision and recall is not optimal. It might be 
            beneficial to tune the model to increase the F-beta score, particularly if there's a need to 
            put more emphasis on precision or recall.

            ### 4. Cohen's Kappa (0.18)
            The Cohen's Kappa score indicates that the agreement between the model's predictions and the actual 
            results is not strong. This means there's a significant scope for improvement. A low Cohen's Kappa 
            score could point to a model that is only slightly better than random guessing, indicating a need 
            for further investigation.

            ### 5. Balanced Accuracy (0.83)
            The balanced accuracy score suggests that the model performs reasonably well across both positive 
            and negative classes. However, given the low precision, this might reflect a bias towards correctly 
            predicting the majority class. This warrants further investigation into potential model bias and 
            class imbalance.

            ### 6. Average Precision (0.11)
            The low average precision score indicates a high rate of false positives, supporting the findings 
            from the low precision score. This could also suggest that the model may not perform well across 
            different thresholds, which could be a concern if there's a need to adjust the decision threshold 
            based on changing business needs.

            ### 7. Feature Importance
            The analysis of feature importance provides insight into the features driving the model's predictions. 
            Torque, Rotational speed, and Air temperature are the most influential, whereas Process temperature 
            has a negative influence. This suggests that as the Process temperature increases, the likelihood 
            of machine failure, as predicted by the model, decreases. This counter-intuitive relationship 
            warrants further investigation.

            ### 8. Hyperparameters (C=2.15, solver='lbfgs', penalty='l2', class_weight='balanced')
            The hyperparameters reveal a focus on regularization (l2 penalty) and addressing class imbalance 
            (balanced class weight). There might be a need to further tune the C parameter to optimize the 
            trade-off between model complexity and regularization. Testing different solvers could also 
            potentially result in better model performance.

            ### Conclusion
            While the model demonstrates a strong ability to identify actual failures, its low precision and 
            average precision scores suggest a need for improvement in reducing false positives. The feature 
            importance analysis has highlighted some unexpected relationships, indicating a need for further 
            data exploration and potential feature engineering. Lastly, there could be benefits from further 
            hyperparameter tuning and testing different model types to potentially improve model performance.

            """
            )

            with st.expander("INVESTIGATE SECOND EXPERIMENT"):
                st.markdown("### SECOND EXPERIMENT")

                st.markdown(
                    """
                ## Experimental Setup: Feature Engineering

                In the second experiment, a more sophisticated feature engineering approach was employed to 
                capture the complex relationships between the raw sensor data. 

                The raw sensor data was transformed into three new features, each designed to encapsulate 
                the underlying physics of the manufacturing process:

                1. **Power**: This feature is a direct product of torque and rotational speed. It represents 
                the mechanical power input to the process, which is a fundamental aspect of the manufacturing 
                operation. By combining torque and rotational speed into a single feature, we aim to capture 
                their synergistic effect on machine failure.

                2. **Strain**: This feature is the product of tool wear and torque. High torque levels combined 
                with a highly worn tool can lead to overstrain, potentially causing a machine failure. This 
                feature is expected to help the model detect situations where the machine is being pushed beyond 
                its limits.

                3. **Temperature Difference**: The difference between the process temperature and the air temperature 
                is used as a feature. This temperature gradient is critical in many manufacturing processes. An 
                abnormal temperature difference could signal a problem with the heat dissipation system or a process 
                anomaly, leading to potential machine failure.

                All the raw sensory data was dropped and only these engineered features were used for the machine 
                learning model in this experiment. This approach emphasizes the belief that these engineered features 
                capture the key factors driving machine failures, and simplifies the model by reducing the dimensionality 
                of the input data.

                In this setup, the model is expected to understand the complex relationships between these features and 
                the machine failure, providing a robust and interpretable predictive tool.

                """
                )

                st.markdown("RESULTS OF THE SECOND EXPERIMENT")

                # Display the resulting model metrics
                initial_result_filepath = "ml/binary_classification/results/logistic_regression_experiment2/metrics.csv"

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/logistic_regression_experiment2/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/logistic_regression_experiment2/feature_importances.csv"
                )

                st.markdown(
                    """

                ### Model Performance

                The performance of the model in this experiment showed a decrease across all metrics compared to the 
                first experiment. Precision fell to 0.09, indicating an increase in false positives. The recall also 
                dropped slightly to 0.74, revealing a diminished ability of the model to identify all machine failures. 
                The F-beta score decreased to 0.30, reflecting the overall decrease in both precision and recall. 
                Cohen's Kappa score, a measure of the level of agreement between the model's predictions and the actual 
                results, also experienced a decline. The balanced accuracy of the model, representing its overall 
                performance across both classes, showed a decrease, and the average precision dropped to 0.08, 
                signifying a higher rate of false positives.

                ### Feature Importance

                Analysis of feature importance showed that the newly introduced strain, which is a product of tool wear 
                and torque, was the most significant feature. This suggests that situations where the machine is being 
                pushed beyond its limits are key indicators of machine failures. The temperature difference, however, 
                had a negative influence on the model's prediction, implying that an increase in this temperature 
                difference leads to a decrease in the predicted likelihood of machine failure. The calculated power, 
                being a product of torque and rotational speed, had a positive influence on the model's predictions, 
                suggesting that higher power inputs are associated with a higher risk of machine failure.

                ### Hyperparameters

                The hyperparameters utilized in this experiment included a C value of 0.0003, indicating a higher degree 
                of regularization; the 'lbfgs' solver, which is optimal for smaller datasets; the 'l2' penalty, adding a 
                regularization term to the model's cost function based on the square magnitude of the coefficients; and 
                the 'balanced' class weight, adjusting weights inversely proportional to class frequencies to handle 
                class imbalance.

                ### Conclusion

                Compared to the initial experiment, the feature engineering approach did not improve the predictive power of 
                the model. In fact, the decrease in model performance across all metrics suggests that further work is 
                required. This could involve exploring different methods of combining the sensor readings, considering 
                additional, potentially relevant features, or using a different machine learning model that might better 
                manage the complexities introduced by the new features.
                
                """
                )

            with st.expander("INVESTIGATE THIRD EXPERIMENT"):
                st.write("### THIRD EXPERIMENT")

                st.markdown(
                    """## Experiment 3: Combined Raw Data and Feature Engineering

                In the third experiment, a hybrid approach was adopted which combines the raw data and the new features 
                created in the second experiment. This decision was made based on the results of the second experiment 
                which showed a decrease in model performance, suggesting potential information loss when using only the 
                engineered features. The aim of this experiment is to leverage the information from both the raw sensor 
                readings and the relationships captured in the engineered features. 

                The combined dataset therefore includes the raw sensor readings (Air temperature, Process temperature, 
                Torque, Tool wear, Rotational speed) as well as the engineered features from the second experiment 
                (power, strain, temperature difference). Power was calculated as a multiplication of torque and rotational 
                speed, strain was a result of multiplying tool wear and torque, and temperature difference was the 
                difference between process and air temperature.

                The model will be trained on this combined dataset, and the results will be compared with the previous 
                experiments to evaluate the effectiveness of this approach.

                """
                )

                st.markdown("RESULTS OF THE THIRD EXPERIMENT")

                # Display the resulting model metrics
                initial_result_filepath = "ml/binary_classification/results/logistic_regression_experiment3/metrics.csv"

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/logistic_regression_experiment3/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/logistic_regression_experiment3/feature_importances.csv"
                )

                st.markdown(
                    """
                    

                The third experiment, which incorporated both the raw data and the engineered features, achieved mixed results. 

                The precision increased to 0.15 from 0.13 in the first experiment, indicating a decrease in the rate of false 
                positives. However, the recall decreased slightly to 0.82 from 0.85, suggesting that the model missed a few 
                more actual failures. The F-beta score, which balances precision and recall, improved to 0.44 from 0.40. These 
                changes suggest that incorporating the engineered features with the raw data improved the model's overall 
                predictive performance.

                The Cohen's Kappa score, measuring the agreement between the model's predictions and the actual results, also 
                improved from 0.18 to 0.22, indicating that the model is better than random guessing, but still has room for 
                improvement.

                The balanced accuracy increased slightly to 0.84 from 0.83, indicating a slight improvement in overall accuracy 
                across both classes. The average precision also improved slightly from 0.11 to 0.13, suggesting that the model 
                is slightly more effective across different thresholds. 

                The most influential feature in the third experiment was 'Torque', followed by 'power' and 'Tool wear'. 
                Interestingly, 'power', an engineered feature, had a negative influence, suggesting that as the power 
                increases, the likelihood of machine failure, as predicted by the model, decreases. This counter-intuitive 
                relationship might warrant further investigation.

                The hyperparameters for the third experiment were quite different from the first one, with a significant 
                increase in the C parameter, suggesting a decrease in the model complexity, and a change in the solver 
                to 'liblinear'.

                In conclusion, the third experiment, which combined raw data with engineered features, showed slight 
                improvements in some metrics compared to the first experiment. However, there's still a need for further 
                refinement in terms of reducing false positives, understanding the counter-intuitive relationship of 
                some features, and tuning the hyperparameters for better performance.

                """
                )

            with st.expander("INVESTIGATE FORTH EXPERIMENT"):
                st.write("### FORTH EXPERIMENT")

                st.markdown(
                    """
                ## Fourth Experiment: Kernel PCA on Raw Sensor Readings

                In the fourth experiment, a new approach was employed to handle the complex relationships 
                suspected in the raw sensor readings. Kernel Principal Component Analysis (Kernel PCA), a 
                powerful technique for dimensionality reduction and capturing non-linear relationships, 
                was used in the preprocessing pipeline. 

                Two features were extracted from the raw sensor readings using Kernel PCA, aiming to simplify 
                the data and potentially improve the model's ability to understand the underlying structure. 
                It's worth noting that the Kernel PCA process is a form of unsupervised learning, meaning it 
                didn't take into account the target variable while transforming the features. 

                This experiment was designed to test whether this transformation could lead to a better 
                performance in the subsequent logistic regression model.
                """
                )

                st.markdown("RESULTS OF THE FORTH EXPERIMENT")

                # Display the resulting model metrics
                initial_result_filepath = "ml/binary_classification/results/logistic_regression_experiment4/metrics.csv"

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/logistic_regression_experiment4/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/logistic_regression_experiment4/feature_importances.csv"
                )

                st.markdown(
                    """
                The fourth experiment attempted to improve upon the initial experiment by utilizing Kernel PCA to transform 
                raw sensor readings into two principal components: PC1 and PC2. These two features served as the only 
                predictors in the subsequent logistic regression model. However, the results from the fourth experiment demonstrated 
                a decrease in performance across all evaluation metrics when compared to the initial experiment.

                Both precision and recall significantly dropped, suggesting that the Kernel PCA transformation may not have 
                successfully retained the critical information from the raw sensor readings necessary for accurate prediction. 
                The F-beta score, which reflects the balance between precision and recall, also saw a decline, indicating a 
                worsening in the model's overall performance. Cohen's Kappa score further supported this conclusion, as it 
                demonstrated a decrease, indicating weaker agreement between the model's predictions and the actual results.

                The model's balanced accuracy and average precision scores both decreased, suggesting a poorer performance across 
                both positive and negative classes and a higher rate of false positives. Furthermore, the feature importance analysis 
                revealed that both principal components extracted through Kernel PCA negatively influenced the model's predictions. 
                However, due to the nature of PCA, it became challenging to interpret these feature importances within the context 
                of the original raw sensor readings.

                The model's hyperparameters revealed a considerable increase in the C parameter, suggesting a more complex model 
                with less regularization compared to the initial experiment. This change may have been an attempt to capture more 
                complexity in the transformed data, but it did not result in an improvement in the model's performance.

                In conclusion, the fourth experiment did not improve the logistic regression model's performance compared to the 
                initial experiment. The decrease in all evaluation metrics and the difficulty interpreting the transformed features 
                indicate that this approach might not be suitable for this particular problem. Further experiments could explore 
                other feature extraction techniques or different types of models to potentially improve model performance.
                                            

                """
                )

        elif selected_model == "Decision Tree":
            st.write("SECOND MODEL")
        elif selected_model == "Random Forest":
            st.write("THIRD MODEL")
        elif selected_model == "SVM":
            st.write("FORTH MODEL")
        elif selected_model == "XGBoost":
            st.write("FIFTH MODEL")


main()
