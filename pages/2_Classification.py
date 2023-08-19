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
import joblib


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


def display_metrics(df_metrics, file_path):
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
                        key=file_path + column,
                    )
            elif column == "Recall" or column == "Balanced Accuracy":
                with col2:
                    card(
                        text=column,
                        title="{:.2f}".format(df_metrics.loc[0, column]),
                        on_click=lambda *args: None,
                        image=data,
                        key=file_path + column,
                    )

            elif column == "Fbeta Score" or column == "Average Precision":
                with col3:
                    card(
                        text=column,
                        title="{:.2f}".format(df_metrics.loc[0, column]),
                        on_click=lambda *args: None,
                        image=data,
                        key=file_path + column,
                    )


def load_and_display_metrics(file_path):
    df_metrics = pd.read_csv(file_path)
    display_metrics(df_metrics, file_path)


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
        try_yourself_tab,
    ) = st.tabs(
        [
            "Introduction",
            "Machine Failure Classification Models",
            "Machine Failure Classification Results",
            "Root Cause Analysis Models",
            "Root Cause Analysis Results",
            "Try Yourself",
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

                st.markdown("## RESULTS OF THE THIRD EXPERIMENT")

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
            st.markdown("# DECISION TREES")

            st.markdown(
                """
            ## INTRODUCTION TO DECISION TREES

            Before diving deep into advanced modeling techniques and intricate feature manipulations, it's crucial to build a 
            foundational understanding of the inherent characteristics present in our dataset. For this purpose, we'll commence 
            our analysis using a Decision Tree model on our raw, unaltered data.

            Decision Trees are intuitive and versatile algorithms, fitting for both classification and regression tasks. The nature 
            of a Decision Tree, where decisions are made based on asking a series of questions, mirrors human decision-making processes, 
            making the model easy to visualize and comprehend.

            So, why have we chosen the Decision Tree for this initial phase? Here are the primary motivations:

            1. **Visual Interpretability**: One of the standout features of Decision Trees is their visual interpretability. We can 
            literally visualize the tree structure, seeing how decisions are made at each node. This will allow us to quickly grasp 
            how our features contribute to predictions, even without any enhancements.
            2. **Feature Importance**: Decision Trees naturally rank features based on their importance in making splits. This offers 
            a preliminary understanding of which features might be the most influential in predicting our target variable, setting the 
            stage for further analysis and feature engineering.
            3. **Flexibility with Data**: Decision Trees can handle both numerical and categorical data, and they aren’t easily thrown 
            off by outliers or missing values. This makes them an ideal candidate for our initial analysis on raw data.

            In essence, this step represents our journey's beginning. Our ambition is to progressively refine and optimize our models, 
            leveraging sophisticated feature engineering, and model tuning strategies. But for now, let's unearth the insights hidden in 
            our untapped data using the Decision Tree!
            """
            )

            st.markdown("## INITIAL RESULTS")

            # Display the resulting model metrics
            initial_result_filepath = (
                "ml/binary_classification/results/decision_tree_experiment1/metrics.csv"
            )

            load_and_display_metrics(file_path=initial_result_filepath)

            # Display the confusion matrix
            load_and_plot_confusion_matrix(
                filename="ml/binary_classification/results/decision_tree_experiment1/confusion_matrix.txt"
            )

            # Load the feature importances and display them
            load_and_plot_feature_importances(
                file_path="ml/binary_classification/results/decision_tree_experiment1/feature_importances.csv"
            )

            st.markdown(
                """
                #### **1. Features**

                The model considers a diverse range of features, which captures both environmental conditions 
                (Air temperature, Process temperature) and operational parameters (Torque, Tool wear, Rotational 
                speed). This breadth ensures that the model is taking into account various aspects of the operational 
                environment which can potentially influence machine failure.

                #### **2. Precision (0.1629) and Recall (0.9344)**

                The decision tree model demonstrates an exceptional recall, meaning it has a keen ability to correctly 
                flag actual machine failures. Conversely, its precision is on the lower side, suggesting that there are 
                scenarios where the model predicts a machine failure when, in reality, none exists. Such false positives 
                could lead to unwarranted intervention, thus affecting operational flow and efficiency. A balance between 
                precision and recall might be sought after, especially depending on the costs associated with incorrect 
                predictions.

                #### **3. F-beta Score (0.4798)**

                Given the disparity between precision and recall, the F-beta score for the model is not particularly high. 
                This reveals a model that currently leans more towards recall. Depending on the operational requirements, 
                we might consider tuning the model to enhance this score, especially if one metric is more valuable than 
                the other in a specific business context.

                #### **4. Cohen's Kappa (0.2378)**

                The obtained Cohen's Kappa score showcases that the agreement between the model predictions and the actual 
                occurrences isn't very robust. This suggests there's considerable room for model refinement. A Cohen's Kappa 
                score at this level implies the model performs better than mere chance but has significant potential for improvement.

                #### **5. Balanced Accuracy (0.8917)**

                The balanced accuracy portrays a model that performs well across both the positive and negative classifications. 
                Nevertheless, the precision score suggests there might be a bias towards predicting one class over the other, 
                which requires more in-depth exploration to understand any underlying model bias or data imbalance.

                #### **6. Average Precision (0.1542)**

                Echoing the precision score, the low average precision emphasizes the model's propensity for false positives. 
                It might be inferred that the model could face challenges across various thresholds, posing issues if business 
                requirements demand adjustment of prediction sensitivity.

                #### **7. Feature Importance**

                Evaluating feature importance reveals pivotal insights:
                - **Rotational speed [rpm]** is the standout influencer, suggesting changes in rotational speed strongly 
                predict machine failure.
                - **Tool wear [min]** and **Torque [Nm]** also play significant roles, indicating the condition of the 
                tool and torque exerted are crucial indicators of potential machine issues.
                - Surprisingly, **Process temperature [K]** doesn't seem to hold any predictive power in this model iteration, 
                which might demand further analysis to understand its relevance.

                #### **8. Hyperparameters (max_depth=7, min_samples_split=0.0951, min_samples_leaf=0.0170, class_weight='balanced')**

                The chosen hyperparameters indicate:
                - The model's depth is controlled to prevent overfitting.
                - Minimum samples for a split and leaf are set to non-default values, showcasing an effort to adjust 
                the model's learning pattern.
                - The 'balanced' class weight suggests an attempt to handle any class imbalances present in the training data.

                #### **Conclusion**

                The decision tree model excels in identifying genuine machine failures but struggles with false positives. 
                Feature importance reveals significant insights about operational parameters' role in predictions. While this 
                is a promising start, there are clear areas for enhancement, especially around precision and the role of 
                specific features. Further experiments, hyperparameter tweaking, and potentially even a different model or 
                ensemble approach might be the next steps in refining the predictions.
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

                st.markdown("## RESULTS OF THE SECOND EXPERIMENT")

                # Display the resulting model metrics
                initial_result_filepath = "ml/binary_classification/results/decision_tree_experiment2/metrics.csv"

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/decision_tree_experiment2/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/decision_tree_experiment2/feature_importances.csv"
                )

                st.markdown(
                    """
                ### **Model Performance**

                The Decision Tree model from the second experiment displayed varied performance across metrics when 
                compared to the inaugural experiment. 

                - **Precision** elevated to 0.2816, signaling a notable decline in false positives.

                - **Recall**, while still fairly robust, saw a minor downtick to 0.8033, hinting at a marginal compromise 
                in detecting all potential machine failures.

                - The **F-beta score** increased to 0.5861, suggesting a more balanced harmony between precision and recall.

                - **Cohen's Kappa** ascended to 0.3894, indicating better agreement between the model's predictions and 
                actual outcomes compared to the first experiment.

                - **Balanced accuracy** displayed a slight decrement, but with a score of 0.8694, the model remains adept 
                at handling both positive and negative classes.

                - **Average precision** swelled to 0.2322, further substantiating the model's enhanced precision.

                ### **Feature Importance**

                Delving into feature significance, a few observations emerged:

                - **Power (0.6108)**: Representing the interplay of torque and rotational speed, power emerged as the most influential feature, 
                affirming the notion that mechanical dynamics critically shape machine health.

                - **Temperature Difference (0.2003)**: While being a significant influencer, its impact suggests a nuanced role in 
                the model's predictions.

                - **Strain (0.1889)**: Indicative of tool stress, strain underscores scenarios where machines might be operating under 
                undue duress, potentially inching closer to failure.

                ### **Hyperparameters**

                The hyperparameters tailored for this experiment are as follows:

                - With a **max_depth** of 46, the model is enabled to grasp intricate patterns, paving the way for nuanced decision-making.

                - **min_samples_split** and **min_samples_leaf**, set to minuscule values, facilitate granular partitions in the decision tree.

                - The continued use of the **'balanced' class weight** underlines the persistent commitment to tackle class imbalances effectively.

                ### **Conclusion**

                Upon juxtaposition with the first experiment, the second iteration—powered by feature engineering—registered clear 
                improvements in certain metrics, particularly precision, F-beta score, and Cohen's Kappa. This attests to the potency of thoughtful 
                feature design, which embeds domain-specific insights into the model, thereby enhancing its predictive prowess. Yet, there remain 
                avenues to optimize the model further, which might entail delving deeper into feature synthesis, hyperparameter refinement, 
                or even exploring alternative modeling strategies.

                """
                )

            with st.expander("INVESTIGATE THIRD EXPERIMENT"):
                st.markdown("### THIRD EXPERIMENT")

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

                st.markdown("## RESULTS OF THE THIRD EXPERIMENT")

                # Display the resulting model metrics
                initial_result_filepath = "ml/binary_classification/results/decision_tree_experiment3/metrics.csv"

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/decision_tree_experiment3/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/decision_tree_experiment3/feature_importances.csv"
                )

                st.markdown(
                    """
                ### **Model Performance**

                The third Decision Tree experiment, which embraced a fusion of raw data and engineered features, 
                manifested significant advancements in model metrics when juxtaposed with the previous two experiments:

                - **Precision** skyrocketed to 0.8182, which intimates a commendable accuracy in predicting machine 
                failures, significantly minimizing false positives.

                - **Recall**, holding steady at 0.7377, conveys that the model still successfully discerns a considerable 
                majority of actual machine failures.

                - The **F-beta score** ascended to 0.7525, representing a balanced performance in both precision and recall metrics.

                - **Cohen's Kappa**, now at a laudable 0.7692, indicates a strong agreement between the model's predictions 
                and the actual outcomes, highlighting its superiority over random chance.

                - The **Balanced accuracy** remains impressive at 0.8663, underscoring the model's robustness across both classes.

                - **Average precision** also exhibited a leap, settling at 0.6116, underscoring the model's reinforced precision 
                across various thresholds.

                ### **Feature Importance**

                Diving deeper into the terrain of feature influence, illuminating insights come to the fore:

                - **Rotational speed (0.3734)** emerged as the leading influencer, emphasizing the pivotal role of mechanical 
                speed in assessing machine health.

                - **Strain (0.3386)** and **Power (0.2045)** continue to hold significant sway, corroborating their roles as 
                critical synthesized indicators of machine status.

                - **Temperature Difference (0.0705)**, although marginally influential, still factors into the model's 
                decision-making process.

                - Interestingly, raw sensory readings like **Tool wear (0.0129)**, **Air temperature (0.000067)**, **Process temperature**, 
                and **Torque** showcased reduced or negligible influence in the predictions.

                ### **Hyperparameters**

                Tailoring the model further with specific hyperparameters:

                - A **max_depth** of 47 was established, allowing the tree to glean intricate relationships within the combined data.

                - **min_samples_split** and **min_samples_leaf**, both fine-tuned, accommodate specific decision-making nodes in the tree.

                - Adhering to the **'balanced' class weight** ensures the model's unwavering focus on treating class imbalances.

                ### **Conclusion**

                Experiment 3, a melange of raw sensory readings and engineered features, heralds a palpable stride in model performance. 
                It affirms the belief that a holistic dataset, which encapsulates direct readings and domain-informed synthetic features, 
                crafts a more potent predictive model. The pronounced influence of both raw and engineered features evinces their collective 
                merit in prediction. Still, there's always room to explore – be it refining the synthesis of new features, fine-tuning 
                hyperparameters, or even mulling over alternative modeling strategies.

                """
                )

            with st.expander("INVESTIGATE FORTH EXPERIMENT"):
                st.markdown("### FORTH EXPERIMENT")

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
                initial_result_filepath = "ml/binary_classification/results/decision_tree_experiment4/metrics.csv"

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/decision_tree_experiment4/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/decision_tree_experiment4/feature_importances.csv"
                )

                st.markdown(
                    """
                ### **Model Performance**

                The fourth Decision Tree experiment took a different pathway, applying Kernel PCA to the 
                raw sensor readings. Let's delve into the outcomes:

                - **Precision** took a nosedive to 0.0606, revealing an accentuated rate of false positives, which may 
                cast doubts on the model's trustworthiness.

                - **Recall** stands at 0.5902, indicating the model's compromised capacity to identify over half of the 
                true machine failures.

                - The **F-beta score** sharply declined to 0.2148, mirroring the dip in both precision and recall.

                - **Cohen's Kappa** settled at a mere 0.0578, denoting a borderline trivial agreement between the model's 
                predictions and actual results.

                - **Balanced Accuracy** still hovers at 0.6512, showcasing a moderate performance across the class spectrum.

                - **Average Precision** spiraled down to 0.0483, mirroring the model's weakened precision throughout various thresholds.

                ### **Feature Importance**

                On the frontier of feature significance, intriguing patterns emerge:

                - **PC2 (0.6851)** ascended as the paramount contributor, signifying its dominance in encapsulating the underlying structure of the data.

                - **PC1 (0.3149)**, although secondary, still holds substantive weight in the prediction process.

                ### **Hyperparameters**

                The model's architecture and behavior were sculpted by several hyperparameters:

                - With a **max_depth** of 47, the model was designed to plumb the depths of the transformed data.

                - The settings of **min_samples_split** and **min_samples_leaf** were notably higher than previous experiments, 
                possibly to address the different nature of the PCA components.

                - The persistence of the **'balanced' class weight** ensures the model’s focus on both classes, despite any imbalances.

                ### **Conclusion**

                The fourth experiment, underpinned by Kernel PCA, ushered in a perceptible dip in performance metrics. While the 
                technique is potent in capturing non-linearities and reducing dimensionality, its applicability in the context of 
                this problem seems debatable. The reduced precision and recall, paired with the minimal influence of PC1, suggests 
                that crucial information may have been lost during the transformation. It's essential to reevaluate whether 
                unsupervised transformations like Kernel PCA are suitable for this dataset or if alternative strategies, such as 
                supervised dimensionality reduction or other feature engineering techniques, could be more beneficial.

            """
                )

        elif selected_model == "Random Forest":
            st.markdown("# RANDOM FOREST")

            st.markdown(
                """
                ## **INTRODUCTION TO RANDOM FORESTS**

                Before navigating the maze of ensemble methods and delving into the deeper layers of our data, it's of paramount importance 
                to grasp the synergistic principles governing the interactions within our dataset. Thus, our narrative unfolds with the 
                introduction of the Random Forest model, harnessing the raw, pristine essence of our data.

                Random Forests, an ensemble of Decision Trees, are known for their robustness and adaptability, making them apt for diverse 
                machine learning challenges, be it classification or regression. By combining multiple trees to make decisions, Random Forests 
                tend to mitigate the overfitting issue inherent in single Decision Trees, providing a more generalized solution.

                What steers our compass towards the Random Forest at this juncture? Here are our guiding stars:

                1. **Reduction of Overfitting**: Unlike a single Decision Tree, which can be overly specific to the training data, Random Forests, 
                by averaging out decisions from multiple trees, offer a balanced perspective, reducing the chances of overfitting.

                2. **Feature Importance on Steroids**: While Decision Trees rank features based on their importance, Random Forests aggregate this 
                ranking across multiple trees. This cumulative insight bestows a more holistic view of feature significance, laying the groundwork 
                for more targeted feature engineering.

                3. **Versatility in Handling Data**: Random Forests inherit the merits of Decision Trees, being adept at managing both categorical 
                and numerical data, and displaying resilience against outliers and missing values. This makes them a potent tool for our maiden 
                voyage into the raw data realm.

                4. **Boosted Accuracy**: Owing to its ensemble nature, Random Forests generally promise a spike in accuracy, as they draw wisdom 
                from multiple decision-making entities, making them less prone to individual biases.

                Charting our course with Random Forests symbolizes an evolved phase in our exploration. As we wade deeper, our quest will be 
                augmented with advanced feature manipulations, ensemble strategies, and meticulous model tuning. But at this moment, let the 
                Random Forest guide us through the troves of knowledge veiled within our untouched data!

            """
            )

            st.markdown("## INITIAL RESULTS")

            # Display the resulting model metrics
            initial_result_filepath = (
                "ml/binary_classification/results/random_forest_experiment1/metrics.csv"
            )

            load_and_display_metrics(file_path=initial_result_filepath)

            # Display the confusion matrix
            load_and_plot_confusion_matrix(
                filename="ml/binary_classification/results/random_forest_experiment1/confusion_matrix.txt"
            )

            # Load the feature importances and display them
            load_and_plot_feature_importances(
                file_path="ml/binary_classification/results/random_forest_experiment1/feature_importances.csv"
            )

            st.markdown(
                """
                ### 1. Features

                The Random Forest model consolidates an array of features, encapsulating both ambient metrics (Air temperature, 
                Process temperature) and operational parameters (Torque, Tool wear, Rotational speed). This comprehensive feature 
                profile ensures that the model holistically evaluates various facets that might impact machine malfunction.

                ### 2. Precision (0.1009) and Recall (0.7377)

                With a strong recall, the Random Forest model displays a competent ability to detect true machine failures. 
                However, its lower precision means there might be cases where the model erroneously flags a machine failure, 
                leading to possible unwarranted interventions. Balancing precision and recall will be essential to ensure 
                smooth operations and avoid unnecessary disruptions.

                ### 3. F-beta Score (0.3261)

                The F-beta score indicates a model that is skewed more towards recall. The moderate score suggests that there's 
                potential for improvement, and refining the model to improve this metric could be beneficial, especially if 
                business contexts require a specific trade-off between precision and recall.

                ### 4. Cohen's Kappa (0.1309)

                The derived Cohen's Kappa metric signifies that the alignment between model predictions and real outcomes 
                could be improved. A score at this level indicates the model is performing better than random chance 
                but showcases substantial room for enhancement.

                ### 5. Balanced Accuracy (0.7654)

                The balanced accuracy, while respectable, suggests the model's well-rounded performance in predicting both 
                positive and negative classes. Still, with the precision in context, further investigation might be needed 
                to understand any underlying biases or tendencies.

                ### 6. Average Precision (0.0824)

                The lower average precision resonates with the precision metric, indicating the model's tendency towards 
                false positives. This highlights potential challenges in fine-tuning prediction sensitivity to cater to 
                specific operational needs.

                ### 7. Feature Importance

                Insights from feature importance reveal:

                - **Rotational speed [rpm]** stands out as the most influential predictor, suggesting that changes 
                in rotational speed are highly indicative of machine failures.
                - **Torque [Nm]** and **Tool wear [min]** have considerable importance, signifying that the torque 
                levels and tool condition play key roles in the prediction landscape.
                - **Air temperature [K]** carries moderate importance, while **Process temperature [K]**, surprisingly, 
                seems to have minimal influence in this model iteration. This might necessitate further investigation to 
                fathom its role or potential correlations.

                ### 8. Hyperparameters (n_estimators=145, max_depth=9, min_samples_split=0.3978, min_samples_leaf=0.1001, 
                # class_weight='balanced')

                The selected hyperparameters indicate:

                - The model uses a reasonably large ensemble with 145 trees, providing diverse perspectives in making predictions.
                - The depth of each tree is controlled at 9 levels to prevent overfitting.
                - The thresholds set for splits and leaf nodes, depicted by the non-default values, showcase a conscious effort 
                to tailor the model's learning dynamics.
                - The 'balanced' class weight choice underscores the model's design to counteract any class imbalances present 
                in the dataset.

                ### Conclusion

                The initial iteration with the Random Forest model demonstrates a commendable knack for identifying machine failures 
                but indicates a propensity for false alarms. Insights derived from feature importance offer valuable clues about the 
                operational metrics' significance. While this forms a strong foundation, there's a tangible avenue for enhancements, 
                particularly concerning precision and understanding specific feature relevance. Future endeavors might encompass more 
                intricate experiments, fine-tuning, or considering other modeling avenues for refining predictions.
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

                st.markdown("## RESULTS OF THE SECOND EXPERIMENT")

                # Display the resulting model metrics
                initial_result_filepath = "ml/binary_classification/results/random_forest_experiment2/metrics.csv"

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/random_forest_experiment2/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/random_forest_experiment2/feature_importances.csv"
                )

                st.markdown(
                    """
                ### Model Performance

                The Random Forest model, in its second experiment, showcased differential performance across metrics in 
                comparison to the initial iteration.

                - **Precision** shot up to 0.1061, implying a reduction in false positives, although there's still 
                potential for improvement.
                
                - **Recall** remained stable at 0.7377, highlighting the model's consistent capability to spot machine 
                failures.
                
                - With an F-beta score of 0.3368, the model showcases a better balance between precision and recall compared 
                to the first experiment.
                
                - **Cohen's Kappa** rose to 0.1397, representing enhanced agreement between the model's forecasts and the 
                observed outcomes.
                
                - **Balanced Accuracy** stands at 0.7711, showing that the model remains competent in handling both failure 
                and non-failure scenarios.
                
                - **Average Precision** has seen an uptick to 0.0863, endorsing the model's improved precision.

                ### Feature Importance

                A dive into the features' influence yields interesting findings:

                - **Strain (0.3636)**: As a representation of tool stress under varying torque conditions, strain underscores 
                situations wherein machinery might be operating under strenuous circumstances, potentially pointing to impending failure.

                - **Temperature Difference (0.3273)**: Highlighting the gradient between process and ambient temperatures, 
                its prominence suggests its nuanced role in foretelling machine health.

                - **Power (0.3091)**: Emanating from the interplay of torque and rotational speed, power's significance resonates 
                with the concept that mechanical power plays a pivotal role in determining machine functionality.

                ### Hyperparameters

                The following hyperparameters were configured for this specific experiment:

                - With a **max_depth of 25**, the ensemble of trees is allowed to explore deeper patterns, enabling intricate 
                decision-making processes.
                
                - The **min_samples_split** set at 0.6117 and **min_samples_leaf** at 0.1064 illustrates the model's intention to 
                make more comprehensive splits and leaves.
                
                - Employing the **'balanced' class weight** remains a testament to the model's approach in countering any class 
                imbalances effectively.

                ### Conclusion

                Contrasting with the initial experiment, the Random Forest's second iteration, driven by ingenious feature engineering, 
                exhibited enhancements in particular metrics, most notably precision, F-beta score, and Cohen's Kappa. This underscores 
                the importance of apt feature creation and refinement in leveraging domain-specific knowledge for model betterment. 
                There are still opportunities to further augment the model's performance. Future directions might encompass deeper 
                feature synthesis, meticulous hyperparameter tuning, or venturing into diverse modeling paradigms.

                """
                )

            with st.expander("INVESTIGATE THIRD EXPERIMENT"):
                st.markdown("### THIRD EXPERIMENT")

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

                st.markdown("## RESULTS OF THE THIRD EXPERIMENT")

                # Display the resulting model metrics
                initial_result_filepath = "ml/binary_classification/results/random_forest_experiment3/metrics.csv"

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/random_forest_experiment3/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/random_forest_experiment3/feature_importances.csv"
                )

                st.markdown(
                    """
                ### Model Performance

                The Random Forest model, in its third experiment, unveiled distinct performance dynamics when set against 
                the backdrop of previous iterations.

                - **Precision**: Experienced an uptick to 0.1203, indicating the model's refined prowess in correctly identifying 
                machine failures without excessive false alarms.
                - **Recall**: Steadfastly anchored at 0.7377, cementing the model's unswerving aptitude in pinpointing 
                potential machine failures.
                - **F-beta score**: At 0.3641, this metric testifies to the model's elevated equilibrium between precision 
                and recall, underscoring its heightened efficacy in forecasting machine glitches.
                - **Cohen's Kappa**: Progressed to 0.1630, mirroring the model's advanced congruence with real-world 
                outcomes against mere chance.
                - **Balanced Accuracy**: Settled at 0.7840, reiterating the model's superior balance in addressing 
                both operational and failure scenarios.
                - **Average Precision**: An ascension to 0.0968 further validates the model's reinforced precision.

                ### Feature Importance

                Probing into the model's feature reliance brings forth enlightening insights:

                - **Rotational speed [rpm] (0.3333)**: This raw metric's prominence underscores its indispensable 
                role in demystifying machine health intricacies.
                - **Strain (0.2778)**: Drawing from tool wear and torque dynamics, strain's heightened influence reaffirms 
                the theory that machinery under duress is a telltale sign of looming failure.
                - **Power (0.2222)**: As an offspring of the synergistic dance between torque and rotational speed, power's 
                pivotal role attests to the inherent relationship between mechanical vigor and machine health.
                - **Torque [Nm] (0.1111)**: Its continuous significance, even in its raw form, fortifies its foundational 
                stature in the manufacturing tableau.

                ### Hyperparameters

                The model's architectural blueprint for this iteration comprised:

                - **max_depth**: Set at 19, sculpting a roadmap for the trees to unravel intricate patterns without 
                convoluting the decision fabric.
                - **min_samples_split**: Calibrated at 0.5350.
                - **min_samples_leaf**: Positioned at 0.1001, sketching the model's blueprint to strike a delicate balance 
                between granularity and potential overfitting.
                - **Class weight**: The resolute adoption of the 'balanced' elucidates the model's unwavering commitment 
                to judiciously manage inherent class disparities.

                ### Conclusion

                When set against its predecessors, the third foray of the Random Forest model, championed by a blend of raw and engineered features, 
                signaled advancements in select metrics. This experiment, rich in its holistic data approach, magnifies the essence of a balanced 
                data tapestry in predictive analytics. While strides have been made, the horizon still holds promise. Forward leaps might envelop 
                deeper feature orchestration, granular hyperparameter alchemy, or the exploration of fresh modeling horizons.

                """
                )

            with st.expander("INVESTIGATE FORTH EXPERIMENT"):
                st.markdown("### FORTH EXPERIMENT")

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
                initial_result_filepath = "ml/binary_classification/results/random_forest_experiment4/metrics.csv"

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/random_forest_experiment4/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/random_forest_experiment4/feature_importances.csv"
                )

                st.markdown(
                    """
                ### Model Performance

                In the fourth experiment, the Random Forest model exhibited a distinct behavior when juxtaposed 
                against the preceding models, specifically hinging on Kernel PCA-processed data.

                - **Precision**: Decreased notably to 0.0608, pointing towards an increased number of false positives.
                
                - **Recall**: Experienced a slight increase to 0.7541, suggesting that despite the decrease in precision, 
                the model's capability to detect machine failures remained robust.
                
                - **F-beta score**: Positioned at 0.2298, the model showcases a lean towards recall, but the overall 
                harmony between precision and recall waned in comparison to the previous iterations.
                
                - **Cohen's Kappa**: Slipped to 0.0594, signaling a lesser degree of alignment between the model's 
                predictions and the actual outcomes compared to prior experiments.
                
                - **Balanced Accuracy**: Hovered around 0.6937, indicating a diminished equilibrium in the model's 
                competence to handle different scenarios uniformly.
                
                - **Average Precision**: Experienced a dip, settling at 0.0533, which aligns with the reduced 
                precision observed.

                ### Feature Importance

                Delving into the dimensionally-reduced feature landscape reveals:

                - **PC1 (0.6528)**: Dominating the feature spectrum, this principal component mirrors the major variance 
                within the dataset post Kernel PCA, encapsulating the significant patterns and relationships from the raw sensor data.
                
                - **PC2 (0.3472)**: Accounting for the next significant chunk of variance, it complements PC1, 
                ensuring most of the informational essence from the raw data is retained.

                ### Hyperparameters

                The architectural nuances of this iteration encompassed:

                - **max_depth**: Capped at 8, hinting at a relatively shallow exploration depth for the constituent trees, 
                perhaps aimed at avoiding overfitting given the reduced feature space.
                
                - **min_samples_split**: Tuned to 0.2279, shaping the tree's bifurcation strategy.
                
                - **min_samples_leaf**: Pinned at 0.2048, reflecting a bias towards preserving broader leaf nodes.
                
                - **Class weight**: Persisting with the 'balanced' stance underscores a continuous effort to equitably address class disparities.

                ### Conclusion

                The fourth endeavor with the Random Forest model, anchored on Kernel PCA preprocessing, evoked a mixed bag of results. 
                While there was a discernible decline in certain metrics, the experiment illuminated the nuances of leveraging 
                sophisticated dimensionality reduction techniques like Kernel PCA. Its unsupervised nature, coupled with the 
                overarching aim to distill complex non-linear relationships, offers intriguing prospects. While this iteration 
                didn't outshine its predecessors in performance, it broadens the analytical canvas, beckoning further exploration 
                into refined transformations or diverse modeling paradigms.

                """
                )

        elif selected_model == "XGBoost":
            st.markdown("# XGBOOST")

            st.markdown(
                """
                # INTRODUCTION TO XGBOOST

                Embarking on a new journey in the realm of gradient boosting models, our path is illuminated by the beacon 
                that is XGBoost. As we endeavor to harness the underlying patterns in our data, XGBoost emerges as a powerful 
                tool, pushing the boundaries of traditional gradient boosting techniques.

                XGBoost, short for eXtreme Gradient Boosting, is a refined concoction of gradient boosting algorithms tailored 
                for speed and performance. Recognized for its efficiency and predictive prowess, XGBoost has positioned itself 
                as the go-to algorithm for many Kaggle competition enthusiasts and industry professionals.

                So, why does our quest align with the direction of XGBoost? The reasons are multi-fold:

                - **Computational Efficiency**: XGBoost, living up to its "extreme" moniker, boasts superior speed and performance, 
                thanks to its ability to parallelize the tree-building process across all available cores. Such efficiency ensures 
                rapid model building and iterative refinement.
                
                - **Regularization Mastery**: Unlike traditional gradient boosting methods, XGBoost incorporates L1 (Lasso Regression) 
                and L2 (Ridge Regression) regularization. This integrated regularization prevents overfitting, ensuring that our model 
                remains versatile and adaptable.
                
                - **Flexibility Across Datasets**: XGBoost stands tall in its ability to handle a variety of data types, be it 
                tabular data, time series, or even textual data. Its prowess extends to both regression and classification tasks, 
                making it a versatile tool in the machine learning arsenal.
                
                - **Handling Imbalanced Datasets**: With its `scale_pos_weight` parameter, XGBoost gracefully addresses class imbalance, 
                which can often skew the predictive capabilities of a model. By adjusting this parameter, we can guide our model to be 
                more sensitive to the minority class, ensuring balanced predictions.
                
                - **Tree Pruning**: While standard gradient boosting uses a depth-first approach, XGBoost prunes trees using a depth-first 
                approach, ensuring the most optimal tree structure. This not only boosts the model's efficiency but also its predictive accuracy.

                With XGBoost as our navigator, we're set to embark on a journey rife with insights and discoveries. This advanced algorithm 
                promises not just predictive accuracy but also offers interpretability, ensuring that our voyage into the data's depths 
                is both enlightening and impactful. As we venture further, XGBoost will be our guiding light, revealing layers of knowledge 
                intricately woven into our data.

            """
            )

            st.markdown("## INITIAL RESULTS")

            # Display the resulting model metrics
            initial_result_filepath = (
                "ml/binary_classification/results/xgboost_experiment1/metrics.csv"
            )

            load_and_display_metrics(file_path=initial_result_filepath)

            # Display the confusion matrix
            load_and_plot_confusion_matrix(
                filename="ml/binary_classification/results/xgboost_experiment1/confusion_matrix.txt"
            )

            # Load the feature importances and display them
            load_and_plot_feature_importances(
                file_path="ml/binary_classification/results/xgboost_experiment1/feature_importances.csv"
            )

            st.markdown(
                """
            ### 1. Features

            For the inaugural experiment using XGBoost, our feature selection primarily hinged on potential 
            indicators of machine failure. This set encompasses both environmental variables, such as Air 
            temperature and Process temperature, and operational metrics like Torque, Tool wear, and Rotational speed.

            ### 2. Precision (0.6176) and Recall (0.6885)

            As we embark on this journey with XGBoost, the model exhibits a commendable recall, hinting at its 
            adeptness in identifying genuine failures. Meanwhile, the precision, though not perfect, reflects a 
            promising start. A more refined precision in future iterations would reduce unwarranted maintenance 
            or operational halts, thereby streamlining efficiency.

            ### 3. F-beta Score (0.6731)

            For our first foray with XGBoost, achieving such a balance between precision and recall, as indicated 
            by the F-beta score, is quite an accomplishment. Refinements in future models might further hone this 
            balance, especially aligned with specific operational goals.

            ### 4. Cohen's Kappa (0.6396)

            Starting with a strong footing, the Cohen's Kappa score for this XGBoost model denotes a substantial 
            agreement between predicted and actual values. It's encouraging to kick off with a model that significantly 
            outperforms random chance.

            ### 5. Balanced Accuracy (0.8376)

            Our maiden XGBoost model's balanced accuracy showcases its robustness across both positive and negative 
            classes. This score, in tandem with the observed precision, underscores the model's holistic grasp of the dataset.

            ### 6. Average Precision (0.4348)

            The model's average precision, right out of the gate, indicates its potential in managing false positives 
            and consistently performing across varied decision thresholds. This adaptability is essential for dynamic 
            business scenarios.

            ### 7. Feature Importance

            An initial glance at the model's drivers reveals:
            - **Torque [Nm]** as the most influential, affirming its pivotal role.
            - **Air temperature [K]** and **Tool wear [min]** are also of paramount importance.
            - **Rotational speed [rpm]** and **Process temperature [K]**, though on the lower side, should not be overlooked. 
            Understanding their intricate relationships with machine failure might be instrumental in future iterations.

            ### 8. Hyperparameters

            For this XGBoost prototype, the chosen hyperparameters are:
            - **learning_rate**: 0.0924, guiding the model's adaptiveness at each iteration.
            - **max_depth**: 19, reflecting the model's capacity for complexity.
            - **subsample**: 0.8744 and **colsample_bytree**: 0.8397, both focusing on sample diversity and curbing overfitting 
            tendencies.

            ### Conclusion

            Our initial exploration with XGBoost has laid a strong foundation. The insights gleaned and the performance metrics 
            are encouraging. As with any maiden endeavor, there exists potential for growth, be it through more nuanced feature 
            explorations, refined engineering strategies, or further hyperparameter tuning.
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

                st.markdown("## RESULTS OF THE SECOND EXPERIMENT")

                # Display the resulting model metrics
                initial_result_filepath = (
                    "ml/binary_classification/results/xgboost_experiment2/metrics.csv"
                )

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/xgboost_experiment2/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/xgboost_experiment2/feature_importances.csv"
                )

                st.markdown(
                    """
                ---

                **Model Performance**

                Compared to the first experiment, this iteration displays a drop across all metrics. Precision 
                has fallen to 0.4752, hinting at a surge in false positives. There's a decrease in recall to 0.7869, 
                signaling the model's subdued capability to pinpoint true machine failures. The F-beta score reflects 
                this trend with a score of 0.6957. Meanwhile, Cohen's Kappa score, an indicator of prediction accuracy, 
                slips to 0.5765. The balanced accuracy, showcasing the model's consistent performance across classes, 
                has also reduced, with average precision dropping to 0.3805, further emphasizing the rise in false positives.

                ---

                **Feature Importance**

                Post-analysis, the second experiment puts a spotlight on the strain, derived from tool wear and torque, 
                emerging as the most influential feature, underscoring those precarious moments leading to machine failures. 
                Conversely, temperature difference seems to inversely influence predictions, suggesting a potential disparity 
                in understanding this relationship. Lastly, the 'power' feature, a culmination of torque and rotational speed, 
                reinforces the hypothesis that heightened power levels correlate with increased machine failure risks.

                ---

                **Hyperparameters**

                For this run, hyperparameters were set with a learning rate of 0.0882, emphasizing gradual model adaptability. 
                The model's depth was increased substantially to 75 layers, while subsample and colsample_bytree values were 
                fine-tuned to 0.8286 and 0.8486 respectively, ensuring diversified sample utilization and curbing potential overfitting.

                ---

                **Conclusion**

                This experiment's shift in strategy, marked by advanced feature engineering, failed to enhance the model's predictive 
                prowess over the initial experiment. The universal dip in performance metrics suggests a need for recalibration. 
                Future undertakings might require a more holistic synthesis of sensor data, incorporation of newer features, or 
                perhaps a pivot to alternative modeling techniques that better handle the introduced feature complexities.

                ---
                """
                )

            with st.expander("INVESTIGATE THIRD EXPERIMENT"):
                st.markdown("### THIRD EXPERIMENT")

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

                st.markdown("## RESULTS OF THE THIRD EXPERIMENT")

                # Display the resulting model metrics
                initial_result_filepath = (
                    "ml/binary_classification/results/xgboost_experiment3/metrics.csv"
                )

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/xgboost_experiment3/confusion_matrix.txt"
                )

                # Load the feature importances and display them
                load_and_plot_feature_importances(
                    file_path="ml/binary_classification/results/xgboost_experiment3/feature_importances.csv"
                )

                st.markdown(
                    """
                ---

                **Model Performance**

                The third experiment witnessed a noticeable uptick across most metrics compared to the first two iterations. 
                Precision, recall, and the F-beta score all converged at 0.8197, indicating a balanced model performance in 
                terms of true positives and negatives. The Cohen's Kappa score, reflecting the model's accuracy in prediction, 
                climbed to 0.8140. Balanced accuracy reached an impressive 0.9070, highlighting the model's refined capability 
                to consistently predict across classes. Meanwhile, average precision settled at 0.6774, suggesting a better 
                balance between precision and recall.

                ---

                **Feature Importance**

                The third experiment's results revealed the strain feature, a derivative of tool wear and torque, as the most influential, 
                albeit with reduced dominance at 28.31521. Power and Rotational speed followed, implying their persistent significance 
                in predicting machine failures. Interestingly, the raw sensor data like Torque, Air temperature, Tool wear, and Process temperature, 
                which were reincorporated, occupied varied ranks in the importance hierarchy, with Air temperature and Process temperature 
                being the least impactful among the featured metrics.

                ---

                **Hyperparameters**

                For this iteration, the model's learning rate was set at a conservative 0.0206, favoring a cautious adaptability pace. 
                The model's depth was maintained near the previous setting at 74 layers, while subsample and colsample_bytree values 
                were adjusted to 0.5691 and 0.9805 respectively. This configuration suggests a strategic sampling of the data and a 
                near-complete feature set inclusion, offering the model a comprehensive perspective.

                ---

                **Conclusion**

                In the third experiment, the synthesis of raw sensor data with engineered features seems to strike a harmonious chord. 
                The marked improvement in performance indicators validates the hypothesis that while engineered features capture 
                intricate relationships, raw data maintains intrinsic, indispensable information. This blended approach has proven to 
                be a robust strategy in the quest for optimal machine failure prediction, but continuous iteration and exploration 
                remain key to further model enhancements.

                ---
                """
                )

            with st.expander("INVESTIGATE FORTH EXPERIMENT"):
                st.markdown("### FORTH EXPERIMENT")

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
                initial_result_filepath = (
                    "ml/binary_classification/results/xgboost_experiment4/metrics.csv"
                )

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/xgboost_experiment4/confusion_matrix.txt"
                )

                # # Load the feature importances and display them
                # load_and_plot_feature_importances(
                #     file_path="ml/binary_classification/results/random_forest_experiment4/feature_importances.csv"
                # )

                st.markdown(
                    """
                ---

                **Model Performance**

                The fourth experiment showed a substantial decline in model performance across all metrics compared to 
                previous iterations. Precision took a significant hit, dipping to a mere 0.0907, indicating a marked 
                rise in false positives. Recall, measuring the model's ability to correctly identify machine failures, 
                reduced to 0.6066. The F-beta score, a balance between precision and recall, also saw a decline, registering 
                at 0.2837. Cohen's Kappa score, gauging the level of agreement between the model's predictions and actual 
                outcomes, fell to a paltry 0.1106. Balanced accuracy, which measures the model's performance across classes, 
                did remain relatively decent at 0.7076, but the average precision plummeted to 0.0670, underlining the model's 
                struggle with an increased rate of false positives.

                ---

                **Feature Importance**

                For this iteration, feature importance couldn't be analyzed in the conventional sense due to the 
                application of Kernel PCA, which transforms the original feature space. Thus, direct interpretation 
                of feature significance based on their original definitions becomes challenging. Instead, the two features 
                extracted using Kernel PCA would represent combinations of the original sensor readings, but their specific 
                relation to the original metrics remains abstract.

                ---
                
                **Conclusion**

                The fourth experiment's use of Kernel PCA on raw sensor readings introduced a new dimensionality reduction 
                technique to the modeling process. While the intent was to harness Kernel PCA's power to simplify complex 
                relationships, the results indicate otherwise. The model's deteriorated performance suggests that the transformation 
                may have lost vital information, or perhaps the xgboost model isn't well-suited to work with the 
                PCA-transformed features. Future endeavors might need to re-examine the approach, perhaps by combining Kernel 
                PCA with a different modeling algorithm or by adjusting the number of components extracted.

                ---
                """
                )

    with try_yourself_tab:
        st.title("SELECT A MACHINE LEARNING MODEL")
        inference_model = st.selectbox(
            label="inference_model",
            options=[
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "XGBoost",
            ],
            label_visibility="hidden",
        )

        # Define the CSS for blinking effect
        st.markdown(
            """
        <style>
            @keyframes blink {
            0% { opacity: 0.0; }
            50% { opacity: 1.0; }
            100% { opacity: 0.0; }
            }

            .blinking-red {
            animation: blink 2s infinite; 
            background-color: red;
            color: white;
            padding: 20px; 
            display: block;       
            width: 100%;          
            border-radius: 15px;  
            font-size: 36px;      
            text-align: center;   
            }

            .blinking-green {
            animation: blink 2s infinite; 
            background-color: green;
            color: white;
            padding: 20px; 
            display: block;       
            width: 100%;          
            border-radius: 15px;  
            font-size: 36px;      
            text-align: center;   
            }

        </style>
        """,
            unsafe_allow_html=True,
        )

        with st.form("binary_classification_form"):
            # Get input from the user
            air_temperature = st.number_input("Air Temperature")
            process_temperature = st.number_input("Process Temperature")
            rotational_speed = st.number_input("Rotational Speed")
            torque = st.number_input("Torque")
            tool_wear = st.number_input("Tool Wear")

            # Create a form submit button
            submitted = st.form_submit_button("Predict")

        # If user presses the submit button
        # TODO Change the predicted class logic
        if submitted:
            # We need to extract other features as well
            power = rotational_speed * torque
            strain = tool_wear * torque
            temp_diff = process_temperature * air_temperature

            # Initiate the binary classification model
            binary_classification_model = None

            if inference_model == "Logistic Regression":
                # Load the logistic regression model
                binary_classification_model = joblib.load(
                    "ml/binary_classification/models/logistic_regression/3.joblib"
                )

                predicted_class = binary_classification_model.predict(
                    [
                        [
                            air_temperature,
                            process_temperature,
                            rotational_speed,
                            torque,
                            tool_wear,
                            power,
                            strain,
                            temp_diff,
                        ]
                    ]
                )

            elif inference_model == "Decision Tree":
                # Load the decision tree model
                binary_classification_model = joblib.load(
                    "ml/binary_classification/models/decision_tree/3.joblib"
                )

                predicted_class = binary_classification_model.predict(
                    [
                        [
                            air_temperature,
                            process_temperature,
                            rotational_speed,
                            torque,
                            tool_wear,
                            power,
                            strain,
                            temp_diff,
                        ]
                    ]
                )

            elif inference_model == "Random Forest":
                # Load the decision tree model
                binary_classification_model = joblib.load(
                    "ml/binary_classification/models/random_forest/3.joblib"
                )

                predicted_class = binary_classification_model.predict(
                    [
                        [
                            air_temperature,
                            process_temperature,
                            rotational_speed,
                            torque,
                            tool_wear,
                            power,
                            strain,
                            temp_diff,
                        ]
                    ]
                )

            elif inference_model == "XGBoost":
                # Load the decision tree model
                binary_classification_model = joblib.load(
                    "ml/binary_classification/models/xgboost/3.joblib"
                )

                predicted_class = binary_classification_model.predict(
                    [
                        [
                            air_temperature,
                            process_temperature,
                            rotational_speed,
                            torque,
                            tool_wear,
                            power,
                            strain,
                            temp_diff,
                        ]
                    ]
                )

            # Display the results in an appealing way
            if predicted_class[0] == 0:
                st.markdown(
                    f'<div class="blinking-green">No Machine Failure</div>',
                    unsafe_allow_html=True,
                )
            elif predicted_class[0] == 1:
                st.markdown(
                    f'<div class="blinking-red">Machine Failure!</div>',
                    unsafe_allow_html=True,
                )


main()
