import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

import plotly.graph_objects as go
import plotly.figure_factory as ff

from streamlit_card import card
from streamlit_image_select import image_select

import base64
import joblib

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

            This is just the starting point. Our goal is to progressively enhance the 
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
            different decision thresholds, which could be a concern if there's a need to adjust the decision threshold 
            based on changing business needs.

            ### 7. Feature Importance
            The analysis of feature importance provides insight into the features driving the model's predictions. 
            Torque, Rotational speed, and Air temperature are the most influential, whereas Process temperature 
            has a negative influence. This suggests that as the Process temperature increases, the likelihood 
            of machine failure, as predicted by the model, decreases. The decrease in Process temperature and
            the increase in Air temperature seems to increase the machine failure as found in the exploratory
            data analysis processes.

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
                st.markdown("## SECOND EXPERIMENT")

                st.markdown(
                    """
                ### Experimental Setup: Feature Engineering

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

                st.markdown("### RESULTS OF THE SECOND EXPERIMENT")

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
                st.write("## THIRD EXPERIMENT")

                st.markdown(
                    """
                ### Experimental Setup: Feature Engineering

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

                st.markdown("### RESULTS OF THE THIRD EXPERIMENT")

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
                ### Model Performance

                The third experiment, incorporating both raw data and the engineered features, showcased mixed results. 
                The precision rose to 0.15 from 0.13 observed in the initial experiment, signaling a decrease in false positives. 
                However, recall slightly fell to 0.82 from 0.85, indicating that the model was less effective in identifying all actual failures.
                The F-beta score, a balance of precision and recall, climbed to 0.44 from 0.40, suggesting a 
                net positive influence of integrating engineered features with raw data. Additionally, Cohen's Kappa score rose 
                from 0.18 to 0.22, portraying better model agreement than mere random guessing, but also hinting at potential improvements. 
                The balanced accuracy showed a minor rise to 0.84 from 0.83, which indicates a marginal enhancement in the overall 
                accuracy across both categories. Meanwhile, average precision also climbed slightly from 0.11 to 0.13, hinting at 
                the model's marginally better performance across varying thresholds.

                ### Feature Importance

                In this experiment, the most dominant feature turned out to be 'Torque', trailed by 'power' and 'Tool wear'. 
                Notably, 'power', an engineered feature, exhibited a negative influence. This indicates that as power escalates, 
                the model predicts a drop in the odds of machine failure — an unexpected relationship demanding further examination.

                ### Hyperparameters

                Distinct from the first experiment, this third trial's hyperparameters underwent significant adjustments. There was 
                a marked rise in the C parameter, alluding to a simpler model, coupled with a solver transition to 'liblinear'.

                ### Conclusion

                In summary, the third test, which combined raw data with new features, showed slight improvements compared to the first 
                test. However, there's still work to do, especially in reducing wrong predictions, understanding odd results from some 
                features, and tweaking settings for the best outcome.

                """
                )

            with st.expander("INVESTIGATE FORTH EXPERIMENT"):
                st.write("## FORTH EXPERIMENT")

                st.markdown(
                    """
                ### Experimental Setup: Feature Engineering

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

                st.markdown("### RESULTS OF THE FORTH EXPERIMENT")

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
                ### Model Performance

                The fourth experiment, which employed Kernel PCA to transform raw sensor readings, revealed a decline in 
                model performance across all evaluation metrics compared to the initial experiment. Precision and recall 
                saw significant drops, hinting that the Kernel PCA transformation might not have effectively retained the 
                critical details from the raw sensor readings for precise predictions. The F-beta score also dipped, showcasing 
                a decline in the overall performance of the model. Additionally, the Cohen's Kappa score decreased, which 
                suggests a weaker correlation between the model's predictions and the actual outcomes. The balanced accuracy 
                and average precision scores, representing the model's overall capability across both classes, both suffered declines.

                ### Feature Importance

                The analysis of the features unearthed that both principal components (PC1 and PC2) extracted via Kernel PCA had a 
                negative impact on the model's predictions. However, interpreting these importances is challenging, given PCA's 
                nature and its detachment from the original sensor readings.

                ### Hyperparameters

                In terms of model settings, there was a noticeable increase in the C parameter in this experiment, implying a shift 
                towards a more complex model with lesser regularization. This adjustment might have been an endeavor to grasp more 
                intricacies in the transformed data, but it didn't translate into better performance.

                ### Conclusion

                To wrap things up, the fourth experiment, which centered around using Kernel PCA, didn't yield better outcomes compared 
                to the initial trial. The all-round decline in evaluation metrics and the complexities in understanding the PCA-transformed 
                features suggest that this strategy might not be the ideal fit for the given challenge. Future experiments might delve into 
                alternative feature extraction techniques or even different model types to seek improvements.

                """
                )

        elif selected_model == "Decision Tree":
            st.markdown("# DECISION TREES")

            st.markdown(
                """
            ## INITIAL ANALYSIS WITH DECISION TREES

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

            This step is just our starting point. We want to keep improving our models, using advanced techniques. But first, let's see 
            what our data can tell us with the Decision Tree!
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

                #### **2. Precision (0.16) and Recall (0.93)**

                The decision tree model demonstrates an exceptional recall, meaning it has a keen ability to correctly 
                flag actual machine failures. Conversely, its precision is on the lower side, suggesting that there are 
                scenarios where the model predicts a machine failure when, in reality, none exists. Such false positives 
                could lead to unwarranted intervention, thus affecting operational flow and efficiency. A balance between 
                precision and recall might be sought after, especially depending on the costs associated with incorrect 
                predictions.

                #### **3. F-beta Score (0.48)**

                Given the disparity between precision and recall, the F-beta score for the model is not particularly high. 
                This reveals a model that currently leans more towards recall. Depending on the operational requirements, 
                we might consider tuning the model to enhance this score, especially if one metric is more valuable than 
                the other in a specific business context.

                #### **4. Cohen's Kappa (0.24)**

                The obtained Cohen's Kappa score showcases that the agreement between the model predictions and the actual 
                occurrences isn't very robust. This suggests there's considerable room for model refinement. A Cohen's Kappa 
                score at this level implies the model performs better than mere chance but has significant potential for improvement.

                #### **5. Balanced Accuracy (0.89)**

                The balanced accuracy portrays a model that performs well across both the positive and negative classifications. 
                Nevertheless, the precision score suggests there might be a bias towards predicting one class over the other, 
                which requires more in-depth exploration to understand any underlying model bias or data imbalance.

                #### **6. Average Precision (0.15)**

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
                st.markdown("## SECOND EXPERIMENT")

                st.markdown(
                    """
                ### Experimental Setup: Feature Engineering

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

                st.markdown("### RESULTS OF THE SECOND EXPERIMENT")

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
                ## Model Performance

                The second experiment, which utilized the Decision Tree model, showed mixed results 
                in performance when benchmarked against the initial experiment. 

                - **Precision** experienced an uptick, indicating fewer false positives.
                
                - **Recall** faced a slight decrease, suggesting the model might miss out on some actual machine failures.

                - **F-beta score** made progress, reflecting a better balance between precision and recall.

                - **Cohen's Kappa** improved, highlighting a stronger alignment between the model's predictions and the actual results.

                - **Balanced accuracy**, despite a minor dip, showcases competency across both positive and negative classes.

                - **Average precision** marked growth, endorsing the model's improved precision capabilities.

                ## Feature Importance

                A closer look at the features reveals:

                - **Power (0.61)**: Being a blend of torque and speed, this emerged as a top influencer, 
                emphasizing the role of mechanical dynamics in assessing machine health.
                
                - **Temperature Difference (0.20)**: Its impact indicates it plays a nuanced role in model predictions.
                
                - **Strain (0.18)**: As a reflection of tool stress, it highlights situations where machines 
                might be under excessive strain, hinting at possible breakdowns.

                ## Hyperparameters

                For this experiment's settings:

                - **max_depth** of 46 allows the model to discern more intricate patterns, enabling more detailed decision-making.

                - **min_samples_split** and **min_samples_leaf**, with their low values, promote finer divisions in the decision tree.

                - The sustained use of the **'balanced' class weight** underscores the ongoing effort to effectively address class imbalances.

                ## Conclusion

                When compared to the first trial, the second test, enriched by feature engineering, displayed clear enhancements in 
                some metrics, such as precision, F-beta score, and Cohen's Kappa. This highlights the impact of well-thought-out 
                feature integration, which infuses specialized insights into the model, boosting its prediction accuracy. However, 
                there's still room for model improvement, which might involve deeper feature exploration, refining hyperparameters, 
                or trying different modeling approaches.

                """
                )

            with st.expander("INVESTIGATE THIRD EXPERIMENT"):
                st.markdown("## THIRD EXPERIMENT")

                st.markdown(
                    """
                ### Experimental Setup: Feature Engineering

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

                st.markdown("### RESULTS OF THE THIRD EXPERIMENT")

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

                - **Precision** skyrocketed to 0.82, which intimates a commendable accuracy in predicting machine 
                failures, significantly minimizing false positives.

                - **Recall**, holding steady at 0.74, conveys that the model still successfully discerns a considerable 
                majority of actual machine failures.

                - The **F-beta score** ascended to 0.75, representing a balanced performance in both precision and recall metrics.

                - **Cohen's Kappa**, now at a laudable 0.77, indicates a strong agreement between the model's predictions 
                and the actual outcomes, highlighting its superiority over random chance.

                - The **Balanced accuracy** remains impressive at 0.87, underscoring the model's robustness across both classes.

                - **Average precision** also exhibited a leap, settling at 0.61, underscoring the model's reinforced precision 
                across various thresholds.

                ### **Feature Importance**

                Diving deeper into the terrain of feature influence, illuminating insights come to the fore:

                - **Rotational speed (0.37)** emerged as the leading influencer, emphasizing the pivotal role of mechanical 
                speed in assessing machine health.

                - **Strain (0.33)** and **Power (0.20)** continue to hold significant sway, corroborating their roles as 
                critical synthesized indicators of machine status.

                - **Temperature Difference (0.07)**, although marginally influential, still factors into the model's 
                decision-making process.

                - Interestingly, raw sensory readings like **Tool wear (0.01)**, **Air temperature (0.01)**, **Process temperature**, 
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
                st.markdown("## FORTH EXPERIMENT")

                st.markdown(
                    """
                ### Experimental Setup: Feature Engineering
                
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

                st.markdown("### RESULTS OF THE FORTH EXPERIMENT")

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

                - **Precision** took a nosedive to 0.06, revealing an accentuated rate of false positives, which may 
                cast doubts on the model's trustworthiness.

                - **Recall** stands at 0.59, indicating the model's compromised capacity to identify over half of the 
                true machine failures.

                - The **F-beta score** sharply declined to 0.21, mirroring the dip in both precision and recall.

                - **Cohen's Kappa** settled at a mere 0.06, denoting a borderline trivial agreement between the model's 
                predictions and actual results.

                - **Balanced Accuracy** still hovers at 0.65, showcasing a moderate performance across the class spectrum.

                - **Average Precision** spiraled down to 0.05, mirroring the model's weakened precision throughout various thresholds.

                ### **Feature Importance**

                Even though 1 principle component is much more higher contributer than the other, it is hard to interpret 
                since the ontributions of raw data to principle components are not explicit.

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
                ## **INITIAL ANALYSIS WITH RANDOM FORESTS**

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

                ### 2. Precision (0.10) and Recall (0.75)

                With a strong recall, the Random Forest model displays a competent ability to detect true machine failures. 
                However, its lower precision means there might be cases where the model erroneously flags a machine failure, 
                leading to possible unwarranted interventions. Balancing precision and recall will be essential to ensure 
                smooth operations and avoid unnecessary disruptions.

                ### 3. F-beta Score (0.33)

                The F-beta score indicates a model that is skewed more towards recall. The moderate score suggests that there's 
                potential for improvement, and refining the model to improve this metric could be beneficial, especially if 
                business contexts require a specific trade-off between precision and recall.

                ### 4. Cohen's Kappa (0.13)

                The derived Cohen's Kappa metric signifies that the alignment between model predictions and real outcomes 
                could be improved. A score at this level indicates the model is performing better than random chance 
                but showcases substantial room for enhancement.

                ### 5. Balanced Accuracy (0.77)

                The balanced accuracy, while respectable, suggests the model's well-rounded performance in predicting both 
                positive and negative classes. Still, with the precision in context, further investigation might be needed 
                to understand any underlying biases or tendencies.

                ### 6. Average Precision (0.08)

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
                ### class_weight='balanced')

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
                st.markdown("## SECOND EXPERIMENT")

                st.markdown(
                    """
                ### Experimental Setup: Feature Engineering

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

                st.markdown("### RESULTS OF THE SECOND EXPERIMENT")

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

                - **Precision** shot up to 0.09, implying a reduction in false positives, although there's still 
                potential for improvement.
                
                - **Recall** remained stable at 0.74, highlighting the model's consistent capability to spot machine 
                failures.
                
                - With an F-beta score of 0.30, the model showcases a better balance between precision and recall compared 
                to the first experiment.
                
                - **Cohen's Kappa** rose to 0.13, representing enhanced agreement between the model's forecasts and the 
                observed outcomes.
                
                - **Balanced Accuracy** stands at 0.77, showing that the model remains competent in handling both failure 
                and non-failure scenarios.
                
                - **Average Precision** has seen an uptick to 0.08, endorsing the model's improved precision.

                ### Feature Importance

                A dive into the features' influence yields interesting findings:

                - **Strain (0.36)**: As a representation of tool stress under varying torque conditions, strain underscores 
                situations wherein machinery might be operating under strenuous circumstances, potentially pointing to impending failure.

                - **Temperature Difference (0.32)**: Highlighting the gradient between process and ambient temperatures, 
                its prominence suggests its nuanced role in foretelling machine health.

                - **Power (0.30)**: Emanating from the interplay of torque and rotational speed, power's significance resonates 
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

                Compared to the initial experiment, the Random Forest's second iteration, with additional feature engineering, 
                showed improvements in certain metrics, including precision, F-beta score, and Cohen's Kappa. This suggests the 
                role of effective feature development in utilizing domain-specific knowledge for model improvement. There remains 
                potential for enhancing the model's performance. Future steps might involve further feature exploration, hyperparameter 
                adjustments, or exploring other modeling approaches.

                """
                )

            with st.expander("INVESTIGATE THIRD EXPERIMENT"):
                st.markdown("## THIRD EXPERIMENT")

                st.markdown(
                    """
                    ### Experimental Setup: Feature Engineering

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

                st.markdown("### RESULTS OF THE THIRD EXPERIMENT")

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

                - **Precision**: Experienced an uptick to 0.12, indicating the model's refined prowess in correctly identifying 
                machine failures without excessive false alarms.
                - **Recall**: Steadfastly anchored at 0.73, cementing the model's unswerving aptitude in pinpointing 
                potential machine failures.
                - **F-beta score**: At 0.36, this metric testifies to the model's elevated equilibrium between precision 
                and recall, underscoring its heightened efficacy in forecasting machine glitches.
                - **Cohen's Kappa**: Progressed to 0.16, mirroring the model's advanced congruence with real-world 
                outcomes against mere chance.
                - **Balanced Accuracy**: Settled at 0.78, reiterating the model's superior balance in addressing 
                both operational and failure scenarios.
                - **Average Precision**: An ascension to 0.09 further validates the model's reinforced precision.

                ### Feature Importance

                Probing into the model's feature reliance brings forth enlightening insights:

                - **Rotational speed [rpm] (0.33)**: This raw metric's prominence underscores its indispensable 
                role in demystifying machine health intricacies.
                - **Strain (0.27)**: Drawing from tool wear and torque dynamics, strain's heightened influence reaffirms 
                the theory that machinery under duress is a telltale sign of looming failure.
                - **Power (0.22)**: As an offspring of the synergistic dance between torque and rotational speed, power's 
                pivotal role attests to the inherent relationship between mechanical vigor and machine health.
                - **Torque [Nm] (0.11)**: Its continuous significance, even in its raw form, fortifies its foundational 
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
                st.markdown("## FOURTH EXPERIMENT")

                st.markdown(
                    """
                ### Fourth Experiment: Kernel PCA on Raw Sensor Readings

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

                st.markdown("### RESULTS OF THE FORTH EXPERIMENT")

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

                - **Precision**: Decreased notably to 0.06, pointing towards an increased number of false positives.
                
                - **Recall**: Experienced a slight increase to 0.75, suggesting that despite the decrease in precision, 
                the model's capability to detect machine failures remained robust.
                
                - **F-beta score**: Positioned at 0.23, the model showcases a lean towards recall, but the overall 
                harmony between precision and recall waned in comparison to the previous iterations.
                
                - **Cohen's Kappa**: Slipped to 0.06, signaling a lesser degree of alignment between the model's 
                predictions and the actual outcomes compared to prior experiments.
                
                - **Balanced Accuracy**: Hovered around 0.69, indicating a diminished equilibrium in the model's 
                competence to handle different scenarios uniformly.
                
                - **Average Precision**: Experienced a dip, settling at 0.05, which aligns with the reduced 
                precision observed.

                ### Feature Importance

                It is hard to estimate the which raw features are important with the principal components.

                ### Hyperparameters

                The architectural nuances of this iteration encompassed:

                - **max_depth**: Capped at 8, hinting at a relatively shallow exploration depth for the constituent trees, 
                perhaps aimed at avoiding overfitting given the reduced feature space.
                
                - **min_samples_split**: Tuned to 0.2279, shaping the tree's bifurcation strategy.
                
                - **min_samples_leaf**: Pinned at 0.2048, reflecting a bias towards preserving broader leaf nodes.
                
                - **Class weight**: Persisting with the 'balanced' stance underscores a continuous effort to equitably address class disparities.

                ### Conclusion

                The fourth try with the Random Forest model, using Kernel PCA preprocessing, gave mixed results. Some metrics dropped, 
                but using advanced methods like Kernel PCA gave us new insights. Even though this try didn't do better than the earlier ones, 
                it shows we can look more into different data methods or other model types.

                """
                )

        elif selected_model == "XGBoost":
            st.markdown("# XGBOOST")

            st.markdown(
                """
                # INITIAL ANALYSIS WITH XGBOOST

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

            ### 2. Precision (0.62) and Recall (0.69)

            As we embark on this journey with XGBoost, the model exhibits a commendable recall, hinting at its 
            adeptness in identifying genuine failures. Meanwhile, the precision, though not perfect, reflects a 
            promising start. A more refined precision in future iterations would reduce unwarranted maintenance 
            or operational halts, thereby streamlining efficiency.

            ### 3. F-beta Score (0.67)

            For our first foray with XGBoost, achieving such a balance between precision and recall, as indicated 
            by the F-beta score, is quite an accomplishment. Refinements in future models might further hone this 
            balance, especially aligned with specific operational goals.

            ### 4. Cohen's Kappa (0.64)

            Starting with a strong footing, the Cohen's Kappa score for this XGBoost model denotes a substantial 
            agreement between predicted and actual values. It's encouraging to kick off with a model that significantly 
            outperforms random chance.

            ### 5. Balanced Accuracy (0.84)

            Our maiden XGBoost model's balanced accuracy showcases its robustness across both positive and negative 
            classes. This score, in tandem with the observed precision, underscores the model's holistic grasp of the dataset.

            ### 6. Average Precision (0.43)

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
                st.markdown("## SECOND EXPERIMENT")

                st.markdown(
                    """
                ### Experimental Setup: Feature Engineering

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

                st.markdown("### RESULTS OF THE SECOND EXPERIMENT")

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
                
                **Model Performance**

                Compared to the first experiment, this iteration displays a drop across all metrics. Precision 
                has fallen to 0.48, hinting at a surge in false positives. There's a decrease in recall to 0.79, 
                signaling the model's subdued capability to pinpoint true machine failures. The F-beta score reflects 
                this trend with a score of 0.70. Meanwhile, Cohen's Kappa score, an indicator of prediction accuracy, 
                slips to 0.58. The balanced accuracy, showcasing the model's consistent performance across classes, 
                has also reduced, with average precision dropping to 0.38, further emphasizing the rise in false positives.

                **Feature Importance**

                Post-analysis, the second experiment puts a spotlight on the strain, derived from tool wear and torque, 
                emerging as the most influential feature, underscoring those precarious moments leading to machine failures. 
                Conversely, temperature difference seems to inversely influence predictions, suggesting a potential disparity 
                in understanding this relationship. Lastly, the 'power' feature, a culmination of torque and rotational speed, 
                reinforces the hypothesis that heightened power levels correlate with increased machine failure risks.

                **Hyperparameters**

                For this run, hyperparameters were set with a learning rate of 0.0882, emphasizing gradual model adaptability. 
                The model's depth was increased substantially to 75 layers, while subsample and colsample_bytree values were 
                fine-tuned to 0.8286 and 0.8486 respectively, ensuring diversified sample utilization and curbing potential overfitting.

                **Conclusion**

                This experiment's shift in strategy, marked by advanced feature engineering, failed to enhance the model's predictive 
                prowess over the initial experiment. The universal dip in performance metrics suggests a need for recalibration. 
                Future undertakings might require a more holistic synthesis of sensor data, incorporation of newer features, or 
                perhaps a pivot to alternative modeling techniques that better handle the introduced feature complexities.

                """
                )

            with st.expander("INVESTIGATE THIRD EXPERIMENT"):
                st.markdown("## THIRD EXPERIMENT")

                st.markdown(
                    """
                ### Experimental Setup: Feature Engineering

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

                st.markdown("### RESULTS OF THE THIRD EXPERIMENT")

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
                **Model Performance**

                The third experiment witnessed a noticeable uptick across most metrics compared to the first two iterations. 
                Precision, recall, and the F-beta score all converged at 0.82, indicating a balanced model performance in 
                terms of true positives and negatives. The Cohen's Kappa score, reflecting the model's accuracy in prediction, 
                climbed to 0.8140. Balanced accuracy reached an impressive 0.91, highlighting the model's refined capability 
                to consistently predict across classes. Meanwhile, average precision settled at 0.68, suggesting a better 
                balance between precision and recall.

                **Feature Importance**

                The third experiment's results revealed the strain feature, a derivative of tool wear and torque, as the most influential, 
                albeit with reduced dominance at 28.31521. Power and Rotational speed followed, implying their persistent significance 
                in predicting machine failures. Interestingly, the raw sensor data like Torque, Air temperature, Tool wear, and Process temperature, 
                which were reincorporated, occupied varied ranks in the importance hierarchy, with Air temperature and Process temperature 
                being the least impactful among the featured metrics.

                **Hyperparameters**

                For this iteration, the model's learning rate was set at a conservative 0.0206, favoring a cautious adaptability pace. 
                The model's depth was maintained near the previous setting at 74 layers, while subsample and colsample_bytree values 
                were adjusted to 0.5691 and 0.9805 respectively. This configuration suggests a strategic sampling of the data and a 
                near-complete feature set inclusion, offering the model a comprehensive perspective.

                **Conclusion**

                In the third experiment, the synthesis of raw sensor data with engineered features seems to strike a harmonious chord. 
                The marked improvement in performance indicators validates the hypothesis that while engineered features capture 
                intricate relationships, raw data maintains intrinsic, indispensable information. This blended approach has proven to 
                be a robust strategy in the quest for optimal machine failure prediction, but continuous iteration and exploration 
                remain key to further model enhancements.

                """
                )

            with st.expander("INVESTIGATE FORTH EXPERIMENT"):
                st.markdown("## FOURTH EXPERIMENT")

                st.markdown(
                    """
                ### Fourth Experiment: Kernel PCA on Raw Sensor Readings

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

                st.markdown("### RESULTS OF THE FORTH EXPERIMENT")

                # Display the resulting model metrics
                initial_result_filepath = (
                    "ml/binary_classification/results/xgboost_experiment4/metrics.csv"
                )

                load_and_display_metrics(file_path=initial_result_filepath)

                # Display the confusion matrix
                load_and_plot_confusion_matrix(
                    filename="ml/binary_classification/results/xgboost_experiment4/confusion_matrix.txt"
                )

                st.markdown(
                    """
                ---

                **Model Performance**

                The fourth experiment showed a substantial decline in model performance across all metrics compared to 
                previous iterations. Precision took a significant hit, dipping to a mere 0.09, indicating a marked 
                rise in false positives. Recall, measuring the model's ability to correctly identify machine failures, 
                reduced to 0.61. The F-beta score, a balance between precision and recall, also saw a decline, registering 
                at 0.28. Cohen's Kappa score, gauging the level of agreement between the model's predictions and actual 
                outcomes, fell to a paltry 0.11. Balanced accuracy, which measures the model's performance across classes, 
                did remain relatively decent at 0.71, but the average precision plummeted to 0.07, underlining the model's 
                struggle with an increased rate of false positives.

                **Feature Importance**

                For this iteration, feature importance couldn't be analyzed in the conventional sense due to the 
                application of Kernel PCA, which transforms the original feature space. Thus, direct interpretation 
                of feature significance based on their original definitions becomes challenging. Instead, the two features 
                extracted using Kernel PCA would represent combinations of the original sensor readings, but their specific 
                relation to the original metrics remains abstract.

                **Conclusion**

                The fourth experiment's use of Kernel PCA on raw sensor readings introduced a new dimensionality reduction 
                technique to the modeling process. While the intent was to harness Kernel PCA's power to simplify complex 
                relationships, the results indicate otherwise. The model's deteriorated performance suggests that the transformation 
                may have lost vital information, or perhaps the xgboost model isn't well-suited to work with the 
                PCA-transformed features. Future endeavors might need to re-examine the approach, perhaps by combining Kernel 
                PCA with a different modeling algorithm or by adjusting the number of components extracted.

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
