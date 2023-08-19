import streamlit as st


def classification_introduction():
    st.markdown(
        """
    # Introduction

    Welcome to this predictive modeling project. In this work, the focus is 
    on the critical realm of industrial machinery. In an increasingly automated 
    world, the efficient and reliable operation of machinery is a crucial factor 
    in maintaining productivity and safety across numerous sectors. Unexpected 
    machinery failure can result in significant financial loss, downtime, and 
    potential safety hazards.

    Through the application of machine learning, this project aims to provide a 
    solution to this pressing issue. Machine learning offers the promise of predictive 
    maintenance, using historical data to anticipate potential problems before they arise. 
    This project is not merely about prediction, but also about understanding the 
    causal factors contributing to machinery failure.

    ## Project Objectives

    The primary objectives of this project are as follows:

    1. **Predictive Maintenance**: The foremost goal is to predict potential machinery 
    failures. The ability to forecast failure allows for the timely scheduling of 
    maintenance tasks, preventing unexpected breakdowns, enhancing operational efficiency, 
    prolonging machine service life, and significantly reducing the financial 
    impact of machinery downtime.

    2. **Fault Diagnosis**: Once a potential failure has been predicted, the next 
    step is to determine the probable causes. Understanding the underlying reasons 
    for failure enables targeted repairs and modifications to be made, prevents 
    repeated breakdowns, and provides invaluable insight into machine operation 
    and performance.

    To achieve these objectives, various machine learning models are being employed, 
    meticulously examining their performance based on a range of factors and features 
    that might affect machinery operation. The ultimate goal is to construct a model 
    that is not only accurate and robust but also offers interpretability, facilitating 
    its practical application in predictive maintenance and fault diagnosis.

    This project represents a significant contribution to the field of predictive 
    maintenance and fault diagnosis. However, its potential applications extend far 
    beyond industrial machinery. The methodologies and insights gained through this 
    work could be applied to numerous other domains, underscoring the extraordinary 
    potential and versatility of machine learning.

    You are invited to explore this project further, as it delves deeper into the 
    fascinating world of machine learning and predictive modeling.

    """
    )

    st.markdown(
        """
    ## Performance Metrics

    Throughout this project, various metrics are used to evaluate the performance of 
    the machine learning models. Understanding these metrics is crucial to comprehend 
    the results and the decisions made during the model development and selection process.

    **Precision**

    Imagine we're trying to find diamonds in a coal mine. Precision asks, "Of all the rocks 
    we predicted to be diamonds, how many were actually diamonds?" High precision means when 
    we predict something is a diamond, we're usually right.

    Formula:
    `Precision = TP / (TP + FP)`

    **Recall**

    On the other hand, recall asks, "Of all the actual diamonds in the mine, how many 
    did we correctly identify?" High recall means we found most of the diamonds, even 
    if we also misclassified some coal as diamonds.

    Formula:
    `Recall = TP / (TP + FN)`

    **F-beta Score**

    This is a way of combining precision and recall into a single number. The F-beta score 
    (where beta > 1) gives more importance to recall than to precision. This means we care 
    more about not missing any actual diamonds than we do about occasionally misclassifying 
    coal as a diamond. In this project, the beta value is set to 2, implying that the cost 
    of missing a machine failure (false negatives) is considered to be twice as significant 
    as the cost of unnecessary maintenance (false positives). This decision is based on the 
    assumption that the consequences of machine failure are more severe than the costs 
    associated with preventive maintenance.

    Formula:
    `F-beta = (1 + beta^2) * (Precision * Recall) / ((beta^2 * Precision) + Recall)`

    **Cohen's Kappa**

    Let's say you and a friend both look at the same rocks and guess whether they're diamonds 
    or coal. Cohen's Kappa measures how much you both agree on your guesses, taking into 
    account the amount of agreement that would happen just by chance. A high Cohen's Kappa 
    means the model's predictions agree well with the actual values, more than what would be 
    expected by chance alone.

    Formula:
    `Kappa = (observed accuracy - expected accuracy) / (1 - expected accuracy)`

    **Balanced Accuracy**

    This is an average of recall obtained on each class. It's useful when the classes are 
    imbalanced (like if there are a lot more coal rocks than diamonds). Balanced accuracy 
    treats all classes equally, regardless of their size. This means we care just as much 
    about correctly identifying diamonds as we do about correctly identifying coal.

    Formula:
    `Balanced Accuracy = (Recall(Class1) + Recall(Class2)) / 2`

    When evaluating our machine learning model, we use these metrics to ensure that our 
    model is not only making accurate predictions, but also the right kind of accurate 
    predictions. We want to be sure that when our model tells us a rock is a diamond, 
    it's usually correct (high precision), and that it can find most of the diamonds 
    in the mine (high recall). The F-beta score, and balanced accuracy help us 
    balance these competing priorities, while Cohen's Kappa confirms that our model's 
    predictions are better than just random guessing.
    """
    )

    st.markdown(
        """
    ## Models Used in the Project

    In this project, several different machine learning models are used. Here's a 
    brief overview of each.

    **Logistic Regression**

    Logistic Regression is a statistical model that uses a logistic function to 
    model a binary dependent variable. In other words, it's a way to predict 
    the odds of being in one class or the other. It's a simple, fast model that 
    is easy to understand and explain, but it may not capture complex patterns 
    as well as some other models.

    **Decision Trees**

    A Decision Tree is a flowchart-like structure in which each internal node 
    represents a "test" on an attribute, each branch represents the outcome of 
    the test, and each leaf node represents a class label. The paths from root 
    to leaf represent classification rules. They are simple to understand and 
    visualize, can handle both numerical and categorical data, but they can 
    easily overfit to the training data.

    **Random Forests**

    Random Forests is a type of ensemble learning method, where a group of weak 
    models come together to form a strong model. In Random Forests, multiple 
    decision trees are created and then voted to make a final prediction. The 
    model is less likely to overfit than a single decision tree, and it can 
    handle a large number of features without needing feature elimination.

    **XGBoost**

    XGBoost stands for eXtreme Gradient Boosting. It's an implementation of gradient 
    boosting machines that pushes the limits of computing power for boosted trees 
    algorithms. It's known for its speed and performance, out-of-the-box performance, 
    and it's the go-to method for many winning teams of machine learning competitions.

    """
    )

    st.markdown(
        """
    ## Feature Importance

    In machine learning, we are often interested in identifying the features that contribute 
    most to our model's predictive power. These are called "important features". By focusing 
    on important features, we can simplify our models (making them easier to interpret and 
    less prone to overfitting), improve computational efficiency, and gain insights into 
    the underlying processes that generate the data.

    There are several ways to measure feature importance, and the best method can depend on 
    the specific model being used. For tree-based models like Decision Trees, Random Forests, 
    and XGBoost, feature importance can be calculated directly from the structure of the trained 
    model. These models make their predictions based on a series of decisions, like "Is Feature 
    A greater than some value?" The more often a feature is used in these decisions, 
    the more important it is.

    In this project, the feature importances are calculated for each model and then saved. 
    This allows for a comparison of which features each model finds important, and how this 
    influences the model's performance.


    """
    )

    st.markdown(
        """
    ## Experimental Design

    The experimental design process for this project is structured around the concept of 
    iterative model development and enhancement. The process unfolds as follows:

    1. **Benchmark Model Training:** The initial step involves training and tuning each of 
    the chosen machine learning models using the raw measurements from the dataset. These 
    models are intended to serve as benchmark models, providing a baseline level of performance 
    against which to compare the performance of improved models.

    2. **Benchmark Model Evaluation:** The performance of the benchmark models is assessed 
    using a suite of metrics. This evaluation provides a comprehensive understanding of 
    each model's strengths and weaknesses.

    3. **Feature Extraction:** After the establishment of the benchmark models, feature 
    extraction methods are applied to the raw data. Feature extraction can assist in 
    reducing the dimensionality of the data and may reveal important structures or patterns 
    that can enhance model performance.

    4. **Enhanced Model Training:** Utilizing the processed data from the feature extraction 
    stage, each of the models is retrained and tuned. The goal at this stage is to see if the 
    feature extraction methods can improve the models' performance relative to the benchmark models.

    5. **Enhanced Model Evaluation:** Finally, the performance of the enhanced models is 
    evaluated using the same suite of metrics as before. By comparing these results to the 
    benchmark results, the impact of the feature extraction methods on model performance can be 
    quantified.

    This experimental design process allows for systematic exploration and quantification 
    of the potential benefits of feature extraction for this classification task. The results 
    obtained will provide valuable insights that can guide future efforts to improve the 
    performance of machine learning models in similar tasks.

    """
    )

    st.markdown(
        """
    ## Interpreting the Results

    Interpreting the results of these experiments involves a comparative analysis of each model's 
    performance across various stages of the experimental design process. 

    The performance of each model, both in the benchmark phase and the enhanced phase (post-feature 
    extraction), will be evaluated using a suite of metrics including the confusion matrix, 
    precision, recall, F-beta score, Cohen's Kappa, and balanced accuracy. 

    Here's how to interpret these results:

    1. **Benchmark vs. Enhanced Model Performance:** Compare the performance of each model before 
    and after feature extraction. If a model's performance improves after feature extraction, 
    this suggests that the feature extraction methods were beneficial for that particular model.

    2. **Model Comparisons:** Evaluate the performance of different models against each other. 
    Some models may perform better on certain metrics than others. By comparing across models, 
    the strengths and weaknesses of each model can be better understood.

    3. **Metrics Analysis:** Examine each of the performance metrics. Some metrics may be more 
    important than others based on the specific context and goals of the project. For instance, 
    if reducing false negatives is a priority, models with high recall or F-beta scores 
    (with beta > 1) might be preferred.

    4. **Feature Importance:** Review the feature importance rankings generated by each model. 
    Features that consistently rank high across multiple models are likely to be particularly 
    important for the task.

    The goal is not just to identify the model that performs best, but also to 
    understand why it performs best and how different aspects of the data and feature extraction 
    methods contribute to its performance. Through this understanding, more effective models 
    can be developed in the future.
    """
    )

    st.markdown(
        """
    ## Project Summary

    This project aims to use machine learning to predict potential machinery failures and diagnose 
    faults in industrial machinery. The primary goals are to prevent unexpected breakdowns, enhance 
    operational efficiency, prolong machine service life, and reduce the financial impact of machinery downtime.

    Various machine learning models are employed, including Logistic Regression, Decision Trees, 
    Random Forests and XGBoost. Each of these models offers different 
    strengths and will be evaluated based on a suite of metrics including precision, recall, F-beta score, 
    Cohen's Kappa, and balanced accuracy.

    An important aspect of this project is understanding feature importance. Identifying the features 
    that most contribute to our model's predictive power can simplify models, improve computational 
    efficiency, and provide insights into the underlying processes that generate the data.

    The experimental design process involves an iterative model development and enhancement approach. 
    It begins with training benchmark models using raw measurements from the dataset. These models are 
    then evaluated, and feature extraction methods are applied to the raw data. The models are retrained 
    and evaluated again using the processed data. This process allows for a systematic exploration of 
    the potential benefits of feature extraction for this classification task.

    Interpreting the results involves comparing the performance of each model before and after feature 
    extraction, evaluating the performance of different models against each other, examining each of 
    the performance metrics, and reviewing the feature importance rankings generated by each model.

    The goal of this project is not just to identify the best-performing model, but also to understand 
    why it performs best and how different aspects of the data and feature extraction methods 
    contribute to its performance. The methodologies and insights gained could be applied to numerous 
    other domains, underscoring the versatility of machine learning.

    """
    )
