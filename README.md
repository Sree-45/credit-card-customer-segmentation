Customer Segmentation using K-Means Clustering

This project demonstrates how to perform customer segmentation using a credit card dataset. The goal is to group customers based on their spending behavior and financial profiles, enabling more targeted marketing strategies.

Project Overview

Objective: Segment customers into distinct groups using unsupervised learning techniques to uncover patterns in credit card usage and financial behavior.

Data Files:

-   credit_card_dataset.csv: The raw dataset containing credit card data.

-   Clustered_Customer_Data.csv: The output dataset with customers assigned to clusters after segmentation.

Analysis Notebook:

-   Customer_Segmentation-final.ipynb: A Jupyter Notebook that details the complete workflow, including data preprocessing, exploratory analysis, feature engineering, K-Means model training, and cluster visualization.

Model File:

-   kmeans_model.pkl: A serialized (pickled) version of the trained K-Means clustering model for predicting customer segments on new data.

Project Structure

-   credit_card_dataset.csv # Raw credit card dataset.

-   Clustered_Customer_Data.csv # Dataset with customer cluster labels.

-   Customer_Segmentation-final.ipynb # Jupyter Notebook with analysis and model development.

-   kmeans_model.pkl # Pickled K-Means clustering model.

Requirements

-   Python 3.x

-   Jupyter Notebook or JupyterLab

-   Python Libraries:

    -   pandas

    -   numpy

    -   matplotlib

    -   seaborn

    -   scikit-learn

Install the necessary packages with:

pip install pandas numpy matplotlib seaborn scikit-learn

How to Run the Project

1.  Clone the Repository: Use the following commands to clone the repository and navigate into it: git clone <https://github.com/yourusername/your-repository-name.git> cd your-repository-name

2.  Launch the Jupyter Notebook: Open the notebook with: jupyter notebook Customer_Segmentation-final.ipynb Execute each cell in sequence to process data, train the model, and analyze the clusters.

3.  Using the Trained Model: The kmeans_model.pkl file can be loaded to predict clusters for new data. For example:

    import pickle import pandas as pd

    Load the pre-trained model
    ==========================Customer Segmentation using K-Means Clustering
==============================================

Overview
--------

This project uses a credit card dataset to perform customer segmentation. The goal is to group customers based on their spending behavior and financial profiles, allowing for more targeted marketing strategies.

Project Objectives
------------------

-   **Identify distinct customer segments:** Use unsupervised learning (K-Means clustering) to uncover patterns in credit card usage and financial behavior.

-   **Provide actionable insights:** Enable data-driven decision-making for targeted marketing strategies.

Data Files
----------

-   **credit_card_dataset.csv:**\
    Raw dataset containing credit card data.

-   **Clustered_Customer_Data.csv:**\
    Processed dataset with assigned cluster labels.

Analysis and Modeling
---------------------

-   **Customer_Segmentation-final.ipynb:**\
    Jupyter Notebook that documents the complete workflow, including:

    -   Data preprocessing

    -   Exploratory data analysis

    -   Feature engineering

    -   K-Means model training

    -   Cluster visualization

-   **kmeans_model.pkl:**\
    Serialized (pickled) version of the trained K-Means clustering model used for predicting customer segments on new data.

Project Structure
-----------------

-   `credit_card_dataset.csv` -- Raw credit card data

-   `Clustered_Customer_Data.csv` -- Data with cluster labels

-   `Customer_Segmentation-final.ipynb` -- Notebook with analysis and model training

-   `kmeans_model.pkl` -- Saved K-Means model

Requirements
------------

-   **Python 3.x**

-   **Jupyter Notebook** or **JupyterLab**

-   **Python Libraries:**

    -   pandas

    -   numpy

    -   matplotlib

    -   seaborn

    -   scikit-learn

Installation
------------

Install the necessary packages using pip:

bash

Copy code

`pip install pandas numpy matplotlib seaborn scikit-learn`

How to Run the Project
----------------------

### 1\. Clone the Repository

Clone this repository to your local machine:

bash

Copy code

`git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name`

### 2\. Launch the Notebook

Open the Jupyter Notebook by running:

bash

Copy code

`jupyter notebook Customer_Segmentation-final.ipynb`

Execute the notebook cells sequentially to process the data, train the model, and analyze clusters.

### 3\. Using the Trained Model

To predict clusters for new data, load the pre-trained model with the following code:

python

Copy code

`import pickle
import pandas as pd

# Load the pre-trained model
with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

# Prepare new data (ensure the format matches the training data)
new_data = pd.DataFrame({
    # 'feature1': [value1],
    # 'feature2': [value2],
    # Add additional features as needed
})

# Predict the cluster
predicted_cluster = kmeans_model.predict(new_data)
print("Predicted cluster:", predicted_cluster)`

Results and Insights
--------------------

-   **Insights into Customer Behavior:**\
    The analysis reveals distinct customer segments based on financial behavior.

-   **Targeted Marketing Strategies:**\
    Identified segments can be used to tailor marketing approaches for each customer group.

-   **Data-Driven Decision Making:**\
    The clustering model supports informed decisions in customer relationship management.

    with open('kmeans_model.pkl', 'rb') as file: kmeans_model = pickle.load(file)

    Prepare new data (ensure the format matches the training data)
    ==============================================================

    new_data = pd.DataFrame({ # 'feature1': [value1], # 'feature2': [value2], # Add additional features as needed })

    Predict the cluster
    ===================

    predicted_cluster = kmeans_model.predict(new_data) print("Predicted cluster:", predicted_cluster)

Results and Insights

The segmentation analysis reveals insights into customer behavior and identifies key segments that can be targeted with customized marketing strategies. This clustering approach aids in understanding the diversity of customer profiles, supporting more informed decisions in customer relationship management.
