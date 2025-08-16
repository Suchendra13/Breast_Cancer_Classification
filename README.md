# Breast Cancer Classification using Logistic Regression
               This project focuses on building a machine learning model to classify breast cancer tumors as either malignant or benign. The model is built using a Logistic Regression algorithm and evaluated on the Wisconsin Breast Cancer dataset, which is available in scikit-learn.

# Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Results](#results)
6. [How to Run](#how-to-run)
7. [Libraries Used](#libraries-used)
8. [Acknowledgements](#acknowledgemenets)

## 1. Project Overview
The notebook covers the entire workflow, from loading and exploring the data to preprocessing, training the model, and evaluating its performance.

## 2. Dataset
The project utilizes the Wisconsin Breast Cancer dataset from sklearn.datasets.

**Number of Samples:** 569

**Number of Features:** 30 (e.g., mean radius, mean texture, mean smoothness)

**Target Classes:** Malignant (0) and Benign (1)

* **Target Distribution:**

**Benign (1):** ~63%

**Malignant (0):** ~37%

## 3. Methodology
The project follows these key steps:

**Data Loading and Exploration:** The dataset is loaded, and its basic properties (shape, target distribution) are examined.

**Train-Test Split:** The data is split into an 80% training set and a 20% testing set to ensure the model is evaluated on unseen data.

**Feature Scaling:** The features are scaled using StandardScaler to normalize the data, which helps improve the performance of the Logistic Regression model.

**Model Training:** A Logistic Regression model is trained on the scaled training data.

**Model Evaluation:** The model's performance is assessed using the test set.

## 4. Exploratory Data Analysis (EDA):
       A correlation heatmap is generated to visualize the relationships between the different features in the dataset.

## 5. Results
The trained **Logistic Regression model** performed very well in classifying the tumors.

* **Accuracy:** `97.37%`

**Classification Report:**

               precision    recall  f1-score   support

   malignant       0.98      0.95      0.96        43
      benign       0.97      0.99      0.98        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
**Confusion Matrix:** The confusion matrix heatmap confirms the high accuracy, showing a low number of misclassifications.

## 6. How to Run

To execute this Jupyter Notebook:

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone [https://github.com/Suchendra13/Breast_Cancer_Classification.git](https://github.com/Suchendra13/Breast_Cancer_Classification.git)
    cd Breast_Cancer_LR
    ```
2.  **Ensure you have Jupyter Notebook installed** or use a compatible IDE (e.g., VS Code with Jupyter extensions).
3.  **Install the required Python libraries**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4.  **Open the Jupyter Notebook**:
    ```bash
    jupyter notebook Breast_Cancer_Classification.ipynb
    ```
5.  **Run all cells** in the notebook.

## 7. Libraries Used

* `pandas`
* `numpy`
* `matplotlib.pyplot`
* `seaborn`
* `sklearn` (specifically `datasets`, `model_selection`, `preprocessing`, `decomposition`, `linear_model`, `metrics`)

## 8. Acknowledgements

* The `load_breast_cancer` dataset is provided by scikit-learn.
