# Customer Churn Prediction

This project aims to predict customer churn using machine learning techniques. The model uses historical data from customers to identify those who are likely to churn (i.e., leave the service), which helps businesses implement targeted retention strategies and improve customer satisfaction.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Source](#data-source)
3. [Setup and Installation](#setup-and-installation)
4. [Technologies Used](#technologies-used)
5. [Model Development](#model-development)
6. [Model Evaluation](#model-evaluation)
7. [How to Use](#how-to-use)
8. [Contributing](#contributing)
9. [License](#license)

## Project Overview
Customer churn is one of the major challenges in subscription-based businesses. The goal of this project is to predict which customers are most likely to churn, allowing businesses to proactively retain valuable customers. We use a variety of machine learning models to predict churn based on customer demographic data, service usage characteristics, and account details.

## Data Source
The dataset used in this project is the **IBM Customer Churn Dataset**, which contains 7043 customer records and 21 features, including:
- **customerID**: Unique identifier for each customer
- **gender**: Gender of the customer (Male/Female)
- **SeniorCitizen**: Whether the customer is a senior citizen (1: Yes, 0: No)
- **tenure**: Number of months the customer has been with the company
- **Churn**: Target variable indicating whether the customer churned (1: Yes, 0: No)

The dataset is used to develop and evaluate machine learning models for customer churn prediction.

## Setup and Installation
Follow these steps to set up and run the project locally:

1. Clone this repository:
   ```bash
   git clone https://github.com/shahabaalam/customer-churn-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd customer-churn-prediction
   ```

3. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

## Technologies Used
- **Python**: Programming language used for data analysis and model development
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning library for model development
- **XGBoost**: Gradient boosting for advanced models
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebooks**: For interactive data exploration and model development

## Model Development
We developed three machine learning models for churn prediction:
- **Logistic Regression**: A baseline model for classification problems.
- **Random Forest**: An ensemble method using multiple decision trees.
- **XGBoost**: A high-performance gradient boosting method.

Each model was trained on the customer dataset, and their performance was evaluated based on precision, recall, accuracy, and ROC-AUC score.

## Model Evaluation
The models were evaluated using key metrics:
- **Precision**: Accuracy of positive predictions (churned customers)
- **Recall**: The ability to correctly identify churned customers
- **F1-Score**: A weighted average of precision and recall
- **Accuracy**: The overall correctness of the model
- **ROC-AUC**: Measures the model's ability to distinguish between churned and non-churned customers

### Performance Summary:
| Model               | Accuracy | ROC-AUC | Precision (Churned) | Recall (Churned) | F1-Score (Churned) |
|---------------------|----------|---------|---------------------|------------------|--------------------|
| **Logistic Regression** | 81%      | 0.86    | 0.67                | 0.58             | 0.62               |
| **Random Forest**      | 80%      | 0.82    | 0.66                | 0.54             | 0.60               |
| **XGBoost**           | 78%      | 0.83    | 0.60                | 0.54             | 0.57               |

## How to Use
1. **Clone the repository** and install dependencies.
2. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
3. **Train the models**: Execute the cells in the notebook to train the models and evaluate their performance.
4. **Predict churn**: Use the trained models to predict whether new customers will churn based on their features.

## Contributing
If you'd like to contribute to this project:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.

## License
This project is licensed under the MIT License.

## References:
1. **Customer Churn Dataset**: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. **Helpful Notebook**: [Customer Churn Prediction Notebook](https://www.kaggle.com/code/bhartiprasad17/customer-churn-prediction/noteboo)
```
