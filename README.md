# BankCustomerChurnPredictor


## Overview

BankCustomerChurnPredictor is a machine learning project aimed at predicting whether a bank customer will stay with the bank or leave (churn). The project utilizes a dataset containing various attributes related to bank customers, such as credit score, account balance, number of products used, and more, to determine the likelihood of customer churn.


## Dataset

The project uses the bank customer churn dataset, which includes the following attributes:

- **Customer ID**: A unique identifier for each customer
- **Surname**: The customer's surname or last name
- **Credit Score**: A numerical value representing the customer's credit score
- **Geography**: The country where the customer resides (France, Spain, or Germany)
- **Gender**: The customer's gender (Male or Female)
- **Age**: The customer's age
- **Tenure**: The number of years the customer has been with the bank
- **Balance**: The customer's account balance
- **NumOfProducts**: The number of bank products the customer uses (e.g., savings account, credit card)
- **HasCrCard**: Whether the customer has a credit card (1 = yes, 0 = no)
- **IsActiveMember**: Whether the customer is an active member (1 = yes, 0 = no)
- **EstimatedSalary**: The estimated salary of the customer
- **Exited**: Whether the customer has churned (1 = yes, 0 = no)


## Project Structure
The project is organized as follows:

```
BankCustomerChurnPredictor/
├── data/
│   ├──train/
│   └── test/
|
├── notebooks/
│   └── BankCustomerChurnPredictor.ipynb
|
├── src/
│   └── preprocessing.py
│   └── model.py
│   └── prediction.py
|
├── README.md
|
├── requirements.txt
|
└── models/
    ├── BankCustomerChurnPredictor.pkl
    └── _BankCustomerChurnPredictor.tf
```


## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/BankCustomerChurnPredictor.git
cd BankCustomerChurnPredictor
```

2. Create a virtual environment and activate it:
```
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```
Install the required dependencies:
```

### Usage

1. Preprocess the data:
```
python src/preprocessing.py
```

2. Train the model:
```
python src/model.py
```

3. Make predictions:
```
python src/prediction.py
```


## Models
This project uses the following models: The models have been trained on the bank customer churn dataset. The models are:

- BankCustomerChurnPredictor.pkl
- _BankCustomerChurnPredictor.tf


## Frontend
The frontend of the project is built with Next.js, a React framework. The frontend provides the following features:
- **Data Loading**: Load training and testing datasets.
- **Model Training**: Train the Random Forest model on the provided datasets.
- **Model Evaluation**: Evaluate the model using accuracy, classification report, and confusion matrix.
- **Model Retraining**: Retrain the model via an API endpoint.
- **Visualizations**: Display evaluation metrics and confusion matrix.
- **Predictions**: Make predictions on new data.


## Technologies
The project uses the following technologies:
- **Frontend**: Next.js, React
- **Backend**: Python, FastAPI
- **Machine Learning**: scikit-learn, TensorFlow
- **Visualization**: Matplotlib, Seaborn


## Frontend Link to the repository
[BankCustomerChurnPredictor](https://github.com/kaybrian/churn-prediction.git)


## live link to the project
[BankCustomerChurnPredictor](https://churn-prediction-two.vercel.app/)



## Author

- [Kayongo Johnson Brian](https://github.com/kaybrian)
