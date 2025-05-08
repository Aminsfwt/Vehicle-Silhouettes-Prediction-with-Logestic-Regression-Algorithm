import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.inspection import permutation_importance


# Read the data
data = pd.read_csv('E:/ML/Intro to Deep Learning/Labs/Codes/Applied ML/Vehicle Silhouettes/vehicle.csv')

#encode categorical variables
le = LabelEncoder()
data['class'] = le.fit_transform(data['class'])

#split the data into features and target variable
X = data.drop(['class'], axis=1)
y = data['class']

# Drop rows with any NaN values in the dataset 
# Get indices of non-NaN rows
non_nan_indices = X.dropna(axis=0).index
X = X.loc[non_nan_indices]
y = y.loc[non_nan_indices]

#split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

#scale the data using standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

"""
train the model with Logistic Regression
Logistic Regression is a linear model for binary classification 
that uses the logistic function to model the probability of a binary outcome.
It is a simple and efficient algorithm that works well for linearly separable data.
Logistic Regression can also be used for multiclass classification using the softmax function.
The softmax function is a generalization of the logistic function 
that maps a vector of real numbers to a probability distribution over multiple classes.

LBFGS (Limited-memory BFGS) is an optimization algorithm for parameter estimation
It's the default solver in scikit-learn because it's:
Generally very robust
Works well for multinomial classification
Handles L2 regularization effectively
"""
logreg_model = LogisticRegression(     ## Use multinomial logistic regression for multiclass classification
    solver= 'lbfgs',               ## Use the LBFGS solver for optimization
    max_iter= 1000                 ## Set the maximum number of iterations for convergence
)

logreg_model.fit(X_train_scaled, y_train)

#Evaluate the model
y_pred = logreg_model.predict(X_test_scaled)

#comute the accuracy of the model
model_accurcey = accuracy_score(y_test, y_pred)
#print(f"Logistic Regression Model Accuracy: {model_accurcey:.2f}")

"""
#compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")

#comute classification report
print(classification_report(y_test, y_pred))
"""

#train using GridSearchCV to find optimal parameters
"""
Key advantages of LBFGS:

Memory efficient since it stores only a few vectors instead of a full Hessian matrix
Works well for small to medium-sized datasets
Generally converges faster than simpler optimizers like gradient descent

Alternative solvers include:
'newton-cg': For multinomial loss
'sag' and 'saga': For large datasets
'liblinear': Default for binary classification
If you're working with binary classification and a relatively small dataset, 
'lbfgs' is often a good default choice as it provides 
a good balance between convergence speed and memory usage.
"""
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],          # Regularization strength
    'solver': ['lbfgs','liblinear', 'saga'],      # Solver compatible with L1/L2
    'max_iter': [100, 500, 1000]                  # Ensure convergence
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

#Get the best model after hyperparameter tunning
best_model = grid_search.best_estimator_

#Get the accuracy after hyperparameter tuning
y_pred = best_model.predict(X_test_scaled)
#print(f"Accuracy after hyperparameter tuning: {accuracy_score(y_test, y_pred):.2f}")

