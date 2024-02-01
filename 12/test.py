import mlflow
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
logged_model = 'runs:/4ff0ce7817474c36b4b17e380bec4ceb/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
diabetes = datasets.load_diabetes()

X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(loaded_model.predict(pd.DataFrame(X_test)))