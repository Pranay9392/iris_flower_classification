# model.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Save the trained model to a file
joblib.dump(knn, 'iris_model.pkl')