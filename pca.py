import numpy as np

def PCA(X , num_components):
    # Center the data around the origin
    X_meaned = X - np.mean(X , axis = 0)
     
    # Create a covariance matrix
    cov_mat = np.cov(X_meaned , rowvar = False)
     
    # Calculate the eigen vectors and eigen values
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    # Sort the eigen vectors based on the eigen values (descending order)
    # The eigen vector with the largest eigen value is the most important principle component. 
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
     
    # Get a handful of components
    eigenvector_subset = sorted_eigenvectors[:,:num_components]
     
    # Project the dataset onto the new PCs. 
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced

import pandas as pd
 
#Get the IRIS dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
 
#prepare the data
x = data.iloc[:,0:4]
 
#prepare the target
target = data.iloc[:,4]
 
#Applying it to PCA function
mat_reduced = PCA(x , 2)
 
#Creating a Pandas DataFrame of reduced Dataset
principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])
 
#Concat it with target variable to create a complete Dataset
principal_df = pd.concat([principal_df , pd.DataFrame(target)] , axis = 1)

import seaborn as sb
import matplotlib.pyplot as plt
 
plt.figure(figsize = (6,6))
sb.scatterplot(data = principal_df , x = 'PC1',y = 'PC2' , hue = 'target' , s = 60 , palette= 'icefire')
plt.show()