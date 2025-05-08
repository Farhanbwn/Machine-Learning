import numpy as np  # numpy is used for numpy arrays
import pandas as pd # pandas is used for creating the data table in a structural way
from sklearn.preprocessing import StandardScaler # StandardScaler is used for standardizing the data
from sklearn.model_selection import train_test_split # train_test_split is used for splitting the data into train and test parts
from sklearn.metrics import accuracy_score # accuracy_score is used to see the accuracy score of the model
from sklearn.impute import SimpleImputer

# ### Read the dataset to pandas dataframe
dataset = pd.read_csv("diabetes_original.csv")

# ### Separating the Dementia status column from the main dataset
dataset_new = dataset.drop(columns='Outcome', axis=1)
Dementia_status = dataset['Outcome']
x = dataset_new
y = Dementia_status

# ### Data Standardization 
Stand = StandardScaler()
x = x.values
Stand.fit(x)
Stand_data = Stand.transform(x)

# ### Splitting the standardized data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=2)

# # Printing the number of train and test data
# print(x.shape, x_train.shape, x_test.shape)

# ### kNN Algorithm Implementation
def knn_predict(x_train, y_train, x_test, k=5):
    predictions = []
    for test_point in x_test:
        # Calculate the distances from the test point to all training points
        distances = np.linalg.norm(x_train - test_point, axis=1)
        
        # Get the indices of the k-nearest neighbors
        k_nearest_indices = np.argsort(distances)[:k]
        
        # Get the corresponding labels for these k-nearest neighbors
        k_nearest_labels = y_train.iloc[k_nearest_indices]  # Using iloc for pandas Series indexing
        
        # Perform majority voting to get the predicted label
        prediction = k_nearest_labels.value_counts().idxmax()  # Majority vote
        predictions.append(prediction)
    
    return np.array(predictions)

# ### Predict on training data
x_train_predit = knn_predict(x_train, y_train, x_train, k=5)
training_data_acc = accuracy_score(x_train_predit, y_train)
print("Accuracy score of x_train is ", training_data_acc)

# ### Predict on test data
x_test_predit = knn_predict(x_train, y_train, x_test, k=5)
testing_data_acc = accuracy_score(x_test_predit, y_test)
print("Accuracy score of x_test is ", testing_data_acc)
