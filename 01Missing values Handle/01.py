import pandas as pd
from sklearn.impute import SimpleImputer

dataset1 = pd.read_csv("diabetes-new-missing-values.csv")
dataset2 = pd.read_csv("diabetes-new-missing-values.csv")
dataset3 = pd.read_csv("diabetes-new-missing-values.csv")

print(dataset1.head())
print(dataset1.isnull().sum())

print("------------Mean------------")
simple_mean = SimpleImputer(strategy='mean')
dataset1['Glucose'] = simple_mean.fit_transform(dataset1[['Glucose']])
dataset1['BloodPressure'] = simple_mean.fit_transform(dataset1[['BloodPressure']])
dataset1['SkinThickness'] = simple_mean.fit_transform(dataset1[['SkinThickness']])
dataset1['Insulin'] = simple_mean.fit_transform(dataset1[['Insulin']])
print(dataset1.head())


print("------------Median------------")
simple_median = SimpleImputer(strategy='median')
dataset2['Glucose'] = simple_median.fit_transform(dataset2[['Glucose']])
dataset2['BloodPressure'] = simple_mean.fit_transform(dataset2[['BloodPressure']])
dataset2['SkinThickness'] = simple_mean.fit_transform(dataset2[['SkinThickness']])
dataset2['Insulin'] = simple_mean.fit_transform(dataset2[['Insulin']])
print(dataset2.head())


print("------------Mode------------")
simple_mode = SimpleImputer(strategy='most_frequent')
dataset3['Glucose'] = simple_mode.fit_transform(dataset3[['Glucose']])
dataset3['BloodPressure'] = simple_mean.fit_transform(dataset3[['BloodPressure']])
dataset3['SkinThickness'] = simple_mean.fit_transform(dataset3[['SkinThickness']])
dataset3['Insulin'] = simple_mean.fit_transform(dataset3[['Insulin']])
print(dataset3.head())

