import pandas as pd
# pandas is for reading files easily
import numpy as np
# numpy is for using arrays(multi-dimensional) , because lists are not so efficient here
import sklearn
# sklearn is a library that has many models and features of machine learning
from sklearn import linear_model
import pickle
# pickle for saving object structures in python
from math import fabs
from math import pow
# matplotlib is for mathematical representation(statistics & calculus) , using animated/static visualizations
# 'style' submodule is for customizing the style of our plot in many predefined style or create ones


# now we read the data using pandas , using the reader of csv files method with a separator ;
data = pd.read_csv('crops-yield.csv', sep=';')
# TIP : after reading the data you can show SOME of them using the data.head(n) method , that shows the
# first n elements
data = data[['AvgRD', 'AvgRF', 'AvgS', 'AvgH', 'AvgT', 'AvgG', 'AvgWS', 'CY']]
# TIP : to get only the attributes you want you pick them in the way above :
# [ ['atr1', 'atr2', ..., 'atrn'] ]
predict = 'CY'


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


clean_dataset(data)

# TIP : this is our LABEL , what we want to get , now it's there in the attributes but we will remove it
# later when we train the model
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
# TIP : we produce numpy.array form pandas.dataframe , drop function removes the 'predict' label which is
# a column (axis=1)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=2)
# this function will shuffle(by default = True) the inputs and outputs splits for training and testing the
# model , with test_size 0.1 , meaning that 0.1 of our data will be tested and 0.9 will be for training
model = linear_model.LinearRegression()
# picking a model from the sub module 'linear_model' and launch it with the linear regression algorithm
model.fit(X_train, Y_train)
# training the model with our data
# AND WE ARE BASICALLY DONE :)


def score_accuracy(true_y, predict_y):
    if len(true_y) == len(predict_y):
        acc_relative_err = 0
        for j in range(len(true_y)):
            acc_relative_err = acc_relative_err + pow((fabs((true_y[j] - predict_y[j]))/fabs((true_y[j]))), 1)
        relative_err = acc_relative_err/len(true_y)
        return (1 - relative_err)*100


data_18 = [[0, 0.1, 267.3, 36.7, 21.3, 22.8, 15.1]]
sum_predict_18 = 0
sum_accuracy = 0
n = 1000
for i in range(n):
    predictions = model.predict(X_test)
    accuracy = score_accuracy(list(Y_test), list(predictions))
    predict_18 = int(model.predict(data_18))
    sum_predict_18 += predict_18
    sum_accuracy += accuracy
avg_predict_18 = sum_predict_18/n
avg_accuracy = sum_accuracy/n

print('The crop Yield in 2018 in AL_Minya was 632 thousand metric tonne, and we predict : ' + str(avg_predict_18))
print('Accuracy of the model is  :  ' + str(avg_accuracy))
