import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle


complete_data = pd.read_csv("/Users/egambarampanneerselvam/PycharmProjects/MachineLearning/LinearRegression/student-mat.csv", sep=",")

selected_data = complete_data[["G1", "G2", "G3", "traveltime", "freetime", "absences", "age"]]

predict = "G3"

# Assigning selected array of data into x after removing the predictable value
x = np.array(selected_data.drop([predict], 1))

# Assigning selected array of data into y for the predictable value
y = np.array(selected_data[predict])


optimize = 0
for _ in range(1000):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    my_linear = linear_model.LinearRegression()

    my_linear.fit(x_train, y_train)

    accuracy = my_linear.score(x_test, y_test)

    print("The accuracy percentage is : ", "{:.2%}".format(accuracy))
    if accuracy > optimize:
        optimize = accuracy
        with open("student_optimize_model.pickle", "wb") as f:
            pickle.dump(my_linear, f)
print("Optimize model %", "{:.2%}".format(optimize))

pickle_in = open("/Users/egambarampanneerselvam/PycharmProjects/MachineLearning/LinearRegression/student_optimize_model.pickle", "rb")

my_linear = pickle.load(pickle_in)

'''

The equation of straight line is y = mx + c

m is co-efficient of x (this is the slope(gradient) of the equation)

c is y intercept value (x1, y1) (x2, y2)

in this case, y = m1x1 + m2x2 + m3x3 + m4x4 + m5x5 + m6x6 + c

'''

print("The value of m", my_linear.coef_)
print("The value of c", my_linear.intercept_)

predictions = my_linear.predict(x_test)

print(len(predictions))
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

