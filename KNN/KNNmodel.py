import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("/Users/egambarampanneerselvam/PycharmProjects/MachineLearning/KNN/car.data")

print(data.head())

lableEncode = preprocessing.LabelEncoder()

buying = lableEncode.fit_transform(list(data["buying"]))
maint = lableEncode.fit_transform(list(data["maint"]))
doors = lableEncode.fit_transform(list(data["doors"]))
persons = lableEncode.fit_transform(list(data["persons"]))
lug_boot = lableEncode.fit_transform(list(data["lug_boot"]))
safety = lableEncode.fit_transform(list(data["safety"]))
dclass = lableEncode.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(dclass)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train, y_train)

KNNmodel = KNeighborsClassifier(n_neighbors=9)

KNNmodel.fit(x_train, y_train)

accuracy = KNNmodel.score(x_test, y_test)

print(accuracy)

predicted = KNNmodel.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]

for i in range(len(predicted)):
    print("Predicted values: ", names[predicted[i]], "Data: ", x_test[i], names[y_test[i]])
    n = KNNmodel.kneighbors([x_test[i]], 7, True)
    print("The distance: ", n)
