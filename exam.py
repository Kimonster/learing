import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import main
from main import LogisticRegression

data = pd.read_csv('D:\\qqdownload\\learnig\\data\\yanweihua.csv')
iris_types = ['setosa','versicolor','virginica']

x_axis = 'PetalLength'
y_axis = 'PetalWidth'

for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type],
                data[y_axis][data['class'] == iris_type],
                label = iris_type)

    
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis,y_axis]].values.reshape((num_examples,2))
y_train = data['class'].values.reshape((num_examples,1))

max_iterations = 10000
polynomial_degree = 0
sinusoid_degree = 0

logisticregression = LogisticRegression(x_train,y_train,polynomial_degree, sinusoid_degree)
# logistic_regression = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
thetas, cost_histories = logisticregression.train(max_iterations)
labels = logisticregression.unique_labels


plt.plot(range(len(cost_histories[0])), cost_histories[0], label =labels[0])
plt.plot(range(len(cost_histories[1])), cost_histories[1], label =labels[1])
plt.plot(range(len(cost_histories[2])), cost_histories[2], label =labels[2])
plt.show()

y_train_predicted = logisticregression.predict(x_train)
precision = np.sum(y_train_predicted == y_train) / y_train.shape[0] * 100
print('precision = {0}%'.format(precision))

x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])
samples = 104

X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)

Z_normal = np.zeros((samples, samples))
Z_yaerror = np.zeros((samples, samples))
Z_xiaerror = np.zeros((samples, samples))


#iris_types = ['setosa','versicolor','virginica']

for x_index,x in enumerate(X):
    for y_index,y in enumerate(Y):
        data = np.array([[x,y]])
        prediction = logisticregression.predict(data)[0][0]
        if prediction == 'setosa':
            Z_normal[x_index, y_index] = 1
        elif prediction == 'versicolor':
            Z_yaerror[x_index, y_index] = 1
        elif prediction == 'virginica':
            Z_xiaerror[x_index, y_index] = 1

for iris_type in iris_types:
    plt.scatter(
        x_train[(y_train == iris_type).flatten(), 0],
        x_train[(y_train == iris_type).flatten(), 1],
        label = iris_type)

plt.contour(X, Y, Z_normal, [0.5], colors = 'blue')
plt.contour(X, Y, Z_yaerror, [0.5], colors = 'red')
plt.contour(X, Y, Z_xiaerror, [0.5], colors = 'green')
# plt.contour(X, Y, Z_normal)
# plt.contour(X, Y, Z_yaerror)
# plt.contour(X, Y, Z_xiaerror)

plt.show()