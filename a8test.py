import csv
from neural import *
import sklearn

def round_to_nearest_half(number, n):
    return round(number * 2) / 2

iris_data = []

with open('iris.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for line in csv_reader:
        if len(line) > 0:
            features = list(map(float, line[:-1]))
            label = [line[-1]] if len(line) > 1 else []
            iris_data.append((features, label))


for i in range(len(iris_data)):
    if iris_data[i][1][0] == 'Iris-setosa':
        iris_data[i][1][0] = 0
    elif iris_data[i][1][0] == 'Iris-versicolor':
        iris_data[i][1][0] = 0.5
    elif iris_data[i][1][0] == 'Iris-virginica':
        iris_data[i][1][0] = 1

iris_nn = NeuralNet(4, 1, 1)

iris_nn.train(iris_data)

iris_res = iris_nn.test_with_expected(iris_data)

for i in range(len(iris_res)):
    iris_res[i][2][0] = round_to_nearest_half(iris_res[i][2][0], 1)
    print(f"{iris_res[i][2][0]}, {round_to_nearest_half(iris_res[i][2][0], 1)}")

corrCount = 0
wroCount = 0

for i in range(len(iris_res)):
    if round_to_nearest_half(iris_res[i][1][0], 1) == iris_res[i][2][0]:
        corrCount += 1
    else:
        wroCount += 1


denom = corrCount + wroCount
print(f"Percent wrong = {round(((wroCount / denom)*100), 1)}%")
print(f"Percent correct = {round(((corrCount / denom)*100), 1)}%")