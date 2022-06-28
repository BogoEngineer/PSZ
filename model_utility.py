import psycopg2
import pandas as pd
import numpy as np
import scipy.spatial
from numpy import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

mu = []
std = []


def h(x, theta):
    ret = np.matmul(x, theta)
    ret.columns = ['price']
    return ret


def cost_function(x, y, theta):
    return ((h(x, theta) - y.to_frame())**2).sum() / (2 * y.shape[0])


def gradient_descent(x, y, theta, learning_rate=0.1, num_epochs=10):
    m = x.shape[0]
    J_all = []

    for _ in range(num_epochs):
        h_x = h(x, theta)

        diff = h_x - y.to_frame()
        cost_ = (1 / m) * (x.T @ diff)  # dJ/dTheta
        theta = theta - (learning_rate) * cost_
        J_all.append(cost_function(x, y, theta)[0])

    return theta, J_all


def plot_cost(J_all, num_epochs):
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(num_epochs, J_all)
    plt.show()


def test_regression(theta, x_test, y_test):
    predictions = theta.T @ x_test.T
    prediction_mean = predictions.mean(axis=1)
    ssr = ((predictions-y_test)**2).to_numpy().sum()
    sst = ((predictions.T-prediction_mean)**2).to_numpy().sum()

    r2_manual = 1 - ssr/sst
    print("Estimated r2: ", r2_manual)

    r2 = r2_score(y_test, predictions.T)
    print("Real r2: ", r2)

def test_knn(prediction, real):
    counter = 0
    for p, r in zip(prediction, real):
        if p == r:
            counter += 1

    return counter / len(prediction) * 100


logging_map = {}


def log_progress(index, length):
    percentage = int(index / length * 100)
    if percentage % 5 == 0 and percentage not in logging_map:
        print('-' * int(percentage / 5) + ' ' + str(percentage) + '% done')
        logging_map[percentage] = True


class ModelUtility:

    def __init__(self, config):
        postgres_config = config['Postgres']
        self.postgres_connection = psycopg2.connect(database=postgres_config['Database'],
                                                    user=postgres_config['Username'],
                                                    password=postgres_config['Password'],
                                                    host=postgres_config['Host'],
                                                    port=postgres_config['Port'])
        self.postgres_cursor = self.postgres_connection.cursor()
        self.file_names = config['FileNames']['DataVisualization'].split(',')

        self.data_limit = config['Regression']['data_limit']
        sql_query = '''SELECT * FROM car WHERE leasing = false AND loan = false LIMIT ''' + self.data_limit
        self.postgres_cursor.execute(sql_query)
        self.data = pd.DataFrame(self.postgres_cursor.fetchall(), columns=config['Data']['Columns'].split(','))
        # print("PRE CHANGE------------------------------------")
        # print(self.data[:10].to_string())
        # print("PRE CHANGE------------------------------------")

        self.filter_data(config['Data']['Filter'].split(','))

        # print("FILTER------------------------------------")
        # print(self.data[:10].to_string())
        # print("FILTER------------------------------------")

        self.encode_data(config['Data']['Encoding'].split(','))

        # print("ENCODING------------------------------------")
        # print(self.data[:10].to_string())
        # print("ENCODING------------------------------------")

        self.normalize()

        # print("END------------------------------------")
        # print(self.data[:10].to_string())
        # print("END------------------------------------")

        self.learning_rate = config['Regression']['learning_rate']
        self.num_epochs = config['Regression']['num_epochs']

        training_columns = self.data.columns.values[:-1]
        self.x, self.y = self.data[training_columns], self.data['price']

        self.classes = {}
        for class_ in config['ClassificationClasses']:
            self.classes[class_] = [int(x) for x in config['ClassificationClasses'][class_].split('-')]

        self.split = int(float(config['Regression']['split']) * self.data.__len__())
        self.metrics = [x for x in config['ClassificationDistances'] if config['ClassificationDistances'][x] == 'True'][
            0]

        self.k = int(config['Classification']['numer_of_neighbours'])

        self.theta = []

    def filter_data(self, needed):
        columns = list(set(needed) ^ set(self.data.columns))
        self.data.drop(columns, axis=1, inplace=True)

    def encode_data(self, columns):
        self.data['seats'] = self.data['seats'].apply(lambda x: int(x[0]))
        self.data['doors'] = self.data['doors'].apply(lambda x: int(x[0]))

        columns = list(set(columns) & set(self.data.columns))  # intersection - common columns

        label_encoder = LabelEncoder()
        for column in columns:
            self.data[column] = label_encoder.fit_transform(self.data[column])

    def normalize(self):
        for column in self.data:
            if column == 'price': continue
            # self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[
            #     column].std()  # mean normalization

            # self.data[column] = (self.data[column] - min(self.data[column])) / (max(self.data[
            #     column]) - min(self.data[column]))  # -1 to 1

            self.data[column] = 2 * (self.data[column] - min(self.data[column])) / (max(self.data[column]) - min(
                self.data[column])) - 1  # 0 to 1

    def linear_regression(self):
        # x_train = self.x[:self.split]
        # y_train = self.y[:self.split]
        # x_test = self.x[self.split:]
        # y_test = self.y[self.split:]
        # y = np.reshape(y.to_numpy(), (self.data['price'].count(), 1))
        # x = np.hstack((np.ones((x.shape[0], 1)), x))

        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.005, random_state=42)

        x_train.columns = [''] * len(x_train.columns)
        x_test.columns = [''] * len(x_test.columns)

        self.theta = random.random_sample((x_train.shape[1], 1))
        self.theta, J_all = gradient_descent(x_train, y_train, self.theta, float(self.learning_rate), int(self.num_epochs))

        # J = cost_function(x_train, y_train, self.theta)
        # print("Cost: ", J)
        print("Parameters: ", self.theta)

        # for testing and plotting cost
        # plot_cost(J_all, [i for i in range(10)])

        test_regression(self.theta, x_test, y_test)

    def k_nearest_neighbours(self):
        y = self.y.apply(lambda x: self.get_knn_class(x))

        # x_train = self.x[:self.split]
        # y_train = y[:self.split]
        #
        # x_test = self.x[self.split:]
        # y_test = y[self.split:]

        x_train, x_test, y_train, y_test = train_test_split(
            self.x, y, test_size=0.005, random_state=42)

        predictions = np.array([])
        counter = 0
        for index, instance in x_test.iterrows():
            log_progress(counter, x_test.__len__())
            counter += 1
            top_k_neighbours = [x.split(',')[0] for x in self.get_distance_vector(x_train, y_train, instance)]
            prediction_dict = {}

            for class_ in top_k_neighbours:
                if class_ in prediction_dict:
                    prediction_dict[class_] += 1
                else:
                    prediction_dict[class_] = 1

            max_appearances = max(prediction_dict, key=lambda x: prediction_dict[x])
            prediction = '???'
            for key in prediction_dict:
                if key == max_appearances:
                    prediction = key
                    break

            # print(prediction_dict)
            predictions = np.append(predictions, prediction)

            # print([y_train[i] for i in np.argpartition(distance_vector, self.k)])

        result = test_knn(predictions, y_test)

        print(result)

    def get_distance_vector(self, x_train, y_train, test_instance):
        distance_vector = np.array([])
        k_nearest_neighbours = np.array([])
        for index, x in x_train.iterrows():
            distance = getattr(scipy.spatial.distance, self.metrics)(x, test_instance)
            if len(k_nearest_neighbours) < self.k:
                k_nearest_neighbours = np.append(k_nearest_neighbours, str(y_train[index]) + ',' + str(distance))
                continue

            max_distance_neighbour = max(k_nearest_neighbours, key=lambda n: float(n.split(',')[1]))
            if distance < float(max_distance_neighbour.split(',')[1]):
                k_nearest_neighbours = k_nearest_neighbours[k_nearest_neighbours != max_distance_neighbour]  # remove furthest neighbour
                k_nearest_neighbours = np.append(k_nearest_neighbours, str(y_train[index]) + ',' + str(distance))

            # distance_vector = np.append(distance_vector,
            #                             getattr(scipy.spatial.distance, self.metrics)(x, test_instance))

        return k_nearest_neighbours

    def get_knn_class(self, price):
        for class_ in self.classes:
            price_range = self.classes[class_]
            if price_range[0] <= price < price_range[1]:
                return class_
