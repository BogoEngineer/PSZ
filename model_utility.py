import psycopg2
import pandas as pd
from sklearn.preprocessing import LabelEncoder


mu = []
std = []


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

        sql_query = '''SELECT * FROM car WHERE leasing = false AND loan = false'''
        self.postgres_cursor.execute(sql_query)
        self.data = pd.DataFrame(self.postgres_cursor.fetchall(), columns=config['Data']['Columns'].split(','))

        print("PRE CHANGE------------------------------------")
        print(self.data[:10].to_string())
        print("PRE CHANGE------------------------------------")

        self.filter_data(config['Data']['Filter'].split(','))

        print("FILTER------------------------------------")
        print(self.data[:10].to_string())
        print("FILTER------------------------------------")

        self.encode_data(config['Data']['Encoding'].split(','))

        print("ENCODING------------------------------------")
        print(self.data[:10].to_string())
        print("ENCODING------------------------------------")

        self.normalize()

        print("END------------------------------------")
        print(self.data[:10].to_string())
        print("END------------------------------------")

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
            self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()  # mean normalization

    # def h(x, theta):
    #     return np.matmul(x, theta)
    #
    # def cost_function(x, y, theta):
    #     return ((h(x, theta) - y).T @ (h(x, theta) - y)) / (2 * y.shape[0])
    #
    # def gradient_descent(x, y, theta, learning_rate=0.1, num_epochs=10):
    #     m = x.shape[0]
    #     J_all = []
    #
    #     for _ in range(num_epochs):
    #         h_x = h(x, theta)
    #         cost_ = (1 / m) * (x.T @ (h_x - y))
    #         theta = theta - (learning_rate) * cost_
    #         J_all.append(cost_function(x, y, theta))
    #
    #     return theta, J_all
    #
    # def plot_cost(J_all, num_epochs):
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Cost')
    #     plt.plot(num_epochs, J_all, 'm', linewidth="5")
    #     plt.show()
    #
    # def test(theta, x):
    #     x[0] = (x[0] - mu[0]) / std[0]
    #     x[1] = (x[1] - mu[1]) / std[1]
    #
    #     y = theta[0] + theta[1] * x[0] + theta[2] * x[1]
    #     print("Price of house: ", y)
    #
    # x, y = load_data("house_price_data.txt")
    # y = np.reshape(y, (46, 1))
    # x = np.hstack((np.ones((x.shape[0], 1)), x))
    # theta = np.zeros((x.shape[1], 1))
    # learning_rate = 0.1
    # num_epochs = 50
    # theta, J_all = gradient_descent(x, y, theta, learning_rate, num_epochs)
    # J = cost_function(x, y, theta)
    # print("Cost: ", J)
    # print("Parameters: ", theta)
    #
    # # for testing and plotting cost
    # n_epochs = []
    # jplot = []
    # count = 0
    # for i in J_all:
    #     jplot.append(i[0][0])
    #     n_epochs.append(count)
    #     count += 1
    # jplot = np.array(jplot)
    # n_epochs = np.array(n_epochs)
    # plot_cost(jplot, n_epochs)
    #
    # test(theta, [1600, 3])
