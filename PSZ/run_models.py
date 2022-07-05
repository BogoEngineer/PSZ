from model_utility import ModelUtility
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

features = config['Data']['columns'].split(',')

print("Insert values for features: ('x' or 0 to skip this feature)")
feature_values = []
parameters = config['TopRegressionModel']
# for feature in parameters:
#     value = input(feature + ': ')
#
#     if feature not in  parameters:
#         continue
#     if value == 'x':
#         feature_values.append(0)
#     else:
#         feature_values.append(value)

feature_values = ['Fiat', 2006, 281600, 'Dizel', 1910, 120, 'Prednji', False, 'Siva', True, 'Domaće tablice ', False, '?', 'Pančevo']

fltr = []
parameter_values = []
for feature in parameters:
    parameter_values.append(parameters[feature])
    fltr.append(feature)

model_utility = ModelUtility(config, True, fltr, feature_values)
model_utility.compute_regression_result(parameter_values)