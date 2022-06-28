from visualization import DataVisualization
from analysis import DataAnalysis
from model_utility import ModelUtility
import configparser

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    # file_names = config['FileNames']
    #
    # # Data Analysis
    # data_analysis_config = config['DataAnalysis']
    # max_parameters = data_analysis_config['max_parameters'].split(',')
    # data_analysis = DataAnalysis(config)
    #
    # for method in data_analysis_config:
    #     if method == 'max_parameters': continue
    #     if data_analysis_config.getboolean(method):
    #         result = getattr(data_analysis.__class__, method)(data_analysis)
    #
    # data_analysis.close()
    #
    # # Data Visualization
    # data_visualization_config = config['DataVisualization']
    # data_visualization = DataVisualization(config)
    #
    # for method in data_visualization_config:
    #     if data_visualization_config.getboolean(method):
    #         getattr(data_visualization.__class__, method)(data_visualization)
    #
    # data_visualization.close()

    mu = ModelUtility(config)

    mu.linear_regression()
    # mu.k_nearest_neighbours()

