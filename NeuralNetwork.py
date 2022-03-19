import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


class NeuralNetwork:
    def __init__(self):
        pass

    def run_test(self, layers, iterations, function, file_name):
        arquivo = np.load(file_name)
        x = arquivo[0]
        y = np.ravel(arquivo[1])

        regr = MLPRegressor(hidden_layer_sizes=layers,
                            max_iter=iterations,
                            activation=function,  # {'identity', 'logistic', 'tanh', 'relu'},
                            solver='adam',
                            learning_rate='adaptive',
                            n_iter_no_change=50)

        regr = regr.fit(x, y)

        estimated_y = regr.predict(x)
        loss = regr.loss_

        return x, y, estimated_y, loss, regr

    def plot(self, regr, y_est, x, y, fig_name='figure'):
        plt.figure(figsize=[14, 7])

        # plot curso original
        plt.subplot(1, 3, 1)
        plt.plot(x, y)

        # plot aprendizagem
        plt.subplot(1, 3, 2)
        plt.plot(regr.loss_curve_)

        # plot regressor
        plt.subplot(1, 3, 3)
        plt.plot(x, y, linewidth=1, color='yellow')
        plt.plot(x, y_est, linewidth=2)

        plt.savefig(fig_name)

        plt.close()
