from NeuralNetwork import NeuralNetwork
import numpy as np


# {'identity', 'logistic', 'tanh', 'relu'}
def simulate(layers, iterations, function, file_name, file):
    nn = NeuralNetwork()

    losses = []
    str_layers = ''.join(map(str, map(lambda x: f'{x}', layers)))

    directory = f'simulations/{file}'

    start = 1
    end = 11

    real_y = []
    sim_y = []

    for th in range(start, end):
        x, y, estimated_y, loss, regr = nn.run_test(
            layers,
            iterations,
            function,
            f'{file_name}.npy'
        )

        plot_name = f'{file_name}__{str_layers}__{iterations}__{function}__{th}th'

        print(f'{directory}/{plot_name}')
        nn.plot(regr, estimated_y, x, y, f'{directory}/{plot_name}')

        losses.append(loss)

        real_y = y
        sim_y = estimated_y

    avrg = np.average(losses)
    std = np.std((real_y, sim_y))

    calc_name = f'avrg_std__{str_layers}__{iterations}__{function}.txt'
    f = open(f'{directory}/{calc_name}', "w")
    f.write(f'average: {avrg}\nstandard deviation: {std}')
    f.close()

    return avrg, std


def run_first_test():
    layers = tuple([5])
    simulate(layers, 10000, 'relu', 'teste1', 1)

    layers = tuple([5, 5])
    simulate(layers, 10000, 'relu', 'teste1', 1)

    layers = tuple([5, 10])
    simulate(layers, 20000, 'relu', 'teste1', 1)


def run_second_test():
    layers = tuple([5])
    simulate(layers, 10000, 'identity', 'teste2', 2)

    layers = tuple([5, 5])
    simulate(layers, 10000, 'identity', 'teste2', 2)

    layers = tuple([5, 10])
    simulate(layers, 20000, 'identity', 'teste2', 2)

    layers = tuple([5, 10])
    simulate(layers, 30000, 'identity', 'teste2', 2)

    layers = tuple([10, 10])
    simulate(layers, 30000, 'identity', 'teste2', 2)


def run_third_test():
    layers = tuple([5])
    simulate(layers, 10000, 'relu', 'teste3', 3)

    layers = tuple([5, 10])
    simulate(layers, 20000, 'relu', 'teste3', 3)

    layers = tuple([5, 10])
    simulate(layers, 50000, 'relu', 'teste3', 3)


def run_fourth_test():
    layers = tuple([5])
    simulate(layers, 10000, 'tanh', 'teste4', 4)

    layers = tuple([5, 5])
    simulate(layers, 10000, 'tanh', 'teste4', 4)

    layers = tuple([10, 10])
    simulate(layers, 30000, 'tanh', 'teste4', 4)

    layers = tuple([20, 20])
    simulate(layers, 60000, 'tanh', 'teste4', 4)

    layers = tuple([20, 20, 20])
    simulate(layers, 60000, 'tanh', 'teste4', 4)


def run_fifth_test():
    layers = tuple([10])
    simulate(layers, 5000, 'tanh', 'teste5', 5)
    simulate(layers, 5000, 'relu', 'teste5', 5)

    layers = tuple([10, 10])
    simulate(layers, 10000, 'tanh', 'teste5', 5)
    simulate(layers, 10000, 'relu', 'teste5', 5)

    layers = tuple([30, 30])
    simulate(layers, 100000, 'tanh', 'teste5', 5)
    simulate(layers, 100000, 'relu', 'teste5', 5)

    layers = tuple([30, 30, 30])
    simulate(layers, 200000, 'tanh', 'teste5', 5)
    simulate(layers, 200000, 'relu', 'teste5', 5)

    layers = tuple([50, 50, 50, 50])
    simulate(layers, 200000, 'tanh', 'teste5', 5)
    simulate(layers, 200000, 'relu', 'teste5', 5)

    layers = tuple([60, 60, 60, 60])
    simulate(layers, 300000, 'tanh', 'teste5', 5)
    simulate(layers, 300000, 'relu', 'teste5', 5)


def main():
    run_first_test()
    print('\n\n')
    run_second_test()
    print('\n\n')
    # run_third_test()
    # print('\n\n')
    # run_fourth_test()
    # print('\n\n')
    # run_fifth_test()
    # print('\n\n')


if __name__ == '__main__':
    main()
