import pickle
import numpy as np
import argparse
from network import rnn
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_neurons', type=int, default=400, help='number of neurons')
    parser.add_argument('--timesteps', type=int, default=400, help='timesteps')
    parser.add_argument('--we', type=float, default=15, help='we')
    parser.add_argument('--wi', type=float, default=6, help='wi')
    parser.add_argument('--de', type=float, default=.5, help='de')
    parser.add_argument('--di', type=float, default=1., help='di')
    parser.add_argument('--mu_s', type=float, default=0.94, help='mu_s')
    parser.add_argument('--sigma_s', type=float, default=0.1, help='sigma_s')
    parser.add_argument('--dt', type=float, default=0.5, help='dt')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    num_neurons = args.num_neurons
    timesteps = args.timesteps
    dt = np.array(args.dt)

    we = args.we
    wi = args.wi
    de = args.de
    di = args.di
    mu_s = args.mu_s
    sigma_s = args.sigma_s

    init_condition = [np.zeros(num_neurons), 0.1 * np.random.rand(num_neurons),
                      0.0001 * np.eye(num_neurons)]
    init_condition[0][175:225]=1

    init_condition = [init.numpy() for init in init_condition]

    rMNN = rnn(num_neurons, we=we, wi=wi, de=de, di=di, init_condition=init_condition)
    for t in range(timesteps):
        rMNN.run(mu_s, sigma_s, dt=dt)
    u, s, c = rMNN.u, rMNN.s, rMNN.cov

    plt.plot(u)
    plt.imshow(c,vmax=1,vmin=-1)
    plt.colorbar()
    plt.show()
