import numpy as np
import sys
import mnn_core.maf

maf = mnn_core.maf.MomentActivation()

class rnn():
    def __init__(self, num_neurons, we=5, wi=1.5, de=0.5, di=1.5, init_condition=None):
        self.num_neurons = num_neurons
        self.weight = self.mexi_mat(we, wi, de, di) * 2 * np.pi / num_neurons
        if init_condition:
            self.u, self.s, self.cov = init_condition
        else:
            x = np.roll(np.arange(0.0, 2 * np.pi, 2 * np.pi / self.num_neurons), int(self.num_neurons / 2))
            self.u = np.exp(np.cos(x) - 1).squeeze(0)
            self.s = np.exp(np.cos(x) - 1).squeeze(0)
            self.cov = np.eye(len(x)).squeeze(0)

    def mexi_mat(self, we, wi, de, di):
        x = np.arange(0.0, 2 * np.pi, 2 * np.pi / self.num_neurons)
        func = lambda x, d: np.exp((np.cos(x) - 1) / d / d)
        y = we * func(x, de) - wi * func(x, di)
        W = np.zeros([self.num_neurons, self.num_neurons])
        for i in range(self.num_neurons):
            W[i, :] = np.roll(y, i)
        return W

    def run(self, u_ext=0.0, s_ext=0., dt=np.array(1)):
        u_bar = np.matmul(self.weight, self.u.reshape(-1, 1))
        cov_bar = np.matmul(np.matmul(self.weight, self.cov), self.weight.T)
        s_bar = np.sqrt(np.diagonal(cov_bar))  # TODO: check if right
        u_bar += u_ext
        s_bar = np.sqrt(s_bar * s_bar + s_ext * s_ext)
        u_bar = u_bar.reshape(-1)
        s_bar = s_bar.reshape(-1)
        u_activated = maf.mean(u_bar, s_bar)
        s_activated = maf.std(u_bar, s_bar)[0]
        psi = maf.chi(u_bar, s_bar) / s_bar * s_activated
        print(self.s.min())
        cov_activated = cov_bar * psi.reshape(-1, 1) * psi.reshape(1, -1)
        cov_activated[np.arange(self.num_neurons), np.arange(self.num_neurons)] = s_activated * s_activated
        self.u, self.s, self.cov = (1 - dt) * self.u + dt * u_activated, (1 - dt) * self.s + dt * s_activated, (
                1 - dt) * self.cov + dt * cov_activated
        self.cov[np.arange(self.num_neurons), np.arange(self.num_neurons)] = self.s * self.s
