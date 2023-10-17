# Self-organization of nonlinearly coupled neural fluctuations into synergistic population codes

This repo contains the Python implementation for the recurrent moment neural network (MNN) in the paper [Self-organization of nonlinearly coupled neural fluctuations into synergistic population codes](https://direct.mit.edu/neco/article/35/11/1820/117580/Self-Organization-of-Nonlinearly-Coupled-Neural).

Run the recurrent moment neural network (MNN) through `bump_attractor.py`:

```sh
python bump_attractor.py
```
You can observe the emergence of a bump attractor with special covariance structures. Additionally, you have the option to vary the value of mu_s within the range of 0.9 to 1.2 in order to generate the system in different phases:

```sh
python bump_attractor.py --mu_s: value of the mu_s
```
Feel free to change other settings of the network including the weight parameters (w_e, w_i, d_e, d_i), the std of the noise (sigma_s), and the initial states.
