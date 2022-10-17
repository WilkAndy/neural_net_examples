# Examples of using neural networks to fit data

This repository provides examples of fitting data using neural networks.  Simple, synthetic data is created, and then modelled using neural networks of differing complexity in order to illustrate how the architecture of the neural network impacts the fit.

## Software requirements

To run the examples in this repository, python3 is needed, along with the following libraries:

- numpy
- pandas
- matplotlib

## The synthetic data

Two sets of synthetic data - "smooth" and "noisy" - are generated using the python script [generate_data.py](generate_data.py):

```
ep = 0.01
noise = 0.2
xvals = np.arange(0, 1 + ep, ep)
smooth = 0.5 + 0.5 * np.sin(1.5 * np.pi * xvals)
noisy = smooth + noise * (np.random.rand(len(xvals)) - 0.5)
```

which is shown visually in the figure below

![The synthetic data](data.svg)

## More



