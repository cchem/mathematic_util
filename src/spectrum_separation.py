import numpy as np
from scipy.optimize import curve_fit
from src.normal_distribution_function import PeakNormalizedGaussian

import matplotlib.pyplot as plt


def gaussian_function(x, i0, m0, s0, i1, m1, s1, i2, m2, s2):
    y = np.zeros_like(x)
    y += PeakNormalizedGaussian(i0, m0, s0)(x)
    y += PeakNormalizedGaussian(i1, m1, s1)(x)
    y += PeakNormalizedGaussian(i2, m2, s2)(x)
    return y


def gaussian_separation_sample():
    x = np.linspace(0, 10, 1000)
    i0, i1, i2 = 3, 2, 4
    m0, m1, m2 = 2, 5, 6
    s0, s1, s2 = 1, 2, 1

    y = np.zeros_like(x)
    y += PeakNormalizedGaussian(i0, m0, s0)(x)
    y += PeakNormalizedGaussian(i1, m1, s1)(x)
    y += PeakNormalizedGaussian(i2, m2, s2)(x)

    opt, _ = curve_fit(gaussian_function, x, y)
    y_opt = gaussian_function(x, *opt)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='original')
    ax.plot(x, y_opt, label='optimized')
    plt.show()


if __name__ == '__main__':
    gaussian_separation_sample()
