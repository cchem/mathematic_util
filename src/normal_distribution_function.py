from typing import Sequence
import pytest

import numpy as np


class AreaNormalizedGaussian:
    def __init__(self, intensity, mu, sigma):
        self.intensity = intensity
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: Sequence):
        x = np.array(x)
        coef = self.intensity * 1 / np.sqrt(2 * np.pi * self.sigma ** 2)
        val = np.exp(-(x - self.mu) ** 2 / 2 / self.sigma ** 2)
        return coef * val


class PeakNormalizedGaussian:
    def __init__(self, intensity, mu, sigma):
        self.intensity = intensity
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: Sequence):
        x = np.array(x)

        coef = self.intensity
        val = np.exp(-(x - self.mu) ** 2 / 2 / self.sigma ** 2)
        return coef * val


def test_area():
    x = np.linspace(0, 10, 1001)

    for area in [1, 3, 7]:
        gaussian = AreaNormalizedGaussian(area, 5, 1.2)
        y = gaussian(x)
        dx = x[1] - x[0]
        calculated_area = sum(y * dx)
        assert calculated_area == pytest.approx(area, 1e-4)


def test_peak():
    x = np.linspace(0, 10, 1001)

    for peak in [1, 3, 7]:
        gaussian = PeakNormalizedGaussian(peak, 5, 1.2)
        y = gaussian(x)

        calculated_peak = max(y)
        assert calculated_peak == pytest.approx(peak, 1e-4)
