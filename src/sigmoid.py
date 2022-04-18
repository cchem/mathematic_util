from typing import Sequence
import numpy as np
import pytest


class Sigmoid:
    def __init__(self, gain, center):
        self.gain = gain
        self.center = center

    def __call__(self, x: Sequence):
        x = np.array(x)
        return 1 / (1 + np.exp(-self.gain * (x - self.center)))


class RangeSigmoid(Sigmoid):
    """
    rangeの左端でおおよそ-1, 右端でおおよそ1になるシグモイド。
    todo: おおよそ0の有効数字を指定できるようにしたい。
    """

    def __init__(self, left, right):
        gain = 10 / abs(right - left)
        center = (right + left) / 2
        super().__init__(gain, center)


def test_range_sigmoid():
    left = -4
    right = 6
    sigmoid = RangeSigmoid(left, right)
    x = np.linspace(left, right, 1001)
    y = sigmoid(x)
    print(y[0])
    print(y[-1])

    assert y[0] + y[-1] == pytest.approx(1)
    assert y[-1] == pytest.approx(1, 1e-2)
