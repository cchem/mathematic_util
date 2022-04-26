from typing import Union
from typing import List

import numpy as np


def is_pareto_front(values, direction: Union[str, List[str]] = 'minimize'):
    """
    copied from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    Find the pareto-efficient points
    :param values: An (n_points, n_cols) array
    :param direction: List of 'minimize' or 'maximize'
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """

    # 文字列の場合は全部の列を同様の処理で。
    if isinstance(direction, str):
        direction = [direction for _ in range(values.shape[1])]

    # 大文字小文字を意識しなくて済むように。
    direction = [i.lower() for i in direction]

    # directionは目的とする変数の数と同じだけ与えられる必要がある。
    if len(direction) != values.shape[1]:
        raise RuntimeError('Size of array is not correct')

    # 'minimize'と'maximize'以外の文字列が入っていた場合はじく。
    if not all([i in {'minimize', 'maximize'} for i in list(set(direction))]):
        raise RuntimeError('Keyword of "direction" must be "minimize" or "maximize"')

    # 最大化するのか最小化するのかの計算
    coef = [1 if i == 'minimize' else -1 for i in direction]
    values = np.array(values)
    for i in range(values.shape[1]):
        values[:, i] *= coef[i]

    # pareto frontの判定
    is_efficient = np.ones(values.shape[0], dtype=bool)
    for i, c in enumerate(values):
        is_efficient[i] = np.all(np.any(values[:i] > c, axis=1)) \
                          and np.all(np.any(values[i + 1:] > c, axis=1))
    return is_efficient


def sample_2d_plot():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    size = 1000
    x = np.random.random(size) * 10
    y = 1 / x * np.random.random(size)

    val = np.vstack([x, y]).T
    mask = is_pareto_front(val, direction='maximize')
    pareto = val[mask, :]
    non_pa = val[~mask, :]
    plt.scatter(non_pa[:, 0], non_pa[:, 1], c='deepskyblue', alpha=0.5, label='Non Pareto')
    plt.scatter(pareto[:, 0], pareto[:, 1], c='lightcoral', alpha=0.5, label='Pareto Front')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    sample_2d_plot()
