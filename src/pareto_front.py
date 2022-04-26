import numpy as np


def is_pareto_front(values):
    """
    copied from https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    Find the pareto-efficient points
    :param values: An (n_points, n_cols) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    values = np.array(values)
    is_efficient = np.ones(values.shape[0], dtype=bool)
    for i, c in enumerate(values):
        is_efficient[i] = np.all(np.any(values[:i] > c, axis=1)) and np.all(np.any(values[i + 1:] > c, axis=1))
    return is_efficient


def sample_2d_plot():
    import matplotlib.pyplot as plt

    np.random.seed(42)
    size = 1000
    x = np.random.random(size) * 10
    y = 1 / x * np.random.random(size)

    val = np.vstack([x, y]).T
    print(val)
    print(val.shape)
    mask = is_pareto_front(val)
    pareto = val[mask, :]
    non_pa = val[~mask, :]
    plt.scatter(pareto[:, 0], pareto[:, 1], c='lightcoral', alpha=0.5, label='Pareto Front')
    plt.scatter(non_pa[:, 0], non_pa[:, 1], c='deepskyblue', alpha=0.5, label='Non Pareto')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    sample_2d_plot()
