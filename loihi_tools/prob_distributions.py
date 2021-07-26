"""
Copyright © 2018 University of Zurich.

"""

import numpy as np


def normal1d_density(x, mu=0, sigma=1, normalized=True):
    """
    Args:
        x, (float): x values at which density is calculated.
        mu (float): Mean of Gaussian.
        sigma (float, optional): Standard deviation Gaussian distribution.
        normalized (bool, optional): Description

    Returns:
        float: (probability) density at a specific distance to the mean of a Gaussian distribution.
    """
    dist_x = x - mu
    if normalized:
        f = 1 / np.sqrt(2 * np.pi * sigma ** 2)
    else:
        f = 1
    density = f * np.exp(-(1 / 2) * (dist_x / sigma) ** 2)
    # print(density)
    return density


def normal2d_density(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1, rho=0, normalized=True):
    """
    Args:
        x, y (float): x and y values at which density is calculated.
        mu_x, mu_y (float): Means of Gaussian in x and y dimension.
        sigma_x, sigma_y (float, optional): Standard deviations of Gaussian distribution.
        rho (float, optional): correlation coefficient of the 2 variables.
        normalized (bool, optional): Description

    Returns:
        float: normal (probability) density at a specific distance to the mean (dist_x,dist_y) of a 2d distribution.
    """
    dist_x = x - mu_x
    dist_y = y - mu_y
    if normalized:
        f1 = (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2)))
    else:
        f1 = 1
    f2 = -(1 / (2 * (1 - rho ** 2)))
    fx = dist_x / sigma_x
    fy = dist_y / sigma_y
    fxy = 2 * fx * fy * rho
    density = f1 * np.exp(f2 * (fx ** 2 + fy ** 2 - fxy))
    # print(density)
    return density


def normal2d_density_array(nrows, ncols, sigma_x=1, sigma_y=1, rho=0, mu_x=None, mu_y=None, normalized=True):
    """Returns a 2d normal density distributuion array of size (nrows, ncols).

    Args:
        ncols, nrows (int): size of the output array.
        sigma_x (int, optional): Description
        sigma_y (int, optional): Description
        rho (float, optional): correlation coefficient of the 2 variables.
        mu_x (None, optional): Description
        mu_y (None, optional): Description
        normalized (boolean, optional): If you set this to False, it will no longer be a
            probability density with an integral of one, but the maximum amplitude (in the middle of the bump) will be 1.
        sigma_x and sigma_y (float, optional): Standard deviations of Gaussian distribution.
        mu_x and mu_y (float, optional): Means of Gaussian distribution.

    Returns:
        ndarray: Description

    Note:
        as the function is vectorized, this is the same as:

        >>> density = np.zeros((nrows+1, ncols+1))
        >>>     i = -1
        >>>     for dx in dist_x:
        >>>         i+= 1
        >>>         j = -1
        >>>         for dy in dist_y:
        >>>         j+=1
        >>>         density[j,i] = normal2d_density(dx, dy, sigma_x, sigma_y, rho, normalized)
    """
    x = np.arange(0, nrows)
    y = np.reshape(np.arange(0, ncols), (ncols, 1))
    # y = x[:, np.newaxis]

    if mu_x is None:
        mu_x = nrows // 2
    if mu_y is None:
        mu_y = ncols // 2

    density = normal2d_density(
        x, y, mu_x, mu_y, sigma_x, sigma_y, rho, normalized)

    return density


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = normal2d_density_array(100, 100, 20, 10, 0.5)
    # img = normal2d_density_array(100, 100, 20, 20, 0.5, 50,50,0)
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    # the sum is almost one, if the sigmas are much smaller than the range
    print(np.sum(img))
