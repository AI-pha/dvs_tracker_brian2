import os
from functools import partial

import numpy as np
from scipy.spatial import distance

flatten = lambda x: np.reshape(x, (np.prod(x.shape)))

eq_dist_linspace = lambda n_pix: np.linspace(0.5, n_pix - 0.5, n_pix) / n_pix

dim2str = lambda dim: str(dim[0]) + 'x' + str(dim[1])

from scipy.spatial import distance_matrix


def create_dist_mat_resize(predims, postdims, normalize='pre'):
    '''

    :param predims:
    :param postdims:
    :param normalize: can be pre, post or both. If it is 'pre', all distances will be given with respect to post
    :return:
    '''
    xbins_pre = np.linspace(1.0 / (2.0 * predims[0]), 1.0 - 1.0 / (2.0 * predims[0]), predims[0])
    ybins_pre = np.linspace(1.0 / (2.0 * predims[1]), 1.0 - 1.0 / (2.0 * predims[1]), predims[1])
    x_pre, y_pre, = np.meshgrid(xbins_pre, ybins_pre)

    if normalize == 'pre':
        x_pre = x_pre.T * postdims[0]
        y_pre = y_pre.T * postdims[1]
    elif normalize == 'post':
        x_pre = x_pre.T * predims[0]
        y_pre = y_pre.T * predims[1]
    else:
        x_pre = x_pre.T
        y_pre = y_pre.T

    xbins_post = np.linspace(1.0 / (2.0 * postdims[0]), 1.0 - 1.0 / (2.0 * postdims[0]), postdims[0])
    ybins_post = np.linspace(1.0 / (2.0 * postdims[1]), 1.0 - 1.0 / (2.0 * postdims[1]), postdims[1])
    x_post, y_post = np.meshgrid(xbins_post, ybins_post)

    if normalize == 'pre':
        x_post = x_post.T * postdims[0]
        y_post = y_post.T * postdims[1]
    elif normalize == 'post':
        x_post = x_post.T * predims[0]
        y_post = y_post.T * predims[1]
    else:
        x_post = x_pre.T
        y_post = y_pre.T

    pre_coordinates = np.asarray(
        list(zip(np.reshape(x_pre, np.prod(x_pre.shape)), np.reshape(y_pre, np.prod(y_pre.shape)))))
    post_coordinates = np.asarray(
        list(zip(np.reshape(x_post, np.prod(x_post.shape)), np.reshape(y_post, np.prod(y_post.shape)))))

    dist_mat = distance_matrix(pre_coordinates, post_coordinates)

    return dist_mat


# predims=(4,2)
# postdims = (4,4)
# distmat = create_dist_mat_resize(predims, postdims)
# plt.figure()
# plt.imshow(distmat)


def create_dist_mat(predims, postdims, dist_metric):
    """
        Precomputes a distance matrix between grid loactions of 2 2d fields
        It stores the result for later usage.
        From the distance matrix, you can get a connection mask and calculate the weight matrix.

        The 2 grids are aligned and rescaled to unity in all dimensions, then, the distance is scaled according
        to the dimensions of the second field (postdims)

    :param predims: dimensions of the first field
    :param postdims: dimensions of the second field
    :param dist_metric: distance metric to apply
    :return:
    """

    filename = './data/dist_matrix_' + dim2str(predims) + '_' + dim2str(postdims) + '_' + dist_metric.__name__ + '.npz'
    if os.path.isfile(filename):
        print('file existst: loading dist mat from file: ', filename)
        with np.load(filename) as loaded:
            dist_mat = loaded['dist_mat']
    else:
        xbins_pre = eq_dist_linspace(predims[0])
        ybins_pre = eq_dist_linspace(predims[1])
        x_pre, y_pre = np.meshgrid(xbins_pre, ybins_pre)

        xbins_post = eq_dist_linspace(postdims[0])
        ybins_post = eq_dist_linspace(postdims[1])
        x_post, y_post = np.meshgrid(xbins_post, ybins_post)

        pre_coordinates = np.asarray(list(zip(flatten(x_pre), flatten(y_pre))))
        post_coordinates = np.asarray(list(zip(flatten(x_post), flatten(y_post))))

        dist_metric = partial(dist_metric, weights=postdims)
        dist_mat = distance.cdist(post_coordinates, pre_coordinates, metric=dist_metric)
        # dist_mat = distance_matrix(pre_coordinates,post_coordinates)

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
            print('created directory', os.path.dirname(filename), 'in', os.path.abspath(os.path.dirname(filename)))
        np.savez(filename, dist_mat=dist_mat)

    return dist_mat


if __name__ == '__main__':
    numX = 32
    numY = 32

    dist_mat = create_dist_mat(predims=(numX, numX), postdims=(numY, numY),
                               dist_metric=wrapped_euclidean_squares)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(dist_mat)

    plt.figure()
    plt.imshow(np.reshape(dist_mat, (numX, numX, numY, numY))[0, 0, :, :])

    plt.figure()
    plt.imshow(np.reshape(dist_mat, (numX, numX, numY, numY))[5, 5, :, :])

    np.sum(np.reshape(dist_mat, (numX, numX, numY, numY))[0, 0, :, :])
