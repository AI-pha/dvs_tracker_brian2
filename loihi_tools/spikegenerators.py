import numpy as np

from loihi_tools.prob_distributions import normal2d_density_array


def delete_doublets(spiketimes, indices):
    len_before = len(spiketimes)
    buff_data = np.vstack((spiketimes, indices)).T
    buff_data[:, 0] = buff_data[:, 0].astype(int)
    _, idx = np.unique(buff_data, axis=0, return_index=True)
    buff_data = buff_data[np.sort(idx), :]

    spiketimes = buff_data[:, 0]
    indices = np.asarray(buff_data[:, 1], dtype=int)

    print(len_before - len(spiketimes), 'spikes removed')
    print(len(spiketimes) / len_before * 100, '% spikes removed')
    return spiketimes, indices


def gaussian2d_poisson_spikes(mean_index, std_index, num_timesteps, numX, numY, frequency):
    """
    Poisson (temporal), Gaussian (spatial) bump input

    :param mean_index:
    :param std_index:
    :param num_timesteps:
    :return:

    #mean_index = (16, 9)
    #std_index = 0.5
    #num_timesteps = 100
    # numX = 30; numY = 30
    # frequency = 5
    """
    gauss_mat = normal2d_density_array(numX, numY, sigma_x=std_index, sigma_y=std_index,
                                       mu_x=mean_index[0], mu_y=mean_index[1], normalized=True)
    frequency = np.asarray([gauss_mat, ]) * frequency
    spike_mat = np.random.poisson(frequency.T, (numX, numY, num_timesteps))

    x, y, timesteps = np.where(spike_mat)
    indices = np.ravel_multi_index((x, y), dims=(numX, numY))

    return indices, timesteps
