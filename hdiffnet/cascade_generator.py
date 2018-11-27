import tensorflow as tf
import numpy as np
from edward.models import Exponential
import edward as ed
from tqdm import tqdm


class GenerativeModel():
    def __init__(self, alpha, T=None):
        self.alpha = alpha
        self.T = T
        pass

    def build_cascade(self, seed, T=None, alpha=None):

        sess = ed.get_session()
        # Store number of nodes
        if not(T):
            T = self.T

        if not(alpha):
            alpha = self.alpha

        time = Exponential(alpha)

        n = time.shape[0]

        # Transpose times and reduce minimum
        times_T = tf.minimum(tf.transpose(time), T)

        # Initialize transmission times to be max time except for seed node
        transmission = tf.ones(n) * T
        transmission = tf.subtract(transmission, tf.one_hot(seed, n) * T)

        # Continually update transmissions
        for _ in range(n):

            # Tile transmission
            transmission_tiled = tf.reshape(tf.tile(transmission, [n]),
                                            [n, n])

            # Add transposed times and tiled transmissions
            potential_transmission = tf.add(transmission_tiled, times_T)

            # Find minimum path from all new
            potential_transmission_row = tf.reduce_min(potential_transmission,
                                                       reduction_indices=[1])

            # Concatenate previous transmission and potential new transmission
            potential_transmission_stack = tf.\
                stack([transmission, potential_transmission_row], axis=0)

            # Take the min of the original and the potential new transmission
            transmission = tf.reduce_min(potential_transmission_stack,
                                         reduction_indices=[0])

        cascade = sess.run(transmission)

        return cascade

    def build_cascade_series(self, seeds, T=None):
        """

        :param seeds:
        :param T:
        :return:
        """
        result = []
        if T:
            for t in tqdm(range(len(seeds))):
                cascade = self.build_cascade(seeds[t], T[t])
                result.append(cascade)
        else:
            for t in tqdm(range(len(seeds))):
                cascade = self.build_cascade(seeds[t])
                result.append(cascade)

        cascades = np.vstack(result)
        return cascades
