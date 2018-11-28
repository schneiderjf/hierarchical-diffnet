import tensorflow as tf
import numpy as np

from tensorflow_probability import edward2 as ed


class GenerativeModel():
    """
    This class defines the generative model for the graphical model
    """
    def __init__(self, alpha, T=None):
        self.alpha = alpha
        self.T = T
        pass

    def build_cascade(self, seed, T=None, alpha=None):
        """
        This function generates a cascades given a seed node, a time horizon
        and a parameter matrix.
        :param seed: int | seed node parameter
        :param T: float
        :param alpha: np.array
        :return: np.array | cascade
        """

        sess = tf.Session()
        # Store number of nodes
        if T is None:
            T = self.T

        if alpha is None:
            alpha = self.alpha

        time = ed.Exponential(alpha)

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
        Builds a list of cascades, when feeded with a list of seed nodes and
        times, seed list and T list must have the same lenght
        :param seeds: list
        :param T: list
        :return: np.array | an array of cascades
        """
        result = []
        if T:
            for t in range(len(seeds)):
                cascade = self.build_cascade(seeds[t], T[t])
                result.append(cascade)
        else:
            for t in range(len(seeds)):
                cascade = self.build_cascade(seeds[t])
                result.append(cascade)

        cascades = np.vstack(result)
        return cascades

    def build_topic_cascades(self, seeds, T, topics):
        """
        Builds a set of cascades, given a topic polarity. The number of
        cascades are determined by the lenght of T and seeds
        :param seeds: list
        :param T: list
        :param topics: np.array | indicator matrix for topic per seed
        :return: np.array | cascades
        """

        alpha = self.alpha

        cascade = []

        for i in range(len(seeds)):

            topic_pos = topics[i, 0]
            if topic_pos == 0:
                tmpCascade = self.build_cascade(seeds[i], T[i], alpha[0])

            elif topic_pos == 1:
                tmpCascade = self.build_cascade(seeds[i], T[i], alpha[1])

            cascade.append(tmpCascade)

        return np.array(cascade)
