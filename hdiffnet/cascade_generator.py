import tensorflow as tf
import numpy as np

from tensorflow_probability import edward2 as ed

from tqdm import tqdm


class GenerativeModel():
    def __init__(self, alpha, T=None):
        self.alpha = alpha
        self.T = T
        pass

    def build_cascade(self, seed, T=None, alpha=None):

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

    def build_topic_cascades(self, seeds, T, topics):

        sess = tf.Session()

        alpha = self.alpha
        nodes = alpha[1].shape[0]

        cascade = []
        for i in tqdm(range(len(seeds))):

            topic_pos = topics[i,0]
            if topic_pos == 0:
                tmpCascade = self.build_cascade(seeds[i], T[i], alpha[0])

            elif topic_pos == 1:
                tmpCascade = self.build_cascade(seeds[i], T[i], alpha[1])

            # order = tmpCascade.argsort()
            # times = tmpCascade[order]
            #
            # cascadeList = []
            #
            # for i in range(nodes):
            #     if times[i] >= T[i]: break
            #     cascadeList.append(float(order[i]))
            #     cascadeList.append(times[i])

            cascade.append(tmpCascade)

        return np.array(cascade)
