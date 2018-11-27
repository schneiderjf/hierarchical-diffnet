import tensorflow as tf
import numpy as np


def infectedCascade(cascade, N, T=10):
    inf = np.zeros((N, N))

    c_nodes = [int(cascade[i * 2]) for i in range(len(cascade) // 2)]
    c_times = [cascade[i * 2 + 1] for i in range(len(cascade) // 2)]

    for i in range(len(c_nodes)):
        for j in range(i):
            if cascade[j] < T:
                inf[(c_nodes[i], c_nodes[j])] = c_times[i] - c_times[j]

    return tf.convert_to_tensor(inf)


def uninfectedCascade(cascade, N, T=6):
    nodes = {s for s in range(N)}
    uninf = np.zeros((N, N))

    c_nodes = [int(cascade[i * 2]) for i in range(len(cascade) // 2)]
    c_times = [cascade[i * 2 + 1] for i in range(len(cascade) // 2)]

    for i in range(len(c_nodes)):
        for j in (nodes - set(c_nodes)):
            uninf[c_nodes[i], j] = T - c_times[i]

    return tf.convert_to_tensor(uninf)


def genInfectedTensor(v, numNodes, T):
    tf_infected = None
    for cascade in v:
        if tf_infected is None:
            tf_infected = tf.expand_dims(infectedCascade(cascade,
                                                         numNodes,
                                                         T), 0)
        else:
            tf_infected = tf.concat([tf_infected,
                                     tf.expand_dims(infectedCascade(cascade,
                                                                    numNodes,
                                                                    T), 0)],
                                    axis=0)
    return tf_infected


def genUninfectedTensor(v, numNodes, T):
    tf_uninfected = None
    for cascade in v:
        if tf_uninfected is None:
            tf_uninfected = tf.expand_dims(uninfectedCascade(cascade,
                                                             numNodes,
                                                             T), 0)
        else:
            tf_uninfected = tf.concat([tf_uninfected,
                                       tf.expand_dims(
                                           uninfectedCascade(cascade,
                                                             numNodes,
                                                             T), 0)],
                                      axis=0)
    return tf_uninfected


def f_psi_3(alpha_tensor, infected):
    infected_sign = tf.cast(tf.sign(infected), tf.float32)

    # Row sum infected
    alpha_tensor_row = tf.reduce_sum(tf.multiply(infected_sign, alpha_tensor),
                                     axis=1)

    # Add 1 to 0 entries so log(1)=0
    alpha_tensor_row_zeros = -tf.cast(tf.sign(alpha_tensor_row),
                                      tf.float32) + 1
    return tf.reduce_sum(
        tf.log(tf.add(alpha_tensor_row, alpha_tensor_row_zeros)))


def f_psi_2(alpha_tensor, uninfected):
    return -tf.reduce_sum(tf.multiply(tf.transpose(alpha_tensor),
                                      tf.cast(uninfected, dtype=tf.float32)))


def f_psi_1(alpha_tensor, infected):
    return -tf.reduce_sum(tf.multiply(alpha_tensor,
                                      tf.cast(infected,
                                              dtype=tf.float32)))


class ProbabilityModel():
    def __init__(self, data, numNodes, T):
        self.data = data
        self.sess = tf.Session()
        self.a = None
        self.numNodes = numNodes
        self.T = T

    def map_estimate_BFGS(self, max_iter=1000):
        sess = self.sess
        max_iter = max_iter

        Inf = genInfectedTensor(self.data, self.numNodes, self.T)
        U = genUninfectedTensor(self.data, self.numNodes, self.T)

        U_ph = tf.placeholder(tf.float32, U.shape)
        I_ph = tf.placeholder(tf.float32, Inf.shape)

        B = tf.Variable(tf.random_uniform(U.shape[1:]), dtype=tf.float32)
        alpha_tensor = tf.nn.sigmoid(B)

        psi_1 = tf.map_fn(lambda x: f_psi_1(alpha_tensor, x), I_ph,
                          dtype=tf.float32)
        psi_2 = tf.map_fn(lambda x: f_psi_2(alpha_tensor, x), U_ph,
                          dtype=tf.float32)
        psi_3 = tf.map_fn(lambda x: f_psi_3(alpha_tensor, x), I_ph,
                          dtype=tf.float32)

        log_p = -(tf.reduce_sum(psi_1) +
                  tf.reduce_sum(psi_2) +
                  tf.reduce_sum(psi_3))

        feed_dict = {U_ph: U.eval(session=sess),
                     I_ph: Inf.eval(session=sess)}

        optimizer = tf.contrib.opt.\
            ScipyOptimizerInterface(log_p,
                                    method='L-BFGS-B',
                                    options={'maxiter': max_iter})

        model = tf.global_variables_initializer()

        sess.run(model)
        optimizer.minimize(sess, feed_dict=feed_dict)

        self.a = alpha_tensor.eval(session=sess)
        return self.a
